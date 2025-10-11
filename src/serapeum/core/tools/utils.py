"""Utilities for converting Python callables into tool schemas and executing tools.

This module provides:

- Docstring: Utilities to parse parameter descriptions from function docstrings.
- FunctionArgument and FunctionConverter: Helpers to convert a Python function
  signature into a Pydantic model schema suitable for tool metadata.
- ToolExecutor: A safe execution harness for tools, with support for
  synchronous/asynchronous calls, optional single-argument auto-unpacking, and
  standardized error outputs.

The examples in this module are written as doctests and are designed to be
self-contained (using local stubs), so they can be executed without any external
infrastructure.

See Also:
    - serapeum.core.tools.models.BaseTool
    - serapeum.core.tools.models.ToolOutput
    - serapeum.core.llm.base.ToolSelection
"""

import re
import json
from dataclasses import dataclass
from inspect import signature, Parameter
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    get_origin,
    get_args,
    Annotated,
    TYPE_CHECKING,
    Sequence
)
import datetime

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from serapeum.core.tools.models import BaseTool, ToolOutput, adapt_to_async_tool
from serapeum.core.tools.models import ToolSelection


if TYPE_CHECKING:
    from serapeum.core.tools.models import BaseTool


class Docstring:
    """Helper to introspect a callable's docstring and signature.

    It extracts the function signature and provides utilities to parse
    parameter descriptions from various docstring styles (Google, Sphinx,
    Javadoc).

    Examples:
    - Create from a simple function and get first-line summary
        ```python
        >>> def add(a: int, b: int) -> int:
        ...     '''Add two integers.
        ...
        ...     Args:
        ...         a (int): First number.
        ...         b (int): Second number.
        ...
        ...     Returns:
        ...         int: Sum of a and b.
        ...     '''
        ...     return a + b
        >>> ds = Docstring(add)
        >>> str(ds.signature).startswith('(a: int, b: int)')
        True

        ```
    """

    def __init__(self, func: Callable[..., Any], *,name: Optional[str] = None) -> None:
        """Initialize a Docstring helper for a given callable.

        Args:
            func (Callable[..., Any]): The callable to introspect.
            name (str | None): Optional name to use instead of ``func.__name__``.
        """
        # self.func = func
        self.name = name or func.__name__
        self.docstring = func.__doc__ or ""
        self.func_signature = signature(func)
        self.func_arguments = set(self.func_signature.parameters.keys())

    @property
    def signature(self):
        """The inspect.Signature of the wrapped callable.

        Returns:
            inspect.Signature: The captured function signature.
        """
        return self.func_signature

    @signature.setter
    def signature(self, sig):
        """Override the stored ``inspect.Signature``.

        Args:
            sig (inspect.Signature): The new signature to store.
        """
        self.func_signature = sig

    def extract_param_docs(self) -> Tuple[dict, set]:
        """Parse parameter descriptions from this callable's docstring.

        Supports Sphinx (``:param name: desc``), Google (``name (type): desc``),
        and Javadoc (``@param name desc``) styles. The first non-conflicting
        description per parameter is kept; names not present in the function
        signature are returned as unknown and ignored by schema enrichment.

        Returns:
            Tuple[dict, set]:
                - A mapping of parameter name to the first non-conflicting
                  description found.
                - A set of unknown parameter names encountered in the docstring.

        Examples:
            - Parse Google-style docstring
                ```python
                >>> def mul(a: int, b: int) -> int:
                ...     '''Multiply two numbers.
                ...
                ...     Args:
                ...         a (int): Left factor.
                ...         b (int): Right factor.
                ...     '''
                ...     return a * b
                >>> ds = Docstring(mul)
                >>> ds.func_arguments == {'a', 'b'}
                True
                >>> docs, unknown = ds.extract_param_docs()
                >>> sorted(docs.items())
                [('a', 'Left factor.'), ('b', 'Right factor.')]
                >>> sorted(unknown)
                []

                ```

        See Also:
            - FunctionConverter: Uses parsed docs to enrich generated schemas.
        """
        raw_param_docs: dict[str, str] = {}
        unknown_params = set()

        def try_add_param(name: str, desc: str) -> None:
            desc = desc.strip()
            if self.func_arguments and name not in self.func_arguments:
                unknown_params.add(name)
                return
            if name in raw_param_docs and raw_param_docs[name] != desc:
                return
            raw_param_docs[name] = desc

        # Sphinx style
        for match in re.finditer(r":param (\w+): (.+)", self.docstring):
            try_add_param(match.group(1), match.group(2))

        # Google style
        for match in re.finditer(
    r"^\s*(\w+)\s*\(.*?\):\s*(.+)$", self.docstring, re.MULTILINE
        ):
            try_add_param(match.group(1), match.group(2))

        # Javadoc style
        for match in re.finditer(r"@param (\w+)\s+(.+)", self.docstring):
            try_add_param(match.group(1), match.group(2))

        return raw_param_docs, unknown_params

    def get_short_summary_line(self) -> str:
        """Return a short summary line for the wrapped callable.

        The summary includes the function name with its signature followed by the
        first non-empty line of the original docstring.

        Returns:
            str: A summary line combining signature and the first docstring line.

        Examples:
            - When the wrapped function has a one-line docstring
                ```python
                >>> def say(name: str) -> str:
                ...     'Greet someone.'
                ...     return f'Hi {name}'
                >>> ds = Docstring(say)
                >>> summary = ds.get_short_summary_line()
                >>> 'say(name: str) -> str' in summary and 'Greet someone.' in summary
                True

                ```
        """
        description = f"{self.name}{self.func_signature}\n"

        doc_lines = self.docstring.strip().splitlines()
        for line in doc_lines:
            if line.strip():
                description += line.strip()
                break
        return description


class FunctionArgument:
    """Adapter for an inspect.Parameter to Pydantic FieldInfo and type.

    This helper extracts the effective parameter type and builds a matching
    pydantic FieldInfo, taking into account typing.Annotated, default values,
    FieldInfo defaults, and datetime formats.

    Args:
        param (inspect.Parameter): The function parameter to convert.

    Examples:
        - From a basic annotated parameter with a default
            ```python
            >>> from inspect import signature
            >>> def f(d: int = 3):
            ...     pass
            >>> param = list(signature(f).parameters.values())[0]
            >>> arg = FunctionArgument(param)
            >>> t, field = arg.to_field()
            >>> (t is int) and (field.default == 3)
            True

            ```
    """
    def __init__(self, param: Parameter) -> None:
        self.param = param
        # Extract type, description, and extras from annotation
        self.param_type = self.param.annotation
        self.description: Optional[str] = None
        self.json_schema_extra: dict[str, Any] = {}
        if self.is_annotated():
            self._extract_annotated_info()

    def is_annotated(self) -> bool:
        """Return True if the parameter uses typing.Annotated.

        Returns:
            bool: True if the annotation origin is Annotated, else False.

        Examples:
            - Basic usage
                ```python
                >>> from inspect import signature
                >>> from typing import Annotated
                >>> def f(x: Annotated[int, 'desc']):
                ...     pass
                >>> param = list(signature(f).parameters.values())[0]
                >>> FunctionArgument(param).is_annotated()
                True

                ```
        """
        return get_origin(self.param.annotation) is Annotated

    def _extract_annotated_info(self):
        """Extract metadata from typing.Annotated annotations.

        For parameters annotated as ``Annotated[T, meta]``, this method sets the
        effective base type to ``T`` and pulls optional metadata from ``meta``.

        Supported metadata values:
        - str: Used as the field description.
        - pydantic.fields.FieldInfo: Copies description and json_schema_extra.

        Examples:
            - Using a string description
                ```python
                >>> from typing import Annotated
                >>> from inspect import signature
                >>> def f(age: Annotated[int, 'User age']):
                ...     pass
                >>> param = list(signature(f).parameters.values())[0]
                >>> fa = FunctionArgument(param)
                >>> fa.param_type is int and fa.description == 'User age'
                True

                ```
            - Using FieldInfo to pass description and extras
                ```python
                >>> from typing import Annotated
                >>> from inspect import signature
                >>> from pydantic.fields import FieldInfo
                >>> def f(ts: Annotated[str, FieldInfo(description='Timestamp', json_schema_extra={'format': 'date-time'})]):
                ...     pass
                >>> param = list(signature(f).parameters.values())[0]
                >>> fa = FunctionArgument(param)
                >>> fa.description, fa.json_schema_extra.get('format')
                ('Timestamp', 'date-time')

                ```
        """
        args = get_args(self.param_type)
        if args:
            self.param_type = args[0]
            if len(args) > 1:
                meta = args[1]
                if isinstance(meta, str):
                    self.description = meta
                elif isinstance(meta, FieldInfo):
                    self.description = meta.description
                    if meta.json_schema_extra and isinstance(meta.json_schema_extra, dict):
                        self.json_schema_extra.update(meta.json_schema_extra)


    def _add_format_if_datetime(self) -> None:
        """Add JSON Schema ``format`` for date/time-like parameter types.

        If the parameter type is ``datetime.date``, ``datetime.datetime`` or
        ``datetime.time``, a ``format`` key is added to ``json_schema_extra``
        (``date``, ``date-time``, or ``time`` respectively) if not already set.

        Examples:
            - Date type gets ``format: 'date'``
                ```python
                >>> from inspect import signature
                >>> def f(x: datetime.date):
                ...     pass
                >>> param = list(signature(f).parameters.values())[0]
                >>> fa = FunctionArgument(param)
                >>> fa._add_format_if_datetime()
                >>> fa.json_schema_extra.get('format')
                'date'

                ```
        """
        if self.param_type == datetime.date:
            self.json_schema_extra.setdefault("format", "date")
        elif self.param_type == datetime.datetime:
            self.json_schema_extra.setdefault("format", "date-time")
        elif self.param_type == datetime.time:
            self.json_schema_extra.setdefault("format", "time")

    def _create_field_info(self) -> FieldInfo:
        """Build a pydantic FieldInfo for this parameter.

        The default/required semantics follow Python and Pydantic rules:
        - If the parameter has no default (``Parameter.empty``), the field is
          required and only description/extras are set.
        - If the default is already a ``FieldInfo``, it is used as-is.
        - Otherwise, the value is used as the default for a new ``FieldInfo``.

        Returns:
            FieldInfo: A FieldInfo configured with description and json_schema_extra.

        Examples:
            - Required field (no default)
                ```python
                >>> from inspect import signature, Parameter
                >>> def f(x: int):
                ...     pass
                >>> param = list(signature(f).parameters.values())[0]
                >>> fi = FunctionArgument(param)._create_field_info()
                >>> fi.is_required()
                True

                ```
        """
        default = self.param.default
        
        if default is Parameter.empty:
            field_info = FieldInfo(description=self.description, json_schema_extra=self.json_schema_extra)
        elif isinstance(default, FieldInfo):
            field_info = default
        else:
            field_info = FieldInfo(default=default, description=self.description, json_schema_extra=self.json_schema_extra)
        
        return field_info

    def to_field(self) -> Tuple[Type[Any], FieldInfo]:
        """Convert this parameter to a (type, FieldInfo) tuple.

        This combines any extracted metadata, applies date/time formats, and
        respects required vs. default semantics.

        Returns:
            Tuple[Type[Any], FieldInfo]: The effective type and constructed FieldInfo.

        Examples:
            - Parameter with Annotated description and default value
                ```python
                >>> from typing import Annotated
                >>> from inspect import signature
                >>> def f(x: Annotated[int, 'Counter'] = 42):
                ...     pass
                >>> param = list(signature(f).parameters.values())[0]
                >>> t, fi = FunctionArgument(param).to_field()
                >>> t is int and fi.default == 42 and fi.description == 'Counter'
                True

                ```
        """
        # Add format for date/datetime/time if applicable
        self._add_format_if_datetime()

        # Fallbacks for missing annotation
        if self.param_type is self.param.empty:
            param_type = Any
        else:
            param_type = self.param_type

        # Build FieldInfo based on default semantics
        field_info = self._create_field_info()
        return param_type, field_info


class FunctionConverter:
    """Build a Pydantic model schema from a Python callable's signature.

    This converter inspects a function's parameters and creates a Pydantic model
    with corresponding fields, honoring typing.Annotated metadata and datetime
    formats. It also supports ignoring specific parameters and injecting
    additional custom fields.

    Args:
        name (str): Name to give the generated Pydantic model class.
        func (Callable[..., Any] | Callable[..., Awaitable[Any]]): The function
            to analyze.
        additional_fields (list[tuple] | None): Optional extra fields to add.
            Each item is either ``(name, type)`` for a required field or
            ``(name, type, default)`` for an optional field with a default.
        ignore_fields (list[str] | None): Names of function parameters to skip.

    Examples:
        - Minimal usage
            ```python
            >>> def greet(name: str):
            ...     return f'Hi {name}'
            >>> schema = FunctionConverter('GreetArgs', greet).to_schema()
            >>> list(schema.model_fields.keys())
            ['name']

            ```
    """
    def __init__(
        self,
        name: str,
        func: Union[Callable[..., Any], Callable[..., Awaitable[Any]]],
         additional_fields: Optional[
             List[Union[Tuple[str, Type, Any], Tuple[str, Type]]]
         ] = None,
         ignore_fields: Optional[List[str]] = None
    ):
        self.name = name
        self.func = func
        self.additional_fields = additional_fields
        self.ignore_fields = ignore_fields

    def to_schema(self) -> Type[BaseModel]:
        """Create a Pydantic model from the wrapped function's parameters.

        The generated model:
        - Respects ``ignore_fields``.
        - Supports ``typing.Annotated`` to extract description and extras.
        - Adds JSON Schema ``format`` for date, datetime, and time.
        - Honors required vs. default vs. FieldInfo defaults.
        - Merges ``additional_fields`` at the end.

        Returns:
            Type[pydantic.BaseModel]: The generated Pydantic model class.

        Examples:
            - Add an extra field and ignore one parameter
                ```python
                >>> def f(a: int, b: str):
                ...     pass
                >>> conv = FunctionConverter(
                ...     'FArgs', f,
                ...     additional_fields=[('c', float, 0.5)],
                ...     ignore_fields=['b']
                ... )
                >>> Model = conv.to_schema()
                >>> sorted(Model.model_fields.keys())
                ['a', 'c']

                ```
        """
        fields = self._collect_fields_from_func_signature()
        fields = self._apply_additional_fields(fields)
        return create_model(self.name, **fields)  # type: ignore

    def _collect_fields_from_func_signature(self) -> dict[str, Tuple[Type[Any], FieldInfo]]:
        """Derive Pydantic fields from the function signature.

        Returns:
            dict[str, tuple[type, FieldInfo]]: Mapping from parameter name to
            a tuple of (type, FieldInfo) describing the Pydantic model field.

        Examples:
            - Respecting ``ignore_fields``
                ```python
                >>> def f(x: int, y: int):
                ...     pass
                >>> conv = FunctionConverter('Args', f, ignore_fields=['y'])
                >>> fields = conv._collect_fields_from_func_signature()
                >>> sorted(fields.keys())
                ['x']

                ```
        """
        fields: dict[str, Tuple[Type[Any], FieldInfo]] = {}
        ignore_fields = self.ignore_fields or []
        params = signature(self.func).parameters
        for param_name, param in params.items():
            if param_name in ignore_fields:
                continue
            argument = FunctionArgument(param)
            field_type, field_info = argument.to_field()
            fields[param_name] = (field_type, field_info)
        return fields

    
    def _apply_additional_fields(self, fields: dict[str, Tuple[Type[Any], FieldInfo]]) -> dict[str, Tuple[Type[Any], FieldInfo]]:
        """Merge ``additional_fields`` into the collected ``fields`` mapping.

        Each entry in ``additional_fields`` is either ``(name, type)`` for a
        required field or ``(name, type, default)`` for a field with a default.

        Args:
            fields (dict[str, tuple[type, FieldInfo]]): Existing fields map to update.

        Returns:
            dict[str, tuple[type, FieldInfo]]: The updated fields mapping.

        Raises:
            ValueError: If any additional field tuple is not of length 2 or 3.

        Examples:
            - Adding required and optional extra fields
                ```python
                >>> def noop():
                ...     pass
                >>> conv = FunctionConverter(
                ...     'X', noop,
                ...     additional_fields=[('a', int), ('b', str, 'x')]
                ... )
                >>> merged = conv._apply_additional_fields({})
                >>> sorted((k, v[0].__name__) for k, v in merged.items())
                [('a', 'int'), ('b', 'str')]

                ```
        """
        additional_fields = self.additional_fields or []
        for field_info in additional_fields:
            if len(field_info) == 3:
                field_info = cast(Tuple[str, Type, Any], field_info)
                field_name, field_type, field_default = field_info
                fields[field_name] = (field_type, FieldInfo(default=field_default))
            elif len(field_info) == 2:
                field_info = cast(Tuple[str, Type], field_info)
                field_name, field_type = field_info
                fields[field_name] = (field_type, FieldInfo())
            else:
                raise ValueError(
                    f"Invalid additional field info: {field_info}. "
                    "Must be a tuple of length 2 or 3."
                )

        return fields


@dataclass
class ExecutionConfig:
    """Configuration flags for tool execution behavior.

    Args:
        verbose (bool): If True, print basic logs at start and end of execution.
        single_arg_auto_unpack (bool): If True, attempt to call tools that
            declare a single parameter by passing the single value positional,
            and fall back to keyword arguments if that fails.
        raise_on_error (bool): If True, re-raise exceptions; otherwise, return a
            standardized error ToolOutput.

    Examples:
        - Default values
            ```python
            >>> cfg = ExecutionConfig()
            >>> (cfg.verbose, cfg.single_arg_auto_unpack, cfg.raise_on_error)
            (False, True, False)

            ```
    """
    verbose: bool = False
    single_arg_auto_unpack: bool = True
    raise_on_error: bool = False


class ToolExecutor:
    """Execute tools safely with optional argument auto-unpacking.

    The executor centralizes synchronous/asynchronous invocation, basic logging,
    and standardized error handling. When ``single_arg_auto_unpack`` is enabled,
    tools that declare exactly one parameter can be called with a single
    positional argument derived from the only key in the arguments mapping.

    Args:
        config (ExecutionConfig | None): Optional execution config. Defaults to
            ``ExecutionConfig()``.

    Examples:
        - Basic synchronous execution with a stub tool
            ```python
            >>> class ToolOutput:
            ...     def __init__(self, content: str, tool_name: str, raw_input=None, raw_output=None, is_error: bool=False):
            ...         self.content = content
            ...         self.tool_name = tool_name
            ...         self.raw_input = raw_input
            ...         self.raw_output = raw_output
            ...         self.is_error = is_error
            >>> class _Meta:
            ...     def __init__(self, name):
            ...         self.name = name
            ...     def get_name(self):
            ...         return self.name
            ...     def get_schema(self):
            ...         return {"properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}}
            >>> class SumTool:
            ...     def __init__(self):
            ...         self.metadata = _Meta("sum")
            ...     def __call__(self, **kwargs):
            ...         return ToolOutput(str(kwargs["x"] + kwargs["y"]), self.metadata.get_name())
            >>> executor = ToolExecutor()
            >>> out = executor.execute(SumTool(), {"x": 2, "y": 3})
            >>> (out.content, out.tool_name)
            ('5', 'sum')

            ```
        - Single-argument auto-unpacking when the tool schema has exactly one property
            ```python
            >>> class ToolOutput:
            ...     def __init__(self, content: str, tool_name: str, raw_input=None, raw_output=None, is_error: bool=False):
            ...         self.content = content
            ...         self.tool_name = tool_name
            ...         self.raw_input = raw_input
            ...         self.raw_output = raw_output
            ...         self.is_error = is_error
            >>> class _Meta:
            ...     def __init__(self, name):
            ...         self.name = name
            ...     def get_name(self):
            ...         return self.name
            ...     def get_schema(self):
            ...         return {"properties": {"data": {"type": "array"}}}
            >>> class EchoTool:
            ...     def __init__(self):
            ...         self.metadata = _Meta("echo")
            ...     def __call__(self, value):
            ...         return ToolOutput(str(value), self.metadata.get_name())
            >>> executor = ToolExecutor()
            >>> out = executor.execute(EchoTool(), {"data": [1, 2, 3]})
            >>> (out.content, out.tool_name)
            ('[1, 2, 3]', 'echo')

            ```
    """

    def __init__(self, config: Optional[ExecutionConfig] = None):
        """Initialize the tool executor.

        Args:
            config: Optional configuration for execution behavior.
                If None, uses default configuration.
        """
        self.config = config or ExecutionConfig()

    def execute(self, tool: BaseTool, arguments: dict) -> ToolOutput:
        """Execute a tool synchronously with error handling.

        Args:
            tool (BaseTool): The tool instance to call. Must expose ``metadata``
                with ``get_name()`` and ``get_schema()`` and be callable.
            arguments (dict): Keyword arguments for the tool.

        Returns:
            ToolOutput: The tool's output, or a standardized error output if
                execution failed and ``raise_on_error`` is False.

        Raises:
            Exception: If ``raise_on_error`` is True and execution fails.

        Examples:
            - Successful execution with a stub tool
                ```python
                >>> class ToolOutput:
                ...     def __init__(self, content: str, tool_name: str, raw_input=None, raw_output=None, is_error: bool=False):
                ...         self.content = content
                ...         self.tool_name = tool_name
                ...         self.raw_input = raw_input
                ...         self.raw_output = raw_output
                ...         self.is_error = is_error
                >>> class _Meta:
                ...     def __init__(self, name):
                ...         self.name = name
                ...     def get_name(self):
                ...         return self.name
                ...     def get_schema(self):
                ...         return {"properties": {"text": {"type": "string"}}}
                >>> class Echo:
                ...     def __init__(self):
                ...         self.metadata = _Meta("echo")
                ...     def __call__(self, **kwargs):
                ...         return ToolOutput(kwargs["text"], self.metadata.get_name())
                >>> out = ToolExecutor().execute(Echo(), {"text": "hi"})
                >>> (out.content, out.tool_name, out.is_error)
                ('hi', 'echo', False)

                ```
        """
        if self.config.verbose:
            self._log_execution_start(tool, arguments)

        try:
            output = self._invoke_tool(tool, arguments)

            if self.config.verbose:
                self._log_execution_result(output)

            return output

        except Exception as e:
            if self.config.raise_on_error:
                raise
            return self._create_error_output(tool, arguments, e)

    async def execute_async(self, tool: BaseTool, arguments: dict) -> ToolOutput:
        """Execute a tool asynchronously with error handling.

        Args:
            tool (BaseTool): The tool to execute.
            arguments (dict): Dictionary of arguments to pass to the tool.

        Returns:
            ToolOutput: The tool's output, or an error output if execution failed.

        Raises:
            Exception: If ``raise_on_error`` is True and execution fails.

        Examples:
            - Creating an executor (avoid running event loops in doctests)
                ```python
                >>> exec_async = ToolExecutor()
                >>> isinstance(exec_async, ToolExecutor)
                True

                ```
        """
        if self.config.verbose:
            self._log_execution_start(tool, arguments)

        async_tool = adapt_to_async_tool(tool)

        try:
            output = await self._invoke_tool_async(async_tool, arguments)

            if self.config.verbose:
                self._log_execution_result(output)

            return output

        except Exception as e:
            if self.config.raise_on_error:
                raise
            return self._create_error_output(tool, arguments, e)

    def execute_with_selection(
        self,
        tool_call: ToolSelection,
        tools: Sequence[BaseTool],
    ) -> ToolOutput:
        """Execute a tool based on a tool selection.

        Args:
            tool_call (ToolSelection): The tool selection containing name and arguments.
            tools (Sequence[BaseTool]): Sequence of available tools.

        Returns:
            ToolOutput: The execution result.

        Raises:
            KeyError: If the selected tool name is not present in ``tools``.

        Examples:
            - Execute using a local ``ToolSelection`` stub
                ```python
                >>> class ToolOutput:
                ...     def __init__(self, content: str, tool_name: str, raw_input=None, raw_output=None, is_error: bool=False):
                ...         self.content = content
                ...         self.tool_name = tool_name
                ...         self.raw_input = raw_input
                ...         self.raw_output = raw_output
                ...         self.is_error = is_error
                >>> class _Meta:
                ...     def __init__(self, name):
                ...         self.name = name
                ...     def get_name(self):
                ...         return self.name
                ...     def get_schema(self):
                ...         return {"properties": {"text": {"type": "string"}}}
                >>> class Echo:
                ...     def __init__(self):
                ...         self.metadata = _Meta("echo")
                ...     def __call__(self, **kwargs):
                ...         return ToolOutput(kwargs["text"], self.metadata.get_name())
                >>> class ToolSelection:
                ...     def __init__(self, tool_name: str, tool_kwargs: dict):
                ...         self.tool_name = tool_name
                ...         self.tool_kwargs = tool_kwargs
                >>> sel = ToolSelection("echo", {"text": "hi"})
                >>> out = ToolExecutor().execute_with_selection(sel, [Echo()])
                >>> (out.content, out.tool_name)
                ('hi', 'echo')

                ```
        """
        tool = self._find_tool_by_name(tool_call.tool_name, tools)
        return self.execute(tool, tool_call.tool_kwargs)

    async def execute_async_with_selection(
            self,
            tool_call: ToolSelection,
            tools: Sequence[BaseTool],
    ) -> ToolOutput:
        """Execute a tool asynchronously based on a tool selection.

        Args:
            tool_call (ToolSelection): The tool selection containing name and arguments.
            tools (Sequence[BaseTool]): Sequence of available tools.

        Returns:
            ToolOutput: The execution result.

        Raises:
            KeyError: If the selected tool is not found in the tools sequence.

        Examples:
            - Creating an executor (avoid running event loops in doctests)
                ```python
                >>> isinstance(ToolExecutor(), ToolExecutor)
                True

                ```
        """
        tool = self._find_tool_by_name(tool_call.tool_name, tools)
        return await self.execute_async(tool, tool_call.tool_kwargs)

    def _invoke_tool(self, tool: BaseTool, arguments: dict) -> ToolOutput:
        """Internal method to invoke tool with argument unpacking logic."""
        if self._should_unpack_single_arg(tool, arguments):
            output = self._try_single_arg_then_kwargs(tool, arguments)
        else:
            output = tool(**arguments)

        return output

    async def _invoke_tool_async(
        self,
        async_tool: BaseTool,
        arguments: dict
    ) -> ToolOutput:
        """Internal method to invoke async tool with argument unpacking logic."""
        if self._should_unpack_single_arg(async_tool, arguments):
            output = await self._try_single_arg_then_kwargs_async(async_tool, arguments)
        else:
            output = await async_tool.acall(**arguments)

        return output

    def _should_unpack_single_arg(self, tool: BaseTool, arguments: dict) -> bool:
        """Determine whether to auto-unpack a single argument.

        Auto-unpacking is allowed when the executor config enables it and when
        both the tool schema and the provided arguments indicate exactly one
        parameter/value.

        Args:
            tool (BaseTool): Tool whose schema is inspected via ``metadata.get_schema()``.
            arguments (dict): The provided arguments mapping.

        Returns:
            bool: True if a single positional value should be passed.

        Examples:
            - Single property in schema and a single provided argument
                ```python
                >>> class _Meta:
                ...     def get_schema(self):
                ...         return {"properties": {"data": {"type": "array"}}}
                >>> class T:
                ...     metadata = _Meta()
                >>> executor = ToolExecutor()
                >>> executor._should_unpack_single_arg(T(), {"data": [1]})
                True

                ```
            - Multiple properties or multiple provided arguments -> do not unpack
                ```python
                >>> class _Meta:
                ...     def get_schema(self):
                ...         return {"properties": {"a": {}, "b": {}}}
                >>> class T:
                ...     metadata = _Meta()
                >>> executor = ToolExecutor()
                >>> executor._should_unpack_single_arg(T(), {"a": 1})
                False

                ```
        """
        if not self.config.single_arg_auto_unpack:
            val = False
        else:
            # get the tool schema and check if it's a single arg tool and that the given arguments are a single arg
            schema = tool.metadata.get_schema()
            val = (
                    len(schema.get("properties", {})) == 1
                    and len(arguments) == 1
            )

        return val

    def _try_single_arg_then_kwargs(
        self,
        tool: BaseTool,
        arguments: dict
    ) -> ToolOutput:
        """Try calling with single unpacked arg, fall back to kwargs."""
        try:
            single_arg = arguments[next(iter(arguments))]
            output = tool(single_arg)
        except Exception:
            # Some tools require kwargs, so try that instead
            output =tool(**arguments)

        return output

    async def _try_single_arg_then_kwargs_async(
        self,
        async_tool: BaseTool,
        arguments: dict
    ) -> ToolOutput:
        """Try calling async with single unpacked arg, fall back to kwargs."""
        try:
            single_arg = arguments[next(iter(arguments))]
            output = await async_tool.acall(single_arg)
        except Exception:
            # Some tools require kwargs, so try that instead
            output = await async_tool.acall(**arguments)

        return output

    def _create_error_output(
        self,
        tool: BaseTool,
        arguments: dict,
        error: Exception
    ) -> ToolOutput:
        """Create a standardized error output."""
        return ToolOutput(
            content=f"Encountered error: {str(error)}",
            tool_name=tool.metadata.get_name(),
            raw_input=arguments,
            raw_output=str(error),
            is_error=True,
        )

    def _find_tool_by_name(
        self,
        name: str,
        tools: Sequence[BaseTool]
    ) -> BaseTool:
        """Find a tool by name from a sequence of tools."""
        tools_by_name = {tool.metadata.name: tool for tool in tools}
        return tools_by_name[name]

    def _log_execution_start(self, tool: BaseTool, arguments: dict) -> None:
        """Log the start of tool execution."""
        arguments_str = json.dumps(arguments)
        print("=== Calling Function ===")
        print(f"Calling function: {tool.metadata.get_name()} with args: {arguments_str}")

    def _log_execution_result(self, output: ToolOutput) -> None:
        """Log the result of tool execution."""
        print("=== Function Output ===")
        print(output.content)
