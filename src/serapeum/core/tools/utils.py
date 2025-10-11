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

    def __init__(self, func: Callable[..., Any], *,name: Optional[str] = None) -> None:
        # self.func = func
        self.name = name or func.__name__
        self.docstring = func.__doc__ or ""
        self.fn_sig = signature(func)
        self.fn_params = set(self.fn_sig.parameters.keys())

    @property
    def signature(self):
        return self.fn_sig

    @signature.setter
    def signature(self, sig):
        self.fn_sig = sig

    def extract_param_docs(self) -> Tuple[dict, set]:
        """Parse parameter descriptions from a docstring in common styles.

        Supports Sphinx (``:param name: desc``), Google (``name (type): desc``),
        and Javadoc (``@param name desc``) styles. Unknown parameters (i.e.,
        names not present in ``fn_params`` when provided) are returned
        separately and ignored in the final schema enrichment.

        Args:
            docstring (str): The docstring text to parse.
            fn_params (Optional[set]): Optional set of valid parameter names.
                When supplied, parameters not in this set are returned in the
                second element of the tuple as ``unknown_params``.

        Returns:
            Tuple[dict, set]:
                - A mapping of parameter name to the first non-conflicting
                  description found.
                - A set of unknown parameter names encountered in the docstring.

        Examples:
            - Google style
                ```python
                >>> from serapeum.core.tools.callable_tool import CallableTool
                >>> doc = '''
                ... Adds two numbers.
                ...
                ... Args:
                ...     a (int): First addend.
                ...     b (int): Second addend.
                ... '''
                >>> param_docs, unknown = CallableTool.extract_param_docs(doc, {"a", "b"})
                >>> print(sorted(param_docs.items()))
                [('a', 'First addend.'), ('b', 'Second addend.')]
                >>> print(sorted(unknown))
                []

                ```

        See Also:
            - CallableTool.from_defaults: Uses this to enrich a generated schema.
        """
        raw_param_docs: dict[str, str] = {}
        unknown_params = set()

        def try_add_param(name: str, desc: str) -> None:
            desc = desc.strip()
            if self.fn_params and name not in self.fn_params:
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
        """Extracts the first line of the docstring."""
        description = f"{self.name}{self.fn_sig}\n"

        doc_lines = self.docstring.strip().splitlines()
        for line in doc_lines:
            if line.strip():
                description += line.strip()
                break
        return description


class FunctionArgument:
    def __init__(self, param: Parameter) -> None:
        self.param = param
        # Extract type, description, and extras from annotation
        self.param_type = self.param.annotation
        self.description: Optional[str] = None
        self.json_schema_extra: dict[str, Any] = {}
        if self.is_annotated():
            self._extract_annotated_info()

    def is_annotated(self)-> bool:
        return get_origin(self.param.annotation) is Annotated

    def _extract_annotated_info(self):
        """Extract base type, description, and json_schema_extra from an annotation.
        Supports typing.Annotated[str | FieldInfo] for metadata.
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
        """Mutates json_schema_extra to include appropriate datetime format if applicable."""
        if self.param_type == datetime.date:
            self.json_schema_extra.setdefault("format", "date")
        elif self.param_type == datetime.datetime:
            self.json_schema_extra.setdefault("format", "date-time")
        elif self.param_type == datetime.time:
            self.json_schema_extra.setdefault("format", "time")

    def _create_field_info(self) -> FieldInfo:
        """Create FieldInfo respecting required/default and FieldInfo defaults."""
        default = self.param.default

        if default is Parameter.empty:
            field_info = FieldInfo(description=self.description, json_schema_extra=self.json_schema_extra)
        elif isinstance(default, FieldInfo):
            field_info = default
        else:
            field_info = FieldInfo(default=default, description=self.description, json_schema_extra=self.json_schema_extra)

        return field_info

    def to_field(self) -> Tuple[Type[Any], FieldInfo]:

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
        """
        Build a Pydantic model schema from the wrapped function signature.

        Behavior is preserved:
        - respects ignore_fields
        - supports typing.Annotated[str|FieldInfo] for description/extra
        - adds format for date, datetime, time
        - handles required vs default vs FieldInfo defaults
        - merges self.additional_fields
        """
        fields = self._collect_fields_from_func_signature()
        fields = self._apply_additional_fields(fields)
        return create_model(self.name, **fields)  # type: ignore

    def _collect_fields_from_func_signature(self) -> dict[str, Tuple[Type[Any], FieldInfo]]:
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
    """Configuration for tool execution behavior."""
    verbose: bool = False
    single_arg_auto_unpack: bool = True
    raise_on_error: bool = False


class ToolExecutor:
    """Handles execution of tools with error handling and argument unpacking.

    This class encapsulates the logic for calling tools safely, handling both
    synchronous and asynchronous execution, with configurable behavior for
    argument unpacking and error handling.

    Examples:
        - Basic synchronous execution
            ```python
            >>> executor = ToolExecutor()
            >>> output = executor.execute(my_tool, {"input": "hello"})
            ```

        - Async execution
            ```python
            >>> executor = ToolExecutor()
            >>> async def my_tool():
            ...     return await executor.execute_async(my_tool, {"input": "hello"})
            >>> asyncio.run(my_tool())
            ```

        - With configuration
            ```python
            >>> config = ExecutionConfig(verbose=True, raise_on_error=True)
            >>> executor = ToolExecutor(config)
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
            tool: The tool to execute.
            arguments: Dictionary of arguments to pass to the tool.

        Returns:
            ToolOutput: The tool's output, or an error output if execution failed.

        Raises:
            Exception: If config.raise_on_error is True and execution fails.
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
            tool: The tool to execute.
            arguments: Dictionary of arguments to pass to the tool.

        Returns:
            ToolOutput: The tool's output, or an error output if execution failed.

        Raises:
            Exception: If config.raise_on_error is True and execution fails.
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
            tool_call: The tool selection containing name and arguments.
            tools: Sequence of available tools.

        Returns:
            ToolOutput: The execution result.

        Raises:
            KeyError: If the selected tool is not found in the tools sequence.
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
            tool_call: The tool selection containing name and arguments.
            tools: Sequence of available tools.

        Returns:
            ToolOutput: The execution result.

        Raises:
            KeyError: If the selected tool is not found in the tools sequence.
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
        """Check if single argument unpacking should be attempted."""
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
