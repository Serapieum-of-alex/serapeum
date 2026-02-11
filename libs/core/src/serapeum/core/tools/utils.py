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
    - serapeum.core.tools.types.BaseTool
    - serapeum.core.tools.types.ToolOutput
    - serapeum.core.tools.types.ToolCallArguments
"""

import datetime
import re
from inspect import Parameter, Signature, signature
from typing import (
    Annotated,
    Any,
    Awaitable,
    Callable,
    Type,
    TypeVar,
    get_args,
    get_origin,
)

from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo

Model = TypeVar("Model", bound=BaseModel)


class Docstring:
    r"""Helper to introspect a callable's docstring and signature.

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
        >>> ds.name
        'add'
        >>> 'Add two integers.' in ds.docstring
        True
        >>> ds.get_short_summary_line()
        'add(a: int, b: int) -> int\nAdd two integers.'
        >>> ds.func_arguments
        {'b', 'a'}

        ```
    - Create from a class and get first-line summary
        ```python
        >>> from pydantic import BaseModel
        >>> class Song(BaseModel):
        ...     '''A music track.
        ...
        ...     Attribute:
        ...         title (str): Title of the song.
        ...     '''
        ...     title: str
        >>> ds = Docstring(Song)
        >>> ds.name
        "Song"
        >>> ds.func_arguments
        {'title'}
        >>> ds.extract_param_docs()
        ({'title': 'Title of the song.'}, set())
        >>> ds.get_short_summary_line()
        'Song(*, title: str) -> None\nA music track.'
        >>> ds.docstring
        'A music track.\n    \n    Attribute:\n        title (str): Title of the song.\n    '

        ```
    """

    def __init__(
        self,
        func: Callable[..., Any] | Type[Model],
        *,
        name: str | None = None,
    ) -> None:
        """Initialize a Docstring helper for a given callable.

        Args:
            func (Callable[..., Any], Type[Model]):
                The base models/callable to introspect.
            name (str | None):
                Optional name to use instead of ``func.__name__``.
        """
        self.name = name or func.__name__
        self.docstring = func.__doc__ or ""
        self.func_signature = signature(func)
        self.func_arguments = set(self.func_signature.parameters.keys())

    @property
    def signature(self) -> Signature:
        """The inspect.Signature of the wrapped callable.

        Returns:
            inspect.Signature: The captured function signature.
        """
        return self.func_signature

    @signature.setter
    def signature(self, sig: Signature) -> None:
        """Override the stored ``inspect.Signature``.

        Args:
            sig (inspect.Signature): The new signature to store.
        """
        self.func_signature = sig

    def extract_param_docs(self) -> tuple[dict[str, str], set[str]]:
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
        """Initialize FunctionArgument with a parameter."""
        self.param = param
        # Extract type, description, and extras from annotation
        self.param_type = self.param.annotation
        self.description: str | None = None
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

    def _extract_annotated_info(self) -> None:
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
                >>> def f(
                ...     ts: Annotated[
                ...         str,
                ...         FieldInfo(
                ...             description='Timestamp',
                ...             json_schema_extra={'format': 'date-time'},
                ...         ),
                ...     ]
                ... ):
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
                    if meta.json_schema_extra and isinstance(
                        meta.json_schema_extra, dict
                    ):
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
            field_info = FieldInfo(
                description=self.description, json_schema_extra=self.json_schema_extra
            )
        elif isinstance(default, FieldInfo):
            field_info = default
        else:
            field_info = FieldInfo(
                default=default,
                description=self.description,
                json_schema_extra=self.json_schema_extra,
            )

        return field_info

    def to_field(self) -> tuple[Type[Any], FieldInfo]:
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
        func: Callable[..., Any] | Callable[..., Awaitable[Any]],
        additional_fields:
            list[tuple[str, type[Any], Any] | tuple[str, type[Any]]] | None = None,
        ignore_fields: list[str] | None = None,
    ):
        """Initialize with function, additional fields, and ignore list."""
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

    def _collect_fields_from_func_signature(
        self,
    ) -> dict[str, tuple[Type[Any], FieldInfo]]:
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
        fields: dict[str, tuple[Type[Any], FieldInfo]] = {}
        ignore_fields = self.ignore_fields or []
        params = signature(self.func).parameters
        for param_name, param in params.items():
            if param_name in ignore_fields:
                continue
            argument = FunctionArgument(param)
            field_type, field_info = argument.to_field()
            fields[param_name] = (field_type, field_info)
        return fields

    def _apply_additional_fields(
        self, fields: dict[str, tuple[Type[Any], FieldInfo]]
    ) -> dict[str, tuple[Type[Any], FieldInfo]]:
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
                >>> sorted([(key, value[0].__name__) for key, value in merged.items()])
                [('a', 'int'), ('b', 'str')]

                ```
        """
        additional_fields = self.additional_fields or []
        for field_info in additional_fields:
            if len(field_info) == 3:
                field_name, field_type, field_default = field_info  # type: ignore[misc]
                fields[field_name] = (field_type, FieldInfo(default=field_default))
            elif len(field_info) == 2:
                field_name, field_type = field_info  # type: ignore[misc]
                fields[field_name] = (field_type, FieldInfo())
            else:
                raise ValueError(
                    f"Invalid additional field info: {field_info}. "
                    "Must be a tuple of length 2 or 3."
                )

        return fields