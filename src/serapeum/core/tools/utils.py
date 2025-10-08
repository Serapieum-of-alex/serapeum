import json
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
from serapeum.core.llm.base import ToolSelection


if TYPE_CHECKING:
    from serapeum.core.tools.models import BaseTool

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


def call_tool(tool: BaseTool, arguments: dict) -> ToolOutput:
    """Call a tool with arguments."""
    try:
        if (
                len(tool.metadata.get_schema()["properties"]) == 1
                and len(arguments) == 1
        ):
            try:
                single_arg = arguments[next(iter(arguments))]
                return tool(single_arg)
            except Exception:
                # some tools will REQUIRE kwargs, so try it
                return tool(**arguments)
        else:
            return tool(**arguments)
    except Exception as e:
        return ToolOutput(
            content="Encountered error: " + str(e),
            tool_name=tool.metadata.get_name(),
            raw_input=arguments,
            raw_output=str(e),
            is_error=True,
        )


async def acall_tool(tool: BaseTool, arguments: dict) -> ToolOutput:
    """Call a tool with arguments asynchronously."""
    async_tool = adapt_to_async_tool(tool)
    try:
        if (
                len(tool.metadata.get_schema()["properties"]) == 1
                and len(arguments) == 1
        ):
            try:
                single_arg = arguments[next(iter(arguments))]
                return await async_tool.acall(single_arg)
            except Exception:
                # some tools will REQUIRE kwargs, so try it
                return await async_tool.acall(**arguments)
        else:
            return await async_tool.acall(**arguments)
    except Exception as e:
        return ToolOutput(
            content="Encountered error: " + str(e),
            tool_name=tool.metadata.get_name(),
            raw_input=arguments,
            raw_output=str(e),
            is_error=True,
        )


def call_tool_with_selection(
    tool_call: ToolSelection,
    tools: Sequence["BaseTool"],
    verbose: bool = False,
) -> ToolOutput:

    tools_by_name = {tool.metadata.name: tool for tool in tools}
    name = tool_call.tool_name
    if verbose:
        arguments_str = json.dumps(tool_call.tool_kwargs)
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = tools_by_name[name]
    output = call_tool(tool, tool_call.tool_kwargs)

    if verbose:
        print("=== Function Output ===")
        print(output.content)

    return output


async def acall_tool_with_selection(
    tool_call: ToolSelection,
    tools: Sequence["BaseTool"],
    verbose: bool = False,
) -> ToolOutput:

    tools_by_name = {tool.metadata.name: tool for tool in tools}
    name = tool_call.tool_name
    if verbose:
        arguments_str = json.dumps(tool_call.tool_kwargs)
        print("=== Calling Function ===")
        print(f"Calling function: {name} with args: {arguments_str}")
    tool = tools_by_name[name]
    output = await acall_tool(tool, tool_call.tool_kwargs)

    if verbose:
        print("=== Function Output ===")
        print(output.content)

    return output
