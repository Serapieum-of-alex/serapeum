import json
from inspect import signature
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

def create_schema_from_function(
    name: str,
    func: Union[Callable[..., Any], Callable[..., Awaitable[Any]]],
    additional_fields: Optional[
        List[Union[Tuple[str, Type, Any], Tuple[str, Type]]]
    ] = None,
    ignore_fields: Optional[List[str]] = None,
) -> Type[BaseModel]:
    """
    Create schema from function.
    - Automatically adds json_schema_extra for basic Python types such as:
        - datetime.date -> format: "date"
        - datetime.datetime -> format: "date-time"
        - datetime.time -> format: "time"
    """
    fields = {}
    ignore_fields = ignore_fields or []
    params = signature(func).parameters

    for param_name in params:
        if param_name in ignore_fields:
            continue

        param_type = params[param_name].annotation
        param_default = params[param_name].default
        description = None
        json_schema_extra: dict[str, Any] = {}

        if get_origin(param_type) is Annotated:
            args = get_args(param_type)
            param_type = args[0]

            if isinstance(args[1], str):
                description = args[1]
            elif isinstance(args[1], FieldInfo):
                description = args[1].description
                if args[1].json_schema_extra and isinstance(
                        args[1].json_schema_extra, dict
                ):
                    json_schema_extra.update(args[1].json_schema_extra)

        # Add format based on param_type
        if param_type == datetime.date:
            json_schema_extra.setdefault("format", "date")
        elif param_type == datetime.datetime:
            json_schema_extra.setdefault("format", "date-time")
        elif param_type == datetime.time:
            json_schema_extra.setdefault("format", "time")

        if param_type is params[param_name].empty:
            param_type = Any

        if param_default is params[param_name].empty:
            # Required field
            fields[param_name] = (
                param_type,
                FieldInfo(description=description, json_schema_extra=json_schema_extra),
            )
        elif isinstance(param_default, FieldInfo):
            # Field with pydantic.Field as default value
            fields[param_name] = (param_type, param_default)
        else:
            fields[param_name] = (
                param_type,
                FieldInfo(
                    default=param_default,
                    description=description,
                    json_schema_extra=json_schema_extra,
                ),
            )

    additional_fields = additional_fields or []
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

    return create_model(name, **fields)  # type: ignore


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
