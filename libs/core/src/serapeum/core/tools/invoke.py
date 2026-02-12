"""Tool invocation utilities."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from serapeum.core.tools.types import (
    BaseTool,
    ToolCallArguments,
    ToolOutput,
    adapt_to_async_tool,
    AsyncBaseTool
)

__all__ = ["ExecutionConfig", "ToolExecutor"]

logger = logging.getLogger(__name__)


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
            ...     def __init__(
            ...         self,
            ...         content: str,
            ...         tool_name: str,
            ...         raw_input=None,
            ...         raw_output=None,
            ...         is_error: bool=False,
            ...     ):
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
            ...         return {
            ...             "properties": {
            ...                 "x": {"type": "integer"},
            ...                 "y": {"type": "integer"},
            ...             }
            ...         }
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
            ...     def __init__(
            ...         self,
            ...         content: str,
            ...         tool_name: str,
            ...         raw_input=None,
            ...         raw_output=None,
            ...         is_error: bool=False,
            ...     ):
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

    def __init__(self, config: ExecutionConfig | None = None):
        """Initialize the tool executor.

        Args:
            config: Optional configuration for execution behavior.
                If None, uses default configuration.
        """
        self.config = config or ExecutionConfig()

    def execute(self, tool: BaseTool, arguments: dict[str, Any]) -> ToolOutput:
        """Execute a tool synchronously with error handling.

        Args:
            tool (BaseTool): The tool instance to call. Must expose ``metadata``
                with ``get_name()`` and ``get_schema()`` and be callable.
            arguments (dict[str, Any]): Keyword arguments for the tool.

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

    async def execute_async(
        self, tool: BaseTool, arguments: dict[str, Any]
    ) -> ToolOutput:
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
        tool_call: ToolCallArguments,
        tools: Sequence[BaseTool],
    ) -> ToolOutput:
        """Execute a tool based on a tool selection.

        Args:
            tool_call (ToolCallArguments):
                The tool selection containing name and arguments.
            tools (Sequence[BaseTool]):
                Sequence of available tools.

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
        tool_call: ToolCallArguments,
        tools: Sequence[BaseTool],
    ) -> ToolOutput:
        """Execute a tool asynchronously based on a tool selection.

        Args:
            tool_call (ToolCallArguments): The tool selection containing name and arguments.
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

    def _invoke_tool(self, tool: BaseTool, arguments: dict[str, Any]) -> ToolOutput:
        """Internal method to invoke tool with argument unpacking logic."""
        if self._should_unpack_single_arg(tool, arguments):
            output = self._try_single_arg_then_kwargs(tool, arguments)
        else:
            output = tool(**arguments)

        return output

    async def _invoke_tool_async(
        self, async_tool: AsyncBaseTool, arguments: dict[str, Any]
    ) -> ToolOutput:
        """Internal method to invoke async tool with argument unpacking logic."""
        if self._should_unpack_single_arg(async_tool, arguments):
            output = await self._try_single_arg_then_kwargs_async(async_tool, arguments)
        else:
            output = await async_tool.acall(**arguments)

        return output

    def _should_unpack_single_arg(
        self, tool: BaseTool, arguments: dict[str, Any]
    ) -> bool:
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
            val = len(schema.get("properties", {})) == 1 and len(arguments) == 1

        return val

    def _try_single_arg_then_kwargs(
        self, tool: BaseTool, arguments: dict[str, Any]
    ) -> ToolOutput:
        """Try calling with single unpacked arg, fall back to kwargs."""
        try:
            single_arg = arguments[next(iter(arguments))]
            output = tool(single_arg)
        except Exception:
            # Some tools require kwargs, so try that instead
            output = tool(**arguments)

        return output

    async def _try_single_arg_then_kwargs_async(
        self, async_tool: AsyncBaseTool, arguments: dict[str, Any]
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
        self, tool: BaseTool, arguments: dict[str, Any], error: Exception
    ) -> ToolOutput:
        """Create a standardized error output."""
        return ToolOutput(
            content=f"Encountered error: {str(error)}",
            tool_name=tool.metadata.get_name(),
            raw_input=arguments,
            raw_output=str(error),
            is_error=True,
        )

    @staticmethod
    def _find_tool_by_name(name: str, tools: Sequence[BaseTool]) -> BaseTool:
        """Find a tool by name from a sequence of tools."""
        tools_by_name = {tool.metadata.get_name(): tool for tool in tools}
        return tools_by_name[name]

    @staticmethod
    def _log_execution_start(tool: BaseTool, arguments: dict[str, Any]) -> None:
        """Log the start of tool execution."""
        arguments_str = json.dumps(arguments)
        logger.info("=== Calling Function ===")
        logger.info(
            f"Calling function: {tool.metadata.get_name()} with args: {arguments_str}"
        )

    @staticmethod
    def _log_execution_result(output: ToolOutput) -> None:
        """Log the result of tool execution."""
        logger.info("=== Function Output ===")
        logger.info(output.content)
