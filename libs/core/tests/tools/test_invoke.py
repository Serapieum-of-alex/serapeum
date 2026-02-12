import asyncio
from typing import Any, Sequence

import pytest
from pydantic import BaseModel

from serapeum.core.tools.invoke import ExecutionConfig, ToolExecutor
from serapeum.core.tools.types import (
    AsyncBaseTool,
    BaseTool,
    ToolCallArguments,
    ToolMetadata,
    ToolOutput,
)


class SingleArgEchoTool(BaseTool):
    """Single-argument echo tool for positional forwarding validation.

    The tool uses the default MinimalToolSchema with a single "input" string field
    from ToolMetadata. When called with a single positional argument (the inner
    string), it echoes the value back.
    """

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(description="Echo a single string.", name="single_echo")

    def __call__(self, input_values: Any) -> ToolOutput:
        if isinstance(input_values, dict):
            text = input_values.get("input", "")
        else:
            text = str(input_values)
        return ToolOutput(tool_name=self.metadata.name or "single_echo", content=text)


class SingleArgKwOnlyTool(BaseTool):
    """Single-argument tool requiring keyword-only argument.

    This exercises the kwargs fallback in ToolExecutor.execute when the positional path fails.
    """

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(description="Requires kw-only arg.", name="single_kw")

    def __call__(self, *, input_values: str) -> ToolOutput:  # type: ignore[override]
        # ensure keyword-only is required from the caller perspective
        return ToolOutput(
            tool_name=self.metadata.name or "single_kw", content=input_values
        )


class TwoArgSumTool(BaseTool):
    """Two-argument tool summing integers via kwargs path.

    The tool uses a custom schema with two properties to exercise kwargs path.
    """

    class Args(BaseModel):
        """Arguments schema for TwoArgSumTool."""

        a: int
        b: int

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(
            description="Sum two numbers.", name="sum2", tool_schema=TwoArgSumTool.Args
        )

    def __call__(self, *, a: int, b: int) -> ToolOutput:  # type: ignore[override]
        return ToolOutput(tool_name=self.metadata.name or "sum2", content=str(a + b))


class ErrorTool(BaseTool):
    """Tool that raises an error for error handling testing."""

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(description="Always raises.", name="boom")

    def __call__(self, input_values: Any) -> ToolOutput:  # type: ignore[override]
        raise RuntimeError("boom")


class AsyncSingleArgKwOnlyTool(AsyncBaseTool):
    """Async variant requiring keyword-only argument for execute_async fallback."""

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(description="Async kw-only.", name="async_single_kw")

    def call(self, *, input_values: str) -> ToolOutput:  # type: ignore[override]
        # sync path delegates to same semantics as async
        return ToolOutput(
            tool_name=self.metadata.name or "async_single_kw", content=input_values
        )

    async def acall(self, *, input_values: str) -> ToolOutput:  # type: ignore[override]
        await asyncio.sleep(0)
        return ToolOutput(
            tool_name=self.metadata.name or "async_single_kw", content=input_values
        )


class AsyncTwoArgTool(AsyncBaseTool):
    """Async tool with two kwargs for execute_async validation."""

    class Args(BaseModel):
        """Arguments schema for AsyncTwoArgTool."""

        a: int
        b: int

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(
            description="Async sum.",
            name="async_sum2",
            tool_schema=AsyncTwoArgTool.Args,
        )

    def call(self, *, a: int, b: int) -> ToolOutput:  # type: ignore[override]
        return ToolOutput(
            tool_name=self.metadata.name or "async_sum2", content=str(a + b)
        )

    async def acall(self, *, a: int, b: int) -> ToolOutput:  # type: ignore[override]
        await asyncio.sleep(0)
        return ToolOutput(
            tool_name=self.metadata.name or "async_sum2", content=str(a + b)
        )


class AsyncErrorTool(AsyncBaseTool):
    """Async tool that raises for error handling testing."""

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(description="Async raises.", name="async_boom")

    def call(self, input_values: Any) -> ToolOutput:  # type: ignore[override]
        raise RuntimeError("boom")

    async def acall(self, input_values: Any) -> ToolOutput:  # type: ignore[override]
        raise RuntimeError("boom")


class TestToolExecutor:
    """Test suite for ToolExecutor."""

    class TestExecute:
        """Tests the ToolExecutor.execute method."""

        def test_single_arg_positional_forwarding(self) -> None:
            """A single-arg tool called with one-arg schema forwards a positional value.

            Inputs:
                - Tool with a single schema property (default MinimalToolSchema) and
                  __call__ that accepts a single positional input.
                - arguments={"input": "hello"}
            Expected:
                - ToolExecutor.execute detects one property and forwards the single value.
                - ToolOutput contains content "hello", is_error False, correct tool name.
            Checks:
                - Output content, is_error flag, and tool_name field.
            """
            tool = SingleArgEchoTool()
            tool_executor = ToolExecutor()
            out = tool_executor.execute(tool, {"input": "hello"})

            assert isinstance(out, ToolOutput)
            assert out.content == "hello"
            assert out.is_error is False
            assert out.tool_name == tool.metadata.name

        def test_single_arg_kwargs_fallback(self) -> None:
            """A single-arg tool requiring kw-only arg uses kwargs fallback path.

            Inputs:
                - Tool with a single schema property (default MinimalToolSchema) whose
                  __call__ signature requires a keyword-only parameter named 'input'.
                - arguments={"input": "world"}
            Expected:
                - The positional attempt fails; kwargs fallback is used successfully.
            Checks:
                - Output content is "world".
            """
            tool = SingleArgKwOnlyTool()
            tool_executor = ToolExecutor()
            out = tool_executor.execute(tool, {"input_values": "world"})

            assert out.content == "world"

        def test_multi_arg_kwargs(self) -> None:
            """A two-argument tool is called via kwargs path.

            Inputs:
                - Tool with custom two-field schema and __call__(*, a: int, b: int).
                - arguments={"a": 2, "b": 3}
            Expected:
                - ToolExecutor.execute uses kwargs path and returns sum as string.
            Checks:
                - Output content equals "5".
            """
            tool = TwoArgSumTool()
            tool_executor = ToolExecutor()
            out = tool_executor.execute(tool, {"a": 2, "b": 3})
            assert out.content == "5"

        def test_error_handling_returns_tool_output(self) -> None:
            """Errors raised by the tool are captured and returned as ToolOutput.

            Inputs:
                - Tool that raises RuntimeError("boom").
                - arguments={"input": "anything"}
            Expected:
                - ToolExecutor.execute returns ToolOutput with is_error=True, content prefixed with
                  "Encountered error:", and raw_input matches provided arguments.
            Checks:
                - is_error flag, content prefix, raw_input equality, and tool_name.
            """
            tool = ErrorTool()
            args = {"input": "anything"}
            tool_executor = ToolExecutor()
            out = tool_executor.execute(tool, args)
            assert out.is_error is True
            assert out.content.startswith("Encountered error: ")
            assert out.raw_input == args
            assert out.tool_name == tool.metadata.name

    class TestExecuteAsync:
        """Tests for ToolExecutor.execute_async method."""

        @pytest.mark.asyncio
        async def test_single_arg_positional_forwarding_async(self) -> None:
            """ToolExecutor.execute_async with a single-arg sync tool forwards positional value via adapter.

            Inputs:
                - Synchronous tool with single schema property and positional __call__.
                - arguments={"input": "hello"}
            Expected:
                - ToolExecutor.execute_async adapts the tool and returns the same content.
            Checks:
                - Output content is "hello" and not error.
            """
            tool = SingleArgEchoTool()
            tool_executor = ToolExecutor()
            out = await tool_executor.execute_async(tool, {"input": "hello"})
            assert out.content == "hello"
            assert out.is_error is False

        @pytest.mark.asyncio
        async def test_single_arg_kwargs_fallback_async(self) -> None:
            """ToolExecutor.execute_async uses kwargs fallback with an async kw-only tool.

            Inputs:
                - Async tool having acall(*, input: str), with single schema property.
                - arguments={"input": "world"}
            Expected:
                - First positional attempt fails due to kw-only; second kwargs attempt succeeds.
            Checks:
                - Output content is "world".
            """
            tool = AsyncSingleArgKwOnlyTool()
            tool_executor = ToolExecutor()
            out = await tool_executor.execute_async(tool, {"input_values": "world"})
            assert out.content == "world"

        @pytest.mark.asyncio
        async def test_multi_arg_kwargs_async(self) -> None:
            """ToolExecutor.execute_async calls async tool with multiple kwargs.

            Inputs:
                - Async tool with two arguments, acall(*, a: int, b: int).
                - arguments={"a": 4, "b": 6}
            Expected:
                - Sum returned as string "10".
            Checks:
                - Output content equals "10".
            """
            tool = AsyncTwoArgTool()
            tool_executor = ToolExecutor()
            out = await tool_executor.execute_async(tool, {"a": 4, "b": 6})
            assert out.content == "10"

        @pytest.mark.asyncio
        async def test_error_handling_async(self) -> None:
            """Errors in async tools are captured and returned as ToolOutput.

            Inputs:
                - Async tool that raises RuntimeError.
            Expected:
                - ToolOutput with is_error=True and error message prefix.
            Checks:
                - is_error is True and content prefix matches.
            """
            tool = AsyncErrorTool()
            tool_executor = ToolExecutor()
            out = await tool_executor.execute_async(tool, {"input": "x"})
            assert out.is_error is True
            assert out.content.startswith("Encountered error: ")

    class TestExecuteWithSelection:
        """Tests for ToolExecutor.execute_with_selection method."""

        def test_calls_correct_tool_and_propagates_output(
            self, capsys: pytest.CaptureFixture[str]
        ) -> None:
            """Ensure the correct tool is selected by name and the output is returned.

            Inputs:
                - ToolCallArguments with tool_name="single_echo" and tool_kwargs={"input": "ok"}.
                - tools list containing SingleArgEchoTool and TwoArgSumTool.
            Expected:
                - ToolExecutor.execute_with_selection selects the echo tool and returns content "ok".
            Checks:
                - Output content equals "ok".
            """
            tools: Sequence[BaseTool] = [SingleArgEchoTool(), TwoArgSumTool()]
            sel = ToolCallArguments(
                tool_id="1", tool_name="single_echo", tool_kwargs={"input": "ok"}
            )
            tool_executor = ToolExecutor()
            out = tool_executor.execute_with_selection(sel, tools)
            assert out.content == "ok"

        def test_verbose_prints_arguments_and_output(
            self, capsys: pytest.CaptureFixture[str]
        ) -> None:
            """Verify verbose mode prints the function call info and output content.

            Inputs:
                - ToolCallArguments for SingleArgEchoTool with input "zzz" and verbose=True.
            Expected:
                - Two sections printed: "=== Calling Function ===" and "=== Function Output ===".
                - The printed lines include tool name and arguments JSON, and the echoed output.
            Checks:
                - capsys output contains the expected substrings.
            """
            tools: Sequence[BaseTool] = [SingleArgEchoTool()]
            sel = ToolCallArguments(
                tool_id="2", tool_name="single_echo", tool_kwargs={"input": "zzz"}
            )
            tool_executor = ToolExecutor(ExecutionConfig(verbose=True))
            out = tool_executor.execute_with_selection(sel, tools)
            captured = capsys.readouterr()
            assert "=== Calling Function ===" in captured.out
            assert "single_echo" in captured.out
            assert '"input": "zzz"' in captured.out
            assert "=== Function Output ===" in captured.out
            assert out.content in captured.out

    class TestExecuteAsyncWithSelection:
        """Tests for ToolExecutor.execute_async_with_selection method."""

        @pytest.mark.asyncio
        async def test_calls_correct_tool_async_and_propagates_output(self) -> None:
            """Ensure async selection calls the right tool and returns the awaited output.

            Inputs:
                - ToolCallArguments targeting AsyncTwoArgTool with a=1, b=2.
            Expected:
                - Output content is "3".
            Checks:
                - Content equals "3".
            """
            tools: Sequence[BaseTool] = [AsyncTwoArgTool()]
            sel = ToolCallArguments(
                tool_id="3", tool_name="async_sum2", tool_kwargs={"a": 1, "b": 2}
            )
            tool_executor = ToolExecutor()
            out = await tool_executor.execute_async_with_selection(sel, tools)
            assert out.content == "3"

        @pytest.mark.asyncio
        async def test_verbose_prints_arguments_and_output_async(
            self, capsys: pytest.CaptureFixture[str]
        ) -> None:
            """Verbose mode prints details for async tool calls as well.

            Inputs:
                - ToolCallArguments for AsyncTwoArgTool with arguments and verbose=True.
            Expected:
                - Printed sections include call details and function output.
            Checks:
                - capsys captured output contains expected substrings.
            """
            tools: Sequence[BaseTool] = [AsyncTwoArgTool()]
            sel = ToolCallArguments(
                tool_id="4", tool_name="async_sum2", tool_kwargs={"a": 2, "b": 5}
            )
            tool_executor = ToolExecutor(ExecutionConfig(verbose=True))
            out = await tool_executor.execute_async_with_selection(sel, tools)
            captured = capsys.readouterr()
            assert "=== Calling Function ===" in captured.out
            assert "async_sum2" in captured.out
            assert '"a": 2' in captured.out and '"b": 5' in captured.out
            assert "=== Function Output ===" in captured.out
            assert out.content in captured.out
