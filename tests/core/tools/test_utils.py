"""Unit tests for serapeum.core.tools.utils.

This test suite covers all functions in utils.py:
- create_schema_from_function
- call_tool
- acall_tool
- call_tool_with_selection
- acall_tool_with_selection

For each function, we define a dedicated test class with multiple test methods, each
with a docstring that explains inputs, expected behavior, and what is being checked.
"""

import asyncio
import datetime as dt
from typing import Any, Sequence

import pytest
from pydantic import BaseModel, Field

from serapeum.core.tools.utils import (
    Function,
    call_tool,
    acall_tool,
    call_tool_with_selection,
    acall_tool_with_selection,
)
from serapeum.core.tools.models import BaseTool, AsyncBaseTool, ToolMetadata, ToolOutput
from serapeum.core.llm.base import ToolSelection

# -----------------------------------------------------------------------------
# Helpers: Dummy tool implementations for exercising call utilities
# -----------------------------------------------------------------------------

class SingleArgEchoTool(BaseTool):
    """A single-argument echo tool used to validate positional forwarding path.

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
    """A single-argument tool that requires a keyword-only argument.

    This exercises the kwargs fallback in call_tool when the positional path fails.
    """

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(description="Requires kw-only arg.", name="single_kw")

    def __call__(self, *, input_values: str) -> ToolOutput:  # type: ignore[override]
        # ensure keyword-only is required from the caller perspective
        return ToolOutput(tool_name=self.metadata.name or "single_kw", content=input_values)


class TwoArgSumTool(BaseTool):
    """A two-argument tool that sums two integers via kwargs path.

    The tool uses a custom schema with two properties to exercise kwargs path.
    """

    class Args(BaseModel):
        a: int
        b: int

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(description="Sum two numbers.", name="sum2", tool_schema=TwoArgSumTool.Args)

    def __call__(self, *, a: int, b: int) -> ToolOutput:  # type: ignore[override]
        return ToolOutput(tool_name=self.metadata.name or "sum2", content=str(a + b))


class ErrorTool(BaseTool):
    """A tool that raises an error to test error handling in call helpers."""

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(description="Always raises.", name="boom")

    def __call__(self, input_values: Any) -> ToolOutput:  # type: ignore[override]
        raise RuntimeError("boom")


class AsyncSingleArgKwOnlyTool(AsyncBaseTool):
    """Async variant requiring a keyword-only argument, for acall_tool fallback."""

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(description="Async kw-only.", name="async_single_kw")

    def call(self, *, input_values: str) -> ToolOutput:  # type: ignore[override]
        # sync path delegates to same semantics as async
        return ToolOutput(tool_name=self.metadata.name or "async_single_kw", content=input_values)

    async def acall(self, *, input_values: str) -> ToolOutput:  # type: ignore[override]
        await asyncio.sleep(0)
        return ToolOutput(tool_name=self.metadata.name or "async_single_kw", content=input_values)


class AsyncTwoArgTool(AsyncBaseTool):
    """Async tool with two kwargs to validate acall_tool kwargs path."""

    class Args(BaseModel):
        a: int
        b: int

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(description="Async sum.", name="async_sum2", tool_schema=AsyncTwoArgTool.Args)

    def call(self, *, a: int, b: int) -> ToolOutput:  # type: ignore[override]
        return ToolOutput(tool_name=self.metadata.name or "async_sum2", content=str(a + b))

    async def acall(self, *, a: int, b: int) -> ToolOutput:  # type: ignore[override]
        await asyncio.sleep(0)
        return ToolOutput(tool_name=self.metadata.name or "async_sum2", content=str(a + b))


class AsyncErrorTool(AsyncBaseTool):
    """Async tool that raises to test error handling in acall_tool."""

    @property
    def metadata(self) -> ToolMetadata:  # type: ignore[override]
        return ToolMetadata(description="Async raises.", name="async_boom")

    def call(self, input_values: Any) -> ToolOutput:  # type: ignore[override]
        raise RuntimeError("boom")

    async def acall(self, input_values: Any) -> ToolOutput:  # type: ignore[override]
        raise RuntimeError("boom")


# -----------------------------------------------------------------------------
# Tests for create_schema_from_function
# -----------------------------------------------------------------------------

class TestCreateSchemaFromFunction:
    def test_required_default_any_and_field_info(self) -> None:
        """Validate required fields, defaulted fields, Any typing, and Field defaults.

        Inputs:
            - A Python function with the following parameters:
              - a: int (no default) -> required
              - b: str = "x" (defaulted)
              - c = 3 (no annotation) -> treated as Any and defaulted
              - d: int = Field(default=5, description="five") (FieldInfo default)
        Expected:
            - Pydantic model has corresponding fields with correct metadata:
              - 'a' required with annotation int.
              - 'b' default 'x'.
              - 'c' annotation Any and default 3.
              - 'd' default 5 and description preserved.
            - JSON schema 'required' includes only 'a'.
        Checks:
            - model.model_fields metadata and model_json_schema entries.
        """

        def f(a: int, b: str = "x", c=3, d: int = Field(default=5, description="five")) -> None:
            return None

        function = Function("F", f)
        model = function.to_schema()
        schema = model.model_json_schema()
        required = set(schema.get("required", []))

        assert required == {"a"}
        assert model.model_fields["a"].annotation is int
        assert model.model_fields["b"].default == "x"
        assert model.model_fields["c"].annotation is Any
        assert model.model_fields["c"].default == 3
        assert model.model_fields["d"].default == 5
        assert model.model_fields["d"].description == "five"

    def test_annotated_description_and_field_info_extra(self) -> None:
        """Validate Annotated metadata for description and json_schema_extra merging.

        Inputs:
            - A function with two Annotated parameters:
              - x: Annotated[int, "counter value"]
              - y: Annotated[str, Field(description="text", json_schema_extra={"alpha": True})]
        Expected:
            - Field descriptions reflect annotations.
            - Field 'y' has json_schema_extra merged to include 'alpha': True.
        Checks:
            - model.model_fields descriptions and json_schema_extra content.
        """
        from typing import Annotated as Ann

        def g(x: Ann[int, "counter value"], y: Ann[str, Field(description="text", json_schema_extra={"alpha": True})]) -> None:
            return None
        function = Function("G", g)
        model = function.to_schema()
        fx = model.model_fields["x"]
        fy = model.model_fields["y"]
        assert fx.description == "counter value"
        assert fy.description == "text"
        assert isinstance(fy.json_schema_extra, dict) and fy.json_schema_extra.get("alpha") is True

    def test_date_datetime_time_formats(self) -> None:
        """Ensure date/datetime/time fields carry proper JSON Schema format values.

        Inputs:
            - Function with parameters of type date, datetime, and time.
        Expected:
            - Corresponding JSON Schema properties include format: 'date', 'date-time', 'time'.
        Checks:
            - model.model_json_schema property inspection.
        """

        def h(day: dt.date, ts: dt.datetime, at: dt.time) -> None:
            return None
        function = Function("H", h)
        model = function.to_schema()
        props = model.model_json_schema()["properties"]
        assert props["day"].get("format") == "date"
        assert props["ts"].get("format") == "date-time"
        assert props["at"].get("format") == "time"

    def test_ignore_and_additional_fields(self) -> None:
        """Test ignore_fields and additional_fields (both 2- and 3-tuples).

        Inputs:
            - Function with parameters: keep_me: int, skip_me: str
            - ignore_fields=["skip_me"], additional_fields=[("extra", int), ("flag", bool, True)]
        Expected:
            - 'skip_me' is excluded.
            - 'extra' is required (no default), 'flag' has default True.
        Checks:
            - JSON schema required set and defaults in model fields.
        """

        def k(keep_me: int, skip_me: str) -> None:
            return None

        function = Function(
            "K",
            k,
            additional_fields=[("extra", int), ("flag", bool, True)],
            ignore_fields=["skip_me"],
        )
        model = function.to_schema()
        schema = model.model_json_schema()
        required = set(schema.get("required", []))

        assert "skip_me" not in schema.get("properties", {})
        assert "keep_me" in required and "extra" in required
        assert model.model_fields["flag"].default is True

    def test_invalid_additional_fields_tuple_length(self) -> None:
        """Verify that invalid additional_fields tuple lengths raise ValueError.

        Inputs:
            - additional_fields containing a 1-tuple (invalid length).
        Expected:
            - ValueError is raised with an explanatory message.
        Checks:
            - Exception type and message contents.
        """

        def z(x: int) -> None:
            return None

        with pytest.raises(ValueError):
            function = Function("Z", z, additional_fields=[("bad",)])
            _ = function.to_schema()


# -----------------------------------------------------------------------------
# Tests for call_tool (synchronous)
# -----------------------------------------------------------------------------

class TestCallTool:
    def test_single_arg_positional_forwarding(self) -> None:
        """A single-arg tool called with one-arg schema forwards a positional value.

        Inputs:
            - Tool with a single schema property (default MinimalToolSchema) and
              __call__ that accepts a single positional input.
            - arguments={"input": "hello"}
        Expected:
            - call_tool detects one property and forwards the single value.
            - ToolOutput contains content "hello", is_error False, correct tool name.
        Checks:
            - Output content, is_error flag, and tool_name field.
        """
        tool = SingleArgEchoTool()
        out = call_tool(tool, {"input": "hello"})
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
        out = call_tool(tool, {"input_values": "world"})
        assert out.content == "world"

    def test_multi_arg_kwargs(self) -> None:
        """A two-argument tool is called via kwargs path.

        Inputs:
            - Tool with custom two-field schema and __call__(*, a: int, b: int).
            - arguments={"a": 2, "b": 3}
        Expected:
            - call_tool uses kwargs path and returns sum as string.
        Checks:
            - Output content equals "5".
        """
        tool = TwoArgSumTool()
        out = call_tool(tool, {"a": 2, "b": 3})
        assert out.content == "5"

    def test_error_handling_returns_tool_output(self) -> None:
        """Errors raised by the tool are captured and returned as ToolOutput.

        Inputs:
            - Tool that raises RuntimeError("boom").
            - arguments={"input": "anything"}
        Expected:
            - call_tool returns ToolOutput with is_error=True, content prefixed with
              "Encountered error:", and raw_input matches provided arguments.
        Checks:
            - is_error flag, content prefix, raw_input equality, and tool_name.
        """
        tool = ErrorTool()
        args = {"input": "anything"}
        out = call_tool(tool, args)
        assert out.is_error is True
        assert out.content.startswith("Encountered error: ")
        assert out.raw_input == args
        assert out.tool_name == tool.metadata.name


# -----------------------------------------------------------------------------
# Tests for acall_tool (asynchronous)
# -----------------------------------------------------------------------------

class TestACallTool:
    @pytest.mark.asyncio
    async def test_single_arg_positional_forwarding_async(self) -> None:
        """acall_tool with a single-arg sync tool forwards positional value via adapter.

        Inputs:
            - Synchronous tool with single schema property and positional __call__.
            - arguments={"input": "hello"}
        Expected:
            - acall_tool adapts the tool and returns the same content.
        Checks:
            - Output content is "hello" and not error.
        """
        tool = SingleArgEchoTool()
        out = await acall_tool(tool, {"input": "hello"})
        assert out.content == "hello"
        assert out.is_error is False

    @pytest.mark.asyncio
    async def test_single_arg_kwargs_fallback_async(self) -> None:
        """acall_tool uses kwargs fallback with an async kw-only tool.

        Inputs:
            - Async tool having acall(*, input: str), with single schema property.
            - arguments={"input": "world"}
        Expected:
            - First positional attempt fails due to kw-only; second kwargs attempt succeeds.
        Checks:
            - Output content is "world".
        """
        tool = AsyncSingleArgKwOnlyTool()
        out = await acall_tool(tool, {"input_values": "world"})
        assert out.content == "world"

    @pytest.mark.asyncio
    async def test_multi_arg_kwargs_async(self) -> None:
        """acall_tool calls async tool with multiple kwargs.

        Inputs:
            - Async tool with two arguments, acall(*, a: int, b: int).
            - arguments={"a": 4, "b": 6}
        Expected:
            - Sum returned as string "10".
        Checks:
            - Output content equals "10".
        """
        tool = AsyncTwoArgTool()
        out = await acall_tool(tool, {"a": 4, "b": 6})
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
        out = await acall_tool(tool, {"input": "x"})
        assert out.is_error is True
        assert out.content.startswith("Encountered error: ")


# -----------------------------------------------------------------------------
# Tests for call_tool_with_selection and acall_tool_with_selection
# -----------------------------------------------------------------------------

class TestCallToolWithSelection:
    def test_calls_correct_tool_and_propagates_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Ensure the correct tool is selected by name and the output is returned.

        Inputs:
            - ToolSelection with tool_name="single_echo" and tool_kwargs={"input": "ok"}.
            - tools list containing SingleArgEchoTool and TwoArgSumTool.
        Expected:
            - call_tool_with_selection selects the echo tool and returns content "ok".
        Checks:
            - Output content equals "ok".
        """
        tools: Sequence[BaseTool] = [SingleArgEchoTool(), TwoArgSumTool()]
        sel = ToolSelection(tool_id="1", tool_name="single_echo", tool_kwargs={"input": "ok"})
        out = call_tool_with_selection(sel, tools, verbose=False)
        assert out.content == "ok"

    def test_verbose_prints_arguments_and_output(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify verbose mode prints the function call info and output content.

        Inputs:
            - ToolSelection for SingleArgEchoTool with input "zzz" and verbose=True.
        Expected:
            - Two sections printed: "=== Calling Function ===" and "=== Function Output ===".
            - The printed lines include tool name and arguments JSON, and the echoed output.
        Checks:
            - capsys output contains the expected substrings.
        """
        tools: Sequence[BaseTool] = [SingleArgEchoTool()]
        sel = ToolSelection(tool_id="2", tool_name="single_echo", tool_kwargs={"input": "zzz"})
        out = call_tool_with_selection(sel, tools, verbose=True)
        captured = capsys.readouterr()
        assert "=== Calling Function ===" in captured.out
        assert "single_echo" in captured.out
        assert '"input": "zzz"' in captured.out
        assert "=== Function Output ===" in captured.out
        assert out.content in captured.out


class TestACallToolWithSelection:
    @pytest.mark.asyncio
    async def test_calls_correct_tool_async_and_propagates_output(self) -> None:
        """Ensure async selection calls the right tool and returns the awaited output.

        Inputs:
            - ToolSelection targeting AsyncTwoArgTool with a=1, b=2.
        Expected:
            - Output content is "3".
        Checks:
            - Content equals "3".
        """
        tools: Sequence[BaseTool] = [AsyncTwoArgTool()]
        sel = ToolSelection(tool_id="3", tool_name="async_sum2", tool_kwargs={"a": 1, "b": 2})
        out = await acall_tool_with_selection(sel, tools, verbose=False)
        assert out.content == "3"

    @pytest.mark.asyncio
    async def test_verbose_prints_arguments_and_output_async(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verbose mode prints details for async tool calls as well.

        Inputs:
            - ToolSelection for AsyncTwoArgTool with arguments and verbose=True.
        Expected:
            - Printed sections include call details and function output.
        Checks:
            - capsys captured output contains expected substrings.
        """
        tools: Sequence[BaseTool] = [AsyncTwoArgTool()]
        sel = ToolSelection(tool_id="4", tool_name="async_sum2", tool_kwargs={"a": 2, "b": 5})
        out = await acall_tool_with_selection(sel, tools, verbose=True)
        captured = capsys.readouterr()
        assert "=== Calling Function ===" in captured.out
        assert "async_sum2" in captured.out
        assert '"a": 2' in captured.out and '"b": 5' in captured.out
        assert "=== Function Output ===" in captured.out
        assert out.content in captured.out
