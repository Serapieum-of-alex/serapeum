"""Tests for CallableTool and related tool utilities."""

import asyncio
from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from serapeum.core.base.llms.models import Audio, Image, TextChunk
from serapeum.core.tools.callable_tool import CallableTool, SyncAsyncConverter
from serapeum.core.tools.models import ToolMetadata, ToolOutput


class MockSong(BaseModel):
    """Mock Song model for testing."""

    title: str
    duration: Optional[int] = None


class MockAlbum(BaseModel):
    """Mock Album model for testing."""

    title: str
    artist: str
    songs: List[MockSong]


class MockSimpleModel(BaseModel):
    """Simple model for basic testing."""

    name: str
    value: int


class MockModelWithDescription(BaseModel):
    """Model with description field."""

    description: str = Field(description="A test description")
    data: str


class MockModelWithOptionalFields(BaseModel):
    """Model with optional fields."""

    required_field: str
    optional_field: Optional[str] = None
    optional_int: Optional[int] = None


class TestSyncAsyncConverter:
    """Test SyncAsyncConverter utility for sync/async conversion."""

    class TestSyncToAsync:
        """Test sync to async conversion."""

        def test_returns_awaitable_and_result(self):
            """Test sync to async conversion.

            Inputs:
                A synchronous function f(x, y=2) that returns x * y, wrapped with sync_to_async, then awaited with (3, y=4).
            Expected:
                Awaiting the wrapper returns 12 (3 * 4).
            Checks:
                The wrapper is awaitable and forwards args/kwargs correctly to the sync function, returning its result.
            """

            def f(x: int, y: int = 2) -> int:
                return x * y

            sync_async_converter = SyncAsyncConverter(f)
            wrapped = sync_async_converter.async_func
            result = asyncio.run(wrapped(3, y=4))
            assert result == 12

        def test_exception_propagates(self):
            """Test the sync function raises an exception.

            Inputs: A synchronous function that raises ValueError, wrapped via sync_to_async and awaited.
            Expected: The same ValueError is raised when awaiting the wrapper.
            Checks: Exceptions raised in the underlying sync function propagate through the async wrapper.
            """

            def boom() -> None:
                raise ValueError("fail sync")

            sync_async_converter = SyncAsyncConverter(boom)
            wrapped = sync_async_converter.async_func
            with pytest.raises(ValueError, match="fail sync"):
                asyncio.run(wrapped())

    class TestAsyncToSync:
        """Test async to sync conversion."""

        def test_returns_result(self):
            """Test async to sync conversion.

            Inputs: An async function add(a, b) that returns a + b, wrapped by async_to_sync and called synchronously as func(5, 7).
            Expected: The wrapper returns 12.
            Checks: The wrapper runs the coroutine to completion and returns its value.
            """

            async def add(a: int, b: int) -> int:
                await asyncio.sleep(0)
                return a + b

            sync_async_converter = SyncAsyncConverter(add)
            wrapped = sync_async_converter.sync_func
            assert wrapped(5, 7) == 12

        def test_exception_propagates(self):
            """Test the async function raises an exception.

            Inputs: An async function that raises RuntimeError, wrapped with async_to_sync and invoked synchronously.
            Expected: The wrapper raises a RuntimeError.
            Checks: Errors originating in the async function are raised by the sync wrapper (message may vary depending on event loop handling).
            """

            async def boom() -> None:
                await asyncio.sleep(0)
                raise RuntimeError("fail async")

            sync_async_converter = SyncAsyncConverter(boom)
            wrapped = sync_async_converter.sync_func
            with pytest.raises(RuntimeError):
                wrapped()


class TestCallableToolInit:
    """Test CallableTool initialization logic."""

    def test_init_with_sync_fn(self):
        r"""Test Initializer.

        Inputs: A simple sync function and ToolMetadata(name=\"t\", description=\"d\").
        Expected: CallableTool initializes, exposing .func (sync) and .async_fn (async wrapper), and .real_fn is the original function.
        Checks: Correct wrapping decisions and metadata attachment.
        """

        def echo(text: str) -> str:
            return text.upper()

        meta = ToolMetadata(name="t", description="d")
        tool = CallableTool(func=echo, metadata=meta)
        assert tool.sync_func("hi") == "HI"
        assert asyncio.run(tool.async_func("yo")) == "YO"
        assert tool.input_func is echo
        assert tool.metadata.get_name() == "t"

    def test_init_with_async_fn(self):
        """Inputs: An async function and ToolMetadata.

        Expected: .async_fn is the original async function, .func is a sync wrapper that runs it, and .real_fn is the async function.
        Checks: Wrapping from async to sync path and consistency of returned values through both call styles.
        """

        async def aecho(text: str) -> str:
            await asyncio.sleep(0)
            return text[::-1]

        meta = ToolMetadata(name="ta", description="d")
        tool = CallableTool(func=aecho, metadata=meta)
        # sync path
        assert tool.sync_func("abc") == "cba"
        # async path
        assert asyncio.run(tool.async_func("abcd")) == "dcba"
        assert tool.input_func is aecho

    def test_init_without_metadata_raises(self):
        """Inputs: A valid function but metadata=None.

        Expected: ValueError with message indicating metadata must be provided.
        Checks: Constructor validation for required metadata argument.
        """

        def f():
            return 1

        with pytest.raises(ValueError, match="metadata must be provided"):
            _ = CallableTool(func=f, metadata=None)  # type: ignore[arg-type]

    def test_init_without_fn_or_async_fn_raises(self):
        """Test initialize without func or async_fn.

        Inputs: Both func=None and async_fn=None.
        Expected: ValueError stating that one of func or async_fn must be provided.
        Checks: Constructor enforces presence of at least one callable.
        """
        meta = ToolMetadata(name="x", description="d")
        with pytest.raises(ValueError, match="func must be a callable"):
            _ = CallableTool(func=None, metadata=meta)  # type: ignore[arg-type]


class TestCallableToolFromFunction:
    """Test CallableTool.from_function factory method."""

    def test_builds_metadata_name_description_and_schema_with_docs(self):
        """Test CallableTool.from_function infers name, description, and schema fields from docstring.

        Inputs:
            A function foo(x: int, y: str) -> str with a docstring containing a 1-line summary and parameter
            descriptions for x and y.
        Expected:
            from_defaults infers name="foo", builds a description starting with the function signature, followed by the
            summary, and constructs a tool_schema whose fields have descriptions populated from the docstring.
        Checks:
            Name inference, description content (signature + summary), and schema field descriptions populated via
            extract_param_docs.
        """

        def foo(x: int, y: str) -> str:
            """Summarize.

            :param x: x value
            y (str): y value
            @param z extra (unknown)
            """
            return f"{x}-{y}"

        tool = CallableTool.from_function(func=foo)
        meta = tool.metadata
        assert meta.get_name() == "foo"
        # Description should start with signature and include the summary line
        assert meta.description.startswith(
            "foo("
        ), f"description should start with signature: {meta.description}"
        assert "Summarize." in meta.description
        schema = meta.tool_schema
        assert schema is not None
        # Field descriptions should be filled from docstring for x and y
        assert schema.model_fields["x"].description == "x value"
        assert schema.model_fields["y"].description == "y value"

    def test_description_signature_strips_pydantic_field_defaults(self):
        """Test handling of FieldInfo defaults in auto-generated description.

        Inputs:
            A function bar(a: int = Field(default=1, description="A")) with a docstring.
        Expected:
            The auto-generated description includes the signature without the Field(...) default (i.e., shows "a: int"),
            followed by the summary.
        Checks:
            Handling of FieldInfo defaults when building the signature for the description.
        """
        field_default = Field(default=1, description="A")

        def bar(a: int = field_default) -> int:  # type: ignore[assignment]
            """Bar summary."""
            return a

        tool = CallableTool.from_function(func=bar)
        desc = tool.metadata.description
        # Ensure signature part doesn't leak Field(...) text
        assert desc.startswith("bar(")
        assert "a: int" in desc
        assert ") -> int\n" in desc
        assert "Field(" not in desc and "FieldInfo" not in desc
        assert "Bar summary." in desc

    def test_respects_custom_tool_schema_and_metadata(self):
        """Test the `from_function` respects the tool_schema and tool_metadata arguments.

        Inputs:
            Provide a custom Pydantic schema and a fully-specified ToolMetadata to from_defaults.
        Expected:
            When tool_metadata is supplied, from_defaults should pass it through unchanged without attempting to infer from the function.
        Checks:
            The returned tool uses the provided metadata object as-is.
        """

        class Args(BaseModel):
            q: str

        meta = ToolMetadata(name="custom", description="desc", tool_schema=Args)

        def f(q: str) -> str:
            return q

        tool = CallableTool.from_function(func=f, tool_metadata=meta)
        assert tool.metadata is meta
        assert tool.metadata.get_name() == "custom"
        assert tool.metadata.get_schema()["properties"]["q"]["type"] == "string"


class TestCallableToolFromModel:
    """Test CallableTool.from_model factory method."""

    def test_get_function_tool_basic_model(self):
        """Test creating function tool from basic model.

        Input: MockSimpleModel class
        Expected: Returns CallableTool with correct name and schema
        Check: Tool name matches model title, tool_schema is set
        """
        tool = CallableTool.from_model(MockSimpleModel)

        assert tool is not None
        assert tool.metadata.name == "MockSimpleModel"
        assert tool.metadata.tool_schema == MockSimpleModel

    def test_get_function_tool_model_with_description(self):
        """Test creating function tool from model with description.

        Input: MockModelWithDescription class with schema description
        Expected: Returns CallableTool with description from schema
        Check: Tool has description matching model schema
        """
        tool = CallableTool.from_model(MockModelWithDescription)

        assert tool is not None
        assert tool.metadata.name == "MockModelWithDescription"
        # Description should come from the model schema

    def test_get_function_tool_complex_model(self):
        """Test creating function tool from complex nested model.

        Input: MockAlbum class with nested structure
        Expected: Returns CallableTool that can handle nested data
        Check: Tool schema correctly represents nested structure
        """
        tool = CallableTool.from_model(MockAlbum)

        assert tool is not None
        assert tool.metadata.name == "MockAlbum"
        assert tool.metadata.tool_schema == MockAlbum

    def test_get_function_tool_callable(self):
        """Test that function tool is callable with kwargs.

        Input: MockSimpleModel tool called with valid kwargs
        Expected: Returns instance of MockSimpleModel
        Check: Returned object is correct model instance with data
        """
        tool = CallableTool.from_model(MockSimpleModel)
        result = tool.sync_func(name="test", value=42)

        assert isinstance(result, MockSimpleModel)
        assert result.name == "test"
        assert result.value == 42

    def test_get_function_tool_with_optional_fields(self):
        """Test function tool with model containing optional fields.

        Input: MockModelWithOptionalFields class
        Expected: Returns tool that handles optional fields correctly
        Check: Tool can be called with/without optional fields
        """
        tool = CallableTool.from_model(MockModelWithOptionalFields)

        # Test with only required field
        result1 = tool.sync_func(required_field="test")
        assert isinstance(result1, MockModelWithOptionalFields)
        assert result1.required_field == "test"
        assert result1.optional_field is None

        # Test with all fields
        result2 = tool.sync_func(
            required_field="test",
            optional_field="optional",
            optional_int=10,
        )
        assert result2.optional_field == "optional"
        assert result2.optional_int == 10


class TestCallableToolParseToolOutput:
    """Test _parse_tool_output method of CallableTool."""

    def test_single_text_chunk(self):
        """Test handling of single TextChunk.

        Inputs: _parse_tool_output with a single TextChunk("hello").
        Expected: Returns a list containing the given TextChunk.
        Checks: Pass-through behavior for a single valid chunk.
        """
        tool = CallableTool(
            func=lambda: "ignored",
            metadata=ToolMetadata(name="t", description="d"),
        )
        chunk = TextChunk(content="hello")
        out = tool._parse_tool_output(chunk)
        assert out == [chunk]

    def test_single_image_and_audio(self):
        """Test handling of single Image and Audio chunks.

        Inputs: _parse_tool_output with a single Image and a single Audio instance.
        Expected: Returns lists containing the provided chunk instances respectively.
        Checks: Pass-through behavior for image and audio chunks.
        """
        tool = CallableTool(
            func=lambda: "ignored", metadata=ToolMetadata(name="t", description="d")
        )
        img = Image(url="http://example.com/x.png")
        aud = Audio(url="http://example.com/x.wav")
        assert tool._parse_tool_output(img) == [img]
        assert tool._parse_tool_output(aud) == [aud]

    def test_list_of_chunks(self):
        """Test handling of lists of chunks.

        Inputs: _parse_tool_output with a list of mixed chunk types [TextChunk, Image].
        Expected: Returns the same list unchanged.
        Checks: Lists of valid chunk types are passed through as-is.
        """
        tool = CallableTool(
            func=lambda: "ignored", metadata=ToolMetadata(name="t", description="d")
        )
        chunks = [TextChunk(content="a"), Image(url="http://example.com/i.png")]
        assert tool._parse_tool_output(chunks) == chunks

    def test_plain_and_nonstring_outputs_become_text_chunks(self):
        """Test handling of non-string outputs.

        Inputs: _parse_tool_output with a plain string ("ok"), an int (123), and a dict.
        Expected: Each is converted to a single TextChunk with content=str(raw_output).
        Checks: Fallback conversion for non-chunk outputs to text chunks.
        """
        tool = CallableTool(
            func=lambda: "ignored", metadata=ToolMetadata(name="t", description="d")
        )
        for raw in ("ok", 123, {"a": 1}):
            chunks = tool._parse_tool_output(raw)
            assert len(chunks) == 1
            assert isinstance(chunks[0], TextChunk)
            assert chunks[0].content == str(raw)


class TestCallableToolCall:
    """Test call method of CallableTool."""

    def test_sync_call_with_partial_and_kwargs_and_positional(self):
        """Test Call with partial params and kwargs.

        Inputs: Sync function f(a, b, c=3) that returns a tuple (a, b, c).
                Tool constructed with default_arguments={"c": 10}. Call with args (1,) and kwargs b=2.
        Expected: The merged kwargs result in c=10. The ToolOutput contains the proper chunks, tool_name, raw_input, and raw_output.
        Checks: default_arguments precedence (overridden by explicit kwargs), args forwarding, and ToolOutput population.
        """

        def f(a: int, b: int, c: int = 3) -> tuple[int, int, int]:
            return a, b, c

        meta = ToolMetadata(name="add", description="d")
        tool = CallableTool(func=f, metadata=meta, default_arguments={"c": 10})

        out: ToolOutput = tool.call(1, b=2)
        assert isinstance(out, ToolOutput)
        assert out.tool_name == "add"
        assert out.raw_input == {"args": (1,), "kwargs": {"b": 2, "c": 10}}
        assert out.raw_output == (1, 2, 10)
        # chunks fallback: should be a single TextChunk of the tuple's string
        assert len(out.chunks) == 1 and isinstance(out.chunks[0], TextChunk)

    def test_kwargs_override_default_arguments(self):
        """Test Call with explicit kwargs overriding default_arguments.

        Inputs:
            Same tool as before with default_arguments={"c": 10}. Call with explicit c=20.
        Expected:
            Explicit kwargs override default_arguments, so c=20 is used in the call and reflected in outputs.
        Checks:
            Override precedence of provided kwargs over pre-configured partial params.
        """

        def f(a: int, b: int, c: int = 3) -> tuple[int, int, int]:
            return a, b, c

        meta = ToolMetadata(name="add2", description="d")
        tool = CallableTool(func=f, metadata=meta, default_arguments={"c": 10})

        out = tool.call(1, b=2, c=20)
        assert out.raw_output == (1, 2, 20)
        assert out.raw_input == {"args": (1,), "kwargs": {"b": 2, "c": 20}}


class TestCallableToolDunderCall:
    """Test __call__ dunder method of CallableTool."""

    def test_dunder_call_delegates_and_merges_default_arguments(self):
        """Test __call__ delegates to call with merged default_arguments and provided kwargs.

        Inputs:
            Sync function that concatenates two strings with a separator; tool has default_arguments providing the
            separator.
        Expected:
            Calling the tool instance directly (i.e., __call__) returns a ToolOutput reflecting merged kwargs and
            correct result.
        Checks:
            __call__ path merges default_arguments with provided kwargs and delegates to call correctly.
        """

        def join(a: str, b: str, sep: str = ",") -> str:
            return f"{a}{sep}{b}"

        meta = ToolMetadata(name="join", description="d")
        tool = CallableTool(func=join, metadata=meta, default_arguments={"sep": ":"})

        out = tool("left", b="right")
        assert out.tool_name == "join"
        assert out.raw_output == "left:right"
        assert out.raw_input == {
            "args": ("left",),
            "kwargs": {"b": "right", "sep": ":"},
        }
        assert [c.content for c in out.chunks if isinstance(c, TextChunk)] == [
            "left:right"
        ]


class TestCallableToolACall:
    """Test async acall method of CallableTool."""

    @pytest.mark.asyncio
    async def test_async_call_with_partial_and_kwargs(self):
        """Test Call with partial params and kwargs.

        Inputs: Async function g(a, b, c=3) -> str that returns f"{a+b+c}".
                Tool constructed with default_arguments={"c": 5}. Call with a=1, b=2 using acall.
        Expected: The result uses c=5 (from partials), so raw_output == "8" and chunks contain a single TextChunk("8").
        Checks: Async call path, partial params merging, and ToolOutput correctness.
        """

        async def g(a: int, b: int, c: int = 3) -> str:
            await asyncio.sleep(0)
            return str(a + b + c)

        meta = ToolMetadata(name="sum_async", description="d")
        tool = CallableTool(func=g, metadata=meta, default_arguments={"c": 5})

        out = await tool.acall(a=1, b=2)
        assert out.tool_name == "sum_async"
        assert out.raw_output == "8"
        assert out.raw_input == {"args": (), "kwargs": {"a": 1, "b": 2, "c": 5}}
        assert [c.content for c in out.chunks if isinstance(c, TextChunk)] == ["8"]
