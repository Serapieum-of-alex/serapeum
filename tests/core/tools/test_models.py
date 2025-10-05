import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from serapeum.core.base.llms.models import TextChunk, Image, Audio
from serapeum.core.tools.models import (
    MinimalToolSchema,
    ToolMetadata,
    ToolOutput,
    BaseTool,
    AsyncBaseTool,
    BaseToolAsyncAdapter,
    adapt_to_async_tool,
)


class DummySyncTool(BaseTool):
    def __init__(self, name: str = "dummy_sync"):
        self._metadata = ToolMetadata(description="sync tool desc", name=name)

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def __call__(self, input_values: Any) -> ToolOutput:
        # Return a ToolOutput echoing the input
        return ToolOutput(
            tool_name=self.metadata.get_name(),
            chunks=[TextChunk(content=str(input_values))],
            raw_input={"input": input_values},
            raw_output=input_values,
        )


class DummyAsyncTool(AsyncBaseTool):
    def __init__(self, name: str = "dummy_async"):
        self._metadata = ToolMetadata(description="async tool desc", name=name)

    @property
    def metadata(self) -> ToolMetadata:
        return self._metadata

    def call(self, input: Any) -> ToolOutput:
        # Synchronous path echoes input
        return ToolOutput(
            tool_name=self.metadata.get_name(),
            chunks=[TextChunk(content=str(input))],
            raw_input={"input": input},
            raw_output=input,
        )

    async def acall(self, input: Any) -> ToolOutput:
        # Async path also echoes input
        await asyncio.sleep(0)
        return self.call(input)


class TestDefaultToolFnSchema:
    def test_valid_input(self):
        """Test creating MinimalToolSchema with a valid string.

        Inputs:
          - input: a valid string ("hello").
        Expected:
          - Model is created successfully with the provided string.
        Checks:
          - The `input` field equals the provided string.
        """
        model = MinimalToolSchema(input="hello")
        assert model.input == "hello"

    def test_invalid_input_type(self):
        """Test that non-string input raises a ValidationError.

        Inputs:
          - input: a non-string value (123).
        Expected:
          - pydantic.ValidationError is raised because `input` must be a string.
        Checks:
          - ValidationError is raised.
        """
        with pytest.raises(ValidationError):
            MinimalToolSchema(input=123)  # type: ignore[arg-type]


class TestToolMetadataGetParametersDict:
    def test_tool_schema_none_uses_default_object(self):
        """Test that when tool_schema is None, a default object schema is returned.

        Inputs:
          - ToolMetadata with tool_schema=None.
        Expected:
          - Returned dict has type object, properties with `input` string and required ["input"].
        Checks:
          - Keys present and values match the default structure.
        """
        meta = ToolMetadata(description="desc", name="tool", tool_schema=None)
        params = meta.get_schema()
        assert params["type"] == "object"
        assert "properties" in params and "input" in params["properties"]
        assert params["properties"]["input"]["type"] == "string"
        assert params["required"] == ["input"]

    def test_default_schema_filtered(self):
        """Test that default Pydantic schema is filtered to allowed keys.

        Inputs:
          - ToolMetadata with tool_schema=MinimalToolSchema.
        Expected:
          - The schema dict only contains filtered keys and has an `input` string property.
        Checks:
          - Only expected keys exist; `input` under properties is present and string-typed.
        """
        meta = ToolMetadata(description="desc", name="tool", tool_schema=MinimalToolSchema)
        params = meta.get_schema()
        for k in params.keys():
            assert k in {"type", "properties", "required", "definitions", "$defs"}
        assert params["properties"]["input"]["type"] == "string"

    def test_nested_schema_includes_defs(self):
        """Test that nested Pydantic models preserve $defs/definitions when present.

        Inputs:
          - A custom schema with a nested sub-model to generate $defs/definitions.
        Expected:
          - get_parameters_dict returns filtered schema that still includes "$defs" or "definitions".
        Checks:
          - Either "$defs" or "definitions" key exists in the returned dict.
        """
        class SubModel(BaseModel):
            x: int

        class MainModel(BaseModel):
            sub: SubModel

        meta = ToolMetadata(description="desc", name="tool", tool_schema=MainModel)
        params = meta.get_schema()
        assert any(key in params for key in ("$defs", "definitions"))
        assert params["type"] == "object"
        assert "sub" in params["properties"]


class TestToolMetadataFnSchemaStr:
    def test_returns_json_string(self):
        """Test tool_schema_str returns a JSON string for a valid schema.

        Inputs:
          - ToolMetadata with default tool_schema.
        Expected:
          - JSON string decodes to the same dict as get_parameters_dict.
        Checks:
          - json.loads(tool_schema_str) equals get_parameters_dict().
        """
        meta = ToolMetadata(description="desc", name="tool")
        params = meta.get_schema()
        json_str = meta.tool_schema_str
        assert isinstance(json_str, str)
        assert json.loads(json_str) == params

    def test_raises_when_schema_none(self):
        """Test that tool_schema_str raises when tool_schema is None.

        Inputs:
          - ToolMetadata with tool_schema=None.
        Expected:
          - ValueError with message "tool_schema is None." is raised.
        Checks:
          - ValueError is raised and message matches.
        """
        meta = ToolMetadata(description="desc", name="tool", tool_schema=None)
        with pytest.raises(ValueError, match="tool_schema is None."):
            _ = meta.tool_schema_str


class TestToolMetadataGetName:
    def test_returns_name(self):
        """Test get_name returns the provided name.

        Inputs:
          - ToolMetadata with a non-None name.
        Expected:
          - get_name returns the same name.
        Checks:
          - Returned value equals the original name.
        """
        meta = ToolMetadata(description="desc", name="mytool")
        assert meta.get_name() == "mytool"

    def test_raises_when_name_none(self):
        """Test get_name raises when name is None.

        Inputs:
          - ToolMetadata with name=None.
        Expected:
          - ValueError with message "name is None." is raised.
        Checks:
          - ValueError is raised and message matches.
        """
        meta = ToolMetadata(description="desc", name=None)
        with pytest.raises(ValueError, match="name is None."):
            _ = meta.get_name()


class TestToolMetadataToOpenAITool:
    def test_returns_correct_structure(self):
        """Test the returned OpenAI tool structure is correct.

        Inputs:
          - ToolMetadata with valid name, description, and default schema.
        Expected:
          - A dict with keys: type='function' and function containing name, description, parameters.
        Checks:
          - Structure and values match expectations and parameters come from get_parameters_dict.
        """
        meta = ToolMetadata(description="desc", name="tool")
        tool = meta.to_openai_tool()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "tool"
        assert tool["function"]["description"] == "desc"
        assert tool["function"]["parameters"] == meta.get_schema()

    def test_raises_on_long_description(self):
        """Test that overly long description raises unless skipping length check.

        Inputs:
          - ToolMetadata with description length 1025.
        Expected:
          - ValueError is raised when skip_length_check=False.
        Checks:
          - Error message about exceeding 1024 characters.
        """
        long_desc = "x" * 1025
        meta = ToolMetadata(description=long_desc, name="tool")
        with pytest.raises(ValueError, match="exceeds maximum length of 1024"):
            _ = meta.to_openai_tool()

    def test_skip_length_check_allows_long_description(self):
        """Test skip_length_check allows a long description without error.

        Inputs:
          - ToolMetadata with long description and skip_length_check=True.
        Expected:
          - No exception and structure is returned.
        Checks:
          - Returned dict structure is valid and contains the provided long description.
        """
        long_desc = "x" * 5000
        meta = ToolMetadata(description=long_desc, name="tool")
        tool = meta.to_openai_tool(skip_length_check=True)
        assert tool["function"]["description"] == long_desc


class TestToolOutput:

    def test_init_with_content_and_raw_input_populates_chunks(self):
        """Test that providing content with raw_input creates a single TextChunk.

        Inputs:
          - tool_name: "t"
          - content: "hello"
          - raw_input: {}
        Expected:
          - chunks contains one TextChunk with content "hello"; content property returns "hello".
        Checks:
          - chunks length, type, and content; content property value.
        """
        out = ToolOutput(tool_name="t", content="hello", raw_input={}, raw_output=None)
        assert len(out.chunks) == 1 and isinstance(out.chunks[0], TextChunk)
        assert out.chunks[0].content == "hello"
        assert out.content == "hello"

    def test_init_with_blocks_and_raw_input(self):
        """Test that providing chunks with raw_input uses the given chunks.

        Inputs:
          - tool_name: "t"
          - chunks: [TextChunk("a"), TextChunk("b")]
          - raw_input: {"x": 1}
        Expected:
          - chunks preserved in the model; content joins text chunks with newline.
        Checks:
          - chunks list equals provided; content equals "a\nb".
        """
        chunks = [TextChunk(content="a"), TextChunk(content="b")]
        out = ToolOutput(tool_name="t", chunks=chunks, raw_input={"x": 1}, raw_output=None)
        assert out.chunks == chunks
        assert out.content == "a\nb"

    def test_init_with_both_content_and_blocks_raises(self):
        """Test that providing both content and chunks raises ValueError.

        Inputs:
          - tool_name: "t"
          - content: "x"
          - chunks: [TextChunk("y")]
          - raw_input: {}
        Expected:
          - ValueError with message about both content and chunks provided.
        Checks:
          - ValueError is raised and message matches.
        """
        with pytest.raises(ValueError, match="Cannot provide both content and chunks"):
            ToolOutput(
                tool_name="t",
                content="x",
                chunks=[TextChunk(content="y")],
                raw_input={},
                raw_output=None,
            )

    def test_content_property_filters_non_text_chunks(self):
        """Test that content getter concatenates only TextChunk contents.

        Inputs:
          - chunks: [TextChunk("a"), Image(url=...), Audio(url=...), TextChunk("b")]
        Expected:
          - content returns "a\nb" (non-text chunks ignored).
        Checks:
          - Exact string equality for content.
        """
        chunks = [
            TextChunk(content="a"),
            Image(url="https://example.com/img.png"),
            Audio(url="https://example.com/snd.mp3"),
            TextChunk(content="b"),
        ]
        out = ToolOutput(tool_name="t", chunks=chunks, raw_input={}, raw_output=None)
        assert out.content == "a\nb"

    def test_content_setter_overwrites_chunks(self):
        """Test that setting content replaces chunks with a single TextChunk.

        Inputs:
          - Start with two TextChunks then set content to "new".
        Expected:
          - chunks becomes a single TextChunk("new"); content equals "new".
        Checks:
          - Length of chunks is one and its content equals "new".
        """
        out = ToolOutput(
            tool_name="t",
            chunks=[TextChunk(content="old1"), TextChunk(content="old2")],
            raw_input={},
            raw_output=None,
        )
        out.content = "new"
        assert len(out.chunks) == 1
        assert isinstance(out.chunks[0], TextChunk)
        assert out.chunks[0].content == "new"
        assert out.content == "new"

    def test_str_returns_content(self):
        """Test that __str__ returns the same as content.

        Inputs:
          - ToolOutput with content "hello" and raw_input {}.
        Expected:
          - str(out) equals "hello".
        Checks:
          - Exact equality between str(out) and out.content.
        """
        out = ToolOutput(tool_name="t", content="hello", raw_input={}, raw_output=None)
        assert str(out) == "hello"


class TestAsyncBaseToolCall:
    """Tests for AsyncBaseTool.__call__ dispatching to call"""

    def test_dunder_call_dispatches_to_call(self):
        """Test that AsyncBaseTool.__call__ dispatches to the sync call method.

        Inputs:
          - A DummyAsyncTool instance; call using callable syntax with input 42.
        Expected:
          - Result is same as calling .call(42): a ToolOutput echoing "42".
        Checks:
          - Content equals "42" and tool_name equals the tool's name.
        """
        tool = DummyAsyncTool(name="async_tool")
        out1 = tool(42)
        out2 = tool.call(42)
        assert out1.content == "42" == out2.content
        assert out1.tool_name == "async_tool" == out2.tool_name


class TestBaseToolAsyncAdapter:
    """Tests for BaseToolAsyncAdapter"""

    @pytest.mark.asyncio
    async def test_metadata_and_call_and_acall(self):
        """Test that the adapter proxies metadata and delegates call/acall.

        Inputs:
          - A DummySyncTool wrapped by BaseToolAsyncAdapter, input value "ping".
        Expected:
          - .metadata matches base tool metadata; .call/.acall both echo "ping".
        Checks:
          - tool_name is base tool name; content equals "ping" for both call and acall.
        """
        base = DummySyncTool(name="sync_base")
        adapter = BaseToolAsyncAdapter(base)
        # metadata proxy
        assert adapter.metadata is base.metadata
        # sync call path
        out_sync = adapter.call("ping")
        assert out_sync.tool_name == "sync_base"
        assert out_sync.content == "ping"
        # async call path runs in a thread
        out_async = await adapter.acall("ping")
        assert out_async.tool_name == "sync_base"
        assert out_async.content == "ping"


class TestAdaptToAsyncTool:
    """Tests for adapt_to_async_tool"""

    @pytest.mark.asyncio
    async def test_returns_same_for_async_tool(self):
        """Test that an AsyncBaseTool is returned unchanged by adapt_to_async_tool.

        Inputs:
          - A DummyAsyncTool instance.
        Expected:
          - adapt_to_async_tool returns the exact same object (identity preserved).
        Checks:
          - The returned object `is` the original tool and acall works.
        """
        async_tool = DummyAsyncTool(name="async")
        adapted = adapt_to_async_tool(async_tool)
        assert adapted is async_tool
        out = await adapted.acall("x")
        assert out.content == "x"

    @pytest.mark.asyncio
    async def test_wraps_sync_tool(self):
        """Test that a sync BaseTool is wrapped into a BaseToolAsyncAdapter.

        Inputs:
          - A DummySyncTool instance.
        Expected:
          - adapt_to_async_tool returns a BaseToolAsyncAdapter whose acall echoes input.
        Checks:
          - isinstance check for adapter and content/value checks on acall output.
        """
        sync_tool = DummySyncTool(name="sync")
        adapted = adapt_to_async_tool(sync_tool)
        assert isinstance(adapted, BaseToolAsyncAdapter)
        out = await adapted.acall(123)
        assert out.tool_name == "sync"
        assert out.content == "123"
