"""Tests for tool models."""

import asyncio
import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from serapeum.core.base.llms.models import Audio, Image, TextChunk
from serapeum.core.tools.models import (
    ArgumentCoercer,
    AsyncBaseTool,
    BaseTool,
    BaseToolAsyncAdapter,
    MinimalToolSchema,
    Schema,
    ToolCallArguments,
    ToolMetadata,
    ToolOutput,
    adapt_to_async_tool,
)


class DummySyncTool(BaseTool):
    """Dummy sync tool for testing."""

    def __init__(self, name: str = "dummy_sync"):
        """Initialize DummySyncTool."""
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
    """Dummy async tool for testing."""

    def __init__(self, name: str = "dummy_async"):
        """Initialize DummyAsyncTool."""
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
    """Test suite for MinimalToolSchema."""

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
    """Test suite for ToolMetadata.get_schema method."""

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


class TestSchema:
    """Test suite for Schema."""

    def test_default_schema_filtered(self):
        """Test that default Pydantic schema is filtered to allowed keys.

        Inputs:
          - ToolMetadata with tool_schema=MinimalToolSchema.
        Expected:
          - The schema dict only contains filtered keys and has an `input` string property.
        Checks:
          - Only expected keys exist; `input` under properties is present and string-typed.
        """
        schema = Schema(full_schema=MinimalToolSchema.model_json_schema())
        params = schema.resolve_references()
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

        schema = Schema(full_schema=MainModel.model_json_schema())
        params = schema.resolve_references()
        assert params == {
            "$defs": {
                "SubModel": {
                    "properties": {"x": {"title": "X", "type": "integer"}},
                    "required": ["x"],
                    "title": "SubModel",
                    "type": "object",
                }
            },
            "properties": {"sub": {"$ref": "#/$defs/SubModel"}},
            "required": ["sub"],
            "type": "object",
        }

        params = schema.resolve_references(inline=True)
        assert params == {
            "properties": {
                "sub": {
                    "properties": {"x": {"title": "X", "type": "integer"}},
                    "required": ["x"],
                    "title": "SubModel",
                    "type": "object",
                }
            },
            "required": ["sub"],
            "type": "object",
        }


class TestToolMetadataFnSchemaStr:
    """Test suite for ToolMetadata.tool_schema_str property."""

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
    """Test suite for ToolMetadata.get_name method."""

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
    """Test suite for ToolMetadata.to_openai_tool method."""

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
        assert (
            tool["function"]["description"]
            == "desc\n\nRequired fields:\n  - input (string)"
        )
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
        assert (
            tool["function"]["description"]
            == f"{long_desc}\n\nRequired fields:\n  - input (string)"
        )


class TestToolOutput:
    """Test suite for ToolOutput."""

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
        out = ToolOutput(
            tool_name="t", chunks=chunks, raw_input={"x": 1}, raw_output=None
        )
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
    """Tests for AsyncBaseTool.__call__ dispatching to call."""

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
    """Tests for BaseToolAsyncAdapter."""

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
    """Tests for adapt_to_async_tool."""

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


class TestToolCallArguments:
    """Test suite for ToolCallArguments."""

    def test_valid_arguments_with_dict_kwargs(self):
        """Validate standard construction with proper types.

        Inputs:
          - tool_id: "call-001" (string)
          - tool_name: "echo" (string)
          - tool_kwargs: {"text": "hi", "count": 2} (dict)
        Expected:
          - Model is created successfully with fields preserved as provided.
        Checks:
          - tool_id and tool_name match inputs.
          - tool_kwargs is the exact same mapping content provided.
        """
        sel = ToolCallArguments(
            tool_id="call-001", tool_name="echo", tool_kwargs={"text": "hi", "count": 2}
        )
        assert sel.tool_id == "call-001"
        assert sel.tool_name == "echo"
        assert sel.tool_kwargs == {"text": "hi", "count": 2}

    def test_non_dict_kwargs_string_coerced_to_empty_dict(self):
        """Ensure non-dict tool_kwargs (string) are coerced to {} by validator.

        Inputs:
          - tool_kwargs: "not-a-dict" (string)
        Expected:
          - tool_kwargs is replaced with an empty dict instead of raising.
        Checks:
          - sel.tool_kwargs == {}.
        """
        sel = ToolCallArguments(
            tool_id="id-1", tool_name="echo", tool_kwargs="not-a-dict"
        )
        assert sel.tool_kwargs == {}

    def test_non_dict_kwargs_list_coerced_to_empty_dict(self):
        """Ensure non-dict tool_kwargs (list) are coerced to {} by validator.

        Inputs:
          - tool_kwargs: [("a", 1)] (list)
        Expected:
          - tool_kwargs is replaced with an empty dict instead of raising.
        Checks:
          - sel.tool_kwargs == {}.
        """
        sel = ToolCallArguments(tool_id="id-2", tool_name="sum", tool_kwargs=[("a", 1)])
        assert sel.tool_kwargs == {}

    def test_none_kwargs_coerced_to_empty_dict(self):
        """Ensure None passed for tool_kwargs is coerced to {}.

        Inputs:
          - tool_kwargs: None
        Expected:
          - tool_kwargs is replaced with {} via the wrap validator.
        Checks:
          - sel.tool_kwargs == {}.
        """
        sel = ToolCallArguments(tool_id="id-3", tool_name="echo", tool_kwargs=None)  # type: ignore[arg-type]
        assert sel.tool_kwargs == {}

    def test_missing_tool_name_raises_validation_error(self):
        """Verify that tool_name is a required field.

        Inputs:
          - tool_id present, tool_kwargs present, tool_name missing.
        Expected:
          - pydantic.ValidationError is raised due to missing required field.
        Checks:
          - Raised exception type is ValidationError.
        """
        with pytest.raises(ValidationError):
            ToolCallArguments(tool_id="only-id", tool_kwargs={})  # type: ignore[call-arg]

    def test_missing_tool_id_raises_validation_error(self):
        """Verify that tool_id is a required field.

        Inputs:
          - tool_name present, tool_kwargs present, tool_id missing.
        Expected:
          - pydantic.ValidationError is raised due to missing required field.
        Checks:
          - Raised exception type is ValidationError.
        """
        with pytest.raises(ValidationError):
            ToolCallArguments(tool_name="only-name", tool_kwargs={})  # type: ignore[call-arg]

    def test_missing_tool_kwargs_raises_validation_error(self):
        """Verify that omitting tool_kwargs entirely is not allowed.

        Inputs:
          - tool_id and tool_name provided; tool_kwargs omitted.
        Expected:
          - pydantic.ValidationError is raised before field validator runs.
        Checks:
          - Raised exception type is ValidationError.
        """
        with pytest.raises(ValidationError):
            ToolCallArguments(tool_id="id", tool_name="echo")  # type: ignore[call-arg]

    def test_nested_tool_kwargs_preserved(self):
        """Validate that nested structures inside tool_kwargs are preserved.

        Inputs:
          - tool_kwargs: {"filters": {"tags": ["x", "y"], "limit": 10}}
        Expected:
          - Nested dict/list structure is accepted as-is.
        Checks:
          - Returned tool_kwargs equals the input mapping deep-equal.
        """
        payload = {"filters": {"tags": ["x", "y"], "limit": 10}}
        sel = ToolCallArguments(
            tool_id="nested", tool_name="search", tool_kwargs=payload
        )
        assert sel.tool_kwargs == payload

    def test_extra_fields_are_ignored(self):
        """Ensure extra/unknown fields do not break model construction.

        Inputs:
          - An extra field `extra_field` included in constructor.
        Expected:
          - Model is created successfully; extra field is ignored by default config.
        Checks:
          - getattr(sel, "extra_field", None) is None (not set as attribute).
        """
        sel = ToolCallArguments(tool_id="e1", tool_name="echo", tool_kwargs={}, extra_field=123)  # type: ignore[call-arg]
        assert getattr(sel, "extra_field", None) is None


class TestArgumentCoercer:
    """Test suite for the ArgumentCoercer class.

    Tests cover all methods and edge cases for coercing LLM-returned arguments
    to match expected tool schema types.
    """

    class TestParseJsonString:
        """Test suite for _parse_json_string method."""

        def test_parse_json_string_with_valid_dict_string(self):
            """Test parsing a valid JSON string containing a dictionary.

            Inputs:
              - raw_arguments: '{"a": 1, "b": "hello"}'
            Expected:
              - Returns parsed dict: {"a": 1, "b": "hello"}
            Checks:
              - Result is a dict with correct keys and values.
            """
            coercer = ArgumentCoercer()
            result = coercer._parse_json_string('{"a": 1, "b": "hello"}')
            assert result == {"a": 1, "b": "hello"}

        def test_parse_json_string_with_dict_input(self):
            """Test that passing a dict directly returns it unchanged.

            Inputs:
              - raw_arguments: {"x": 10, "y": 20} (already a dict)
            Expected:
              - Returns the same dict unchanged.
            Checks:
              - Result equals input dict.
            """
            coercer = ArgumentCoercer()
            input_dict = {"x": 10, "y": 20}
            result = coercer._parse_json_string(input_dict)
            assert result == input_dict

        def test_parse_json_string_with_invalid_json(self):
            """Test that invalid JSON string returns empty dict.

            Inputs:
              - raw_arguments: '{invalid json}' (malformed JSON)
            Expected:
              - Returns empty dict {} to handle gracefully.
            Checks:
              - Result is an empty dict.
            """
            coercer = ArgumentCoercer()
            result = coercer._parse_json_string("{invalid json}")
            assert result == {}

        def test_parse_json_string_with_json_array(self):
            """Test that JSON array string returns empty dict (not a dict).

            Inputs:
              - raw_arguments: '[1, 2, 3]' (JSON array, not object)
            Expected:
              - Returns empty dict because result is not a dictionary.
            Checks:
              - Result is an empty dict.
            """
            coercer = ArgumentCoercer()
            result = coercer._parse_json_string("[1, 2, 3]")
            assert result == {}

        def test_parse_json_string_with_empty_string(self):
            """Test that empty string returns empty dict.

            Inputs:
              - raw_arguments: '' (empty string)
            Expected:
              - Returns empty dict due to JSON parsing failure.
            Checks:
              - Result is an empty dict.
            """
            coercer = ArgumentCoercer()
            result = coercer._parse_json_string("")
            assert result == {}

        def test_parse_json_string_with_non_string_non_dict(self):
            """Test that non-string, non-dict input returns empty dict.

            Inputs:
              - raw_arguments: 123 (integer)
            Expected:
              - Returns empty dict as fallback.
            Checks:
              - Result is an empty dict.
            """
            coercer = ArgumentCoercer()
            result = coercer._parse_json_string(123)  # type: ignore[arg-type]
            assert result == {}

    class TestTryParseStringValue:
        """Test suite for _try_parse_string_value method."""

        def test_try_parse_string_value_with_json_list(self):
            """Test parsing a string containing a JSON list.

            Inputs:
              - value: '[1, 2, 3]' (JSON array string)
            Expected:
              - Returns parsed list: [1, 2, 3]
            Checks:
              - Result is a list with correct values.
            """
            coercer = ArgumentCoercer()
            result = coercer._try_parse_string_value("[1, 2, 3]")
            assert result == [1, 2, 3]

        def test_try_parse_string_value_with_json_dict(self):
            """Test parsing a string containing a JSON object.

            Inputs:
              - value: '{"key": "value"}' (JSON object string)
            Expected:
              - Returns parsed dict: {"key": "value"}
            Checks:
              - Result is a dict with correct key-value.
            """
            coercer = ArgumentCoercer()
            result = coercer._try_parse_string_value('{"key": "value"}')
            assert result == {"key": "value"}

        def test_try_parse_string_value_with_json_number(self):
            """Test parsing a string containing a JSON number.

            Inputs:
              - value: '42' (JSON number string)
            Expected:
              - Returns parsed int: 42
            Checks:
              - Result is an integer.
            """
            coercer = ArgumentCoercer()
            result = coercer._try_parse_string_value("42")
            assert result == 42

        def test_try_parse_string_value_with_json_boolean(self):
            """Test parsing a string containing a JSON boolean.

            Inputs:
              - value: 'true' (JSON boolean string)
            Expected:
              - Returns parsed bool: True
            Checks:
              - Result is True.
            """
            coercer = ArgumentCoercer()
            result = coercer._try_parse_string_value("true")
            assert result is True

        def test_try_parse_string_value_with_plain_string(self):
            """Test that plain text string returns unchanged.

            Inputs:
              - value: 'hello world' (not valid JSON)
            Expected:
              - Returns the original string unchanged.
            Checks:
              - Result equals input string.
            """
            coercer = ArgumentCoercer()
            result = coercer._try_parse_string_value("hello world")
            assert result == "hello world"

        def test_try_parse_string_value_with_non_string(self):
            """Test that non-string values return unchanged.

            Inputs:
              - value: [1, 2, 3] (already a list)
            Expected:
              - Returns the list unchanged.
            Checks:
              - Result equals input list.
            """
            coercer = ArgumentCoercer()
            input_list = [1, 2, 3]
            result = coercer._try_parse_string_value(input_list)
            assert result == input_list

        def test_try_parse_string_value_with_nested_json(self):
            """Test parsing deeply nested JSON string.

            Inputs:
              - value: '{"outer": {"inner": [1, 2, 3]}}'
            Expected:
              - Returns parsed nested structure.
            Checks:
              - Result matches expected nested dict.
            """
            coercer = ArgumentCoercer()
            result = coercer._try_parse_string_value('{"outer": {"inner": [1, 2, 3]}}')
            assert result == {"outer": {"inner": [1, 2, 3]}}

    class TestsParseStringFields:
        """Test suite for _parse_string_fields method."""

        def test_parse_string_fields_with_mixed_types(self):
            """Test parsing dict with mixed value types.

            Inputs:
              - argument_dict: {"a": "[1,2,3]", "b": "plain", "c": 42}
            Expected:
              - "a" is parsed to list [1,2,3]
              - "b" remains string "plain"
              - "c" remains int 42
            Checks:
              - Result has correct types for each field.
            """
            coercer = ArgumentCoercer()
            input_dict = {"a": "[1,2,3]", "b": "plain", "c": 42}
            result = coercer._parse_string_fields(input_dict)
            assert result == {"a": [1, 2, 3], "b": "plain", "c": 42}

        def test_parse_string_fields_with_all_json_strings(self):
            """Test parsing dict where all values are JSON strings.

            Inputs:
              - argument_dict: {"nums": "[10, 20]", "flag": "true", "nested": '{"x": 1}'}
            Expected:
              - All values are parsed to their JSON representations.
            Checks:
              - nums becomes list, flag becomes bool, nested becomes dict.
            """
            coercer = ArgumentCoercer()
            input_dict = {"nums": "[10, 20]", "flag": "true", "nested": '{"x": 1}'}
            result = coercer._parse_string_fields(input_dict)
            assert result == {"nums": [10, 20], "flag": True, "nested": {"x": 1}}

        def test_parse_string_fields_with_empty_dict(self):
            """Test parsing empty dict returns empty dict.

            Inputs:
              - argument_dict: {}
            Expected:
              - Returns empty dict.
            Checks:
              - Result is empty dict.
            """
            coercer = ArgumentCoercer()
            result = coercer._parse_string_fields({})
            assert result == {}

        def test_parse_string_fields_preserves_non_string_values(self):
            """Test that non-string values are preserved unchanged.

            Inputs:
              - argument_dict: {"list": [1, 2], "dict": {"a": 1}, "int": 5}
            Expected:
              - All values remain unchanged.
            Checks:
              - Result equals input dict.
            """
            coercer = ArgumentCoercer()
            input_dict = {"list": [1, 2], "dict": {"a": 1}, "int": 5}
            result = coercer._parse_string_fields(input_dict)
            assert result == input_dict

    class TestsValidateWithSchema:
        """Test suite for _validate_with_schema method."""

        def test_validate_with_schema_no_schema(self):
            """Test validation when no schema is provided returns input unchanged.

            Inputs:
              - tool_schema: None
              - coerced_dict: {"a": "5", "b": "text"}
            Expected:
              - Returns input dict unchanged (no validation).
            Checks:
              - Result equals input dict.
            """
            coercer = ArgumentCoercer(tool_schema=None)
            input_dict = {"a": "5", "b": "text"}
            result = coercer._validate_with_schema(input_dict)
            assert result == input_dict

        def test_validate_with_schema_coerces_types(self):
            """Test that Pydantic schema coerces string types to expected types.

            Inputs:
              - tool_schema: Model with int and bool fields
              - coerced_dict: {"count": "42", "active": "true"}
            Expected:
              - Pydantic coerces "42" to int 42 and "true" to bool True.
            Checks:
              - Result has correct types after validation.
            """

            class TestSchema(BaseModel):
                count: int
                active: bool

            coercer = ArgumentCoercer(tool_schema=TestSchema)
            input_dict = {"count": "42", "active": "true"}
            result = coercer._validate_with_schema(input_dict)
            assert result == {"count": 42, "active": True}

        def test_validate_with_schema_handles_list_fields(self):
            """Test that Pydantic schema validates list fields correctly.

            Inputs:
              - tool_schema: Model with list[float] field
              - coerced_dict: {"numbers": [1.0, 2.0, 3.0]}
            Expected:
              - List is validated and returned correctly.
            Checks:
              - Result has list field with correct values.
            """

            class TestSchema(BaseModel):
                numbers: list[float]

            coercer = ArgumentCoercer(tool_schema=TestSchema)
            input_dict = {"numbers": [1.0, 2.0, 3.0]}
            result = coercer._validate_with_schema(input_dict)
            assert result == {"numbers": [1.0, 2.0, 3.0]}

        def test_validate_with_schema_validation_fails_returns_input(self):
            """Test that validation failure returns input dict unchanged.

            Inputs:
              - tool_schema: Model requiring specific fields
              - coerced_dict: {"wrong_field": "value"} (invalid for schema)
            Expected:
              - Validation fails, returns input dict as fallback.
            Checks:
              - Result equals input dict (fallback behavior).
            """

            class TestSchema(BaseModel):
                required_field: str

            coercer = ArgumentCoercer(tool_schema=TestSchema)
            input_dict = {"wrong_field": "value"}
            result = coercer._validate_with_schema(input_dict)
            assert result == input_dict

        def test_validate_with_schema_with_optional_fields(self):
            """Test validation with schema containing optional fields.

            Inputs:
              - tool_schema: Model with optional field
              - coerced_dict: {"name": "test"} (missing optional field)
            Expected:
              - Validation succeeds, optional field gets default value.
            Checks:
              - Result has default value for optional field.
            """

            class TestSchema(BaseModel):
                name: str
                count: int = 0

            coercer = ArgumentCoercer(tool_schema=TestSchema)
            input_dict = {"name": "test"}
            result = coercer._validate_with_schema(input_dict)
            assert result == {"name": "test", "count": 0}

    class TestIntegration:
        """Test suite for ArgumentCoercer integration tests."""

        def test_coerce_integration_json_string_to_typed_dict(self):
            """Test full coercion pipeline: JSON string -> parsed -> validated.

            Inputs:
              - raw_arguments: '{"numbers": "[1.0, 2.0]", "operation": "sum"}'
              - tool_schema: Model with list[float] and str fields
            Expected:
              - JSON string is parsed, nested list is parsed, types are validated.
            Checks:
              - Result has correctly typed fields.
            """

            class ToolSchema(BaseModel):
                numbers: list[float]
                operation: str

            coercer = ArgumentCoercer(tool_schema=ToolSchema)
            raw_args = '{"numbers": "[1.0, 2.0]", "operation": "sum"}'
            result = coercer.coerce(raw_args)
            assert result == {"numbers": [1.0, 2.0], "operation": "sum"}

        def test_coerce_integration_dict_with_string_fields(self):
            """Test coercion with dict input containing stringified fields.

            Inputs:
              - raw_arguments: {"count": "5", "items": "[10, 20, 30]"}
              - tool_schema: Model with int and list[int] fields
            Expected:
              - String fields are parsed and types are coerced.
            Checks:
              - count is int 5, items is list [10, 20, 30].
            """

            class ToolSchema(BaseModel):
                count: int
                items: list[int]

            coercer = ArgumentCoercer(tool_schema=ToolSchema)
            raw_args = {"count": "5", "items": "[10, 20, 30]"}
            result = coercer.coerce(raw_args)
            assert result == {"count": 5, "items": [10, 20, 30]}

        def test_coerce_integration_no_schema(self):
            """Test coercion without schema does JSON parsing only.

            Inputs:
              - raw_arguments: '{"data": "[1, 2, 3]", "name": "test"}'
              - tool_schema: None
            Expected:
              - JSON is parsed, nested string is parsed, but no type coercion.
            Checks:
              - Result has parsed JSON structure.
            """
            coercer = ArgumentCoercer(tool_schema=None)
            raw_args = '{"data": "[1, 2, 3]", "name": "test"}'
            result = coercer.coerce(raw_args)
            assert result == {"data": [1, 2, 3], "name": "test"}

        def test_coerce_integration_invalid_json_returns_empty_dict(self):
            """Test that invalid JSON returns empty dict gracefully.

            Inputs:
              - raw_arguments: '{broken json' (malformed)
            Expected:
              - Returns empty dict without raising exception.
            Checks:
              - Result is empty dict.
            """
            coercer = ArgumentCoercer()
            result = coercer.coerce("{broken json")
            assert result == {}

        def test_coerce_integration_complex_nested_structure(self):
            """Test coercion with complex nested structure.

            Inputs:
              - raw_arguments: dict with nested dicts and lists as strings
              - tool_schema: Model with nested structure
            Expected:
              - All nested strings are parsed correctly.
            Checks:
              - Result has fully parsed nested structure.
            """

            class NestedSchema(BaseModel):
                config: dict[str, Any]
                values: list[int]

            coercer = ArgumentCoercer(tool_schema=NestedSchema)
            raw_args = {"config": '{"key": "value"}', "values": "[1, 2, 3]"}
            result = coercer.coerce(raw_args)
            assert result == {"config": {"key": "value"}, "values": [1, 2, 3]}

        def test_coerce_integration_preserves_already_correct_types(self):
            """Test that already-correct types are preserved through coercion.

            Inputs:
              - raw_arguments: {"count": 42, "items": [1, 2, 3], "name": "test"}
              - tool_schema: Matching schema
            Expected:
              - All values pass through unchanged.
            Checks:
              - Result equals input.
            """

            class ToolSchema(BaseModel):
                count: int
                items: list[int]
                name: str

            coercer = ArgumentCoercer(tool_schema=ToolSchema)
            raw_args = {"count": 42, "items": [1, 2, 3], "name": "test"}
            result = coercer.coerce(raw_args)
            assert result == {"count": 42, "items": [1, 2, 3], "name": "test"}

        def test_coerce_integration_empty_dict_returns_empty_dict(self):
            """Test that empty dict input returns empty dict.

            Inputs:
              - raw_arguments: {}
            Expected:
              - Returns empty dict.
            Checks:
              - Result is empty dict.
            """
            coercer = ArgumentCoercer()
            result = coercer.coerce({})
            assert result == {}
