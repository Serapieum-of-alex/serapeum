"""Comprehensive tests for serapeum.openai.converters — ChatMessageConverter,
ResponsesMessageConverter, and to_openai_message_dicts.

These serve as behavioural golden tests prior to a syntax refactor.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from serapeum.core.base.llms.types import (
    Audio,
    DocumentBlock,
    Image,
    Message,
    MessageRole,
    TextChunk,
    ThinkingBlock,
    ToolCallBlock,
)
from serapeum.openai.parsers import (
    ChatFormat,
    ResponsesFormat,
    _rewrite_system_to_developer,
    _should_null_content,
    _strip_none_keys,
    ChatMessageConverter,
    ResponsesMessageConverter,
    to_openai_message_dicts,
)


def _make_document_block() -> DocumentBlock:
    """Create a DocumentBlock with pre-encoded base64 data so no I/O is needed
    for the as_base64 mock."""
    block = MagicMock(spec=DocumentBlock)
    block.title = "report.pdf"
    block.as_base64.return_value = ("cGRmZGF0YQ==", "application/pdf")
    # Make isinstance checks work
    block.__class__ = DocumentBlock
    return block


def _make_image_inline_block() -> Image:
    """Create an Image block that has no URL, so the converter calls as_data_uri."""
    block = MagicMock(spec=Image)
    block.url = None
    block.detail = None
    block.as_data_uri.return_value = "data:image/png;base64,abc123"
    block.__class__ = Image
    return block


def _make_audio_block() -> Audio:
    """Create an Audio block whose resolve_audio returns a BytesIO with base64 text."""
    block = MagicMock(spec=Audio)
    block.format = "mp3"
    block.resolve_audio.return_value = BytesIO(b"YXVkaW9kYXRh")
    block.__class__ = Audio
    return block



@pytest.mark.unit
class TestHelperFunctions:
    """Tests for the private helper functions used by the converters."""

    def test_rewrite_system_to_developer_with_o3_mini(self) -> None:
        """System role is rewritten to developer for o3-mini (in O1_MODELS, not in
        O1_MODELS_WITHOUT_FUNCTION_CALLING)."""
        msg: dict[str, Any] = {"role": "system", "content": "you are helpful"}
        _rewrite_system_to_developer(msg, "o3-mini")
        assert msg["role"] == "developer", (
            "Expected system role to be rewritten to developer for o3-mini"
        )

    def test_rewrite_system_to_developer_no_rewrite_for_o1_mini(self) -> None:
        """System role is NOT rewritten for o1-mini (in O1_MODELS_WITHOUT_FUNCTION_CALLING)."""
        msg: dict[str, Any] = {"role": "system", "content": "you are helpful"}
        _rewrite_system_to_developer(msg, "o1-mini")
        assert msg["role"] == "system", (
            "Expected system role to remain unchanged for o1-mini"
        )

    def test_rewrite_system_to_developer_no_rewrite_for_non_o1_model(self) -> None:
        """System role is NOT rewritten for a regular model like gpt-4o."""
        msg: dict[str, Any] = {"role": "system", "content": "you are helpful"}
        _rewrite_system_to_developer(msg, "gpt-4o")
        assert msg["role"] == "system", (
            "Expected system role to remain unchanged for gpt-4o"
        )

    def test_rewrite_system_to_developer_no_rewrite_when_model_is_none(self) -> None:
        """System role is NOT rewritten when model is None."""
        msg: dict[str, Any] = {"role": "system", "content": "x"}
        _rewrite_system_to_developer(msg, None)
        assert msg["role"] == "system", (
            "Expected system role to remain unchanged when model is None"
        )

    def test_rewrite_system_to_developer_no_rewrite_for_non_system_role(self) -> None:
        """Non-system roles are never rewritten, even for o3-mini."""
        msg: dict[str, Any] = {"role": "user", "content": "hi"}
        _rewrite_system_to_developer(msg, "o3-mini")
        assert msg["role"] == "user", (
            "Expected user role to remain unchanged even for o3-mini"
        )

    def test_strip_none_keys_removes_none_values(self) -> None:
        """Keys with None values are removed when drop_none is True."""
        msg: dict[str, Any] = {"role": "user", "content": None, "tool_calls": None}
        _strip_none_keys(msg, drop_none=True)
        assert "content" not in msg, "Expected content key to be stripped"
        assert "tool_calls" not in msg, "Expected tool_calls key to be stripped"
        assert msg["role"] == "user", "Expected role key to remain"

    def test_strip_none_keys_noop_when_false(self) -> None:
        """Keys with None values are kept when drop_none is False."""
        msg: dict[str, Any] = {"role": "user", "content": None}
        _strip_none_keys(msg, drop_none=False)
        assert "content" in msg, "Expected content key to remain when drop_none=False"

    def test_should_null_content_assistant_with_tool_calls(self) -> None:
        """Returns True for assistant with tool_calls in additional_kwargs."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="")],
            additional_kwargs={"tool_calls": [{"id": "1"}]},
        )
        result = _should_null_content(message, has_tool_calls=False)
        assert result is True, (
            "Expected True when assistant has tool_calls in additional_kwargs"
        )

    def test_should_null_content_assistant_with_function_call(self) -> None:
        """Returns True for assistant with function_call in additional_kwargs."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="")],
            additional_kwargs={"function_call": {"name": "f"}},
        )
        result = _should_null_content(message, has_tool_calls=False)
        assert result is True, (
            "Expected True when assistant has function_call in additional_kwargs"
        )

    def test_should_null_content_assistant_with_has_tool_calls_flag(self) -> None:
        """Returns True for assistant when has_tool_calls is True."""
        message = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")])
        result = _should_null_content(message, has_tool_calls=True)
        assert result is True, (
            "Expected True when has_tool_calls flag is True for assistant"
        )

    def test_should_null_content_user_message_returns_false(self) -> None:
        """Returns False for non-assistant roles even with tool_calls."""
        message = Message(
            role=MessageRole.USER,
            additional_kwargs={"tool_calls": [{"id": "1"}]},
            chunks=[TextChunk(content="hi")],
        )
        result = _should_null_content(message, has_tool_calls=True)
        assert result is False, (
            "Expected False for user messages regardless of tool_calls"
        )

    def test_should_null_content_assistant_without_any_tool_info(self) -> None:
        """Returns False for assistant without tool_calls, function_call, or has_tool_calls."""
        message = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="hello")])
        result = _should_null_content(message, has_tool_calls=False)
        assert result is False, (
            "Expected False when assistant has no tool-related data"
        )



@pytest.mark.unit
class TestChatFormat:
    """Tests for ChatFormat static methods."""

    def test_text(self) -> None:
        """ChatFormat.text returns a text content dict."""
        block = TextChunk(content="hello")
        result = ChatFormat.text(block)
        assert result == {"type": "text", "text": "hello"}, (
            "Expected text dict with content"
        )

    def test_document(self) -> None:
        """ChatFormat.document returns a file content dict with base64 data URI."""
        block = _make_document_block()
        result = ChatFormat.document(block)
        assert result == {
            "type": "file",
            "filename": "report.pdf",
            "file_data": "data:application/pdf;base64,cGRmZGF0YQ==",
        }, "Expected file dict with base64-encoded data"

    def test_image_with_url(self) -> None:
        """ChatFormat.image with URL returns image_url dict without detail."""
        block = Image(url="https://example.com/img.png")
        result = ChatFormat.image(block)
        assert result["type"] == "image_url", "Expected type to be image_url"
        assert result["image_url"]["url"] == "https://example.com/img.png", (
            "Expected URL to be passed through"
        )
        assert "detail" not in result["image_url"], (
            "Expected no detail key when detail is None"
        )

    def test_image_with_url_and_detail(self) -> None:
        """ChatFormat.image with URL and detail includes detail."""
        block = Image(url="https://example.com/img.png", detail="low")
        result = ChatFormat.image(block)
        assert result["image_url"]["detail"] == "low", (
            "Expected detail to be included"
        )

    def test_image_inline(self) -> None:
        """ChatFormat.image with no URL calls as_data_uri."""
        block = _make_image_inline_block()
        result = ChatFormat.image(block)
        assert result["image_url"]["url"] == "data:image/png;base64,abc123", (
            "Expected as_data_uri result as URL"
        )

    def test_audio(self) -> None:
        """ChatFormat.audio returns input_audio dict."""
        block = _make_audio_block()
        result = ChatFormat.audio(block)
        assert result["type"] == "input_audio", "Expected type to be input_audio"
        assert result["input_audio"]["data"] == "YXVkaW9kYXRh", (
            "Expected base64 audio data"
        )
        assert result["input_audio"]["format"] == "mp3", (
            "Expected audio format to be mp3"
        )

    def test_tool_call(self) -> None:
        """ChatFormat.tool_call returns function tool call dict."""
        block = ToolCallBlock(
            tool_call_id="call_1", tool_name="search", tool_kwargs={"q": "test"}
        )
        result = ChatFormat.tool_call(block)
        assert result == {
            "type": "function",
            "function": {"name": "search", "arguments": {"q": "test"}},
            "id": "call_1",
        }, "Expected function-style tool call dict"



@pytest.mark.unit
class TestResponsesFormat:
    """Tests for ResponsesFormat static methods."""

    def test_text_user_role(self) -> None:
        """ResponsesFormat.text for user role returns input_text type."""
        block = TextChunk(content="hello")
        result = ResponsesFormat.text(block, "user")
        assert result == {"type": "input_text", "text": "hello"}, (
            "Expected input_text for user role"
        )

    def test_text_assistant_role(self) -> None:
        """ResponsesFormat.text for assistant role returns output_text type."""
        block = TextChunk(content="hello")
        result = ResponsesFormat.text(block, "assistant")
        assert result == {"type": "output_text", "text": "hello"}, (
            "Expected output_text for assistant role"
        )

    def test_document(self) -> None:
        """ResponsesFormat.document returns input_file dict."""
        block = _make_document_block()
        result = ResponsesFormat.document(block)
        assert result["type"] == "input_file", "Expected type to be input_file"
        assert result["filename"] == "report.pdf", "Expected filename"
        assert "base64" in result["file_data"], "Expected base64 in file_data"

    def test_image_with_url(self) -> None:
        """ResponsesFormat.image with URL returns input_image dict with detail."""
        block = Image(url="https://example.com/img.png")
        result = ResponsesFormat.image(block)
        assert result["type"] == "input_image", "Expected type input_image"
        assert result["image_url"] == "https://example.com/img.png", (
            "Expected URL string"
        )
        assert result["detail"] == "auto", (
            "Expected detail to default to 'auto' when not specified"
        )

    def test_image_with_url_and_detail(self) -> None:
        """ResponsesFormat.image with explicit detail passes it through."""
        block = Image(url="https://example.com/img.png", detail="high")
        result = ResponsesFormat.image(block)
        assert result["detail"] == "high", "Expected detail to be 'high'"

    def test_image_inline(self) -> None:
        """ResponsesFormat.image with no URL calls as_data_uri."""
        block = _make_image_inline_block()
        result = ResponsesFormat.image(block)
        assert result["image_url"] == "data:image/png;base64,abc123", (
            "Expected data URI for inline image"
        )

    def test_thinking_with_content_and_id(self) -> None:
        """ResponsesFormat.thinking with content and id returns reasoning dict."""
        block = ThinkingBlock(
            content="let me think",
            additional_information={"id": "think_1"},
        )
        result = ResponsesFormat.thinking(block)
        assert result is not None, "Expected non-None result for valid thinking block"
        assert result["type"] == "reasoning", "Expected type reasoning"
        assert result["id"] == "think_1", "Expected id from additional_information"
        assert result["summary"] == [{"type": "summary_text", "text": "let me think"}], (
            "Expected summary list with content"
        )

    def test_thinking_without_content_returns_none(self) -> None:
        """ResponsesFormat.thinking without content returns None."""
        block = ThinkingBlock(content=None, additional_information={"id": "think_1"})
        result = ResponsesFormat.thinking(block)
        assert result is None, "Expected None when thinking block has no content"

    def test_thinking_without_id_returns_none(self) -> None:
        """ResponsesFormat.thinking without id in additional_information returns None."""
        block = ThinkingBlock(content="reasoning", additional_information={})
        result = ResponsesFormat.thinking(block)
        assert result is None, "Expected None when thinking block has no id"

    def test_tool_call(self) -> None:
        """ResponsesFormat.tool_call returns function_call dict."""
        block = ToolCallBlock(
            tool_call_id="call_1", tool_name="search", tool_kwargs={"q": "test"}
        )
        result = ResponsesFormat.tool_call(block)
        assert result == {
            "type": "function_call",
            "arguments": {"q": "test"},
            "call_id": "call_1",
            "name": "search",
        }, "Expected Responses API function_call dict"



@pytest.mark.unit
class TestChatMessageConverterBuild:
    """Tests for ChatMessageConverter.build() (Chat Completions API)."""

    # -- Reference audio short-circuit --

    def test_reference_audio_id_short_circuit(self) -> None:
        """Assistant message with reference_audio_id returns early with audio dict."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="")],
            additional_kwargs={"reference_audio_id": "audio_123"},
        )
        result = ChatMessageConverter(message).build()
        assert result == {
            "role": "assistant",
            "audio": {"id": "audio_123"},
        }, "Expected short-circuit dict with audio id"

    def test_reference_audio_id_ignored_for_non_assistant(self) -> None:
        """reference_audio_id in user message does NOT trigger short-circuit."""
        message = Message(
            role=MessageRole.USER,
            chunks=[TextChunk(content="hi")],
            additional_kwargs={"reference_audio_id": "audio_123"},
        )
        result = ChatMessageConverter(message).build()
        assert "audio" not in result, (
            "Expected no audio key for non-assistant message"
        )

    def test_reference_audio_with_o3_mini_rewrites_role(self) -> None:
        """Reference audio path still rewrites system-to-developer for O1 models.
        (Though system+reference_audio is unlikely, the code path applies.)"""
        # This tests the code path but the role is assistant so no rewrite occurs.
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="")],
            additional_kwargs={"reference_audio_id": "audio_123"},
        )
        result = ChatMessageConverter(message, model="o3-mini").build()
        assert result["role"] == "assistant", (
            "Expected assistant role to remain unchanged"
        )

    def test_reference_audio_with_drop_none(self) -> None:
        """Reference audio path respects drop_none."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="")],
            additional_kwargs={"reference_audio_id": "audio_123"},
        )
        result = ChatMessageConverter(message, drop_none=True).build()
        # No None values exist in this dict, so drop_none is a no-op here
        assert "role" in result, "Expected role key to remain"
        assert "audio" in result, "Expected audio key to remain"

    # -- TextChunk blocks --

    def test_system_message_string_content(self) -> None:
        """System messages always use string content."""
        message = Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="you are helpful")])
        result = ChatMessageConverter(message).build()
        assert result == {"role": "system", "content": "you are helpful"}, (
            "Expected string content for system message"
        )

    def test_tool_role_message_string_content(self) -> None:
        """Tool messages use string content and include tool_call_id."""
        message = Message(
            role=MessageRole.TOOL,
            chunks=[TextChunk(content="result data")],
            additional_kwargs={"tool_call_id": "call_1"},
        )
        result = ChatMessageConverter(message).build()
        assert result["role"] == "tool", "Expected tool role"
        assert result["content"] == "result data", (
            "Expected string content for tool message"
        )
        assert result["tool_call_id"] == "call_1", (
            "Expected tool_call_id to be included"
        )

    def test_user_message_only_text_chunks_uses_string(self) -> None:
        """User message with only TextChunk blocks uses string content."""
        message = Message(
            role=MessageRole.USER,
            chunks=[TextChunk(content="hello"), TextChunk(content=" world")],
        )
        result = ChatMessageConverter(message).build()
        assert result["content"] == "hello world", (
            "Expected concatenated string content for text-only user message"
        )

    # -- ToolCallBlock blocks --

    def test_tool_call_blocks_in_chunks(self) -> None:
        """ToolCallBlock chunks populate tool_calls list."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ToolCallBlock(
                    tool_call_id="call_1",
                    tool_name="search",
                    tool_kwargs={"q": "test"},
                ),
            ],
        )
        result = ChatMessageConverter(message).build()
        assert "tool_calls" in result, "Expected tool_calls key in result"
        assert len(result["tool_calls"]) == 1, "Expected one tool call"
        assert result["tool_calls"][0]["id"] == "call_1", (
            "Expected correct tool call id"
        )
        assert result["content"] is None, (
            "Expected None content for assistant with only tool calls"
        )

    def test_tool_call_block_error_propagates(self) -> None:
        """If ToolCallBlock conversion raises, the exception propagates."""
        bad_block = MagicMock(spec=ToolCallBlock)
        bad_block.__class__ = ToolCallBlock
        bad_block.tool_call_id = None

        with patch.object(ChatFormat, "tool_call", side_effect=ValueError("bad")):
            message = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="hi")])
            message.chunks.append(bad_block)  # type: ignore[arg-type]
            with pytest.raises(ValueError, match="bad"):
                ChatMessageConverter(message).build()

    # -- Document blocks --

    def test_document_block(self) -> None:
        """DocumentBlock is converted to a file content dict."""
        doc_block = DocumentBlock(
            data=b"cGRmZGF0YQ==", title="report.pdf", document_mimetype="application/pdf"
        )
        message = Message(
            role=MessageRole.USER,
            chunks=[TextChunk(content="see doc"), doc_block],
        )
        with patch.object(
            DocumentBlock, "as_base64", return_value=("cGRmZGF0YQ==", "application/pdf")
        ):
            result = ChatMessageConverter(message).build()
        # User with mixed blocks should use list content
        assert isinstance(result["content"], list), (
            "Expected list content for user message with mixed block types"
        )
        file_items = [c for c in result["content"] if c.get("type") == "file"]
        assert len(file_items) == 1, "Expected one file content item"
        assert file_items[0]["filename"] == "report.pdf", "Expected document filename"

    # -- Audio blocks --

    def test_audio_block(self) -> None:
        """Audio block is converted to input_audio dict."""
        audio_block = Audio(url="https://example.com/audio.mp3", format="mp3")
        message = Message(
            role=MessageRole.USER,
            chunks=[TextChunk(content="listen"), audio_block],
        )
        with patch.object(
            Audio, "resolve_audio", return_value=BytesIO(b"YXVkaW9kYXRh")
        ):
            result = ChatMessageConverter(message).build()
        assert isinstance(result["content"], list), (
            "Expected list content for user message with audio block"
        )
        audio_items = [c for c in result["content"] if c.get("type") == "input_audio"]
        assert len(audio_items) == 1, "Expected one input_audio content item"
        assert audio_items[0]["input_audio"]["format"] == "mp3", (
            "Expected mp3 format in audio dict"
        )

    # -- Image with inline data --

    def test_image_inline_data(self) -> None:
        """Image block without URL calls as_data_uri for content list."""
        img_block = Image(content=b"raw_png_bytes", image_mimetype="image/png")
        message = Message(
            role=MessageRole.USER,
            chunks=[TextChunk(content="see image"), img_block],
        )
        with patch.object(
            Image, "as_data_uri", return_value="data:image/png;base64,abc123"
        ):
            result = ChatMessageConverter(message).build()
        assert isinstance(result["content"], list), (
            "Expected list content for user with inline image"
        )
        img_items = [c for c in result["content"] if c.get("type") == "image_url"]
        assert len(img_items) == 1, "Expected one image_url content item"
        assert img_items[0]["image_url"]["url"] == "data:image/png;base64,abc123", (
            "Expected data URI for inline image"
        )

    # -- Content nulling for assistant messages --

    def test_assistant_with_tool_calls_nulls_empty_content(self) -> None:
        """Assistant message with ToolCallBlock and no text has content=None."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ToolCallBlock(
                    tool_call_id="call_1",
                    tool_name="f",
                    tool_kwargs={},
                ),
            ],
        )
        result = ChatMessageConverter(message).build()
        assert result["content"] is None, (
            "Expected None content for assistant with only tool calls"
        )

    def test_assistant_with_text_and_tool_calls_keeps_content(self) -> None:
        """Assistant message with text AND ToolCallBlock keeps the text content."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                TextChunk(content="Here are results"),
                ToolCallBlock(
                    tool_call_id="call_1",
                    tool_name="f",
                    tool_kwargs={},
                ),
            ],
        )
        result = ChatMessageConverter(message).build()
        assert result["content"] == "Here are results", (
            "Expected text content to be preserved alongside tool calls"
        )

    # -- O1 model system→developer rewrite --

    def test_system_role_rewritten_to_developer_for_o3_mini(self) -> None:
        """System role is rewritten to developer when model is o3-mini."""
        message = Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="be helpful")])
        result = ChatMessageConverter(message, model="o3-mini").build()
        assert result["role"] == "developer", (
            "Expected system to be rewritten to developer for o3-mini"
        )

    def test_system_role_not_rewritten_for_o1_mini(self) -> None:
        """System role is NOT rewritten for o1-mini (in O1_MODELS_WITHOUT_FUNCTION_CALLING)."""
        message = Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="be helpful")])
        result = ChatMessageConverter(message, model="o1-mini").build()
        assert result["role"] == "system", (
            "Expected system role to remain for o1-mini"
        )

    def test_system_role_not_rewritten_for_gpt4o(self) -> None:
        """System role is NOT rewritten for standard models."""
        message = Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="be helpful")])
        result = ChatMessageConverter(message, model="gpt-4o").build()
        assert result["role"] == "system", (
            "Expected system role to remain for gpt-4o"
        )

    # -- drop_none --

    def test_drop_none_strips_none_values(self) -> None:
        """When drop_none=True, keys with None values are removed."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="")],
            additional_kwargs={"function_call": {"name": "f"}},
        )
        result = ChatMessageConverter(message, drop_none=True).build()
        assert "content" not in result, (
            "Expected content key to be stripped when None and drop_none=True"
        )

    def test_drop_none_false_keeps_none_values(self) -> None:
        """When drop_none=False (default), None values remain."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="")],
            additional_kwargs={"function_call": {"name": "f"}},
        )
        result = ChatMessageConverter(message, drop_none=False).build()
        assert "content" in result, (
            "Expected content key to remain when drop_none=False"
        )
        assert result["content"] is None, (
            "Expected content to be None (not stripped)"
        )

    # -- Legacy additional_kwargs merge --

    def test_legacy_tool_calls_merge_when_no_tool_call_blocks(self) -> None:
        """Legacy tool_calls in additional_kwargs merge into dict when no ToolCallBlock chunks."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="")],
            additional_kwargs={
                "tool_calls": [{"id": "tc_1", "type": "function", "function": {"name": "f"}}],
            },
        )
        result = ChatMessageConverter(message).build()
        assert "tool_calls" in result, "Expected legacy tool_calls to be merged"
        assert result["tool_calls"][0]["id"] == "tc_1", (
            "Expected correct tool call id from legacy kwargs"
        )

    def test_legacy_tool_calls_not_merged_when_tool_call_blocks_exist(self) -> None:
        """Legacy tool_calls in additional_kwargs are NOT merged when ToolCallBlock chunks exist."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ToolCallBlock(
                    tool_call_id="call_chunk",
                    tool_name="search",
                    tool_kwargs={},
                ),
            ],
            additional_kwargs={
                "tool_calls": [{"id": "tc_legacy", "type": "function", "function": {"name": "f"}}],
            },
        )
        result = ChatMessageConverter(message).build()
        # The tool_calls should come from the chunks, not from additional_kwargs
        assert len(result["tool_calls"]) == 1, "Expected one tool call from chunks"
        assert result["tool_calls"][0]["id"] == "call_chunk", (
            "Expected tool call from chunks, not legacy kwargs"
        )

    # -- tool_call_id passthrough --

    def test_tool_call_id_passthrough(self) -> None:
        """tool_call_id in additional_kwargs is always added to the dict."""
        message = Message(
            role=MessageRole.TOOL,
            chunks=[TextChunk(content="output")],
            additional_kwargs={"tool_call_id": "call_42"},
        )
        result = ChatMessageConverter(message).build()
        assert result["tool_call_id"] == "call_42", (
            "Expected tool_call_id to be added to dict"
        )

    # -- Unsupported block type --

    def test_unsupported_block_type_raises_value_error(self) -> None:
        """An unrecognized block type raises ValueError."""
        message = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="placeholder")])
        # Inject a mock block that is not handled by any isinstance check
        fake_block = MagicMock()
        fake_block.__class__ = type("FakeBlock", (), {})
        message.chunks.append(fake_block)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unsupported content block type"):
            ChatMessageConverter(message).build()

    def test_thinking_block_skipped_in_chat(self) -> None:
        """ThinkingBlock is silently skipped in Chat Completions path."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ThinkingBlock(
                    content="reasoning", additional_information={"id": "t1"}
                ),
                TextChunk(content="answer"),
            ],
        )
        result = ChatMessageConverter(message).build()
        assert result["content"] == "answer", (
            "Expected only text content, ThinkingBlock should be skipped"
        )

    # -- Multiple text chunks concatenation --

    def test_multiple_text_chunks_concatenated(self) -> None:
        """Multiple TextChunk blocks concatenate into content_txt."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                TextChunk(content="Hello "),
                TextChunk(content="world"),
            ],
        )
        result = ChatMessageConverter(message).build()
        assert result["content"] == "Hello world", (
            "Expected concatenated text from multiple TextChunks"
        )

    # -- Assistant with image uses string content (not list) --

    def test_assistant_with_mixed_blocks_uses_string_content(self) -> None:
        """Assistant messages with mixed blocks use string content (not list)."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                TextChunk(content="look at this"),
                Image(url="https://example.com/img.png"),
            ],
        )
        result = ChatMessageConverter(message).build()
        assert result["content"] == "look at this", (
            "Expected string content for assistant, even with mixed blocks"
        )



@pytest.mark.unit
class TestToOpenaiResponsesMessageDict:
    """Tests for ResponsesMessageConverter (Responses API)."""

    # -- TextChunk --

    def test_user_text_returns_dict(self) -> None:
        """Plain text user message returns a dict with role and string content."""
        message = Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")])
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, dict), "Expected dict for user message"
        assert result["role"] == "user", "Expected user role"
        assert result["content"] == "hello", "Expected string content"

    def test_assistant_text_returns_dict(self) -> None:
        """Assistant text message returns a dict with role and string content."""
        message = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="response")])
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, dict), "Expected dict for assistant message"
        assert result["role"] == "assistant", "Expected assistant role"
        assert result["content"] == "response", "Expected string content"

    # -- ThinkingBlock --

    def test_thinking_block_with_valid_content(self) -> None:
        """ThinkingBlock with content and id produces reasoning in the result."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ThinkingBlock(
                    content="let me think",
                    additional_information={"id": "think_1"},
                ),
                TextChunk(content="answer"),
            ],
        )
        result = ResponsesMessageConverter(message).build()
        # Should be a list: [reasoning_dict, message_dict]
        assert isinstance(result, list), (
            "Expected list result when reasoning is present"
        )
        assert result[0]["type"] == "reasoning", (
            "Expected first item to be reasoning"
        )
        assert result[1]["role"] == "assistant", (
            "Expected second item to be the message dict"
        )

    def test_thinking_block_without_content_is_skipped(self) -> None:
        """ThinkingBlock without content is skipped (returns just the message dict).

        Because the chunks list contains a ThinkingBlock (not only TextChunks),
        the ``all(isinstance(block, TextChunk) ...)`` check is False, so
        content is a list of output_text dicts rather than a bare string.
        """
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ThinkingBlock(content=None, additional_information={"id": "t1"}),
                TextChunk(content="answer"),
            ],
        )
        result = ResponsesMessageConverter(message).build()
        # No reasoning, just the message dict
        assert isinstance(result, dict), (
            "Expected dict when thinking block is skipped"
        )
        # Content is list form because chunks contain non-TextChunk block
        assert isinstance(result["content"], list), (
            "Expected list content when chunks include non-TextChunk types"
        )
        text_items = [c for c in result["content"] if c.get("type") == "output_text"]
        assert len(text_items) == 1, "Expected one output_text item"
        assert text_items[0]["text"] == "answer", "Expected answer text in output_text item"

    # -- ToolCallBlock --

    def test_tool_call_blocks_produce_function_call_list(self) -> None:
        """ToolCallBlock chunks produce a list of function_call dicts."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ToolCallBlock(
                    tool_call_id="call_1",
                    tool_name="search",
                    tool_kwargs='{"q": "test"}',
                ),
            ],
        )
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, list), "Expected list for tool call result"
        assert len(result) == 1, "Expected one function_call dict"
        assert result[0]["type"] == "function_call", (
            "Expected function_call type"
        )
        assert result[0]["call_id"] == "call_1", "Expected correct call_id"

    def test_tool_call_with_reasoning(self) -> None:
        """ToolCallBlock with reasoning prepends reasoning items."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ThinkingBlock(
                    content="hmm",
                    additional_information={"id": "t1"},
                ),
                ToolCallBlock(
                    tool_call_id="call_1",
                    tool_name="f",
                    tool_kwargs={},
                ),
            ],
        )
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, list), "Expected list result"
        assert result[0]["type"] == "reasoning", "Expected reasoning first"
        assert result[1]["type"] == "function_call", (
            "Expected function_call second"
        )

    # -- DocumentBlock --

    def test_document_block_responses(self) -> None:
        """DocumentBlock in Responses format uses input_file type."""
        doc_block = DocumentBlock(
            data=b"cGRmZGF0YQ==", title="report.pdf", document_mimetype="application/pdf"
        )
        message = Message(
            role=MessageRole.USER,
            chunks=[TextChunk(content="see"), doc_block],
        )
        with patch.object(
            DocumentBlock, "as_base64", return_value=("cGRmZGF0YQ==", "application/pdf")
        ):
            result = ResponsesMessageConverter(message).build()
        # User with mixed blocks returns list content in a dict
        assert isinstance(result, dict), "Expected dict for user with mixed blocks"
        assert isinstance(result["content"], list), "Expected list content"
        file_items = [c for c in result["content"] if c.get("type") == "input_file"]
        assert len(file_items) == 1, "Expected one input_file item"

    # -- Image --

    def test_image_url_responses(self) -> None:
        """Image with URL in Responses format returns input_image dict."""
        message = Message(
            role=MessageRole.USER,
            chunks=[
                TextChunk(content="see"),
                Image(url="https://example.com/img.png"),
            ],
        )
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, dict), "Expected dict for user with mixed blocks"
        assert isinstance(result["content"], list), "Expected list content"
        img_items = [c for c in result["content"] if c.get("type") == "input_image"]
        assert len(img_items) == 1, "Expected one input_image item"
        assert img_items[0]["detail"] == "auto", "Expected detail to default to auto"

    # -- Chunk tool calls take precedence over legacy --

    def test_chunk_tool_calls_take_precedence_over_legacy(self) -> None:
        """ToolCallBlock chunks take precedence over legacy additional_kwargs tool_calls."""
        legacy_call = {"type": "function", "id": "legacy_1", "function": {"name": "f"}}
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ToolCallBlock(
                    tool_call_id="chunk_1",
                    tool_name="g",
                    tool_kwargs={},
                ),
            ],
            additional_kwargs={"tool_calls": [legacy_call]},
        )
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, list), "Expected list result"
        assert any(
            item.get("call_id") == "chunk_1" for item in result
        ), "Expected chunk-based tool call to take precedence"

    def test_legacy_tool_calls_used_when_no_chunks(self) -> None:
        """Legacy tool_calls are used when no ToolCallBlock chunks exist."""
        legacy_call = {"type": "function", "id": "legacy_1", "function": {"name": "f"}}
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="")],
            additional_kwargs={"tool_calls": [legacy_call]},
        )
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, list), "Expected list result"
        assert any(
            item.get("id") == "legacy_1" for item in result
        ), "Expected legacy tool call in result"

    # -- Tool message --

    def test_tool_message_with_call_id(self) -> None:
        """Tool message returns function_call_output dict."""
        message = Message(
            role=MessageRole.TOOL,
            chunks=[TextChunk(content="output data")],
            additional_kwargs={"tool_call_id": "call_1"},
        )
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, dict), "Expected dict for tool message"
        assert result["type"] == "function_call_output", (
            "Expected function_call_output type"
        )
        assert result["output"] == "output data", "Expected output text"
        assert result["call_id"] == "call_1", "Expected call_id"

    def test_tool_message_with_alternative_call_id_key(self) -> None:
        """Tool message also accepts call_id key in additional_kwargs."""
        message = Message(
            role=MessageRole.TOOL,
            chunks=[TextChunk(content="output")],
            additional_kwargs={"call_id": "call_2"},
        )
        result = ResponsesMessageConverter(message).build()
        assert result["call_id"] == "call_2", (
            "Expected call_id from alternative key"
        )

    def test_tool_message_missing_call_id_raises(self) -> None:
        """Tool message without tool_call_id or call_id raises ValueError."""
        message = Message(
            role=MessageRole.TOOL,
            chunks=[TextChunk(content="output")],
        )
        with pytest.raises(ValueError, match="tool_call_id or call_id is required"):
            ResponsesMessageConverter(message).build()

    # -- Content nulling for assistant --

    def test_assistant_empty_content_with_tool_calls_nulled(self) -> None:
        """Assistant with empty content and function_call in additional_kwargs gets None content."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="")],
            additional_kwargs={"function_call": {"name": "f"}},
        )
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, dict), "Expected dict result"
        assert result["content"] is None, (
            "Expected None content for empty assistant with function_call"
        )

    # -- System/developer role uses string content --

    def test_system_message_string_content(self) -> None:
        """System message in Responses format uses string content."""
        message = Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="be helpful")])
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, dict), "Expected dict for system message"
        assert result["content"] == "be helpful", (
            "Expected string content for system"
        )

    # -- system→developer rewrite --

    def test_system_rewritten_to_developer(self) -> None:
        """System role is unconditionally rewritten to developer in Responses API."""
        message = Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="be helpful")])
        result = ResponsesMessageConverter(message, model="o3-mini").build()
        assert isinstance(result, dict), "Expected dict result"
        assert result["role"] == "developer", (
            "Expected system to be unconditionally rewritten to developer"
        )

    # -- drop_none --

    def test_drop_none_strips_none_values(self) -> None:
        """drop_none=True strips None-valued keys from the result dict."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="")],
            additional_kwargs={"function_call": {"name": "f"}},
        )
        result = ResponsesMessageConverter(message, drop_none=True).build()
        assert isinstance(result, dict), "Expected dict result"
        assert "content" not in result, (
            "Expected content key stripped when None and drop_none=True"
        )

    # -- Unsupported block type --

    def test_audio_block_raises_in_responses_api(self) -> None:
        """Audio block raises ValueError in Responses API path."""
        audio_block = _make_audio_block()
        message = Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])
        message.chunks.append(audio_block)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Audio blocks are not supported in the Responses API"):
            ResponsesMessageConverter(message).build()

    # -- User message with only text returns dict with string content --

    def test_user_only_text_returns_dict_with_string_content(self) -> None:
        """User message with multiple TextChunks returns dict with concatenated string content."""
        message = Message(
            role=MessageRole.USER,
            chunks=[TextChunk(content="part1 "), TextChunk(content="part2")],
        )
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, dict), "Expected dict for text-only user message"
        assert result["role"] == "user", "Expected user role"
        assert result["content"] == "part1 part2", (
            "Expected concatenated string content for text-only user message"
        )

    # -- User with mixed blocks returns dict with list content --

    def test_user_mixed_blocks_returns_dict_with_list(self) -> None:
        """User message with text and image returns dict with list content."""
        message = Message(
            role=MessageRole.USER,
            chunks=[
                TextChunk(content="look"),
                Image(url="https://example.com/img.png"),
            ],
        )
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, dict), "Expected dict for user with mixed blocks"
        assert isinstance(result["content"], list), "Expected list content"
        assert result["role"] == "user", "Expected user role"

    # -- Reasoning + message dict pattern --

    def test_reasoning_prepended_to_message_dict(self) -> None:
        """When reasoning blocks exist, result is [reasoning, ..., message_dict].

        Because chunks contain ThinkingBlock (not only TextChunks), the content
        of the message dict is the list form (output_text items), not a bare string.
        """
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ThinkingBlock(content="hmm", additional_information={"id": "t1"}),
                TextChunk(content="answer"),
            ],
        )
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, list), "Expected list result with reasoning"
        assert len(result) == 2, "Expected reasoning + message dict"
        assert result[0]["type"] == "reasoning", "Expected reasoning item first"
        assert result[1]["role"] == "assistant", "Expected message dict second"
        # Content is list form because chunks include non-TextChunk block
        assert isinstance(result[1]["content"], list), (
            "Expected list content when chunks include ThinkingBlock"
        )
        text_items = [c for c in result[1]["content"] if c.get("type") == "output_text"]
        assert len(text_items) == 1, "Expected one output_text item"
        assert text_items[0]["text"] == "answer", "Expected answer text"

    # -- Assistant with all-text chunks uses string content --

    def test_assistant_all_text_chunks_uses_string(self) -> None:
        """Assistant with only TextChunk uses string content (not list)."""
        message = Message(
            role=MessageRole.ASSISTANT,
            chunks=[TextChunk(content="hello"), TextChunk(content=" world")],
        )
        result = ResponsesMessageConverter(message).build()
        assert isinstance(result, dict), "Expected dict for assistant"
        assert result["content"] == "hello world", (
            "Expected string content for all-text assistant"
        )



@pytest.mark.unit
class TestToOpenaiMessageDicts:
    """Tests for to_openai_message_dicts (batch converter)."""

    # -- Chat path --

    def test_chat_path_delegates_to_message_dict(self) -> None:
        """Chat path (is_responses_api=False) delegates to to_openai_message_dict."""
        messages = [
            Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="system")]),
            Message(role=MessageRole.USER, chunks=[TextChunk(content="user")]),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=False)
        assert isinstance(result, list), "Expected list result"
        assert len(result) == 2, "Expected two message dicts"
        assert result[0]["role"] == "system", "Expected system role first"
        assert result[1]["role"] == "user", "Expected user role second"

    def test_chat_path_passes_model(self) -> None:
        """Chat path passes model parameter through."""
        messages = [Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="system")])]
        result = to_openai_message_dicts(messages, model="o3-mini")
        assert result[0]["role"] == "developer", (
            "Expected system rewritten to developer via model param"
        )

    def test_chat_path_passes_drop_none(self) -> None:
        """Chat path passes drop_none parameter through."""
        messages = [
            Message(
                role=MessageRole.ASSISTANT,
                chunks=[TextChunk(content="")],
                additional_kwargs={"function_call": {"name": "f"}},
            ),
        ]
        result = to_openai_message_dicts(messages, drop_none=True)
        assert "content" not in result[0], (
            "Expected content key stripped with drop_none=True"
        )

    def test_chat_path_empty_messages(self) -> None:
        """Chat path with empty messages returns empty list."""
        result = to_openai_message_dicts([])
        assert result == [], "Expected empty list for empty input"

    # -- Responses path --

    def test_responses_path_delegates_to_responses_dict(self) -> None:
        """Responses path (is_responses_api=True) delegates to responses converter."""
        messages = [
            Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="system")]),
            Message(role=MessageRole.USER, chunks=[TextChunk(content="user")]),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert isinstance(result, list), "Expected list result"
        # System should be rewritten to developer (unconditional in Responses API)
        developer_msgs = [m for m in result if isinstance(m, dict) and m.get("role") == "developer"]
        assert len(developer_msgs) == 1, (
            "Expected system message rewritten to developer in responses path"
        )

    def test_responses_path_list_results_are_extended(self) -> None:
        """Responses path: list results from converter are extended (not appended)."""
        messages = [
            Message(
                role=MessageRole.ASSISTANT,
                chunks=[
                    ThinkingBlock(content="hmm", additional_information={"id": "t1"}),
                    TextChunk(content="answer"),
                ],
            ),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert isinstance(result, list), "Expected list result"
        # Should be flat: [reasoning_dict, message_dict], not [[reasoning_dict, message_dict]]
        assert all(isinstance(item, dict) for item in result), (
            "Expected flat list of dicts (extended, not appended)"
        )

    def test_responses_path_string_results_wrapped(self) -> None:
        """Responses path: string results wrapped in role=user dict."""
        messages = [
            Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="system")]),
            Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")]),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert isinstance(result, list), "Expected list result (not bare string)"
        # With two messages, the single-user-message shortcut doesn't apply
        user_msgs = [m for m in result if isinstance(m, dict) and m.get("role") == "user"]
        assert len(user_msgs) == 1, "Expected one user message dict"
        assert user_msgs[0]["content"] == "hello", (
            "Expected string content wrapped in user dict"
        )

    def test_responses_path_single_user_message_bare_string(self) -> None:
        """Responses path: single user message returns bare string shortcut."""
        messages = [Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")])]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert result == "hello", (
            "Expected bare string for single user message in responses path"
        )

    def test_responses_path_system_rewritten_to_developer(self) -> None:
        """Responses path unconditionally rewrites system→developer."""
        messages = [Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="x")])]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert isinstance(result, list), "Expected list result"
        assert result[0]["role"] == "developer", (
            "Expected developer role (unconditional in Responses API)"
        )

    def test_responses_path_empty_messages(self) -> None:
        """Responses path with empty messages returns empty list."""
        result = to_openai_message_dicts([], is_responses_api=True)
        assert result == [], "Expected empty list for empty input"

    def test_responses_path_dict_results_are_appended(self) -> None:
        """Responses path: dict results from converter are appended normally."""
        messages = [
            Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="hello")]),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert isinstance(result, list), "Expected list result"
        assert len(result) == 1, "Expected one dict item"
        assert result[0]["role"] == "assistant", "Expected assistant dict"

    def test_responses_path_multiple_messages_mix(self) -> None:
        """Responses path handles a mix of message types correctly."""
        messages = [
            Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="system prompt")]),
            Message(role=MessageRole.USER, chunks=[TextChunk(content="question")]),
            Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="answer")]),
        ]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert isinstance(result, list), "Expected list result"
        # system -> developer (unconditional), user -> wrapped dict, assistant -> dict
        roles = [m["role"] for m in result if isinstance(m, dict)]
        assert "developer" in roles, "Expected developer role for system message"
        assert "user" in roles, "Expected user role"
        assert "assistant" in roles, "Expected assistant role"

    def test_responses_path_single_non_user_message_not_bare_string(self) -> None:
        """Responses path: single assistant message does NOT return bare string."""
        messages = [Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="hello")])]
        result = to_openai_message_dicts(messages, is_responses_api=True)
        assert isinstance(result, list), (
            "Expected list (not string) for single assistant message"
        )
        assert result[0]["role"] == "assistant", "Expected assistant role in dict"


@pytest.mark.unit
class TestChatMessageConverterMethods:
    """Tests for ChatMessageConverter private methods in isolation."""

    def test_init_stores_message_and_defaults(self) -> None:
        """Test __init__ stores message and sets defaults."""
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])
        conv = ChatMessageConverter(msg)
        assert conv._message is msg, "Message not stored"
        assert conv._model is None, f"Expected model=None, got {conv._model}"
        assert conv._drop_none is False, f"Expected drop_none=False, got {conv._drop_none}"
        assert conv._content == [], f"Expected empty content, got {conv._content}"
        assert conv._content_txt == "", f"Expected empty content_txt, got {conv._content_txt!r}"
        assert conv._tool_call_dicts == [], f"Expected empty tool_call_dicts, got {conv._tool_call_dicts}"

    def test_init_with_all_params(self) -> None:
        """Test __init__ stores model and drop_none when provided."""
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])
        conv = ChatMessageConverter(msg, model="gpt-4o", drop_none=True)
        assert conv._model == "gpt-4o", f"Expected model 'gpt-4o', got {conv._model}"
        assert conv._drop_none is True, f"Expected drop_none=True, got {conv._drop_none}"

    def test_try_audio_reference_assistant_with_audio_id(self) -> None:
        """Test _try_audio_reference returns dict for assistant with reference_audio_id."""
        msg = Message(
            role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")],
            additional_kwargs={"reference_audio_id": "audio_123"},
        )
        result = ChatMessageConverter(msg)._try_audio_reference()
        assert result is not None, "Expected a dict, got None"
        assert result["role"] == "assistant", f"Expected role 'assistant', got {result['role']}"
        assert result["audio"] == {"id": "audio_123"}, f"Unexpected audio: {result['audio']}"

    def test_try_audio_reference_none_for_user_role(self) -> None:
        """Test _try_audio_reference returns None for user role even with audio_id."""
        msg = Message(
            role=MessageRole.USER, chunks=[TextChunk(content="")],
            additional_kwargs={"reference_audio_id": "audio_123"},
        )
        result = ChatMessageConverter(msg)._try_audio_reference()
        assert result is None, f"Expected None for user role, got {result}"

    def test_try_audio_reference_none_without_audio_id(self) -> None:
        """Test _try_audio_reference returns None when no reference_audio_id."""
        msg = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="hello")])
        result = ChatMessageConverter(msg)._try_audio_reference()
        assert result is None, f"Expected None, got {result}"

    def test_try_audio_reference_none_for_empty_kwargs(self) -> None:
        """Test _try_audio_reference returns None when additional_kwargs is empty."""
        msg = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")])
        result = ChatMessageConverter(msg)._try_audio_reference()
        assert result is None, f"Expected None, got {result}"

    def test_process_blocks_text_chunk(self) -> None:
        """Test _process_blocks populates _content and _content_txt for TextChunks."""
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")])
        conv = ChatMessageConverter(msg)
        conv._process_blocks()
        assert len(conv._content) == 1, f"Expected 1 content item, got {len(conv._content)}"
        assert conv._content[0] == {"type": "text", "text": "hello"}, (
            f"Unexpected content dict: {conv._content[0]}"
        )
        assert conv._content_txt == "hello", f"Expected 'hello', got {conv._content_txt!r}"

    def test_process_blocks_multiple_text_chunks_accumulate(self) -> None:
        """Test _process_blocks accumulates multiple TextChunks."""
        msg = Message(
            role=MessageRole.USER,
            chunks=[TextChunk(content="hello "), TextChunk(content="world")],
        )
        conv = ChatMessageConverter(msg)
        conv._process_blocks()
        assert len(conv._content) == 2, f"Expected 2 content items, got {len(conv._content)}"
        assert conv._content_txt == "hello world", f"Expected 'hello world', got {conv._content_txt!r}"

    def test_process_blocks_tool_call_block(self) -> None:
        """Test _process_blocks appends ToolCallBlocks to _tool_call_dicts only."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            chunks=[ToolCallBlock(tool_call_id="call_1", tool_name="fn", tool_kwargs={"a": 1})],
        )
        conv = ChatMessageConverter(msg)
        conv._process_blocks()
        assert len(conv._tool_call_dicts) == 1, f"Expected 1 tool call, got {len(conv._tool_call_dicts)}"
        assert conv._tool_call_dicts[0]["id"] == "call_1", "Tool call id mismatch"
        assert conv._content == [], "ToolCallBlocks should not go to _content"

    def test_process_blocks_tool_call_error_propagates(self) -> None:
        """Test _process_blocks lets ToolCallBlock conversion errors propagate."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            chunks=[ToolCallBlock(tool_call_id="call_1", tool_name="fn", tool_kwargs={"a": 1})],
        )
        conv = ChatMessageConverter(msg)
        with patch.object(ChatFormat, "tool_call", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                conv._process_blocks()

    def test_process_blocks_document_via_dispatch(self) -> None:
        """Test _process_blocks dispatches DocumentBlock through content_converters."""
        doc = DocumentBlock(data=b"cGRmZGF0YQ==", title="report.pdf", document_mimetype="application/pdf")
        msg = Message(role=MessageRole.USER, chunks=[doc])
        conv = ChatMessageConverter(msg)
        conv._process_blocks()
        assert len(conv._content) == 1, f"Expected 1 content item, got {len(conv._content)}"
        assert conv._content[0]["type"] == "file", f"Expected type 'file', got {conv._content[0]['type']}"

    def test_process_blocks_unsupported_type_raises(self) -> None:
        """Test _process_blocks raises ValueError for unsupported block types."""
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="placeholder")])
        conv = ChatMessageConverter(msg)
        fake_block = MagicMock()
        fake_block.__class__ = type("FakeBlock", (), {})
        conv._message.chunks.append(fake_block)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unsupported content block type"):
            conv._process_blocks()

    def test_process_blocks_thinking_block_skipped(self) -> None:
        """Test _process_blocks skips ThinkingBlock in Chat Completions path."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ThinkingBlock(content="thought", additional_information={"id": "t1"}),
                TextChunk(content="answer"),
            ],
        )
        conv = ChatMessageConverter(msg)
        conv._process_blocks()
        assert conv._content_txt == "answer", "ThinkingBlock should be skipped"
        assert len(conv._content) == 1, "Only TextChunk should produce content"

    def test_process_blocks_empty_chunks(self) -> None:
        """Test _process_blocks with no chunks leaves state unchanged."""
        msg = Message(role=MessageRole.USER, chunks=[])
        conv = ChatMessageConverter(msg)
        conv._process_blocks()
        assert conv._content == [], f"Expected empty content, got {conv._content}"
        assert conv._content_txt == "", f"Expected empty text, got {conv._content_txt!r}"
        assert conv._tool_call_dicts == [], f"Expected empty tool calls, got {conv._tool_call_dicts}"

    def test_resolve_content_string_for_assistant(self) -> None:
        """Test _resolve_content returns string for assistant role."""
        msg = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="answer")])
        conv = ChatMessageConverter(msg)
        conv._content = [{"type": "text", "text": "answer"}]
        conv._content_txt = "answer"
        assert conv._resolve_content() == "answer", "Assistant should get string content"

    def test_resolve_content_string_for_system(self) -> None:
        """Test _resolve_content returns string for system role."""
        msg = Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="prompt")])
        conv = ChatMessageConverter(msg)
        conv._content_txt = "prompt"
        assert conv._resolve_content() == "prompt", "System should get string content"

    def test_resolve_content_string_for_tool(self) -> None:
        """Test _resolve_content returns string for tool role."""
        msg = Message(
            role=MessageRole.TOOL,
            chunks=[TextChunk(content="result")],
            additional_kwargs={"tool_call_id": "c1"},
        )
        conv = ChatMessageConverter(msg)
        conv._content_txt = "result"
        assert conv._resolve_content() == "result", "Tool should get string content"

    def test_resolve_content_list_for_user_mixed_blocks(self) -> None:
        """Test _resolve_content returns list for user with mixed block types."""
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="placeholder")])
        conv = ChatMessageConverter(msg)
        mock_img = MagicMock(spec=Image)
        mock_img.__class__ = Image
        conv._message = MagicMock(
            chunks=[TextChunk(content="look"), mock_img],
            role=MessageRole.USER,
            additional_kwargs={},
        )
        content_list = [{"type": "text", "text": "look"}, {"type": "image_url"}]
        conv._content = content_list
        conv._content_txt = "look"
        result = conv._resolve_content()
        assert result is content_list, f"Expected list content, got {type(result)}"

    def test_resolve_content_string_for_user_only_text(self) -> None:
        """Test _resolve_content returns string for user with only TextChunks."""
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")])
        conv = ChatMessageConverter(msg)
        conv._content_txt = "hello"
        assert conv._resolve_content() == "hello", "User with only text should get string"

    def test_resolve_content_none_for_assistant_with_tool_calls(self) -> None:
        """Test _resolve_content returns None for assistant with tool calls and empty text."""
        msg = Message(
            role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")],
            additional_kwargs={"tool_calls": [{"id": "c1"}]},
        )
        conv = ChatMessageConverter(msg)
        conv._content_txt = ""
        conv._tool_call_dicts = [{"id": "c1", "type": "function"}]
        assert conv._resolve_content() is None, "Empty assistant with tool calls should be None"

    def test_resolve_content_empty_string_without_tool_calls(self) -> None:
        """Test _resolve_content returns empty string for assistant without tool info."""
        msg = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")])
        conv = ChatMessageConverter(msg)
        conv._content_txt = ""
        assert conv._resolve_content() == "", "Empty assistant without tools keeps empty string"

    def test_assemble_basic(self) -> None:
        """Test _assemble builds dict with role and content, no tool_calls key."""
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])
        conv = ChatMessageConverter(msg)
        conv._content_txt = "hi"
        result = conv._assemble()
        assert result["role"] == "user", f"Expected 'user', got {result['role']}"
        assert result["content"] == "hi", f"Expected 'hi', got {result['content']}"
        assert "tool_calls" not in result, "tool_calls should not be present when empty"

    def test_assemble_includes_tool_calls(self) -> None:
        """Test _assemble includes tool_calls key when tool_call_dicts is non-empty."""
        msg = Message(
            role=MessageRole.ASSISTANT,
            chunks=[ToolCallBlock(tool_call_id="c1", tool_name="fn", tool_kwargs={})],
        )
        conv = ChatMessageConverter(msg)
        conv._content_txt = ""
        conv._tool_call_dicts = [{"id": "c1", "type": "function", "function": {"name": "fn"}}]
        result = conv._assemble()
        assert "tool_calls" in result, "Expected tool_calls key in result"
        assert len(result["tool_calls"]) == 1, f"Expected 1 tool call, got {len(result['tool_calls'])}"

    def test_merge_legacy_tool_calls_when_no_blocks(self) -> None:
        """Test _merge_legacy_kwargs merges tool_calls when no ToolCallBlock chunks."""
        legacy_calls = [{"id": "c1", "function": {"name": "fn"}}]
        msg = Message(
            role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")],
            additional_kwargs={"tool_calls": legacy_calls},
        )
        conv = ChatMessageConverter(msg)
        result: dict[str, Any] = {"role": "assistant", "content": None}
        conv._merge_legacy_kwargs(result)
        assert result["tool_calls"] == legacy_calls, f"Unexpected tool_calls: {result.get('tool_calls')}"

    def test_merge_legacy_function_call_when_no_blocks(self) -> None:
        """Test _merge_legacy_kwargs merges function_call when no ToolCallBlock chunks."""
        fn_call = {"name": "get_weather", "arguments": '{"city": "Paris"}'}
        msg = Message(
            role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")],
            additional_kwargs={"function_call": fn_call},
        )
        conv = ChatMessageConverter(msg)
        result: dict[str, Any] = {"role": "assistant", "content": None}
        conv._merge_legacy_kwargs(result)
        assert "function_call" in result, "Legacy function_call should be merged"

    def test_merge_legacy_skipped_when_tool_call_blocks_exist(self) -> None:
        """Test _merge_legacy_kwargs skips merge when _tool_call_dicts is non-empty."""
        msg = Message(
            role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")],
            additional_kwargs={"tool_calls": [{"id": "legacy"}]},
        )
        conv = ChatMessageConverter(msg)
        conv._tool_call_dicts = [{"id": "from_block"}]
        result: dict[str, Any] = {"role": "assistant", "content": None}
        conv._merge_legacy_kwargs(result)
        assert "tool_calls" not in result, "Legacy should not be merged when blocks exist"

    def test_merge_tool_call_id_passthrough(self) -> None:
        """Test _merge_legacy_kwargs passes through tool_call_id."""
        msg = Message(
            role=MessageRole.TOOL, chunks=[TextChunk(content="result")],
            additional_kwargs={"tool_call_id": "call_42"},
        )
        conv = ChatMessageConverter(msg)
        result: dict[str, Any] = {"role": "tool", "content": "result"}
        conv._merge_legacy_kwargs(result)
        assert result["tool_call_id"] == "call_42", f"Expected 'call_42', got {result.get('tool_call_id')}"

    def test_merge_no_tool_call_id_does_nothing(self) -> None:
        """Test _merge_legacy_kwargs does nothing when no tool_call_id."""
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])
        conv = ChatMessageConverter(msg)
        result: dict[str, Any] = {"role": "user", "content": "hi"}
        conv._merge_legacy_kwargs(result)
        assert "tool_call_id" not in result, "tool_call_id should not be added"

    def test_build_audio_path_skips_process_blocks(self) -> None:
        """Test build() with audio reference skips block processing entirely."""
        msg = Message(
            role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")],
            additional_kwargs={"reference_audio_id": "aud_1"},
        )
        conv = ChatMessageConverter(msg)
        result = conv.build()
        assert result["audio"] == {"id": "aud_1"}, f"Unexpected audio: {result.get('audio')}"
        assert conv._content == [], "Content should be empty — _process_blocks was skipped"
        assert conv._tool_call_dicts == [], "Tool calls should be empty"

    def test_build_normal_path(self) -> None:
        """Test build() normal path populates state and returns correct dict."""
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="test")])
        conv = ChatMessageConverter(msg)
        result = conv.build()
        assert result["role"] == "user", f"Expected 'user', got {result['role']}"
        assert result["content"] == "test", f"Expected 'test', got {result['content']}"
        assert len(conv._content) == 1, "Expected _process_blocks to populate _content"


@pytest.mark.unit
class TestResponsesMessageConverterMethods:
    """Tests for ResponsesMessageConverter private methods in isolation."""

    def test_init_stores_message_and_defaults(self) -> None:
        """Test __init__ stores message and sets default empty state.

        Test scenario:
            Construct with only a message; verify all internal lists are
            empty and optional params default to None/False.
        """
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])
        conv = ResponsesMessageConverter(msg)
        assert conv._message is msg, "Message not stored"
        assert conv._model is None, f"Expected model=None, got {conv._model}"
        assert conv._drop_none is False, f"Expected drop_none=False, got {conv._drop_none}"
        assert conv._content == [], f"Expected empty content, got {conv._content}"
        assert conv._content_txt == "", f"Expected empty content_txt, got {conv._content_txt!r}"
        assert conv._tool_call_dicts == [], f"Expected empty tool_call_dicts, got {conv._tool_call_dicts}"
        assert conv._reasoning == [], f"Expected empty reasoning, got {conv._reasoning}"

    def test_init_with_all_params(self) -> None:
        """Test __init__ stores model and drop_none when provided.

        Test scenario:
            Construct with model='o3-mini' and drop_none=True; verify both
            are stored correctly.
        """
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])
        conv = ResponsesMessageConverter(msg, model="o3-mini", drop_none=True)
        assert conv._model == "o3-mini", f"Expected model 'o3-mini', got {conv._model}"
        assert conv._drop_none is True, f"Expected drop_none=True, got {conv._drop_none}"

    def test_process_blocks_text_chunk_user_role(self) -> None:
        """Test _process_blocks produces input_text for user role TextChunk.

        Test scenario:
            User message with a single TextChunk; verify _content gets an
            input_text dict and _content_txt accumulates the text.
        """
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")])
        conv = ResponsesMessageConverter(msg)
        conv._process_blocks()
        assert len(conv._content) == 1, f"Expected 1 content item, got {len(conv._content)}"
        assert conv._content[0] == {"type": "input_text", "text": "hello"}, (
            f"Unexpected content dict: {conv._content[0]}"
        )
        assert conv._content_txt == "hello", f"Expected 'hello', got {conv._content_txt!r}"

    def test_process_blocks_text_chunk_assistant_role(self) -> None:
        """Test _process_blocks produces output_text for assistant role TextChunk.

        Test scenario:
            Assistant message with a single TextChunk; verify _content gets
            an output_text dict (not input_text).
        """
        msg = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="response")])
        conv = ResponsesMessageConverter(msg)
        conv._process_blocks()
        assert conv._content[0]["type"] == "output_text", (
            f"Expected output_text for assistant, got {conv._content[0]['type']}"
        )

    def test_process_blocks_multiple_text_chunks_accumulate(self) -> None:
        """Test _process_blocks accumulates multiple TextChunks.

        Test scenario:
            User message with two TextChunks; verify _content has 2 items
            and _content_txt is the concatenation.
        """
        msg = Message(
            role=MessageRole.USER,
            chunks=[TextChunk(content="hello "), TextChunk(content="world")],
        )
        conv = ResponsesMessageConverter(msg)
        conv._process_blocks()
        assert len(conv._content) == 2, f"Expected 2 content items, got {len(conv._content)}"
        assert conv._content_txt == "hello world", f"Expected 'hello world', got {conv._content_txt!r}"

    def test_process_blocks_thinking_block_with_content(self) -> None:
        """Test _process_blocks appends ThinkingBlock with content to _reasoning.

        Test scenario:
            Assistant message with a ThinkingBlock that has content and id;
            verify it is added to _reasoning (not _content).
        """
        msg = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ThinkingBlock(content="thinking hard", additional_information={"id": "t1"}),
                TextChunk(content="answer"),
            ],
        )
        conv = ResponsesMessageConverter(msg)
        conv._process_blocks()
        assert len(conv._reasoning) == 1, f"Expected 1 reasoning item, got {len(conv._reasoning)}"
        assert conv._reasoning[0]["type"] == "reasoning", (
            f"Expected reasoning type, got {conv._reasoning[0]['type']}"
        )
        assert conv._reasoning[0]["id"] == "t1", f"Expected id 't1', got {conv._reasoning[0]['id']}"

    def test_process_blocks_thinking_block_without_content_skipped(self) -> None:
        """Test _process_blocks skips ThinkingBlock with no content.

        Test scenario:
            ThinkingBlock with content=None returns None from
            ResponsesFormat.thinking, so _reasoning stays empty.
        """
        msg = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ThinkingBlock(content=None, additional_information={"id": "t1"}),
                TextChunk(content="answer"),
            ],
        )
        conv = ResponsesMessageConverter(msg)
        conv._process_blocks()
        assert conv._reasoning == [], f"Expected empty reasoning, got {conv._reasoning}"
        assert len(conv._content) == 1, "Expected only the TextChunk in content"

    def test_process_blocks_tool_call_block(self) -> None:
        """Test _process_blocks appends ToolCallBlock to _tool_call_dicts.

        Test scenario:
            Assistant message with a ToolCallBlock; verify it goes to
            _tool_call_dicts (not _content) as a function_call dict.
        """
        msg = Message(
            role=MessageRole.ASSISTANT,
            chunks=[ToolCallBlock(tool_call_id="call_1", tool_name="fn", tool_kwargs={"a": 1})],
        )
        conv = ResponsesMessageConverter(msg)
        conv._process_blocks()
        assert len(conv._tool_call_dicts) == 1, (
            f"Expected 1 tool call dict, got {len(conv._tool_call_dicts)}"
        )
        assert conv._tool_call_dicts[0]["type"] == "function_call", (
            f"Expected function_call type, got {conv._tool_call_dicts[0]['type']}"
        )
        assert conv._tool_call_dicts[0]["call_id"] == "call_1", "Tool call id mismatch"
        assert conv._content == [], "ToolCallBlocks should not go to _content"

    def test_process_blocks_document_via_dispatch(self) -> None:
        """Test _process_blocks dispatches DocumentBlock through content_converters.

        Test scenario:
            User message with a real DocumentBlock; verify it is dispatched
            via ResponsesFormat.content_converters and produces input_file dict.
        """
        doc = DocumentBlock(
            data=b"cGRmZGF0YQ==", title="report.pdf", document_mimetype="application/pdf",
        )
        msg = Message(role=MessageRole.USER, chunks=[doc])
        conv = ResponsesMessageConverter(msg)
        conv._process_blocks()
        assert len(conv._content) == 1, f"Expected 1 content item, got {len(conv._content)}"
        assert conv._content[0]["type"] == "input_file", (
            f"Expected input_file type, got {conv._content[0]['type']}"
        )

    def test_process_blocks_unsupported_type_raises(self) -> None:
        """Test _process_blocks raises ValueError for unsupported block types.

        Test scenario:
            Audio block triggers explicit ValueError for Responses API.
        """
        audio_block = _make_audio_block()
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="placeholder")])
        conv = ResponsesMessageConverter(msg)
        conv._message = MagicMock(
            chunks=[audio_block], role=MessageRole.USER, additional_kwargs={},
        )
        with pytest.raises(ValueError, match="Audio blocks are not supported in the Responses API"):
            conv._process_blocks()

    def test_process_blocks_empty_chunks(self) -> None:
        """Test _process_blocks with no chunks leaves all state empty.

        Test scenario:
            Message with empty chunks list; verify all internal lists remain
            at their default empty state.
        """
        msg = Message(role=MessageRole.USER, chunks=[])
        conv = ResponsesMessageConverter(msg)
        conv._process_blocks()
        assert conv._content == [], f"Expected empty content, got {conv._content}"
        assert conv._content_txt == "", f"Expected empty text, got {conv._content_txt!r}"
        assert conv._tool_call_dicts == [], f"Expected empty tool calls, got {conv._tool_call_dicts}"
        assert conv._reasoning == [], f"Expected empty reasoning, got {conv._reasoning}"

    def test_resolve_content_none_for_assistant_with_function_call(self) -> None:
        """Test _resolve_content returns None for empty assistant with function_call.

        Test scenario:
            Assistant message with empty content and function_call in kwargs;
            _should_null_content returns True, so content becomes None.
        """
        msg = Message(
            role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")],
            additional_kwargs={"function_call": {"name": "f"}},
        )
        conv = ResponsesMessageConverter(msg)
        conv._content_txt = ""
        result = conv._resolve_content()
        assert result is None, f"Expected None for null-content assistant, got {result}"

    def test_resolve_content_string_for_system_role(self) -> None:
        """Test _resolve_content returns string for system role.

        Test scenario:
            System message with text content; the system role is in the
            string-content roles, so content stays as string.
        """
        msg = Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="be helpful")])
        conv = ResponsesMessageConverter(msg)
        conv._content_txt = "be helpful"
        result = conv._resolve_content()
        assert result == "be helpful", f"Expected string content, got {result}"

    def test_resolve_content_string_for_all_text_user(self) -> None:
        """Test _resolve_content returns string when all chunks are TextChunk.

        Test scenario:
            User message with only TextChunks; the all(isinstance TextChunk)
            check passes, so content stays as the concatenated string.
        """
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")])
        conv = ResponsesMessageConverter(msg)
        conv._content_txt = "hello"
        result = conv._resolve_content()
        assert result == "hello", f"Expected string content, got {result}"

    def test_resolve_content_list_for_mixed_block_user(self) -> None:
        """Test _resolve_content returns list for user with non-TextChunk blocks.

        Test scenario:
            User message with TextChunk + Image; the all(isinstance TextChunk)
            check fails, so content is the list form.
        """
        msg = Message(
            role=MessageRole.USER,
            chunks=[TextChunk(content="look"), Image(url="https://example.com/img.png")],
        )
        conv = ResponsesMessageConverter(msg)
        content_list = [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "https://example.com/img.png", "detail": "auto"},
        ]
        conv._content = content_list
        conv._content_txt = "look"
        result = conv._resolve_content()
        assert result is content_list, f"Expected list content, got {type(result)}"

    def test_resolve_content_list_for_assistant_with_thinking(self) -> None:
        """Test _resolve_content returns list for assistant with ThinkingBlock.

        Test scenario:
            Assistant chunks include ThinkingBlock + TextChunk; since not all
            chunks are TextChunk, content is the list form.
        """
        msg = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ThinkingBlock(content="hmm", additional_information={"id": "t1"}),
                TextChunk(content="answer"),
            ],
        )
        conv = ResponsesMessageConverter(msg)
        content_list = [{"type": "output_text", "text": "answer"}]
        conv._content = content_list
        conv._content_txt = "answer"
        result = conv._resolve_content()
        assert result is content_list, f"Expected list content, got {type(result)}"

    def test_resolve_content_empty_string_no_tool_calls(self) -> None:
        """Test _resolve_content returns empty string when no tool info present.

        Test scenario:
            Assistant with empty content and no tool_calls/function_call;
            _should_null_content returns False, so content stays as empty string.
        """
        msg = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")])
        conv = ResponsesMessageConverter(msg)
        conv._content_txt = ""
        result = conv._resolve_content()
        assert result == "", f"Expected empty string, got {result!r}"

    def test_assemble_tool_output_with_tool_call_id(self) -> None:
        """Test _assemble_tool_output uses tool_call_id from additional_kwargs.

        Test scenario:
            Tool message with tool_call_id in kwargs; produces a
            function_call_output dict with the correct call_id and output.
        """
        msg = Message(
            role=MessageRole.TOOL, chunks=[TextChunk(content="result data")],
            additional_kwargs={"tool_call_id": "call_42"},
        )
        conv = ResponsesMessageConverter(msg)
        conv._content_txt = "result data"
        result = conv._assemble_tool_output()
        assert result == {
            "type": "function_call_output",
            "output": "result data",
            "call_id": "call_42",
        }, f"Unexpected tool output dict: {result}"

    def test_assemble_tool_output_with_call_id_fallback(self) -> None:
        """Test _assemble_tool_output falls back to call_id key.

        Test scenario:
            Tool message with call_id (not tool_call_id) in kwargs;
            the fallback .get("call_id") should be used.
        """
        msg = Message(
            role=MessageRole.TOOL, chunks=[TextChunk(content="output")],
            additional_kwargs={"call_id": "call_99"},
        )
        conv = ResponsesMessageConverter(msg)
        conv._content_txt = "output"
        result = conv._assemble_tool_output()
        assert result["call_id"] == "call_99", f"Expected call_id 'call_99', got {result['call_id']}"

    def test_assemble_tool_output_missing_call_id_raises(self) -> None:
        """Test _assemble_tool_output raises ValueError when no call id present.

        Test scenario:
            Tool message with neither tool_call_id nor call_id; should raise
            ValueError with a descriptive message.
        """
        msg = Message(role=MessageRole.TOOL, chunks=[TextChunk(content="output")])
        conv = ResponsesMessageConverter(msg)
        conv._content_txt = "output"
        with pytest.raises(ValueError, match="tool_call_id or call_id is required"):
            conv._assemble_tool_output()

    def test_assemble_message_dict_basic(self) -> None:
        """Test _assemble_message_dict produces a basic role+content dict.

        Test scenario:
            Assistant message with string content and no reasoning; returns
            a plain dict (not wrapped in a list).
        """
        msg = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="answer")])
        conv = ResponsesMessageConverter(msg)
        conv._content_txt = "answer"
        result = conv._assemble_message_dict()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result["role"] == "assistant", f"Expected assistant role, got {result['role']}"
        assert result["content"] == "answer", f"Expected 'answer', got {result['content']}"

    def test_assemble_message_dict_with_reasoning_returns_list(self) -> None:
        """Test _assemble_message_dict prepends reasoning items.

        Test scenario:
            Converter has _reasoning populated; result should be a list
            of [reasoning_dict, message_dict].
        """
        msg = Message(
            role=MessageRole.ASSISTANT,
            chunks=[
                ThinkingBlock(content="hmm", additional_information={"id": "t1"}),
                TextChunk(content="answer"),
            ],
        )
        conv = ResponsesMessageConverter(msg)
        conv._content = [{"type": "output_text", "text": "answer"}]
        conv._content_txt = "answer"
        conv._reasoning = [{"type": "reasoning", "id": "t1", "summary": [{"type": "summary_text", "text": "hmm"}]}]
        result = conv._assemble_message_dict()
        assert isinstance(result, list), f"Expected list when reasoning present, got {type(result)}"
        assert len(result) == 2, f"Expected 2 items, got {len(result)}"
        assert result[0]["type"] == "reasoning", f"Expected reasoning first, got {result[0].get('type')}"
        assert result[1]["role"] == "assistant", f"Expected message dict second, got {result[1].get('role')}"

    def test_assemble_message_dict_rewrites_system_to_developer(self) -> None:
        """Test _assemble_message_dict unconditionally rewrites system→developer.

        Test scenario:
            System message; the message dict should have role='developer'
            regardless of model (Responses API always uses developer role).
        """
        msg = Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="be helpful")])
        conv = ResponsesMessageConverter(msg)
        conv._content_txt = "be helpful"
        result = conv._assemble_message_dict()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result["role"] == "developer", f"Expected developer role, got {result['role']}"

    def test_assemble_message_dict_strips_none_with_drop_none(self) -> None:
        """Test _assemble_message_dict strips None content when drop_none=True.

        Test scenario:
            Assistant with null content and drop_none=True; the 'content' key
            should be removed from the result dict.
        """
        msg = Message(
            role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")],
            additional_kwargs={"function_call": {"name": "f"}},
        )
        conv = ResponsesMessageConverter(msg, drop_none=True)
        conv._content_txt = ""
        result = conv._assemble_message_dict()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert "content" not in result, f"Expected 'content' stripped, got keys: {list(result.keys())}"

    def test_assemble_chunk_tool_calls_take_precedence(self) -> None:
        """Test _assemble prioritises chunk-based tool calls over legacy kwargs.

        Test scenario:
            Assistant with both _tool_call_dicts and additional_kwargs tool_calls;
            _assemble should return the chunk-based calls.
        """
        legacy_call = {"type": "function", "id": "legacy_1", "function": {"name": "f"}}
        msg = Message(
            role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")],
            additional_kwargs={"tool_calls": [legacy_call]},
        )
        conv = ResponsesMessageConverter(msg)
        conv._tool_call_dicts = [{"type": "function_call", "call_id": "chunk_1"}]
        result = conv._assemble()
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert any(item.get("call_id") == "chunk_1" for item in result), (
            "Expected chunk-based tool call to take precedence"
        )

    def test_assemble_legacy_tool_calls_fallback(self) -> None:
        """Test _assemble uses legacy tool_calls when no chunk tool calls exist.

        Test scenario:
            Assistant with tool_calls in additional_kwargs but empty _tool_call_dicts;
            _assemble should return the legacy calls.
        """
        legacy_call = {"type": "function", "id": "legacy_1", "function": {"name": "f"}}
        msg = Message(
            role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")],
            additional_kwargs={"tool_calls": [legacy_call]},
        )
        conv = ResponsesMessageConverter(msg)
        result = conv._assemble()
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert any(item.get("id") == "legacy_1" for item in result), (
            "Expected legacy tool call in result"
        )

    def test_assemble_legacy_tool_calls_with_reasoning(self) -> None:
        """Test _assemble prepends reasoning to legacy tool calls.

        Test scenario:
            Assistant with both reasoning items and legacy tool_calls; result
            should be [reasoning, ..., legacy_call, ...].
        """
        legacy_call = {"type": "function", "id": "lg_1", "function": {"name": "f"}}
        msg = Message(
            role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")],
            additional_kwargs={"tool_calls": [legacy_call]},
        )
        conv = ResponsesMessageConverter(msg)
        conv._reasoning = [{"type": "reasoning", "id": "t1", "summary": []}]
        result = conv._assemble()
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert result[0]["type"] == "reasoning", f"Expected reasoning first, got {result[0].get('type')}"
        assert result[1]["id"] == "lg_1", f"Expected legacy call second, got {result[1]}"

    def test_assemble_tool_call_dicts_path(self) -> None:
        """Test _assemble returns tool_call_dicts when no legacy tool_calls.

        Test scenario:
            Assistant with ToolCallBlock-based _tool_call_dicts populated and
            no legacy tool_calls in kwargs.
        """
        msg = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="")])
        conv = ResponsesMessageConverter(msg)
        conv._tool_call_dicts = [
            {"type": "function_call", "call_id": "c1", "name": "fn", "arguments": "{}"},
        ]
        result = conv._assemble()
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert result[0]["type"] == "function_call", f"Expected function_call, got {result[0].get('type')}"

    def test_assemble_tool_role_path(self) -> None:
        """Test _assemble delegates to _assemble_tool_output for tool role.

        Test scenario:
            Tool message with call_id; _assemble should produce a
            function_call_output dict.
        """
        msg = Message(
            role=MessageRole.TOOL, chunks=[TextChunk(content="result")],
            additional_kwargs={"tool_call_id": "call_1"},
        )
        conv = ResponsesMessageConverter(msg)
        conv._content_txt = "result"
        result = conv._assemble()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result["type"] == "function_call_output", (
            f"Expected function_call_output, got {result.get('type')}"
        )

    def test_assemble_plain_text_user_path(self) -> None:
        """Test _assemble returns message dict for plain text user.

        Test scenario:
            User message with only input_text content items; _assemble
            falls through to _assemble_message_dict and returns a dict.
        """
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")])
        conv = ResponsesMessageConverter(msg)
        conv._content = [{"type": "input_text", "text": "hello"}]
        conv._content_txt = "hello"
        result = conv._assemble()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result["role"] == "user", f"Expected user role, got {result.get('role')}"
        assert result["content"] == "hello", f"Expected 'hello', got {result.get('content')}"

    def test_assemble_default_message_dict_path(self) -> None:
        """Test _assemble falls through to _assemble_message_dict for other cases.

        Test scenario:
            Assistant message with text content; no tool calls, not user role,
            so falls through to _assemble_message_dict.
        """
        msg = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="answer")])
        conv = ResponsesMessageConverter(msg)
        conv._content = [{"type": "output_text", "text": "answer"}]
        conv._content_txt = "answer"
        result = conv._assemble()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result["role"] == "assistant", f"Expected assistant role, got {result.get('role')}"

    def test_assemble_user_mixed_blocks_not_plain_text(self) -> None:
        """Test _assemble falls to message dict for user with non-text content items.

        Test scenario:
            User message with input_text + input_image; the all(input_text)
            check fails, so falls through to _assemble_message_dict.
        """
        msg = Message(
            role=MessageRole.USER,
            chunks=[TextChunk(content="look"), Image(url="https://example.com/img.png")],
        )
        conv = ResponsesMessageConverter(msg)
        conv._content = [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "https://example.com/img.png", "detail": "auto"},
        ]
        conv._content_txt = "look"
        result = conv._assemble()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert isinstance(result["content"], list), f"Expected list content, got {type(result['content'])}"

    def test_build_populates_state_and_returns(self) -> None:
        """Test build() orchestrates _process_blocks then _assemble.

        Test scenario:
            User message with text; verify build() populates internal state
            via _process_blocks and returns the assembled dict result.
        """
        msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="test")])
        conv = ResponsesMessageConverter(msg)
        result = conv.build()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result["role"] == "user", f"Expected user role, got {result.get('role')}"
        assert result["content"] == "test", f"Expected 'test', got {result.get('content')}"
        assert len(conv._content) == 1, "Expected _process_blocks to populate _content"
        assert conv._content_txt == "test", f"Expected 'test' in _content_txt, got {conv._content_txt!r}"

    def test_build_assistant_with_tool_calls(self) -> None:
        """Test build() end-to-end for assistant with ToolCallBlocks.

        Test scenario:
            Assistant with a ToolCallBlock chunk; verify the full pipeline
            produces a list with the function_call dict.
        """
        msg = Message(
            role=MessageRole.ASSISTANT,
            chunks=[ToolCallBlock(tool_call_id="c1", tool_name="search", tool_kwargs='{"q": "x"}')],
        )
        conv = ResponsesMessageConverter(msg)
        result = conv.build()
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        assert result[0]["type"] == "function_call", f"Expected function_call, got {result[0].get('type')}"
        assert result[0]["call_id"] == "c1", f"Expected call_id 'c1', got {result[0].get('call_id')}"

    @pytest.mark.parametrize(
        "model_name",
        ["o3-mini", "gpt-4o", "o1-mini", None],
        ids=["o3-mini", "gpt-4o", "o1-mini", "none"],
    )
    def test_build_system_always_rewritten_to_developer(self, model_name: str | None) -> None:
        """Test build() always rewrites system→developer regardless of model.

        Args:
            model_name: Any model name or None.

        Test scenario:
            The Responses API always uses 'developer' instead of 'system'.
            This is unconditional — not tied to any model family.
        """
        msg = Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="prompt")])
        result = ResponsesMessageConverter(msg, model=model_name).build()
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
        assert result["role"] == "developer", (
            f"Expected developer role (unconditional), got {result['role']}"
        )
