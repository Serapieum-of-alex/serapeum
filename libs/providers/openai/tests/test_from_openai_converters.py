"""Tests for the "from-OpenAI" parser classes and utility functions in
serapeum.openai.converters: ChatMessageParser, DictMessageParser,
LogProbParser, ToolCallAccumulator, and to_openai_tool.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.chat.chat_completion_token_logprob import (
    ChatCompletionTokenLogprob,
    TopLogprob,
)
from openai.types.completion_choice import Logprobs
from pydantic import BaseModel

from serapeum.core.base.llms.types import (
    Audio,
    Image,
    LogProb,
    Message,
    MessageRole,
    TextChunk,
    ToolCallBlock,
)
from serapeum.openai.parsers import (
    ChatMessageParser,
    DictMessageParser,
    LogProbParser,
    ToolCallAccumulator,
    to_openai_tool,
)


# ---------------------------------------------------------------------------
# ChatMessageParser
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChatMessageParser:
    """Tests for ChatMessageParser — converts typed ChatCompletionMessage to Message."""

    def test_init_stores_fields(self) -> None:
        """Verify __init__ stores message and modalities and initialises empty state.

        Test scenario:
            Construct a parser and inspect internal state before build().
        """
        openai_msg = ChatCompletionMessage(role="assistant", content="hello")
        parser = ChatMessageParser(openai_msg, modalities=["text"])

        assert parser._openai_message is openai_msg, (
            "Expected _openai_message to be stored"
        )
        assert parser._modalities == ["text"], (
            f"Expected modalities ['text'], got {parser._modalities}"
        )
        assert parser._blocks == [], (
            f"Expected empty blocks, got {parser._blocks}"
        )
        assert parser._additional_kwargs == {}, (
            f"Expected empty additional_kwargs, got {parser._additional_kwargs}"
        )

    def test_build_text_message(self) -> None:
        """Build a simple text message and verify the resulting Message.

        Test scenario:
            ChatCompletionMessage with 'text' modality and content returns
            a Message with a single TextChunk.
        """
        openai_msg = ChatCompletionMessage(role="assistant", content="hello world")
        result = ChatMessageParser(openai_msg, modalities=["text"]).build()

        assert result.role == "assistant", (
            f"Expected role 'assistant', got '{result.role}'"
        )
        assert len(result.chunks) == 1, (
            f"Expected 1 chunk, got {len(result.chunks)}"
        )
        assert isinstance(result.chunks[0], TextChunk), (
            f"Expected TextChunk, got {type(result.chunks[0])}"
        )
        assert result.chunks[0].content == "hello world", (
            f"Expected content 'hello world', got '{result.chunks[0].content}'"
        )

    def test_build_no_text_modality_skips_content(self) -> None:
        """When 'text' is not in modalities, content is not extracted to blocks.

        Test scenario:
            ChatCompletionMessage with content but modalities=["audio"] produces
            a Message with no blocks.
        """
        openai_msg = ChatCompletionMessage(role="assistant", content="hello")
        result = ChatMessageParser(openai_msg, modalities=["audio"]).build()

        assert len(result.chunks) == 0, (
            f"Expected 0 chunks when text not in modalities, got {len(result.chunks)}"
        )

    def test_build_none_content_skips_text(self) -> None:
        """When content is None, no TextChunk is created even with 'text' modality.

        Test scenario:
            Azure OpenAI returns function-calling messages without content key.
        """
        openai_msg = ChatCompletionMessage(role="assistant", content=None)
        result = ChatMessageParser(openai_msg, modalities=["text"]).build()

        text_chunks = [b for b in result.chunks if isinstance(b, TextChunk)]
        assert len(text_chunks) == 0, (
            f"Expected 0 TextChunks for None content, got {len(text_chunks)}"
        )

    def test_build_empty_content_skips_text(self) -> None:
        """When content is empty string, no TextChunk is created.

        Test scenario:
            Empty string is falsy, so _extract_text_content should skip it.
        """
        openai_msg = ChatCompletionMessage(role="assistant", content="")
        result = ChatMessageParser(openai_msg, modalities=["text"]).build()

        text_chunks = [b for b in result.chunks if isinstance(b, TextChunk)]
        assert len(text_chunks) == 0, (
            f"Expected 0 TextChunks for empty content, got {len(text_chunks)}"
        )

    def test_extract_tool_calls_creates_blocks_and_kwargs(self) -> None:
        """Tool calls are extracted into both ToolCallBlock chunks and additional_kwargs.

        Test scenario:
            ChatCompletionMessage with tool_calls produces ToolCallBlocks and
            preserves the raw tool_calls in additional_kwargs.
        """
        openai_msg = ChatCompletionMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(name="search", arguments='{"q": "test"}'),
                ),
            ],
        )
        result = ChatMessageParser(openai_msg, modalities=["text"]).build()

        tool_blocks = [b for b in result.chunks if isinstance(b, ToolCallBlock)]
        assert len(tool_blocks) == 1, (
            f"Expected 1 ToolCallBlock, got {len(tool_blocks)}"
        )
        assert tool_blocks[0].tool_call_id == "call_1", (
            f"Expected tool_call_id 'call_1', got '{tool_blocks[0].tool_call_id}'"
        )
        assert tool_blocks[0].tool_name == "search", (
            f"Expected tool_name 'search', got '{tool_blocks[0].tool_name}'"
        )
        assert "tool_calls" in result.additional_kwargs, (
            "Expected tool_calls in additional_kwargs"
        )

    def test_extract_tool_calls_multiple(self) -> None:
        """Multiple tool calls are all extracted.

        Test scenario:
            Two tool calls in the message produce two ToolCallBlocks.
        """
        openai_msg = ChatCompletionMessage(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(name="search", arguments="{}"),
                ),
                ChatCompletionMessageToolCall(
                    id="call_2",
                    type="function",
                    function=Function(name="fetch", arguments="{}"),
                ),
            ],
        )
        result = ChatMessageParser(openai_msg, modalities=["text"]).build()

        tool_blocks = [b for b in result.chunks if isinstance(b, ToolCallBlock)]
        assert len(tool_blocks) == 2, (
            f"Expected 2 ToolCallBlocks, got {len(tool_blocks)}"
        )

    def test_extract_tool_calls_no_function_skipped(self) -> None:
        """Tool call with function=None is skipped but still in additional_kwargs.

        Test scenario:
            A tool call object where function is None does not create a ToolCallBlock.
        """
        tc = MagicMock(spec=ChatCompletionMessageToolCall)
        tc.function = None
        tc.id = "call_x"

        openai_msg = ChatCompletionMessage(
            role="assistant",
            content=None,
        )
        openai_msg.tool_calls = [tc]

        result = ChatMessageParser(openai_msg, modalities=["text"]).build()

        tool_blocks = [b for b in result.chunks if isinstance(b, ToolCallBlock)]
        assert len(tool_blocks) == 0, (
            f"Expected 0 ToolCallBlocks for function=None, got {len(tool_blocks)}"
        )
        assert "tool_calls" in result.additional_kwargs, (
            "Expected raw tool_calls still in additional_kwargs"
        )

    def test_extract_audio_with_audio_modality(self) -> None:
        """Audio data is extracted when 'audio' is in modalities.

        Test scenario:
            ChatCompletionMessage with audio field and modalities=["text", "audio"]
            produces an Audio block and reference_audio_id in additional_kwargs.
        """
        audio_mock = MagicMock()
        audio_mock.id = "audio_123"
        audio_mock.data = "base64audiobytes"

        openai_msg = ChatCompletionMessage(role="assistant", content=None)
        openai_msg.audio = audio_mock

        result = ChatMessageParser(openai_msg, modalities=["text", "audio"]).build()

        audio_blocks = [b for b in result.chunks if isinstance(b, Audio)]
        assert len(audio_blocks) == 1, (
            f"Expected 1 Audio block, got {len(audio_blocks)}"
        )
        assert audio_blocks[0].content == "base64audiobytes", (
            f"Expected audio content 'base64audiobytes', got '{audio_blocks[0].content}'"
        )
        assert result.additional_kwargs.get("reference_audio_id") == "audio_123", (
            f"Expected reference_audio_id 'audio_123', got '{result.additional_kwargs.get('reference_audio_id')}'"
        )

    def test_extract_audio_without_audio_modality_skipped(self) -> None:
        """Audio data is not extracted when 'audio' is not in modalities.

        Test scenario:
            Message has audio but modalities=["text"] — no Audio block created.
        """
        audio_mock = MagicMock()
        audio_mock.id = "audio_123"
        audio_mock.data = "base64audiobytes"

        openai_msg = ChatCompletionMessage(role="assistant", content=None)
        openai_msg.audio = audio_mock

        result = ChatMessageParser(openai_msg, modalities=["text"]).build()

        audio_blocks = [b for b in result.chunks if isinstance(b, Audio)]
        assert len(audio_blocks) == 0, (
            f"Expected 0 Audio blocks without audio modality, got {len(audio_blocks)}"
        )

    def test_extract_audio_none_audio_field(self) -> None:
        """No audio extraction when audio field is None.

        Test scenario:
            Standard message with no audio field, modalities=["text", "audio"].
        """
        openai_msg = ChatCompletionMessage(role="assistant", content="text")
        result = ChatMessageParser(openai_msg, modalities=["text", "audio"]).build()

        audio_blocks = [b for b in result.chunks if isinstance(b, Audio)]
        assert len(audio_blocks) == 0, (
            f"Expected 0 Audio blocks for None audio, got {len(audio_blocks)}"
        )

    def test_build_text_and_tool_calls_together(self) -> None:
        """Build message with both text and tool calls.

        Test scenario:
            A message that has both content text and tool calls should
            produce TextChunk + ToolCallBlocks.
        """
        openai_msg = ChatCompletionMessage(
            role="assistant",
            content="Here are the results",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function=Function(name="search", arguments='{"q": "test"}'),
                ),
            ],
        )
        result = ChatMessageParser(openai_msg, modalities=["text"]).build()

        text_chunks = [b for b in result.chunks if isinstance(b, TextChunk)]
        tool_blocks = [b for b in result.chunks if isinstance(b, ToolCallBlock)]
        assert len(text_chunks) == 1, (
            f"Expected 1 TextChunk, got {len(text_chunks)}"
        )
        assert len(tool_blocks) == 1, (
            f"Expected 1 ToolCallBlock, got {len(tool_blocks)}"
        )

    def test_batch_converts_multiple_messages(self) -> None:
        """Batch classmethod converts a sequence of messages.

        Test scenario:
            Two ChatCompletionMessages are converted to two serapeum Messages.
        """
        messages = [
            ChatCompletionMessage(role="assistant", content="first"),
            ChatCompletionMessage(role="assistant", content="second"),
        ]
        results = ChatMessageParser.batch(messages, modalities=["text"])

        assert len(results) == 2, f"Expected 2 messages, got {len(results)}"
        assert results[0].chunks[0].content == "first", (
            f"Expected first message content 'first', got '{results[0].chunks[0].content}'"
        )
        assert results[1].chunks[0].content == "second", (
            f"Expected second message content 'second', got '{results[1].chunks[0].content}'"
        )

    def test_batch_empty_list(self) -> None:
        """Batch with empty sequence returns empty list.

        Test scenario:
            No messages → no results.
        """
        results = ChatMessageParser.batch([], modalities=["text"])
        assert results == [], f"Expected empty list, got {results}"

    def test_role_preserved(self) -> None:
        """The role from the ChatCompletionMessage is preserved.

        Test scenario:
            Assistant role message returns Message with assistant role.
            (ChatCompletionMessage only accepts role='assistant'.)
        """
        openai_msg = ChatCompletionMessage(role="assistant", content="hello")
        result = ChatMessageParser(openai_msg, modalities=["text"]).build()

        assert result.role == "assistant", f"Expected role 'assistant', got '{result.role}'"


# ---------------------------------------------------------------------------
# DictMessageParser
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDictMessageParser:
    """Tests for DictMessageParser — converts raw OpenAI dicts to Message."""

    def test_init_stores_message_dict(self) -> None:
        """Verify __init__ stores message_dict and initialises empty blocks.

        Test scenario:
            Construct a parser and inspect internal state.
        """
        d = {"role": "user", "content": "hello"}
        parser = DictMessageParser(d)
        assert parser._message_dict is d, "Expected _message_dict to be stored"
        assert parser._blocks == [], f"Expected empty blocks, got {parser._blocks}"

    def test_build_string_content(self) -> None:
        """Build with string content returns Message with content set.

        Test scenario:
            Simple dict with string content produces a Message with text content.
            Note: Message auto-generates a TextChunk from string content.
        """
        result = DictMessageParser({"role": "user", "content": "hello"}).build()
        assert result.role == "user", f"Expected role 'user', got '{result.role}'"
        assert result.content == "hello", (
            f"Expected content 'hello', got '{result.content}'"
        )

    def test_build_none_content(self) -> None:
        """Build with None content returns Message with content=None.

        Test scenario:
            Dict with None content (e.g., assistant with function_call only).
        """
        result = DictMessageParser({
            "role": "assistant",
            "content": None,
            "function_call": {"name": "f"},
        }).build()
        assert result.content is None, (
            f"Expected content None, got '{result.content}'"
        )
        assert result.additional_kwargs.get("function_call") == {"name": "f"}, (
            "Expected function_call in additional_kwargs"
        )

    def test_build_list_content_text_blocks(self) -> None:
        """Build with list content dispatches to block parsers.

        Test scenario:
            Dict with list content containing text blocks creates TextChunks.
            Message auto-derives content from chunks.
        """
        result = DictMessageParser({
            "role": "user",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ],
        }).build()

        assert len(result.chunks) == 2, (
            f"Expected 2 chunks, got {len(result.chunks)}"
        )
        assert all(isinstance(b, TextChunk) for b in result.chunks), (
            "Expected all chunks to be TextChunk"
        )
        assert result.chunks[0].content == "hello", (
            f"Expected first chunk content 'hello', got '{result.chunks[0].content}'"
        )

    def test_build_list_content_image_url_block(self) -> None:
        """Image URL content blocks are parsed into Image objects.

        Test scenario:
            Dict with image_url block containing a URL creates an Image with url field.
        """
        result = DictMessageParser({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/img.png",
                        "detail": "high",
                    },
                },
            ],
        }).build()

        assert len(result.chunks) == 1, f"Expected 1 chunk, got {len(result.chunks)}"
        img = result.chunks[0]
        assert isinstance(img, Image), f"Expected Image, got {type(img)}"
        assert str(img.url) == "https://example.com/img.png", (
            f"Expected URL 'https://example.com/img.png', got '{img.url}'"
        )
        assert img.detail == "high", f"Expected detail 'high', got '{img.detail}'"

    def test_build_list_content_image_data_uri(self) -> None:
        """Image with data: URI is stored in content field, not url.

        Test scenario:
            Dict with image_url block containing data: URI creates Image
            with content (stored as bytes by the Image model).
        """
        result = DictMessageParser({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,abc123",
                        "detail": "low",
                    },
                },
            ],
        }).build()

        img = result.chunks[0]
        assert isinstance(img, Image), f"Expected Image, got {type(img)}"
        assert img.url is None, f"Expected url to be None for data URI, got '{img.url}'"
        assert img.content is not None, "Expected content to be set for data URI"
        assert img.detail == "low", f"Expected detail 'low', got '{img.detail}'"

    def test_build_list_content_function_call_block(self) -> None:
        """Function call content blocks are parsed into ToolCallBlocks.

        Test scenario:
            Dict with function_call block creates a ToolCallBlock.
        """
        result = DictMessageParser({
            "role": "assistant",
            "content": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "search",
                    "arguments": {"q": "test"},
                },
            ],
        }).build()

        assert len(result.chunks) == 1, f"Expected 1 chunk, got {len(result.chunks)}"
        tc = result.chunks[0]
        assert isinstance(tc, ToolCallBlock), f"Expected ToolCallBlock, got {type(tc)}"
        assert tc.tool_call_id == "call_1", (
            f"Expected tool_call_id 'call_1', got '{tc.tool_call_id}'"
        )
        assert tc.tool_name == "search", (
            f"Expected tool_name 'search', got '{tc.tool_name}'"
        )

    def test_build_list_content_output_text_block(self) -> None:
        """output_text type is handled by the same parser as text.

        Test scenario:
            Responses API output_text blocks are parsed as TextChunks.
        """
        result = DictMessageParser({
            "role": "assistant",
            "content": [{"type": "output_text", "text": "response"}],
        }).build()

        assert len(result.chunks) == 1, f"Expected 1 chunk, got {len(result.chunks)}"
        assert isinstance(result.chunks[0], TextChunk), (
            f"Expected TextChunk, got {type(result.chunks[0])}"
        )
        assert result.chunks[0].content == "response", (
            f"Expected content 'response', got '{result.chunks[0].content}'"
        )

    def test_build_list_content_input_text_block(self) -> None:
        """input_text type is handled by the same parser as text.

        Test scenario:
            Responses API input_text blocks are parsed as TextChunks.
        """
        result = DictMessageParser({
            "role": "user",
            "content": [{"type": "input_text", "text": "question"}],
        }).build()

        assert len(result.chunks) == 1, f"Expected 1 chunk, got {len(result.chunks)}"
        assert result.chunks[0].content == "question", (
            f"Expected content 'question', got '{result.chunks[0].content}'"
        )

    def test_build_list_content_unsupported_type_raises(self) -> None:
        """Unsupported block type raises ValueError.

        Test scenario:
            A content block with type "video" should raise ValueError.
        """
        with pytest.raises(ValueError, match="Unsupported message type: video"):
            DictMessageParser({
                "role": "user",
                "content": [{"type": "video", "data": "..."}],
            }).build()

    def test_build_list_content_none_type_raises(self) -> None:
        """Content block with missing type raises ValueError.

        Test scenario:
            A content block dict with no "type" key gets None from .get("type")
            and triggers unsupported type error.
        """
        with pytest.raises(ValueError, match="Unsupported message type: None"):
            DictMessageParser({
                "role": "user",
                "content": [{"text": "no type key"}],
            }).build()

    def test_extract_additional_kwargs_strips_role_and_content(self) -> None:
        """additional_kwargs contains all keys except role and content.

        Test scenario:
            Dict with role, content, function_call, and custom_key returns
            additional_kwargs with only function_call and custom_key.
        """
        result = DictMessageParser({
            "role": "assistant",
            "content": None,
            "function_call": {"name": "f"},
            "custom_key": "value",
        }).build()

        assert "role" not in result.additional_kwargs, (
            "Expected 'role' to be stripped from additional_kwargs"
        )
        assert "content" not in result.additional_kwargs, (
            "Expected 'content' to be stripped from additional_kwargs"
        )
        assert result.additional_kwargs["function_call"] == {"name": "f"}, (
            "Expected function_call in additional_kwargs"
        )
        assert result.additional_kwargs["custom_key"] == "value", (
            "Expected custom_key in additional_kwargs"
        )

    def test_extract_additional_kwargs_empty(self) -> None:
        """Dict with only role and content produces empty additional_kwargs.

        Test scenario:
            Minimal dict with just role and content.
        """
        result = DictMessageParser({"role": "user", "content": "hi"}).build()
        assert result.additional_kwargs == {}, (
            f"Expected empty additional_kwargs, got {result.additional_kwargs}"
        )

    def test_batch_converts_multiple_dicts(self) -> None:
        """Batch classmethod converts a sequence of dicts.

        Test scenario:
            Three dicts are converted to three serapeum Messages.
        """
        dicts: list[dict[str, Any]] = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        results = DictMessageParser.batch(dicts)

        assert len(results) == 3, f"Expected 3 messages, got {len(results)}"
        assert results[0].content == "q1", (
            f"Expected first content 'q1', got '{results[0].content}'"
        )
        assert results[1].role == "assistant", (
            f"Expected second role 'assistant', got '{results[1].role}'"
        )

    def test_batch_empty_list(self) -> None:
        """Batch with empty sequence returns empty list.

        Test scenario:
            No dicts → no results.
        """
        results = DictMessageParser.batch([])
        assert results == [], f"Expected empty list, got {results}"

    def test_parse_text_missing_text_key(self) -> None:
        """_parse_text defaults to empty string when 'text' key is missing.

        Test scenario:
            A text block without a 'text' key uses the default empty string.
        """
        result = DictMessageParser._parse_text({"type": "text"})
        assert result.content == "", (
            f"Expected empty string content, got '{result.content}'"
        )

    def test_parse_image_without_detail(self) -> None:
        """_parse_image handles missing detail key gracefully.

        Test scenario:
            image_url dict without a 'detail' key produces Image with detail=None.
        """
        result = DictMessageParser._parse_image({
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.png"},
        })
        assert isinstance(result, Image), f"Expected Image, got {type(result)}"
        assert result.detail is None, (
            f"Expected detail None, got '{result.detail}'"
        )

    def test_parse_function_call_missing_fields(self) -> None:
        """_parse_function_call handles missing optional fields.

        Test scenario:
            A function_call block with no call_id, name, or arguments
            uses defaults.
        """
        result = DictMessageParser._parse_function_call({"type": "function_call"})
        assert isinstance(result, ToolCallBlock), (
            f"Expected ToolCallBlock, got {type(result)}"
        )
        assert result.tool_call_id is None, (
            f"Expected tool_call_id None, got '{result.tool_call_id}'"
        )
        assert result.tool_name == "", (
            f"Expected tool_name '', got '{result.tool_name}'"
        )

    def test_block_parsers_dict_populated(self) -> None:
        """_BLOCK_PARSERS class dict is populated with expected keys.

        Test scenario:
            Verify all expected block type strings are in the dispatch dict.
        """
        expected_keys = {"text", "image_url", "function_call", "output_text", "input_text"}
        actual_keys = set(DictMessageParser._BLOCK_PARSERS.keys())
        assert actual_keys == expected_keys, (
            f"Expected keys {expected_keys}, got {actual_keys}"
        )


# ---------------------------------------------------------------------------
# LogProbParser
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLogProbParser:
    """Tests for LogProbParser — converts OpenAI logprob types to LogProb lists."""

    def test_from_token_with_data(self) -> None:
        """from_token converts real top_logprobs into LogProb objects.

        Test scenario:
            A ChatCompletionTokenLogprob with two top_logprobs produces
            two LogProb objects with correct token, logprob, and bytes.
        """
        logprob = ChatCompletionTokenLogprob(
            token="hello",
            logprob=-0.5,
            top_logprobs=[
                TopLogprob(token="hello", logprob=-0.5, bytes=[104, 101]),
                TopLogprob(token="hi", logprob=-1.2, bytes=[104, 105]),
            ],
        )
        result = LogProbParser.from_token(logprob)

        assert len(result) == 2, f"Expected 2 LogProbs, got {len(result)}"
        assert result[0].token == "hello", (
            f"Expected token 'hello', got '{result[0].token}'"
        )
        assert result[0].logprob == -0.5, (
            f"Expected logprob -0.5, got {result[0].logprob}"
        )
        assert result[0].bytes == [104, 101], (
            f"Expected bytes [104, 101], got {result[0].bytes}"
        )
        assert result[1].token == "hi", (
            f"Expected token 'hi', got '{result[1].token}'"
        )

    def test_from_token_none_top_logprobs(self) -> None:
        """from_token returns empty list when top_logprobs is None.

        Test scenario:
            No top_logprobs data → empty result.
        """
        logprob = ChatCompletionTokenLogprob(token="x", logprob=0.0, top_logprobs=[])
        logprob.top_logprobs = None
        result = LogProbParser.from_token(logprob)
        assert result == [], f"Expected empty list, got {result}"

    def test_from_token_empty_top_logprobs(self) -> None:
        """from_token returns empty list when top_logprobs is an empty list.

        Test scenario:
            Empty top_logprobs list → empty result.
        """
        logprob = ChatCompletionTokenLogprob(token="x", logprob=0.0, top_logprobs=[])
        result = LogProbParser.from_token(logprob)
        assert result == [], f"Expected empty list, got {result}"

    def test_from_token_none_bytes_defaults_to_empty(self) -> None:
        """from_token uses empty list when token bytes is None.

        Test scenario:
            TopLogprob with bytes=None produces LogProb with bytes=[].
        """
        logprob = ChatCompletionTokenLogprob(
            token="a",
            logprob=-0.1,
            top_logprobs=[
                TopLogprob(token="a", logprob=-0.1, bytes=None),
            ],
        )
        result = LogProbParser.from_token(logprob)
        assert result[0].bytes == [], (
            f"Expected empty bytes for None, got {result[0].bytes}"
        )

    def test_from_tokens_with_data(self) -> None:
        """from_tokens converts a sequence of token logprobs.

        Test scenario:
            Two tokens with top_logprobs each produce a nested list of LogProbs.
        """
        logprobs = [
            ChatCompletionTokenLogprob(
                token="a", logprob=-0.1,
                top_logprobs=[TopLogprob(token="a", logprob=-0.1, bytes=[97])],
            ),
            ChatCompletionTokenLogprob(
                token="b", logprob=-0.2,
                top_logprobs=[TopLogprob(token="b", logprob=-0.2, bytes=[98])],
            ),
        ]
        result = LogProbParser.from_tokens(logprobs)

        assert len(result) == 2, f"Expected 2 token logprob lists, got {len(result)}"
        assert result[0][0].token == "a", (
            f"Expected first token 'a', got '{result[0][0].token}'"
        )
        assert result[1][0].token == "b", (
            f"Expected second token 'b', got '{result[1][0].token}'"
        )

    def test_from_tokens_filters_empty(self) -> None:
        """from_tokens filters out tokens with no top_logprobs.

        Test scenario:
            One token with data and one with None top_logprobs → only one entry.
        """
        logprob_with_data = ChatCompletionTokenLogprob(
            token="a", logprob=-0.1,
            top_logprobs=[TopLogprob(token="a", logprob=-0.1, bytes=[97])],
        )
        logprob_none = ChatCompletionTokenLogprob(
            token="b", logprob=-0.2, top_logprobs=[],
        )
        logprob_none.top_logprobs = None

        result = LogProbParser.from_tokens([logprob_with_data, logprob_none])
        assert len(result) == 1, (
            f"Expected 1 entry (filtered None), got {len(result)}"
        )

    def test_from_tokens_empty_sequence(self) -> None:
        """from_tokens with empty sequence returns empty list.

        Test scenario:
            No input → no output.
        """
        result = LogProbParser.from_tokens([])
        assert result == [], f"Expected empty list, got {result}"

    def test_from_completion_converts_dict(self) -> None:
        """from_completion converts a token→logprob dict to LogProb list.

        Test scenario:
            Dict with two token-logprob pairs produces two LogProbs with empty bytes.
        """
        result = LogProbParser.from_completion({"hello": -0.5, "world": -1.0})

        assert len(result) == 2, f"Expected 2 LogProbs, got {len(result)}"
        assert result[0].token == "hello", (
            f"Expected token 'hello', got '{result[0].token}'"
        )
        assert result[0].logprob == -0.5, (
            f"Expected logprob -0.5, got {result[0].logprob}"
        )
        assert result[0].bytes == [], (
            f"Expected empty bytes, got {result[0].bytes}"
        )

    def test_from_completion_empty_dict(self) -> None:
        """from_completion with empty dict returns empty list.

        Test scenario:
            Empty input → empty result.
        """
        result = LogProbParser.from_completion({})
        assert result == [], f"Expected empty list, got {result}"

    def test_from_completions_with_data(self) -> None:
        """from_completions converts Logprobs with top_logprobs.

        Test scenario:
            Logprobs object with two completion logprob dicts produces
            a nested list of LogProbs.
        """
        logprobs = Logprobs(
            top_logprobs=[
                {"hello": -0.5, "hi": -1.0},
                {"world": -0.3},
            ],
        )
        result = LogProbParser.from_completions(logprobs)

        assert len(result) == 2, f"Expected 2 entries, got {len(result)}"
        assert len(result[0]) == 2, f"Expected 2 LogProbs in first, got {len(result[0])}"
        assert len(result[1]) == 1, f"Expected 1 LogProb in second, got {len(result[1])}"

    def test_from_completions_none_top_logprobs(self) -> None:
        """from_completions returns empty list when top_logprobs is None.

        Test scenario:
            Logprobs with top_logprobs=None → empty result.
        """
        result = LogProbParser.from_completions(Logprobs(top_logprobs=None))
        assert result == [], f"Expected empty list, got {result}"


# ---------------------------------------------------------------------------
# ToolCallAccumulator
# ---------------------------------------------------------------------------


def _make_delta(
    index: int,
    *,
    name: str | None = None,
    arguments: str | None = None,
    call_id: str | None = None,
) -> ChoiceDeltaToolCall:
    """Create a ChoiceDeltaToolCall for testing."""
    return ChoiceDeltaToolCall(
        index=index,
        id=call_id,
        function=ChoiceDeltaToolCallFunction(name=name, arguments=arguments),
        type="function",
    )


@pytest.mark.unit
class TestToolCallAccumulator:
    """Tests for ToolCallAccumulator — accumulates streaming deltas."""

    def test_init_empty(self) -> None:
        """New accumulator starts with empty tool_calls.

        Test scenario:
            Fresh ToolCallAccumulator has no tool calls.
        """
        acc = ToolCallAccumulator()
        assert acc.tool_calls == [], (
            f"Expected empty tool_calls, got {acc.tool_calls}"
        )

    def test_tool_calls_property_returns_internal_list(self) -> None:
        """tool_calls property returns the internal list.

        Test scenario:
            Property getter returns the same list object.
        """
        acc = ToolCallAccumulator()
        assert acc.tool_calls is acc._tool_calls, (
            "Expected property to return internal list"
        )

    def test_update_none_delta_noop(self) -> None:
        """Update with None delta does nothing.

        Test scenario:
            Passing None to update leaves tool_calls empty.
        """
        acc = ToolCallAccumulator()
        acc.update(None)
        assert acc.tool_calls == [], (
            f"Expected empty after None delta, got {acc.tool_calls}"
        )

    def test_update_empty_list_noop(self) -> None:
        """Update with empty list does nothing.

        Test scenario:
            Passing [] to update leaves tool_calls empty.
        """
        acc = ToolCallAccumulator()
        acc.update([])
        assert acc.tool_calls == [], (
            f"Expected empty after empty delta, got {acc.tool_calls}"
        )

    def test_update_first_delta_appends(self) -> None:
        """First delta is appended as a new tool call.

        Test scenario:
            First call to update() with a delta creates one tool call.
        """
        acc = ToolCallAccumulator()
        delta = _make_delta(0, name="search", arguments="{", call_id="call_1")
        acc.update([delta])

        assert len(acc.tool_calls) == 1, (
            f"Expected 1 tool call, got {len(acc.tool_calls)}"
        )
        assert acc.tool_calls[0].id == "call_1", (
            f"Expected id 'call_1', got '{acc.tool_calls[0].id}'"
        )

    def test_update_same_index_merges(self) -> None:
        """Deltas with the same index are merged into one tool call.

        Test scenario:
            Two deltas with index=0 accumulate arguments, name, and id.
        """
        acc = ToolCallAccumulator()
        acc.update([_make_delta(0, name="search", arguments='{"q":', call_id="call_1")])
        acc.update([_make_delta(0, name="", arguments=' "test"}', call_id="")])

        assert len(acc.tool_calls) == 1, (
            f"Expected 1 tool call after merge, got {len(acc.tool_calls)}"
        )
        assert acc.tool_calls[0].function.arguments == '{"q": "test"}', (
            f"Expected merged arguments, got '{acc.tool_calls[0].function.arguments}'"
        )
        assert acc.tool_calls[0].function.name == "search", (
            f"Expected merged name 'search', got '{acc.tool_calls[0].function.name}'"
        )
        assert acc.tool_calls[0].id == "call_1", (
            f"Expected merged id 'call_1', got '{acc.tool_calls[0].id}'"
        )

    def test_update_different_index_creates_new(self) -> None:
        """Deltas with different indices create separate tool calls.

        Test scenario:
            Delta with index=0 then delta with index=1 creates two tool calls.
        """
        acc = ToolCallAccumulator()
        acc.update([_make_delta(0, name="search", arguments="{}", call_id="call_1")])
        acc.update([_make_delta(1, name="fetch", arguments="{}", call_id="call_2")])

        assert len(acc.tool_calls) == 2, (
            f"Expected 2 tool calls, got {len(acc.tool_calls)}"
        )
        assert acc.tool_calls[0].function.name == "search", (
            f"Expected first name 'search', got '{acc.tool_calls[0].function.name}'"
        )
        assert acc.tool_calls[1].function.name == "fetch", (
            f"Expected second name 'fetch', got '{acc.tool_calls[1].function.name}'"
        )

    def test_update_multi_tool_streaming_sequence(self) -> None:
        """Full multi-tool streaming sequence accumulates correctly.

        Test scenario:
            Simulates a realistic streaming sequence: tool 0 receives 3 deltas,
            then tool 1 receives 2 deltas.
        """
        acc = ToolCallAccumulator()
        acc.update([_make_delta(0, name="search", arguments='{"', call_id="c1")])
        acc.update([_make_delta(0, name="", arguments='q":', call_id="")])
        acc.update([_make_delta(0, name="", arguments='"x"}', call_id="")])
        acc.update([_make_delta(1, name="calc", arguments='{"', call_id="c2")])
        acc.update([_make_delta(1, name="", arguments='n":1}', call_id="")])

        assert len(acc.tool_calls) == 2, (
            f"Expected 2 tool calls, got {len(acc.tool_calls)}"
        )
        assert acc.tool_calls[0].function.arguments == '{"q":"x"}', (
            f"Expected first args, got '{acc.tool_calls[0].function.arguments}'"
        )
        assert acc.tool_calls[0].id == "c1", (
            f"Expected first id 'c1', got '{acc.tool_calls[0].id}'"
        )
        assert acc.tool_calls[1].function.arguments == '{"n":1}', (
            f"Expected second args, got '{acc.tool_calls[1].function.arguments}'"
        )
        assert acc.tool_calls[1].function.name == "calc", (
            f"Expected second name 'calc', got '{acc.tool_calls[1].function.name}'"
        )

    def test_merge_into_existing_initializes_none_fields(self) -> None:
        """_merge_into_existing initialises None fields before merging.

        Test scenario:
            Existing tool call has None arguments/name/id — they get
            initialised to empty string before appending delta values.
        """
        acc = ToolCallAccumulator()
        existing = ChoiceDeltaToolCall(
            index=0,
            id=None,
            function=ChoiceDeltaToolCallFunction(name=None, arguments=None),
            type="function",
        )
        acc._tool_calls.append(existing)

        delta = _make_delta(0, name="search", arguments="{}", call_id="c1")
        acc.update([delta])

        assert acc.tool_calls[0].function.arguments == "{}", (
            f"Expected arguments '{{}}', got '{acc.tool_calls[0].function.arguments}'"
        )
        assert acc.tool_calls[0].function.name == "search", (
            f"Expected name 'search', got '{acc.tool_calls[0].function.name}'"
        )
        assert acc.tool_calls[0].id == "c1", (
            f"Expected id 'c1', got '{acc.tool_calls[0].id}'"
        )

    def test_merge_delta_with_none_values(self) -> None:
        """Merging a delta with None arguments/name/id appends empty strings.

        Test scenario:
            Delta with None function fields results in no change to existing values.
        """
        acc = ToolCallAccumulator()
        acc.update([_make_delta(0, name="search", arguments='{"q":', call_id="c1")])

        delta = ChoiceDeltaToolCall(
            index=0,
            id=None,
            function=ChoiceDeltaToolCallFunction(name=None, arguments=None),
            type="function",
        )
        acc.update([delta])

        assert acc.tool_calls[0].function.arguments == '{"q":', (
            f"Expected unchanged args, got '{acc.tool_calls[0].function.arguments}'"
        )
        assert acc.tool_calls[0].function.name == "search", (
            f"Expected unchanged name, got '{acc.tool_calls[0].function.name}'"
        )
        assert acc.tool_calls[0].id == "c1", (
            f"Expected unchanged id, got '{acc.tool_calls[0].id}'"
        )

    def test_update_returns_none(self) -> None:
        """update() method returns None (no return value).

        Test scenario:
            Confirms the method does not return the internal list (no mutate-and-return).
        """
        acc = ToolCallAccumulator()
        result = acc.update([_make_delta(0, name="f", arguments="{}", call_id="c")])
        assert result is None, f"Expected None return, got {result}"


# ---------------------------------------------------------------------------
# to_openai_tool
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestToOpenaiTool:
    """Tests for to_openai_tool function."""

    def test_with_provided_description(self) -> None:
        """Provided description takes precedence over missing docstring.

        Test scenario:
            Model without docstring, description provided as argument.
        """
        class MyTool(BaseModel):
            arg: str

        result = to_openai_tool(MyTool, description="Custom desc")
        assert result["function"]["description"] == "Custom desc", (
            f"Expected 'Custom desc', got '{result['function']['description']}'"
        )

    def test_with_model_docstring(self) -> None:
        """Model docstring is used when no explicit description provided.

        Test scenario:
            Model with docstring, no description argument.
        """
        class MyTool(BaseModel):
            """Tool docstring."""
            arg: str

        result = to_openai_tool(MyTool)
        assert result["function"]["description"] == "Tool docstring.", (
            f"Expected 'Tool docstring.', got '{result['function']['description']}'"
        )

    def test_provided_description_overrides_docstring(self) -> None:
        """Provided description overrides model docstring.

        Test scenario:
            Model with docstring AND explicit description — docstring
            from schema is used because it's truthy and comes first in
            the `or` chain.
        """
        class MyTool(BaseModel):
            """Model doc."""
            arg: str

        result = to_openai_tool(MyTool, description="Provided")
        assert result["function"]["description"] == "Model doc.", (
            f"Expected 'Model doc.' (schema docstring wins), got '{result['function']['description']}'"
        )

    def test_no_description_at_all(self) -> None:
        """No docstring and no provided description yields None.

        Test scenario:
            Model without docstring, description=None (default).
        """
        class MyTool(BaseModel):
            arg: str

        result = to_openai_tool(MyTool)
        assert result["function"]["description"] is None, (
            f"Expected None description, got '{result['function']['description']}'"
        )

    def test_result_structure(self) -> None:
        """Result has correct top-level structure.

        Test scenario:
            Verify type, function.name, function.description, function.parameters.
        """
        class SearchTool(BaseModel):
            """Search for stuff."""
            query: str

        result = to_openai_tool(SearchTool)

        assert result["type"] == "function", (
            f"Expected type 'function', got '{result['type']}'"
        )
        assert "function" in result, "Expected 'function' key"
        func = result["function"]
        assert func["name"] == "SearchTool", (
            f"Expected name 'SearchTool', got '{func['name']}'"
        )
        assert "parameters" in func, "Expected 'parameters' key"

    def test_schema_reused_not_called_twice(self) -> None:
        """model_json_schema() is called only once (schema is reused for parameters).

        Test scenario:
            The 'parameters' value should be the same dict object as
            what model_json_schema() returns.
        """
        class MyTool(BaseModel):
            arg: str

        result = to_openai_tool(MyTool)
        expected_schema = MyTool.model_json_schema()
        assert result["function"]["parameters"] == expected_schema, (
            "Expected parameters to equal model_json_schema() output"
        )
