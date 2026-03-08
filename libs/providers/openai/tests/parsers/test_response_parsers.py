"""Comprehensive tests for the Responses API converter classes in
serapeum.openai.converters: _build_reasoning_content, ResponsesOutputParser,
and ResponsesStreamAccumulator.

These classes were extracted from OpenAIResponses into converters.py as part of
the R1/R2 refactoring plan.
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai.types.responses import (
    ResponseCodeInterpreterToolCall,
    ResponseCompletedEvent,
    ResponseComputerToolCall,
    ResponseCreatedEvent,
    ResponseFileSearchCallCompletedEvent,
    ResponseFileSearchToolCall,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseImageGenCallPartialImageEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseOutputTextAnnotationAddedEvent,
)
from openai.types.responses import Response
from openai.types.responses.response_output_item import ImageGenerationCall, McpCall
from openai.types.responses.response_reasoning_item import Content, Summary

from serapeum.core.base.llms.types import (
    ChatResponse,
    Image,
    Message,
    MessageRole,
    TextChunk,
    ThinkingBlock,
    ToolCallBlock,
)
from serapeum.openai.parsers import (
    ResponsesOutputParser,
    ResponsesStreamAccumulator,
    _build_reasoning_content,
)


def _make_response(
    response_id: str = "resp_1",
    output: list | None = None,
    usage: dict | None = None,
) -> Response:
    """Create a minimal Response via model_construct to bypass Pydantic validation."""
    return Response.model_construct(
        id=response_id,
        created_at=1700000000,
        model="gpt-4o-mini",
        object="response",
        output=output or [],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        usage=usage,
    )


# ---------------------------------------------------------------------------
# _build_reasoning_content
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildReasoningContent:
    """Tests for _build_reasoning_content module-level helper."""

    def test_content_only(self) -> None:
        """Test extracting text from content field only.

        Test scenario:
            Reasoning item with content but no summary returns joined content text.
        """
        item = ResponseReasoningItem(
            id="r1",
            type="reasoning",
            content=[
                Content(text="step one", type="reasoning_text"),
                Content(text="step two", type="reasoning_text"),
            ],
            summary=[],
        )
        result = _build_reasoning_content(item)
        assert result == "step one\nstep two", (
            f"Expected joined content, got {result!r}"
        )

    def test_summary_only(self) -> None:
        """Test extracting text from summary field only.

        Test scenario:
            Reasoning item with summary but no content returns joined summary text.
        """
        item = ResponseReasoningItem(
            id="r1",
            type="reasoning",
            content=None,
            summary=[
                Summary(text="conclusion A", type="summary_text"),
                Summary(text="conclusion B", type="summary_text"),
            ],
            encrypted_content=None,
            status=None,
        )
        result = _build_reasoning_content(item)
        assert result == "conclusion A\nconclusion B", (
            f"Expected joined summary, got {result!r}"
        )

    def test_content_and_summary_combined(self) -> None:
        """Test combining content and summary fields.

        Test scenario:
            Reasoning item with both content and summary returns content followed
            by summary, separated by newline.
        """
        item = ResponseReasoningItem(
            id="r1",
            type="reasoning",
            content=[Content(text="thinking", type="reasoning_text")],
            summary=[Summary(text="summary", type="summary_text")],
            encrypted_content=None,
            status=None,
        )
        result = _build_reasoning_content(item)
        assert result == "thinking\nsummary", (
            f"Expected content+summary, got {result!r}"
        )

    def test_empty_content_and_summary(self) -> None:
        """Test that empty lists for both fields returns None.

        Test scenario:
            Reasoning item with empty content and empty summary lists returns None.
        """
        item = ResponseReasoningItem(
            id="r1",
            type="reasoning",
            content=[],
            summary=[],
            encrypted_content=None,
            status=None,
        )
        result = _build_reasoning_content(item)
        assert result is None, f"Expected None for empty content/summary, got {result!r}"

    def test_none_content_and_empty_summary(self) -> None:
        """Test that None content and empty summary returns None.

        Test scenario:
            Reasoning item with content=None and summary=[] returns None.
        """
        item = ResponseReasoningItem(
            id="r1",
            type="reasoning",
            content=None,
            summary=[],
        )
        result = _build_reasoning_content(item)
        assert result is None, f"Expected None, got {result!r}"

    def test_single_content_item(self) -> None:
        """Test single content item returns just that text.

        Test scenario:
            One content item, no summary — returns the text without newlines.
        """
        item = ResponseReasoningItem(
            id="r1",
            type="reasoning",
            content=[Content(text="only one", type="reasoning_text")],
            summary=[],
        )
        result = _build_reasoning_content(item)
        assert result == "only one", f"Expected 'only one', got {result!r}"

    def test_multiple_summary_items(self) -> None:
        """Test multiple summary items without content.

        Test scenario:
            Three summary items joined by newlines.
        """
        item = ResponseReasoningItem(
            id="r1",
            type="reasoning",
            content=None,
            summary=[
                Summary(text="a", type="summary_text"),
                Summary(text="b", type="summary_text"),
                Summary(text="c", type="summary_text"),
            ],
            encrypted_content=None,
            status=None,
        )
        result = _build_reasoning_content(item)
        assert result == "a\nb\nc", f"Expected 'a\\nb\\nc', got {result!r}"


# ---------------------------------------------------------------------------
# ResponsesOutputParser
# ---------------------------------------------------------------------------


def _make_output_message(
    text: str = "Hello",
    annotations: list | None = None,
    refusal: str | None = None,
) -> ResponseOutputMessage:
    """Create a ResponseOutputMessage with text content."""
    content_kwargs: dict[str, Any] = {"text": text, "type": "output_text"}
    if annotations is not None:
        content_kwargs["annotations"] = annotations
    else:
        content_kwargs["annotations"] = []
    return ResponseOutputMessage(
        id="msg_1",
        content=[ResponseOutputText(**content_kwargs)],
        role="assistant",
        status="completed",
        type="message",
    )


def _make_function_tool_call(
    name: str = "search",
    call_id: str = "call_1",
    arguments: str = '{"q": "test"}',
) -> ResponseFunctionToolCall:
    """Create a ResponseFunctionToolCall."""
    return ResponseFunctionToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        type="function_call",
        status="completed",
    )


def _make_reasoning_item(
    content_texts: list[str] | None = None,
    summary_texts: list[str] | None = None,
    item_id: str = "r1",
) -> ResponseReasoningItem:
    """Create a ResponseReasoningItem with optional content and summary."""
    content = (
        [Content(text=t, type="reasoning_text") for t in content_texts]
        if content_texts
        else None
    )
    summary = (
        [Summary(text=t, type="summary_text") for t in summary_texts]
        if summary_texts
        else []
    )
    return ResponseReasoningItem(
        id=item_id,
        type="reasoning",
        content=content,
        summary=summary,
    )


@pytest.mark.unit
class TestResponsesOutputParser:
    """Tests for ResponsesOutputParser — parses Responses API output items into ChatResponse."""

    def test_empty_output(self) -> None:
        """Test parsing empty output list returns empty ChatResponse.

        Test scenario:
            Empty output list produces a ChatResponse with assistant role,
            no chunks, and empty built_in_tool_calls.
        """
        result = ResponsesOutputParser([]).build()
        assert isinstance(result, ChatResponse), (
            f"Expected ChatResponse, got {type(result)}"
        )
        assert result.message.role == MessageRole.ASSISTANT, (
            f"Expected ASSISTANT role, got {result.message.role}"
        )
        assert result.message.chunks == [], (
            f"Expected empty chunks, got {result.message.chunks}"
        )
        assert result.additional_kwargs["built_in_tool_calls"] == [], (
            f"Expected empty built_in_tool_calls, got {result.additional_kwargs}"
        )

    def test_text_message(self) -> None:
        """Test parsing a single text message output.

        Test scenario:
            One ResponseOutputMessage with text produces a single TextChunk.
        """
        output = [_make_output_message(text="Hello world")]
        result = ResponsesOutputParser(output).build()
        assert len(result.message.chunks) == 1, (
            f"Expected 1 chunk, got {len(result.message.chunks)}"
        )
        assert isinstance(result.message.chunks[0], TextChunk), (
            f"Expected TextChunk, got {type(result.message.chunks[0])}"
        )
        assert result.message.chunks[0].content == "Hello world", (
            f"Expected 'Hello world', got {result.message.chunks[0].content}"
        )

    def test_message_with_annotations(self) -> None:
        """Test that annotations from message content are stored in additional_kwargs.

        Test scenario:
            Output message with annotations list stores them in additional_kwargs.
            Uses the default empty annotations from _make_output_message.
        """
        msg = _make_output_message(text="cited text")
        result = ResponsesOutputParser([msg]).build()
        assert "annotations" in result.additional_kwargs, (
            f"Expected annotations key in additional_kwargs, got {result.additional_kwargs}"
        )
        assert isinstance(result.additional_kwargs["annotations"], list), (
            f"Expected list annotations, got {type(result.additional_kwargs['annotations'])}"
        )

    def test_function_tool_call(self) -> None:
        """Test parsing a function tool call output.

        Test scenario:
            Single ResponseFunctionToolCall produces a ToolCallBlock with correct fields.
        """
        tool_call = _make_function_tool_call(
            name="search", call_id="call_42", arguments='{"q": "test"}'
        )
        result = ResponsesOutputParser([tool_call]).build()
        assert len(result.message.chunks) == 1, (
            f"Expected 1 chunk, got {len(result.message.chunks)}"
        )
        block = result.message.chunks[0]
        assert isinstance(block, ToolCallBlock), (
            f"Expected ToolCallBlock, got {type(block)}"
        )
        assert block.tool_name == "search", (
            f"Expected tool_name 'search', got {block.tool_name}"
        )
        assert block.tool_call_id == "call_42", (
            f"Expected tool_call_id 'call_42', got {block.tool_call_id}"
        )
        assert block.tool_kwargs == '{"q": "test"}', (
            f"Expected arguments string, got {block.tool_kwargs}"
        )

    def test_reasoning_item(self) -> None:
        """Test parsing a reasoning item into a ThinkingBlock.

        Test scenario:
            Single reasoning item with content produces a ThinkingBlock.
        """
        reasoning = _make_reasoning_item(content_texts=["let me think"])
        result = ResponsesOutputParser([reasoning]).build()
        assert len(result.message.chunks) == 1, (
            f"Expected 1 chunk, got {len(result.message.chunks)}"
        )
        block = result.message.chunks[0]
        assert isinstance(block, ThinkingBlock), (
            f"Expected ThinkingBlock, got {type(block)}"
        )
        assert block.content == "let me think", (
            f"Expected 'let me think', got {block.content}"
        )

    def test_reasoning_additional_information(self) -> None:
        """Test that reasoning item stores additional_information excluding content/summary.

        Test scenario:
            ThinkingBlock additional_information contains id, type, etc. but NOT
            content or summary (those are used for the text content).
        """
        reasoning = _make_reasoning_item(
            content_texts=["thinking"], item_id="r42"
        )
        result = ResponsesOutputParser([reasoning]).build()
        block = result.message.chunks[0]
        assert isinstance(block, ThinkingBlock), (
            f"Expected ThinkingBlock, got {type(block)}"
        )
        info = block.additional_information
        assert info["id"] == "r42", f"Expected id 'r42', got {info.get('id')}"
        assert info["type"] == "reasoning", (
            f"Expected type 'reasoning', got {info.get('type')}"
        )
        assert "content" not in info, "content should be excluded from additional_information"
        assert "summary" not in info, "summary should be excluded from additional_information"

    def test_image_generation_success(self) -> None:
        """Test parsing a successful image generation call.

        Test scenario:
            ImageGenerationCall with status != 'failed' and base64 result produces
            an Image chunk and appends to built_in_tool_calls.
        """
        b64_data = base64.b64encode(b"fake_png_data").decode()
        img_call = ImageGenerationCall(
            id="img_1",
            type="image_generation_call",
            status="completed",
            result=b64_data,
        )
        result = ResponsesOutputParser([img_call]).build()
        image_chunks = [c for c in result.message.chunks if isinstance(c, Image)]
        assert len(image_chunks) == 1, (
            f"Expected 1 Image chunk, got {len(image_chunks)}"
        )
        assert image_chunks[0].content is not None, "Expected non-None image content"
        assert len(result.additional_kwargs["built_in_tool_calls"]) == 1, (
            "Expected img_call in built_in_tool_calls"
        )

    def test_image_generation_failed(self) -> None:
        """Test that failed image generation is not added.

        Test scenario:
            ImageGenerationCall with status='failed' produces no chunks
            and is not added to built_in_tool_calls.
        """
        img_call = ImageGenerationCall(
            id="img_1",
            type="image_generation_call",
            status="failed",
            result=None,
        )
        result = ResponsesOutputParser([img_call]).build()
        assert result.message.chunks == [], (
            f"Expected no chunks for failed image, got {result.message.chunks}"
        )
        assert result.additional_kwargs["built_in_tool_calls"] == [], (
            "Failed image should not be in built_in_tool_calls"
        )

    def test_image_generation_no_result(self) -> None:
        """Test image generation with status not failed but result=None.

        Test scenario:
            ImageGenerationCall with status 'in_progress' and no result: appended to
            built_in_tool_calls but no Image chunk created.
        """
        img_call = ImageGenerationCall(
            id="img_1",
            type="image_generation_call",
            status="in_progress",
            result=None,
        )
        result = ResponsesOutputParser([img_call]).build()
        image_chunks = [c for c in result.message.chunks if isinstance(c, Image)]
        assert image_chunks == [], "Expected no Image chunks when result is None"
        assert len(result.additional_kwargs["built_in_tool_calls"]) == 1, (
            "Expected img_call in built_in_tool_calls even without result"
        )

    @pytest.mark.parametrize(
        "tool_type,tool_factory",
        [
            ("file_search", lambda: ResponseFileSearchToolCall(
                id="fs_1", type="file_search_call", status="completed",
                queries=["test"], results=None,
            )),
            ("web_search", lambda: ResponseFunctionWebSearch(
                id="ws_1", type="web_search_call", status="completed",
                action={"type": "search", "query": "test"},
            )),
            ("code_interpreter", lambda: ResponseCodeInterpreterToolCall(
                id="ci_1", type="code_interpreter_call", status="completed",
                container_id="ctr_1", code="print('hi')", outputs=[],
            )),
        ],
        ids=["file_search", "web_search", "code_interpreter"],
    )
    def test_built_in_tool_types(self, tool_type: str, tool_factory: Any) -> None:
        """Test that built-in tool types go to additional_kwargs, not message chunks.

        Args:
            tool_type: Description of the tool type for test identification.
            tool_factory: Factory callable producing the tool call instance.

        Test scenario:
            Built-in tool calls (file_search, web_search, code_interpreter, etc.)
            are stored in additional_kwargs['built_in_tool_calls'] only.
        """
        tool_call = tool_factory()
        result = ResponsesOutputParser([tool_call]).build()
        assert result.message.chunks == [], (
            f"Expected no chunks for {tool_type}, got {result.message.chunks}"
        )
        assert len(result.additional_kwargs["built_in_tool_calls"]) == 1, (
            f"Expected 1 built_in_tool_call for {tool_type}"
        )
        assert result.additional_kwargs["built_in_tool_calls"][0] is tool_call, (
            f"Expected the exact {tool_type} tool call object"
        )

    def test_mixed_output_items(self) -> None:
        """Test parsing a mix of output item types.

        Test scenario:
            Output with reasoning, text message, function tool call, and file search
            produces the correct combination of chunks and additional_kwargs.
        """
        output = [
            _make_reasoning_item(content_texts=["thinking"]),
            _make_output_message(text="answer"),
            _make_function_tool_call(name="fn", call_id="c1", arguments="{}"),
            ResponseFileSearchToolCall(
                id="fs_1", type="file_search_call", status="completed",
                queries=["q"],
            ),
        ]
        result = ResponsesOutputParser(output).build()

        thinking_blocks = [c for c in result.message.chunks if isinstance(c, ThinkingBlock)]
        text_chunks = [c for c in result.message.chunks if isinstance(c, TextChunk)]
        tool_calls = [c for c in result.message.chunks if isinstance(c, ToolCallBlock)]

        assert len(thinking_blocks) == 1, f"Expected 1 ThinkingBlock, got {len(thinking_blocks)}"
        assert len(text_chunks) == 1, f"Expected 1 TextChunk, got {len(text_chunks)}"
        assert len(tool_calls) == 1, f"Expected 1 ToolCallBlock, got {len(tool_calls)}"
        assert len(result.additional_kwargs["built_in_tool_calls"]) == 1, (
            "Expected 1 built-in tool call"
        )

    def test_multiple_function_tool_calls(self) -> None:
        """Test parsing multiple function tool calls.

        Test scenario:
            Two function tool calls produce two ToolCallBlock chunks.
        """
        output = [
            _make_function_tool_call(name="fn_a", call_id="c1", arguments='{"a":1}'),
            _make_function_tool_call(name="fn_b", call_id="c2", arguments='{"b":2}'),
        ]
        result = ResponsesOutputParser(output).build()
        tool_calls = [c for c in result.message.chunks if isinstance(c, ToolCallBlock)]
        assert len(tool_calls) == 2, f"Expected 2 ToolCallBlocks, got {len(tool_calls)}"
        assert tool_calls[0].tool_name == "fn_a", (
            f"Expected fn_a, got {tool_calls[0].tool_name}"
        )
        assert tool_calls[1].tool_name == "fn_b", (
            f"Expected fn_b, got {tool_calls[1].tool_name}"
        )

    def test_multiple_reasoning_items(self) -> None:
        """Test parsing multiple reasoning items.

        Test scenario:
            Four reasoning items with varying content/summary combos produce
            four ThinkingBlocks with correct text.
        """
        output = [
            _make_reasoning_item(content_texts=["hello world", "this is a test"]),
            _make_reasoning_item(content_texts=["another test"]),
            _make_reasoning_item(content_texts=["another test"], summary_texts=["hello"]),
            _make_reasoning_item(summary_texts=["hello", "world"]),
        ]
        result = ResponsesOutputParser(output).build()
        thinking = [c for c in result.message.chunks if isinstance(c, ThinkingBlock)]
        assert len(thinking) == 4, f"Expected 4 ThinkingBlocks, got {len(thinking)}"
        assert thinking[0].content == "hello world\nthis is a test", (
            f"Unexpected content[0]: {thinking[0].content!r}"
        )
        assert thinking[1].content == "another test", (
            f"Unexpected content[1]: {thinking[1].content!r}"
        )
        assert thinking[2].content == "another test\nhello", (
            f"Unexpected content[2]: {thinking[2].content!r}"
        )
        assert thinking[3].content == "hello\nworld", (
            f"Unexpected content[3]: {thinking[3].content!r}"
        )

    def test_message_role_is_always_assistant(self) -> None:
        """Test that the result message role is always ASSISTANT.

        Test scenario:
            Regardless of output contents, the ChatResponse message role is ASSISTANT.
        """
        result = ResponsesOutputParser([_make_output_message()]).build()
        assert result.message.role == MessageRole.ASSISTANT, (
            f"Expected ASSISTANT role, got {result.message.role}"
        )

    def test_message_with_refusal(self) -> None:
        """Test that refusal from message content part is stored in additional_kwargs.

        Test scenario:
            A ResponseOutputRefusal content part with a refusal string is
            detected by hasattr(part, "refusal") and stored in additional_kwargs.
        """
        from openai.types.responses import ResponseOutputRefusal

        refusal_part = ResponseOutputRefusal(
            refusal="I cannot help with that",
            type="refusal",
        )
        msg = ResponseOutputMessage(
            id="msg_ref",
            content=[refusal_part],
            role="assistant",
            status="completed",
            type="message",
        )
        result = ResponsesOutputParser([msg]).build()
        assert result.additional_kwargs.get("refusal") == "I cannot help with that", (
            f"Expected refusal in additional_kwargs, got {result.additional_kwargs}"
        )

    def test_built_in_tool_types_tuple_coverage(self) -> None:
        """Test the _BUILT_IN_TOOL_TYPES class attribute is correct.

        Test scenario:
            Verify the tuple contains the expected 5 types.
        """
        expected = (
            ResponseCodeInterpreterToolCall,
            ResponseComputerToolCall,
            ResponseFileSearchToolCall,
            ResponseFunctionWebSearch,
            McpCall,
        )
        assert ResponsesOutputParser._BUILT_IN_TOOL_TYPES == expected, (
            f"Expected {expected}, got {ResponsesOutputParser._BUILT_IN_TOOL_TYPES}"
        )


# ---------------------------------------------------------------------------
# ResponsesStreamAccumulator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResponsesStreamAccumulator:
    """Tests for ResponsesStreamAccumulator — accumulates Responses API streaming events."""

    @pytest.fixture()
    def accumulator(self) -> ResponsesStreamAccumulator:
        """Create a default accumulator with tracking disabled.

        Returns:
            ResponsesStreamAccumulator: Default instance.
        """
        return ResponsesStreamAccumulator()

    @pytest.fixture()
    def tracking_accumulator(self) -> ResponsesStreamAccumulator:
        """Create an accumulator with response tracking enabled.

        Returns:
            ResponsesStreamAccumulator: Instance with track_previous_responses=True.
        """
        return ResponsesStreamAccumulator(
            track_previous_responses=True,
            previous_response_id="prev_123",
        )

    def test_init_defaults(self) -> None:
        """Test default initialization state.

        Test scenario:
            Default constructor sets empty state, no tracking.
        """
        acc = ResponsesStreamAccumulator()
        assert acc.built_in_tool_calls == [], (
            f"Expected empty built_in_tool_calls, got {acc.built_in_tool_calls}"
        )
        assert acc.previous_response_id is None, (
            f"Expected None previous_response_id, got {acc.previous_response_id}"
        )
        assert acc.additional_kwargs == {"built_in_tool_calls": []}, (
            f"Expected default additional_kwargs, got {acc.additional_kwargs}"
        )

    def test_init_with_tracking(self) -> None:
        """Test initialization with tracking enabled.

        Test scenario:
            Constructor with track_previous_responses=True and initial ID.
        """
        acc = ResponsesStreamAccumulator(
            track_previous_responses=True,
            previous_response_id="resp_abc",
        )
        assert acc.previous_response_id == "resp_abc", (
            f"Expected 'resp_abc', got {acc.previous_response_id}"
        )

    def test_text_delta_event(self, accumulator: ResponsesStreamAccumulator) -> None:
        """Test handling a text delta event.

        Test scenario:
            ResponseTextDeltaEvent produces a TextChunk block and delta string.
        """
        event = ResponseTextDeltaEvent(
            content_index=0,
            item_id="item_1",
            output_index=0,
            delta="Hello",
            type="response.output_text.delta",
            sequence_number=1,
            logprobs=[],
        )
        blocks, delta = accumulator.update(event)
        assert delta == "Hello", f"Expected delta 'Hello', got {delta!r}"
        assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
        assert isinstance(blocks[0], TextChunk), (
            f"Expected TextChunk, got {type(blocks[0])}"
        )
        assert blocks[0].content == "Hello", (
            f"Expected content 'Hello', got {blocks[0].content}"
        )

    def test_response_created_event_with_tracking(
        self, tracking_accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that ResponseCreatedEvent updates previous_response_id when tracking.

        Test scenario:
            With tracking enabled, ResponseCreatedEvent sets the response ID.
        """
        response = _make_response(response_id="resp_new_123")
        event = ResponseCreatedEvent(
            response=response,
            type="response.created",
            sequence_number=0,
        )
        blocks, delta = tracking_accumulator.update(event)
        assert blocks == [], f"Expected no blocks, got {blocks}"
        assert delta == "", f"Expected empty delta, got {delta!r}"
        assert tracking_accumulator.previous_response_id == "resp_new_123", (
            f"Expected 'resp_new_123', got {tracking_accumulator.previous_response_id}"
        )

    def test_response_created_event_without_tracking(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that ResponseCreatedEvent does NOT update ID when tracking is off.

        Test scenario:
            Without tracking, ResponseCreatedEvent does not change previous_response_id.
        """
        response = _make_response(response_id="resp_ignored")
        event = ResponseCreatedEvent(
            response=response,
            type="response.created",
            sequence_number=0,
        )
        accumulator.update(event)
        assert accumulator.previous_response_id is None, (
            f"Expected None, got {accumulator.previous_response_id}"
        )

    def test_response_in_progress_event_with_tracking(
        self, tracking_accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that ResponseInProgressEvent updates previous_response_id when tracking.

        Test scenario:
            With tracking enabled, ResponseInProgressEvent sets the response ID.
        """
        response = _make_response(response_id="resp_prog_456")
        event = ResponseInProgressEvent(
            response=response,
            type="response.in_progress",
            sequence_number=1,
        )
        blocks, delta = tracking_accumulator.update(event)
        assert blocks == [], f"Expected no blocks, got {blocks}"
        assert delta == "", f"Expected empty delta, got {delta!r}"
        assert tracking_accumulator.previous_response_id == "resp_prog_456", (
            f"Expected 'resp_prog_456', got {tracking_accumulator.previous_response_id}"
        )

    def test_output_item_added_function_tool_call(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that OutputItemAdded with function tool call sets _current_tool_call.

        Test scenario:
            Adding a ResponseFunctionToolCall item sets up tracking for argument deltas.
        """
        tool_call = ResponseFunctionToolCall(
            id="tc_1",
            call_id="call_1",
            type="function_call",
            name="search",
            arguments="",
            status="in_progress",
        )
        event = ResponseOutputItemAddedEvent(
            item=tool_call,
            output_index=0,
            sequence_number=1,
            type="response.output_item.added",
        )
        blocks, delta = accumulator.update(event)
        assert blocks == [], f"Expected no blocks, got {blocks}"
        assert delta == "", f"Expected empty delta, got {delta!r}"
        assert accumulator._current_tool_call is tool_call, (
            "Expected _current_tool_call to be set"
        )

    def test_output_item_added_non_function_ignored(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that OutputItemAdded with non-function item does not set _current_tool_call.

        Test scenario:
            Adding a non-function item (e.g. message) does not affect _current_tool_call.
        """
        msg_item = _make_output_message()
        event = ResponseOutputItemAddedEvent(
            item=msg_item,
            output_index=0,
            sequence_number=1,
            type="response.output_item.added",
        )
        accumulator.update(event)
        assert accumulator._current_tool_call is None, (
            "Expected _current_tool_call to remain None"
        )

    def test_function_call_arguments_delta(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that argument deltas accumulate on the current tool call.

        Test scenario:
            After setting up a tool call, argument delta events append to its arguments.
        """
        tool_call = ResponseFunctionToolCall(
            id="tc_1", call_id="call_1", type="function_call",
            name="search", arguments="", status="in_progress",
        )
        accumulator._current_tool_call = tool_call

        event = ResponseFunctionCallArgumentsDeltaEvent(
            item_id="tc_1",
            output_index=0,
            type="response.function_call_arguments.delta",
            delta='{"q": "te',
            sequence_number=2,
        )
        blocks, delta = accumulator.update(event)
        assert blocks == [], f"Expected no blocks during delta, got {blocks}"
        assert delta == "", f"Expected empty delta, got {delta!r}"
        assert tool_call.arguments == '{"q": "te', (
            f"Expected accumulated arguments, got {tool_call.arguments!r}"
        )

    def test_function_call_arguments_delta_no_current_tool(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that argument delta with no current tool call is a no-op.

        Test scenario:
            Receiving an arguments delta when _current_tool_call is None does nothing.
        """
        event = ResponseFunctionCallArgumentsDeltaEvent(
            item_id="tc_1",
            output_index=0,
            type="response.function_call_arguments.delta",
            delta='{"q": "test"}',
            sequence_number=2,
        )
        blocks, delta = accumulator.update(event)
        assert blocks == [], f"Expected no blocks, got {blocks}"
        assert delta == "", f"Expected empty delta, got {delta!r}"

    def test_function_call_arguments_done(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that arguments done event produces a ToolCallBlock and resets current tool.

        Test scenario:
            After setting up a tool call, arguments done event finalizes it into a
            ToolCallBlock, sets status to 'completed', and clears _current_tool_call.
        """
        tool_call = ResponseFunctionToolCall(
            id="tc_1", call_id="call_1", type="function_call",
            name="search", arguments='{"q": "te', status="in_progress",
        )
        accumulator._current_tool_call = tool_call

        event = ResponseFunctionCallArgumentsDoneEvent(
            name="search",
            item_id="tc_1",
            output_index=0,
            type="response.function_call_arguments.done",
            arguments='{"q": "test"}',
            sequence_number=3,
        )
        blocks, delta = accumulator.update(event)
        assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
        assert isinstance(blocks[0], ToolCallBlock), (
            f"Expected ToolCallBlock, got {type(blocks[0])}"
        )
        assert blocks[0].tool_name == "search", (
            f"Expected 'search', got {blocks[0].tool_name}"
        )
        assert blocks[0].tool_kwargs == '{"q": "test"}', (
            f"Expected final arguments, got {blocks[0].tool_kwargs}"
        )
        assert blocks[0].tool_call_id == "call_1", (
            f"Expected 'call_1', got {blocks[0].tool_call_id}"
        )
        assert accumulator._current_tool_call is None, (
            "Expected _current_tool_call to be reset to None"
        )

    def test_function_call_arguments_done_no_current_tool(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that arguments done with no current tool call is a no-op.

        Test scenario:
            Receiving arguments done when _current_tool_call is None produces no blocks.
        """
        event = ResponseFunctionCallArgumentsDoneEvent(
            name="search",
            item_id="tc_1",
            output_index=0,
            type="response.function_call_arguments.done",
            arguments='{"q": "test"}',
            sequence_number=3,
        )
        blocks, delta = accumulator.update(event)
        assert blocks == [], f"Expected no blocks, got {blocks}"

    def test_image_gen_partial_event(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test handling partial image generation event.

        Test scenario:
            ResponseImageGenCallPartialImageEvent with base64 data produces an
            Image chunk with decoded content and detail set to the index.
        """
        b64_data = base64.b64encode(b"partial_img").decode()
        event = ResponseImageGenCallPartialImageEvent(
            item_id="img_1",
            output_index=0,
            type="response.image_generation_call.partial_image",
            partial_image_b64=b64_data,
            partial_image_index=0,
            sequence_number=5,
        )
        blocks, delta = accumulator.update(event)
        assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
        assert isinstance(blocks[0], Image), f"Expected Image, got {type(blocks[0])}"
        assert blocks[0].content is not None, "Expected non-None image content"
        assert blocks[0].detail == "id_0", (
            f"Expected detail 'id_0', got {blocks[0].detail}"
        )

    def test_image_gen_partial_event_empty_b64(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that empty partial_image_b64 produces no blocks.

        Test scenario:
            When partial_image_b64 is empty string, no Image chunk is created.
        """
        event = ResponseImageGenCallPartialImageEvent(
            item_id="img_1",
            output_index=0,
            type="response.image_generation_call.partial_image",
            partial_image_b64="",
            partial_image_index=0,
            sequence_number=5,
        )
        blocks, delta = accumulator.update(event)
        assert blocks == [], f"Expected no blocks for empty b64, got {blocks}"

    def test_text_annotation_added_event(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that annotation added event stores annotation in additional_kwargs.

        Test scenario:
            ResponseOutputTextAnnotationAddedEvent appends annotation to
            additional_kwargs['annotations'].
        """
        annotation = {"type": "url_citation", "url": "https://example.com"}
        event = ResponseOutputTextAnnotationAddedEvent(
            item_id="item_1",
            output_index=0,
            content_index=0,
            annotation_index=0,
            type="response.output_text.annotation.added",
            annotation=annotation,
            sequence_number=6,
        )
        accumulator.update(event)
        assert "annotations" in accumulator.additional_kwargs, (
            "Expected annotations key in additional_kwargs"
        )
        assert accumulator.additional_kwargs["annotations"] == [annotation], (
            f"Expected annotation list, got {accumulator.additional_kwargs['annotations']}"
        )

    def test_multiple_annotations_accumulate(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that multiple annotation events accumulate.

        Test scenario:
            Two annotation events produce a list with two entries.
        """
        for i in range(2):
            event = ResponseOutputTextAnnotationAddedEvent(
                item_id="item_1",
                output_index=0,
                content_index=0,
                annotation_index=i,
                type="response.output_text.annotation.added",
                annotation={"type": f"cite_{i}"},
                sequence_number=6 + i,
            )
            accumulator.update(event)
        assert len(accumulator.additional_kwargs["annotations"]) == 2, (
            f"Expected 2 annotations, got {len(accumulator.additional_kwargs['annotations'])}"
        )

    def test_file_search_completed_event(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that file search completed event is stored in built_in_tool_calls.

        Test scenario:
            ResponseFileSearchCallCompletedEvent is appended to built_in_tool_calls.
        """
        event = ResponseFileSearchCallCompletedEvent(
            item_id="fs_1",
            output_index=0,
            type="response.file_search_call.completed",
            sequence_number=7,
        )
        accumulator.update(event)
        assert len(accumulator.built_in_tool_calls) == 1, (
            f"Expected 1 built-in tool call, got {len(accumulator.built_in_tool_calls)}"
        )
        assert accumulator.built_in_tool_calls[0] is event, (
            "Expected the exact event object"
        )

    def test_web_search_completed_event(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that web search completed event is stored in built_in_tool_calls.

        Test scenario:
            ResponseWebSearchCallCompletedEvent is appended to built_in_tool_calls.
        """
        event = ResponseWebSearchCallCompletedEvent(
            item_id="ws_1",
            output_index=0,
            type="response.web_search_call.completed",
            sequence_number=8,
        )
        accumulator.update(event)
        assert len(accumulator.built_in_tool_calls) == 1, (
            f"Expected 1 built-in tool call, got {len(accumulator.built_in_tool_calls)}"
        )

    def test_output_item_done_reasoning(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that OutputItemDone with reasoning item produces ThinkingBlock.

        Test scenario:
            ResponseOutputItemDoneEvent with a ResponseReasoningItem produces
            a ThinkingBlock with correct content and additional_information.
        """
        reasoning = _make_reasoning_item(
            content_texts=["step 1", "step 2"], item_id="r1"
        )
        event = ResponseOutputItemDoneEvent(
            item=reasoning,
            output_index=0,
            sequence_number=9,
            type="response.output_item.done",
        )
        blocks, delta = accumulator.update(event)
        assert len(blocks) == 1, f"Expected 1 block, got {len(blocks)}"
        assert isinstance(blocks[0], ThinkingBlock), (
            f"Expected ThinkingBlock, got {type(blocks[0])}"
        )
        assert blocks[0].content == "step 1\nstep 2", (
            f"Expected 'step 1\\nstep 2', got {blocks[0].content}"
        )
        info = blocks[0].additional_information
        assert "content" not in info, "content should be excluded"
        assert "summary" not in info, "summary should be excluded"

    def test_output_item_done_non_reasoning(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that OutputItemDone with non-reasoning item produces no blocks.

        Test scenario:
            ResponseOutputItemDoneEvent with a message item (not reasoning) is ignored.
        """
        msg = _make_output_message()
        event = ResponseOutputItemDoneEvent(
            item=msg,
            output_index=0,
            sequence_number=9,
            type="response.output_item.done",
        )
        blocks, delta = accumulator.update(event)
        assert blocks == [], f"Expected no blocks, got {blocks}"

    def test_completed_event_stores_usage(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that ResponseCompletedEvent stores usage in additional_kwargs.

        Test scenario:
            Completed event with usage info stores it in additional_kwargs['usage'].
        """
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        response = _make_response(
            response_id="resp_done",
            output=[_make_output_message(text="done")],
            usage=usage,
        )
        event = ResponseCompletedEvent(
            response=response,
            type="response.completed",
            sequence_number=99,
        )
        blocks, delta = accumulator.update(event)
        stored_usage = accumulator._additional_kwargs.get("usage")
        assert stored_usage is not None, "Expected usage to be stored"
        assert stored_usage.prompt_tokens == 10, (
            f"Expected prompt_tokens=10, got {stored_usage.prompt_tokens}"
        )
        assert stored_usage.completion_tokens == 20, (
            f"Expected completion_tokens=20, got {stored_usage.completion_tokens}"
        )
        assert len(blocks) > 0, "Expected blocks from completed event's output parsing"

    def test_completed_event_parses_output(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that ResponseCompletedEvent delegates to ResponsesOutputParser.

        Test scenario:
            Completed event's response.output is parsed by ResponsesOutputParser
            and the resulting chunks are returned as blocks.
        """
        response = _make_response(
            response_id="resp_done",
            output=[
                _make_output_message(text="final answer"),
                _make_function_tool_call(name="fn", call_id="c1", arguments="{}"),
            ],
        )
        event = ResponseCompletedEvent(
            response=response,
            type="response.completed",
            sequence_number=99,
        )
        blocks, delta = accumulator.update(event)
        text_chunks = [b for b in blocks if isinstance(b, TextChunk)]
        tool_blocks = [b for b in blocks if isinstance(b, ToolCallBlock)]
        assert len(text_chunks) == 1, f"Expected 1 TextChunk, got {len(text_chunks)}"
        assert text_chunks[0].content == "final answer", (
            f"Expected 'final answer', got {text_chunks[0].content}"
        )
        assert len(tool_blocks) == 1, f"Expected 1 ToolCallBlock, got {len(tool_blocks)}"

    def test_additional_kwargs_property_merges_built_in(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that additional_kwargs property merges built_in_tool_calls.

        Test scenario:
            After adding a file search event, additional_kwargs includes
            built_in_tool_calls with the event.
        """
        event = ResponseFileSearchCallCompletedEvent(
            item_id="fs_1",
            output_index=0,
            type="response.file_search_call.completed",
            sequence_number=7,
        )
        accumulator.update(event)
        kwargs = accumulator.additional_kwargs
        assert "built_in_tool_calls" in kwargs, "Expected built_in_tool_calls key"
        assert len(kwargs["built_in_tool_calls"]) == 1, (
            f"Expected 1 entry, got {len(kwargs['built_in_tool_calls'])}"
        )

    def test_additional_kwargs_returns_fresh_dict(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that additional_kwargs returns a new dict each call.

        Test scenario:
            Two calls to additional_kwargs return equal but not identical dicts.
        """
        kwargs1 = accumulator.additional_kwargs
        kwargs2 = accumulator.additional_kwargs
        assert kwargs1 == kwargs2, "Expected equal dicts"
        assert kwargs1 is not kwargs2, "Expected different dict objects"

    def test_full_function_call_lifecycle(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test the complete lifecycle of a function call through streaming events.

        Test scenario:
            OutputItemAdded → ArgumentsDelta (x2) → ArgumentsDone simulates a
            realistic function call streaming sequence.
        """
        tool_call = ResponseFunctionToolCall(
            id="tc_1", call_id="call_1", type="function_call",
            name="get_weather", arguments="", status="in_progress",
        )

        accumulator.update(ResponseOutputItemAddedEvent(
            item=tool_call,
            output_index=0,
            sequence_number=1,
            type="response.output_item.added",
        ))

        accumulator.update(ResponseFunctionCallArgumentsDeltaEvent(
            item_id="tc_1",
            output_index=0,
            type="response.function_call_arguments.delta",
            delta='{"city":',
            sequence_number=2,
        ))

        accumulator.update(ResponseFunctionCallArgumentsDeltaEvent(
            item_id="tc_1",
            output_index=0,
            type="response.function_call_arguments.delta",
            delta=' "Paris"}',
            sequence_number=3,
        ))

        blocks, delta = accumulator.update(ResponseFunctionCallArgumentsDoneEvent(
            name="get_weather",
            item_id="tc_1",
            output_index=0,
            type="response.function_call_arguments.done",
            arguments='{"city": "Paris"}',
            sequence_number=4,
        ))

        assert len(blocks) == 1, f"Expected 1 block on done, got {len(blocks)}"
        assert isinstance(blocks[0], ToolCallBlock), (
            f"Expected ToolCallBlock, got {type(blocks[0])}"
        )
        assert blocks[0].tool_name == "get_weather", (
            f"Expected 'get_weather', got {blocks[0].tool_name}"
        )
        assert blocks[0].tool_kwargs == '{"city": "Paris"}', (
            f"Expected final args, got {blocks[0].tool_kwargs}"
        )

    def test_unhandled_event_type_returns_empty(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that unrecognized event types return empty blocks and delta.

        Test scenario:
            An event type that doesn't match any handler produces ([], "").
        """
        mock_event = MagicMock(spec=ResponseStreamEvent)
        mock_event.__class__ = type("UnknownEvent", (), {})
        blocks, delta = accumulator.update(mock_event)
        assert blocks == [], f"Expected empty blocks, got {blocks}"
        assert delta == "", f"Expected empty delta, got {delta!r}"

    def test_multiple_text_deltas_accumulate_independently(
        self, accumulator: ResponsesStreamAccumulator
    ) -> None:
        """Test that each text delta produces its own independent block.

        Test scenario:
            Three text deltas produce three separate TextChunk blocks and deltas.
        """
        all_blocks = []
        all_deltas = []
        for text in ["Hello", " ", "world"]:
            event = ResponseTextDeltaEvent(
                content_index=0,
                item_id="item_1",
                output_index=0,
                delta=text,
                type="response.output_text.delta",
                sequence_number=1,
                logprobs=[],
            )
            blocks, delta = accumulator.update(event)
            all_blocks.extend(blocks)
            all_deltas.append(delta)

        assert len(all_blocks) == 3, f"Expected 3 blocks, got {len(all_blocks)}"
        assert all_deltas == ["Hello", " ", "world"], (
            f"Expected ['Hello', ' ', 'world'], got {all_deltas}"
        )
