"""Inbound Responses API parsers — OpenAI Responses output to serapeum types.

Classes:

- :class:`ResponsesOutputParser` — parses ``Response.output`` items into a ``ChatResponse``
- :class:`ResponsesStreamAccumulator` — accumulates streaming ``ResponseStreamEvent`` instances
- :func:`_build_reasoning_content` — extracts text from a ``ResponseReasoningItem``
"""

from __future__ import annotations

import base64
from typing import Any

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
    ResponseOutputItem,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseReasoningItem,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseWebSearchCallCompletedEvent,
)
from openai.types.responses.response_output_item import ImageGenerationCall, McpCall

from serapeum.core.llms import (
    ChatResponse,
    ContentBlock,
    Image,
    Message,
    MessageRole,
    TextChunk,
    ThinkingBlock,
    ToolCallBlock,
)

def _build_reasoning_content(item: ResponseReasoningItem) -> str | None:
    """Extract text from a reasoning item's content and summary fields.

    Args:
        item: A ``ResponseReasoningItem`` from the Responses API.

    Returns:
        Concatenated text from content and summary, or ``None`` if both are empty.
    """
    content: str | None = None
    if item.content:
        content = "\n".join([i.text for i in item.content])
    if item.summary:
        if content:
            content += "\n" + "\n".join([i.text for i in item.summary])
        else:
            content = "\n".join([i.text for i in item.summary])
    return content


class ResponsesOutputParser:
    """Parse OpenAI Responses API output items into a :class:`~serapeum.core.llms.ChatResponse`.

    Handles all ``ResponseOutputItem`` subtypes returned by the Responses API:

    - ``ResponseOutputMessage`` — text content, annotations, refusal
    - ``ResponseFunctionToolCall`` — function tool calls → :class:`~serapeum.core.llms.ToolCallBlock`
    - ``ResponseReasoningItem`` — reasoning content/summary → :class:`~serapeum.core.llms.ThinkingBlock`
    - ``ImageGenerationCall`` — base64 image → :class:`~serapeum.core.llms.Image`
    - Built-in tool calls (file search, web search, code interpreter, computer, MCP) →
      stored in ``additional_kwargs["built_in_tool_calls"]``

    This is the Responses API counterpart of :class:`ChatMessageParser`, which
    handles Chat Completions API responses.

    Examples:
        - Parse a Responses API output (requires SDK objects)
            ```python
            >>> # response = client.responses.create(...)  # doctest: +SKIP
            >>> # chat_response = ResponsesOutputParser.parse(response.output)

            ```

    See Also:
        :class:`ChatMessageParser`: Parser for Chat Completions API responses.
        :class:`ResponsesStreamAccumulator`: Streaming counterpart for event-based parsing.
    """

    _BUILT_IN_TOOL_TYPES = (
        ResponseCodeInterpreterToolCall,
        ResponseComputerToolCall,
        ResponseFileSearchToolCall,
        ResponseFunctionWebSearch,
        McpCall,
    )

    def __init__(self, output: list[ResponseOutputItem]) -> None:
        self._output = output
        self._message = Message(role=MessageRole.ASSISTANT)
        self._additional_kwargs: dict[str, Any] = {"built_in_tool_calls": []}

    def build(self) -> ChatResponse:
        """Parse the output items and return a ChatResponse.

        Returns:
            A :class:`~serapeum.core.llms.ChatResponse` with the parsed message, content blocks,
            and additional kwargs (``built_in_tool_calls``, ``annotations``, ``refusal``).
        """
        for item in self._output:
            if isinstance(item, ResponseOutputMessage):
                self._parse_message(item)
            elif isinstance(item, ImageGenerationCall):
                self._parse_image_generation(item)
            elif isinstance(item, self._BUILT_IN_TOOL_TYPES):
                self._additional_kwargs["built_in_tool_calls"].append(item)
            elif isinstance(item, ResponseFunctionToolCall):
                self._parse_function_tool_call(item)
            elif isinstance(item, ResponseReasoningItem):
                self._parse_reasoning(item)
        return ChatResponse(message=self._message, additional_kwargs=self._additional_kwargs)

    def _parse_message(self, item: ResponseOutputMessage) -> None:
        blocks: list[ContentBlock] = []
        for part in item.content:
            if hasattr(part, "text"):
                blocks.append(TextChunk(content=part.text))
            if hasattr(part, "annotations"):
                self._additional_kwargs["annotations"] = part.annotations
            if hasattr(part, "refusal"):
                self._additional_kwargs["refusal"] = part.refusal
        self._message.chunks.extend(blocks)

    def _parse_image_generation(self, item: ImageGenerationCall) -> None:
        if item.status != "failed":
            self._additional_kwargs["built_in_tool_calls"].append(item)
            if item.result is not None:
                image_bytes = base64.b64decode(item.result)
                self._message.chunks.append(Image(content=image_bytes))

    def _parse_function_tool_call(self, item: ResponseFunctionToolCall) -> None:
        self._message.chunks.append(
            ToolCallBlock(
                tool_name=item.name,
                tool_call_id=item.call_id,
                tool_kwargs=item.arguments,
            )
        )

    def _parse_reasoning(self, item: ResponseReasoningItem) -> None:
        self._message.chunks.append(
            ThinkingBlock(
                content=_build_reasoning_content(item),
                additional_information=item.model_dump(
                    exclude={"content", "summary"}
                ),
            )
        )


class ResponsesStreamAccumulator:
    """Accumulate streaming Responses API events into content blocks.

    Encapsulates the mutable state needed during Responses API streaming:
    built-in tool calls, the current in-progress function tool call,
    additional kwargs (annotations, usage), and the previous response ID.

    This is the Responses API counterpart of :class:`ToolCallAccumulator`,
    which handles Chat Completions streaming deltas.

    Args:
        track_previous_responses: Whether to track the response ID for
            stateful conversation continuation.
        previous_response_id: Initial previous response ID, if any.

    Examples:
        - Accumulate events from a streaming response
            ```python
            >>> accumulator = ResponsesStreamAccumulator()
            >>> # for event in stream:  # doctest: +SKIP
            >>> #     blocks, delta = accumulator.update(event)

            ```

    See Also:
        :class:`ToolCallAccumulator`: Streaming accumulator for Chat Completions API.
        :class:`ResponsesOutputParser`: Non-streaming parser for complete responses.
    """

    def __init__(
        self,
        track_previous_responses: bool = False,
        previous_response_id: str | None = None,
    ) -> None:
        self._built_in_tool_calls: list[Any] = []
        self._additional_kwargs: dict[str, Any] = {"built_in_tool_calls": []}
        self._current_tool_call: ResponseFunctionToolCall | None = None
        self._previous_response_id: str | None = previous_response_id
        self._track_previous_responses: bool = track_previous_responses

    @property
    def built_in_tool_calls(self) -> list[Any]:
        """The accumulated built-in tool call events."""
        return self._built_in_tool_calls

    @property
    def additional_kwargs(self) -> dict[str, Any]:
        """Additional kwargs accumulated from events (annotations, usage, built-in tool calls).

        Returns a fresh dict each call, merging ``built_in_tool_calls`` when non-empty.
        """
        result = dict(self._additional_kwargs)
        if self._built_in_tool_calls:
            result["built_in_tool_calls"] = self._built_in_tool_calls
        return result

    @property
    def previous_response_id(self) -> str | None:
        """The most recently observed response ID (for stateful tracking)."""
        return self._previous_response_id

    def update(self, event: ResponseStreamEvent) -> tuple[list[ContentBlock], str]:
        """Process a single streaming event and update internal state.

        Args:
            event: A ``ResponseStreamEvent`` from the Responses API stream.

        Returns:
            A tuple of ``(blocks, delta)`` where ``blocks`` is a list of content
            blocks produced by this event and ``delta`` is the text delta string
            (empty if not a text event).
        """
        delta = ""
        blocks: list[ContentBlock] = []

        if isinstance(event, (ResponseCreatedEvent, ResponseInProgressEvent)):
            if self._track_previous_responses:
                self._previous_response_id = event.response.id
        elif isinstance(event, ResponseOutputItemAddedEvent):
            if isinstance(event.item, ResponseFunctionToolCall):
                self._current_tool_call = event.item
        elif isinstance(event, ResponseTextDeltaEvent):
            delta = event.delta
            blocks.append(TextChunk(content=delta))
        elif isinstance(event, ResponseImageGenCallPartialImageEvent):
            if event.partial_image_b64:
                blocks.append(
                    Image(
                        content=base64.b64decode(event.partial_image_b64),
                        detail=f"id_{event.partial_image_index}",
                    )
                )
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            if self._current_tool_call is not None:
                self._current_tool_call.arguments += event.delta
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            if self._current_tool_call is not None:
                self._current_tool_call.arguments = event.arguments
                self._current_tool_call.status = "completed"
                blocks.append(
                    ToolCallBlock(
                        tool_name=self._current_tool_call.name,
                        tool_kwargs=self._current_tool_call.arguments,
                        tool_call_id=self._current_tool_call.call_id,
                    )
                )
                self._current_tool_call = None
        elif isinstance(event, ResponseOutputTextAnnotationAddedEvent):
            annotations = self._additional_kwargs.get("annotations", [])
            annotations.append(event.annotation)
            self._additional_kwargs["annotations"] = annotations
        elif isinstance(event, ResponseFileSearchCallCompletedEvent):
            self._built_in_tool_calls.append(event)
        elif isinstance(event, ResponseWebSearchCallCompletedEvent):
            self._built_in_tool_calls.append(event)
        elif isinstance(event, ResponseOutputItemDoneEvent):
            if isinstance(event.item, ResponseReasoningItem):
                blocks.append(
                    ThinkingBlock(
                        content=_build_reasoning_content(event.item),
                        additional_information=event.item.model_dump(
                            exclude={"content", "summary"}
                        ),
                    )
                )
        elif isinstance(event, ResponseCompletedEvent):
            if hasattr(event, "response") and hasattr(event.response, "usage"):
                self._additional_kwargs["usage"] = event.response.usage
            resp = ResponsesOutputParser(event.response.output).build()
            blocks = resp.message.chunks

        return (blocks, delta)
