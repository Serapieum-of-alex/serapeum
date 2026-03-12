"""Parsers that convert OpenAI Responses API output into serapeum types.

This module bridges the gap between the OpenAI Responses API wire format
and the provider-agnostic :mod:`serapeum.core.llms` data model.  It provides
two complementary strategies:

- :class:`ResponsesOutputParser` -- parse a **complete** ``Response.output``
  list into a single :class:`~serapeum.core.llms.ChatResponse`.
- :class:`ResponsesStreamAccumulator` -- incrementally accumulate
  ``ResponseStreamEvent`` instances emitted during streaming and produce
  content blocks on each event.

Helper:

- :func:`_build_reasoning_content` -- extract concatenated text from a
  ``ResponseReasoningItem``'s ``content`` and ``summary`` fields.
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
    """Extract and concatenate text from a reasoning item's ``content`` and ``summary`` fields.

    When both ``content`` and ``summary`` are present, their texts are joined
    with a newline separator (content first, then summary).  If only one is
    present, that field's text is returned alone.

    Args:
        item: A :class:`~openai.types.responses.ResponseReasoningItem` from
            the Responses API.

    Returns:
        The concatenated reasoning text, or ``None`` when both ``content``
        and ``summary`` are empty or absent.
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
    """Parse a complete OpenAI Responses API output into a :class:`~serapeum.core.llms.ChatResponse`.

    Iterates over the ``ResponseOutputItem`` list returned in
    ``Response.output`` and converts each subtype into the corresponding
    serapeum content block:

    +-------------------------------------+----------------------------------------------------+
    | Responses API type                  | Serapeum type                                      |
    +=====================================+====================================================+
    | ``ResponseOutputMessage``           | :class:`~serapeum.core.llms.TextChunk`             |
    +-------------------------------------+----------------------------------------------------+
    | ``ResponseFunctionToolCall``        | :class:`~serapeum.core.llms.ToolCallBlock`         |
    +-------------------------------------+----------------------------------------------------+
    | ``ResponseReasoningItem``           | :class:`~serapeum.core.llms.ThinkingBlock`         |
    +-------------------------------------+----------------------------------------------------+
    | ``ImageGenerationCall``             | :class:`~serapeum.core.llms.Image`                 |
    +-------------------------------------+----------------------------------------------------+
    | Built-in tool calls (file search,   | Stored in                                          |
    | web search, code interpreter,       | ``additional_kwargs["built_in_tool_calls"]``        |
    | computer, MCP)                      |                                                    |
    +-------------------------------------+----------------------------------------------------+

    Annotations and refusal strings are stored in ``additional_kwargs``
    under the keys ``"annotations"`` and ``"refusal"`` respectively.

    Args:
        output: List of ``ResponseOutputItem`` instances from the Responses API
            response (i.e. ``response.output``).

    Examples:
        - Build a ChatResponse from a Responses API response:
            ```python
            # response = client.responses.create(
            #     model="gpt-4o",
            #     input="Hello!"
            # )
            # parser = ResponsesOutputParser(response.output)
            # chat_response = parser.build()
            # chat_response.message.content

            ```

        - Access tool calls from the parsed response:
            ```python
            # parser = ResponsesOutputParser(response.output)
            # chat_response = parser.build()
            # tool_blocks = [
            #     chunk for chunk in chat_response.message.chunks
            #     if hasattr(chunk, 'tool_name')
            # ]
            # tool_blocks[0].tool_name

            ```

    See Also:
        :class:`ResponsesStreamAccumulator`: Streaming counterpart that
            processes events incrementally.
    """

    _BUILT_IN_TOOL_TYPES = (
        ResponseCodeInterpreterToolCall,
        ResponseComputerToolCall,
        ResponseFileSearchToolCall,
        ResponseFunctionWebSearch,
        McpCall,
    )

    def __init__(self, output: list[ResponseOutputItem]) -> None:
        """Initialise the parser with a list of Responses API output items.

        Args:
            output: The ``response.output`` list from an OpenAI Responses API
                response object.
        """
        self._output = output
        self._message = Message(role=MessageRole.ASSISTANT)
        self._additional_kwargs: dict[str, Any] = {"built_in_tool_calls": []}

    def build(self) -> ChatResponse:
        """Iterate over all output items and assemble a :class:`~serapeum.core.llms.ChatResponse`.

        Each ``ResponseOutputItem`` in the stored output list is dispatched to
        the appropriate private handler (``_parse_message``,
        ``_parse_function_tool_call``, etc.).  The resulting content blocks are
        collected into a single :class:`~serapeum.core.llms.Message` with role
        ``ASSISTANT``.

        Returns:
            A :class:`~serapeum.core.llms.ChatResponse` whose ``message``
            carries the parsed content blocks and whose
            ``additional_kwargs`` dict may contain:

            - ``"built_in_tool_calls"`` -- list of built-in tool call objects
              (file search, web search, code interpreter, etc.).
            - ``"annotations"`` -- response text annotations, if present.
            - ``"refusal"`` -- refusal string, if the model declined.

        Examples:
            - Parse output and inspect the message content:
                ```python
                # parser = ResponsesOutputParser(response.output)
                # result = parser.build()
                # result.message.role
                # result.message.content

                ```
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
        return ChatResponse(
            message=self._message, additional_kwargs=self._additional_kwargs
        )

    def _parse_message(self, item: ResponseOutputMessage) -> None:
        """Extract text chunks, annotations, and refusal from a message output item."""
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
        """Decode a base64 image result and append an Image block (skips failed calls)."""
        if item.status != "failed":
            self._additional_kwargs["built_in_tool_calls"].append(item)
            if item.result is not None:
                image_bytes = base64.b64decode(item.result)
                self._message.chunks.append(Image(content=image_bytes))

    def _parse_function_tool_call(self, item: ResponseFunctionToolCall) -> None:
        """Convert a function tool call into a ToolCallBlock and append it to the message."""
        self._message.chunks.append(
            ToolCallBlock(
                tool_name=item.name,
                tool_call_id=item.call_id,
                tool_kwargs=item.arguments,
            )
        )

    def _parse_reasoning(self, item: ResponseReasoningItem) -> None:
        """Convert a reasoning item into a ThinkingBlock and append it to the message."""
        self._message.chunks.append(
            ThinkingBlock(
                content=_build_reasoning_content(item),
                additional_information=item.model_dump(exclude={"content", "summary"}),
            )
        )


class ResponsesStreamAccumulator:
    """Accumulate streaming Responses API events into serapeum content blocks.

    Encapsulates the mutable state needed while iterating over a Responses API
    server-sent-event stream.  On each call to :meth:`update`, the
    accumulator processes a single :class:`~openai.types.responses.ResponseStreamEvent`
    and returns the content blocks and text delta it produced.

    Tracked state includes:

    - **Built-in tool calls** (file search, web search, etc.) -- collected in
      :attr:`built_in_tool_calls`.
    - **Current in-progress function tool call** -- argument deltas are
      appended until the call completes, at which point a
      :class:`~serapeum.core.llms.ToolCallBlock` is emitted.
    - **Additional kwargs** (annotations, usage) -- accessible via
      :attr:`additional_kwargs`.
    - **Previous response ID** -- optionally tracked for stateful
      conversation continuation.

    Args:
        track_previous_responses: If ``True``, store the response ID from
            ``ResponseCreatedEvent`` / ``ResponseInProgressEvent`` so that
            subsequent requests can reference it.
        previous_response_id: An initial response ID to seed the tracker
            (useful when resuming an existing conversation).

    Examples:
        - Process a streaming response event by event:
            ```python
            from serapeum.openai.parsers.response_parsers import (
                ResponsesStreamAccumulator,
            )
            acc = ResponsesStreamAccumulator(track_previous_responses=True)
            # stream = client.responses.create(..., stream=True)
            # for event in stream:
            #     blocks, delta = acc.update(event)
            #     if delta:
            #         print(delta, end="")

            ```

        - Retrieve accumulated metadata after the stream ends:
            ```python
            from serapeum.openai.parsers.response_parsers import (
                ResponsesStreamAccumulator,
            )
            acc = ResponsesStreamAccumulator()
            # ... process events ...
            # acc.additional_kwargs
            # acc.built_in_tool_calls
            # acc.previous_response_id

            ```

    See Also:
        :class:`ResponsesOutputParser`: Non-streaming parser for complete
            ``Response.output`` lists.
    """

    def __init__(
        self,
        track_previous_responses: bool = False,
        previous_response_id: str | None = None,
    ) -> None:
        """Initialise the accumulator with optional response-ID tracking.

        Args:
            track_previous_responses: If ``True``, capture the response ID
                from created/in-progress events so callers can chain
                follow-up requests.
            previous_response_id: Seed value for :attr:`previous_response_id`.
        """
        self._built_in_tool_calls: list[Any] = []
        self._additional_kwargs: dict[str, Any] = {"built_in_tool_calls": []}
        self._current_tool_call: ResponseFunctionToolCall | None = None
        self._previous_response_id: str | None = previous_response_id
        self._track_previous_responses: bool = track_previous_responses

    @property
    def built_in_tool_calls(self) -> list[Any]:
        """List of accumulated built-in tool-call events.

        Built-in tool calls include file-search completions, web-search
        completions, and other non-function tool invocations that are
        collected as events arrive.

        Returns:
            A mutable list of tool-call event objects gathered so far.

        Examples:
            - Inspect built-in tool calls after streaming:
                ```python
                >>> from serapeum.openai.parsers.response_parsers import (
                ...     ResponsesStreamAccumulator,
                ... )
                >>> acc = ResponsesStreamAccumulator()
                >>> acc.built_in_tool_calls
                []

                ```
        """
        return self._built_in_tool_calls

    @property
    def additional_kwargs(self) -> dict[str, Any]:
        """Metadata accumulated from stream events.

        Returns a **fresh** dict on each access.  When
        :attr:`built_in_tool_calls` is non-empty, it is merged into the
        returned dict under the ``"built_in_tool_calls"`` key.

        Possible keys include:

        - ``"built_in_tool_calls"`` -- list of built-in tool call objects.
        - ``"annotations"`` -- text annotations added during streaming.
        - ``"usage"`` -- token usage reported by the ``ResponseCompletedEvent``.

        Returns:
            A dict of additional metadata suitable for passing into
            :class:`~serapeum.core.llms.ChatResponse`.

        Examples:
            - Access additional kwargs from a fresh accumulator:
                ```python
                >>> from serapeum.openai.parsers.response_parsers import (
                ...     ResponsesStreamAccumulator,
                ... )
                >>> acc = ResponsesStreamAccumulator()
                >>> "built_in_tool_calls" in acc.additional_kwargs
                True

                ```
        """
        result = dict(self._additional_kwargs)
        if self._built_in_tool_calls:
            result["built_in_tool_calls"] = self._built_in_tool_calls
        return result

    @property
    def previous_response_id(self) -> str | None:
        """The most recently observed response ID, or ``None``.

        Only populated when the accumulator was created with
        ``track_previous_responses=True`` and a
        ``ResponseCreatedEvent`` or ``ResponseInProgressEvent`` has been
        processed.

        Returns:
            The response ID string, or ``None`` if tracking is disabled or
            no response event has been seen yet.

        Examples:
            - Check the response ID before any events:
                ```python
                >>> from serapeum.openai.parsers.response_parsers import (
                ...     ResponsesStreamAccumulator,
                ... )
                >>> acc = ResponsesStreamAccumulator()
                >>> acc.previous_response_id is None
                True

                ```

            - Seed with an initial response ID:
                ```python
                >>> from serapeum.openai.parsers.response_parsers import (
                ...     ResponsesStreamAccumulator,
                ... )
                >>> acc = ResponsesStreamAccumulator(
                ...     previous_response_id="resp_abc123"
                ... )
                >>> acc.previous_response_id
                'resp_abc123'

                ```
        """
        return self._previous_response_id

    def update(self, event: ResponseStreamEvent) -> tuple[list[ContentBlock], str]:
        """Process a single streaming event and return produced content.

        Dispatches the event by type and updates internal state accordingly.
        The following event types produce visible output:

        - ``ResponseTextDeltaEvent`` -- yields a
          :class:`~serapeum.core.llms.TextChunk` and a non-empty *delta*.
        - ``ResponseFunctionCallArgumentsDoneEvent`` -- yields a
          :class:`~serapeum.core.llms.ToolCallBlock` for the completed call.
        - ``ResponseImageGenCallPartialImageEvent`` -- yields an
          :class:`~serapeum.core.llms.Image` with the partial base64 data.
        - ``ResponseOutputItemDoneEvent`` (reasoning) -- yields a
          :class:`~serapeum.core.llms.ThinkingBlock`.
        - ``ResponseCompletedEvent`` -- yields the full set of blocks from the
          completed response (parsed via :class:`ResponsesOutputParser`).

        All other event types update internal bookkeeping (tool-call argument
        accumulation, annotation collection, response-ID tracking) without
        producing blocks.

        Args:
            event: A single
                :class:`~openai.types.responses.ResponseStreamEvent` from the
                server-sent-event stream.

        Returns:
            A two-element tuple ``(blocks, delta)`` where:

            - *blocks* is a (possibly empty) list of
              :class:`~serapeum.core.llms.ContentBlock` instances produced by
              this event.
            - *delta* is the raw text delta string (non-empty only for
              ``ResponseTextDeltaEvent``; empty string otherwise).

        Examples:
            - Process a text delta event:
                ```python
                # acc = ResponsesStreamAccumulator()
                # blocks, delta = acc.update(text_delta_event)
                # delta  # the raw text fragment
                # blocks[0].content  # same text as a TextChunk

                ```
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
