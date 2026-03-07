"""Bidirectional converters between serapeum Message types and OpenAI API formats.

This module provides two conversion directions:

**To OpenAI** (serapeum ``Message`` → OpenAI request format):

- :class:`ChatMessageConverter` — converts a ``Message`` to a Chat Completions API dict
- :class:`ResponsesMessageConverter` — converts a ``Message`` to a Responses API dict or list
- :func:`to_openai_message_dicts` — top-level dispatcher that selects the correct converter

**From OpenAI** (OpenAI response format → serapeum ``Message``):

- :class:`ChatMessageParser` — parses a typed ``ChatCompletionMessage`` into a ``Message``
- :class:`DictMessageParser` — parses a raw dict (e.g. Responses API round-trips) into a ``Message``
- :class:`LogProbParser` — converts OpenAI logprob types to :class:`~serapeum.core.llms.LogProb` lists
- :class:`ToolCallAccumulator` — accumulates streaming ``ChoiceDeltaToolCall`` chunks
- :class:`ResponsesOutputParser` — parses Responses API output items into a ``ChatResponse``
- :class:`ResponsesStreamAccumulator` — accumulates Responses API streaming events

**Block format namespaces**:

- :class:`ChatFormat` — static converters from serapeum blocks to Chat Completions API dicts
- :class:`ResponsesFormat` — static converters from serapeum blocks to Responses API dicts

**Utility**:

- :func:`to_openai_tool` — converts a Pydantic model class to an OpenAI function tool spec
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from typing import Any, Sequence, Type, cast

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_choice import Logprobs
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
from pydantic import BaseModel

from serapeum.core.llms import (
    Audio,
    ChatResponse,
    ContentBlock,
    DocumentBlock,
    Image,
    LogProb,
    Message,
    MessageRole,
    TextChunk,
    ThinkingBlock,
    ToolCallBlock,
)
from serapeum.openai.data.models import O1_MODELS, O1_MODELS_WITHOUT_FUNCTION_CALLING

logger = logging.getLogger(__name__)

OpenAIToolCall = ChatCompletionMessageToolCall | ChoiceDeltaToolCall

# ---------------------------------------------------------------------------
# Shared block-level helpers
# ---------------------------------------------------------------------------


def _rewrite_system_to_developer(
    message_dict: dict[str, Any], model: str | None
) -> None:
    """Rewrite the ``system`` role to ``developer`` for O1-family models that support function calling.

    OpenAI's o3-mini and similar O1-family models that support function calling
    expect the ``developer`` role instead of ``system``. This function mutates
    *message_dict* in place when the conditions are met.

    The rewrite is **not** applied for:

    - Models outside :data:`~serapeum.openai.models.O1_MODELS`
    - Models in :data:`~serapeum.openai.models.O1_MODELS_WITHOUT_FUNCTION_CALLING`
      (e.g. ``o1-mini``, ``o1-preview``) which do not support system prompts at all
    - When *model* is ``None``
    - When the role is not ``"system"``

    Args:
        message_dict: The message dict to mutate in place. Must contain a ``"role"`` key.
        model: The OpenAI model name. Pass ``None`` to skip rewriting entirely.

    Examples:
        - Rewrites for o3-mini
            ```python
            >>> d = {"role": "system", "content": "You are helpful."}
            >>> _rewrite_system_to_developer(d, "o3-mini")
            >>> d["role"]
            'developer'

            ```
        - No-op for a standard model
            ```python
            >>> d = {"role": "system", "content": "You are helpful."}
            >>> _rewrite_system_to_developer(d, "gpt-4o")
            >>> d["role"]
            'system'

            ```
        - No-op when model is None
            ```python
            >>> d = {"role": "system", "content": "You are helpful."}
            >>> _rewrite_system_to_developer(d, None)
            >>> d["role"]
            'system'

            ```

    See Also:
        :data:`~serapeum.openai.models.O1_MODELS`: Set of O1-family model identifiers.
        :data:`~serapeum.openai.models.O1_MODELS_WITHOUT_FUNCTION_CALLING`: Models excluded from rewriting.
    """
    if (
        model is not None
        and model in O1_MODELS
        and model not in O1_MODELS_WITHOUT_FUNCTION_CALLING
        and message_dict.get("role") == "system"
    ):
        message_dict["role"] = "developer"


def _strip_none_keys(message_dict: dict[str, Any], drop_none: bool) -> None:
    """Remove all keys whose value is ``None`` from *message_dict* when *drop_none* is set.

    This is a no-op when *drop_none* is ``False``. The dict is mutated in place.

    Args:
        message_dict: The dict to strip None-valued keys from.
        drop_none: When ``True``, removes every key whose value is ``None``.
            When ``False``, the dict is left unchanged.

    Examples:
        - Removes None values when drop_none is True
            ```python
            >>> d = {"role": "assistant", "content": None, "tool_calls": None}
            >>> _strip_none_keys(d, drop_none=True)
            >>> sorted(d.keys())
            ['role']

            ```
        - Leaves None values intact when drop_none is False
            ```python
            >>> d = {"role": "assistant", "content": None}
            >>> _strip_none_keys(d, drop_none=False)
            >>> d["content"] is None
            True

            ```
    """
    if drop_none:
        for key in [k for k, v in message_dict.items() if v is None]:
            message_dict.pop(key)


def _should_null_content(message: Message, has_tool_calls: bool) -> bool:
    """Return ``True`` when an assistant message's content should be sent as ``None``.

    The OpenAI Chat Completions API requires ``content: null`` for assistant
    messages that consist exclusively of tool calls or function calls.  A message
    qualifies when **all** of the following hold:

    1. The role is ``assistant``
    2. At least one of the following is true:

       - ``"function_call"`` is present in ``message.additional_kwargs``
       - ``"tool_calls"`` is present in ``message.additional_kwargs``
       - *has_tool_calls* is ``True`` (the message has :class:`~serapeum.core.llms.ToolCallBlock` chunks)

    Args:
        message: The serapeum message to inspect.
        has_tool_calls: Whether the caller has already detected ``ToolCallBlock``
            chunks in the message. Avoids re-iterating chunks.

    Returns:
        ``True`` if content should be nulled; ``False`` otherwise.

    Examples:
        - Returns True for assistant with tool_calls in additional_kwargs
            ```python
            >>> from serapeum.core.llms import Message, MessageRole
            >>> msg = Message(
            ...     role=MessageRole.ASSISTANT,
            ...     content="",
            ...     additional_kwargs={"tool_calls": [{"id": "c1"}]},
            ... )
            >>> _should_null_content(msg, has_tool_calls=False)
            True

            ```
        - Returns False for user messages regardless of tool_calls
            ```python
            >>> msg = Message(role=MessageRole.USER, content="hi")
            >>> _should_null_content(msg, has_tool_calls=True)
            False

            ```
        - Returns False for assistant with no tool information
            ```python
            >>> msg = Message(role=MessageRole.ASSISTANT, content="hello")
            >>> _should_null_content(msg, has_tool_calls=False)
            False

            ```

    See Also:
        :meth:`ChatMessageConverter._resolve_content`: Uses this to decide whether to emit ``None``.
        :meth:`ResponsesMessageConverter._resolve_content`: Same check for the Responses API path.
    """
    return (
        message.role == MessageRole.ASSISTANT
        and (
            "function_call" in message.additional_kwargs
            or "tool_calls" in message.additional_kwargs
            or has_tool_calls
        )
    )


class ChatFormat:
    """Namespace of static converters from serapeum content blocks to Chat Completions API dicts.

    All methods accept a single content block and return a dict conforming to
    the ``content`` array item schema used by the OpenAI Chat Completions API.

    The :attr:`content_converters` class attribute maps block *types* to their
    converter callables, enabling generic dispatch for blocks that are not
    explicitly handled by :class:`ChatMessageConverter`.

    Examples:
        - Convert a text chunk
            ```python
            >>> from serapeum.core.llms import TextChunk
            >>> ChatFormat.text(TextChunk(content="hello"))
            {'type': 'text', 'text': 'hello'}

            ```
        - Convert an image with a URL
            ```python
            >>> from serapeum.core.llms import Image
            >>> result = ChatFormat.image(Image(url="https://example.com/img.png"))
            >>> result["type"]
            'image_url'

            ```

    See Also:
        :class:`ResponsesFormat`: Equivalent namespace for the Responses API.
        :class:`ChatMessageConverter`: Uses these converters when building message dicts.
    """

    @staticmethod
    def text(block: TextChunk) -> dict[str, Any]:
        """Convert a :class:`~serapeum.core.llms.TextChunk` to a Chat Completions text dict.

        Args:
            block: The text chunk to convert.

        Returns:
            A dict with ``{"type": "text", "text": <content>}``.

        Examples:
            - Basic conversion
                ```python
                >>> from serapeum.core.llms import TextChunk
                >>> ChatFormat.text(TextChunk(content="hello world"))
                {'type': 'text', 'text': 'hello world'}

                ```
        """
        return {"type": "text", "text": block.content}

    @staticmethod
    def document(block: DocumentBlock) -> dict[str, Any]:
        """Convert a :class:`~serapeum.core.llms.DocumentBlock` to a Chat Completions file dict.

        The document's binary content is base64-encoded and embedded as a data URI.

        Args:
            block: The document block to convert. Must implement ``as_base64()``
                returning ``(b64_string, mimetype)``.

        Returns:
            A dict with keys ``type``, ``filename``, and ``file_data`` (data URI).

        Examples:
            - Convert a PDF document (skipped — requires DocumentBlock I/O)
                ```python
                >>> # result = ChatFormat.document(doc_block)  # doctest: +SKIP
                >>> # result["type"]
                >>> # 'file'

                ```
        """
        b64_string, mimetype = block.as_base64()
        return {
            "type": "file",
            "filename": block.title,
            "file_data": f"data:{mimetype};base64,{b64_string}",
        }

    @staticmethod
    def image(block: Image) -> dict[str, Any]:
        """Convert an :class:`~serapeum.core.llms.Image` block to a Chat Completions image_url dict.

        When the block has a URL, it is used directly. When the block has inline
        content (no URL), :meth:`~serapeum.core.llms.Image.as_data_uri` is called to
        produce a ``data:`` URI. The optional ``detail`` field is only included when
        it is set on the block.

        Args:
            block: The image block to convert.

        Returns:
            A dict with ``{"type": "image_url", "image_url": {...}}``.
            The inner dict contains ``url`` and optionally ``detail``.

        Examples:
            - Image from URL (no detail)
                ```python
                >>> from serapeum.core.llms import Image
                >>> result = ChatFormat.image(Image(url="https://example.com/img.png"))
                >>> result["image_url"]["url"]
                'https://example.com/img.png'
                >>> "detail" in result["image_url"]
                False

                ```
            - Image from URL with detail
                ```python
                >>> result = ChatFormat.image(
                ...     Image(url="https://example.com/img.png", detail="low")
                ... )
                >>> result["image_url"]["detail"]
                'low'

                ```
        """
        if block.url:
            image_url: dict[str, Any] = {"url": str(block.url)}
        else:
            image_url = {"url": block.as_data_uri()}
        if block.detail:
            image_url["detail"] = block.detail
        return {"type": "image_url", "image_url": image_url}

    @staticmethod
    def audio(block: Audio) -> dict[str, Any]:
        """Convert an :class:`~serapeum.core.llms.Audio` block to a Chat Completions input_audio dict.

        The audio is resolved via :meth:`~serapeum.core.llms.Audio.resolve_audio` with
        ``as_base64=True`` and read into a UTF-8 string.

        Args:
            block: The audio block to convert. Must implement ``resolve_audio()``.

        Returns:
            A dict with ``{"type": "input_audio", "input_audio": {"data": ..., "format": ...}}``.

        Examples:
            - Convert an audio block (skipped — requires resolve_audio I/O)
                ```python
                >>> # result = ChatFormat.audio(audio_block)  # doctest: +SKIP
                >>> # result["type"]
                >>> # 'input_audio'

                ```
        """
        audio_bytes = block.resolve_audio(as_base64=True).read()
        audio_str = audio_bytes.decode("utf-8")
        return {
            "type": "input_audio",
            "input_audio": {"data": audio_str, "format": block.format},
        }

    @staticmethod
    def tool_call(block: ToolCallBlock) -> dict[str, Any]:
        """Convert a :class:`~serapeum.core.llms.ToolCallBlock` to a Chat Completions function call dict.

        Args:
            block: The tool call block to convert.

        Returns:
            A dict with ``{"type": "function", "function": {...}, "id": ...}``.

        Examples:
            - Convert a tool call block
                ```python
                >>> from serapeum.core.llms import ToolCallBlock
                >>> result = ChatFormat.tool_call(
                ...     ToolCallBlock(tool_call_id="c1", tool_name="search", tool_kwargs={"q": "x"})
                ... )
                >>> result["type"]
                'function'
                >>> result["id"]
                'c1'
                >>> result["function"]["name"]
                'search'

                ```
        """
        return {
            "type": "function",
            "function": {
                "name": block.tool_name,
                "arguments": block.tool_kwargs,
            },
            "id": block.tool_call_id,
        }

    # noinspection PyUnresolvedReferences
    content_converters: dict[type, Callable[..., dict[str, Any]]] = {
        TextChunk: text.__func__,
        DocumentBlock: document.__func__,
        Image: image.__func__,
        Audio: audio.__func__,
    }
    """Dispatch table mapping block types to their converter callables.

    Used by :class:`ChatMessageConverter` to convert block types that are
    not explicitly handled via ``isinstance`` checks. Access the unbound
    function via the ``.__func__`` trick because the methods are staticmethods
    at class-definition time.
    """


class ResponsesFormat:
    """Namespace of static converters from serapeum content blocks to Responses API dicts.

    The Responses API has a different schema than the Chat Completions API:
    text blocks carry ``input_text`` / ``output_text`` type tags (role-dependent),
    images use a flat ``image_url`` string rather than a nested dict, and thinking
    blocks map to ``reasoning`` items.

    The :attr:`content_converters` dispatch table covers ``DocumentBlock`` and
    ``Image`` only; ``TextChunk`` and ``ThinkingBlock`` and ``ToolCallBlock`` are
    handled explicitly in :class:`ResponsesMessageConverter`.

    Examples:
        - Text block for user role produces input_text
            ```python
            >>> from serapeum.core.llms import TextChunk
            >>> ResponsesFormat.text(TextChunk(content="hello"), "user")
            {'type': 'input_text', 'text': 'hello'}

            ```
        - Text block for assistant role produces output_text
            ```python
            >>> ResponsesFormat.text(TextChunk(content="hello"), "assistant")
            {'type': 'output_text', 'text': 'hello'}

            ```

    See Also:
        :class:`ChatFormat`: Equivalent namespace for the Chat Completions API.
        :class:`ResponsesMessageConverter`: Uses these converters when building Responses dicts.
    """

    @staticmethod
    def text(block: TextChunk, role: str) -> dict[str, Any]:
        """Convert a :class:`~serapeum.core.llms.TextChunk` to a Responses API text dict.

        The type tag is ``"input_text"`` for user messages and ``"output_text"``
        for all other roles (assistant, system, etc.).

        Args:
            block: The text chunk to convert.
            role: The message role string (e.g. ``"user"``, ``"assistant"``).

        Returns:
            A dict with ``{"type": "input_text" | "output_text", "text": <content>}``.

        Examples:
            - User role → input_text
                ```python
                >>> from serapeum.core.llms import TextChunk
                >>> ResponsesFormat.text(TextChunk(content="Hi"), "user")
                {'type': 'input_text', 'text': 'Hi'}

                ```
            - Assistant role → output_text
                ```python
                >>> ResponsesFormat.text(TextChunk(content="Hi"), "assistant")
                {'type': 'output_text', 'text': 'Hi'}

                ```
        """
        text_type = "input_text" if role == "user" else "output_text"
        return {"type": text_type, "text": block.content}

    @staticmethod
    def document(block: DocumentBlock) -> dict[str, Any]:
        """Convert a :class:`~serapeum.core.llms.DocumentBlock` to a Responses API file dict.

        Args:
            block: The document block to convert.

        Returns:
            A dict with keys ``type`` (``"input_file"``), ``filename``, and ``file_data``.

        Examples:
            - Convert a document block (skipped — requires I/O)
                ```python
                >>> # result = ResponsesFormat.document(doc_block)  # doctest: +SKIP
                >>> # result["type"]
                >>> # 'input_file'

                ```
        """
        b64_string, mimetype = block.as_base64()
        return {
            "type": "input_file",
            "filename": block.title,
            "file_data": f"data:{mimetype};base64,{b64_string}",
        }

    @staticmethod
    def image(block: Image) -> dict[str, Any]:
        """Convert an :class:`~serapeum.core.llms.Image` block to a Responses API image dict.

        Unlike the Chat Completions equivalent, the Responses API uses a flat
        ``image_url`` string rather than a nested object, and always includes
        ``detail`` (defaulting to ``"auto"``).

        Args:
            block: The image block to convert.

        Returns:
            A dict with ``{"type": "input_image", "image_url": <str>, "detail": <str>}``.

        Examples:
            - Image with URL, no explicit detail
                ```python
                >>> from serapeum.core.llms import Image
                >>> result = ResponsesFormat.image(Image(url="https://example.com/img.png"))
                >>> result["type"]
                'input_image'
                >>> result["image_url"]
                'https://example.com/img.png'
                >>> result["detail"]
                'auto'

                ```
            - Image with explicit detail
                ```python
                >>> result = ResponsesFormat.image(
                ...     Image(url="https://example.com/img.png", detail="high")
                ... )
                >>> result["detail"]
                'high'

                ```
        """
        if block.url:
            url_str = str(block.url)
        else:
            url_str = block.as_data_uri()
        return {
            "type": "input_image",
            "image_url": url_str,
            "detail": block.detail or "auto",
        }

    @staticmethod
    def thinking(block: ThinkingBlock) -> dict[str, Any] | None:
        """Convert a :class:`~serapeum.core.llms.ThinkingBlock` to a Responses API reasoning dict.

        Returns ``None`` (and the item is silently dropped) when either:

        - ``block.content`` is falsy (empty or ``None``)
        - ``block.additional_information`` does not contain an ``"id"`` key

        Both conditions must hold for a non-``None`` result to be produced.

        Args:
            block: The thinking block to convert.

        Returns:
            A reasoning dict with ``type``, ``id``, and ``summary`` fields,
            or ``None`` if the block lacks the required data.

        Examples:
            - Valid thinking block returns a reasoning dict
                ```python
                >>> from serapeum.core.llms import ThinkingBlock
                >>> result = ResponsesFormat.thinking(
                ...     ThinkingBlock(content="step 1", additional_information={"id": "r1"})
                ... )
                >>> result["type"]
                'reasoning'
                >>> result["id"]
                'r1'
                >>> result["summary"]
                [{'type': 'summary_text', 'text': 'step 1'}]

                ```
            - Returns None when content is missing
                ```python
                >>> ResponsesFormat.thinking(
                ...     ThinkingBlock(content=None, additional_information={"id": "r1"})
                ... ) is None
                True

                ```
            - Returns None when id is missing
                ```python
                >>> ResponsesFormat.thinking(
                ...     ThinkingBlock(content="step 1", additional_information={})
                ... ) is None
                True

                ```
        """
        if block.content and "id" in block.additional_information:
            return {
                "type": "reasoning",
                "id": block.additional_information["id"],
                "summary": [
                    {"type": "summary_text", "text": block.content or ""}
                ],
            }
        return None

    @staticmethod
    def tool_call(block: ToolCallBlock) -> dict[str, Any]:
        """Convert a :class:`~serapeum.core.llms.ToolCallBlock` to a Responses API function_call dict.

        The Responses API uses ``"function_call"`` type (not ``"function"`` as in Chat Completions)
        and uses ``call_id`` instead of ``id``.

        Args:
            block: The tool call block to convert.

        Returns:
            A dict with ``{"type": "function_call", "arguments": ..., "call_id": ..., "name": ...}``.

        Examples:
            - Convert a tool call
                ```python
                >>> from serapeum.core.llms import ToolCallBlock
                >>> result = ResponsesFormat.tool_call(
                ...     ToolCallBlock(tool_call_id="c1", tool_name="search", tool_kwargs={"q": "x"})
                ... )
                >>> result["type"]
                'function_call'
                >>> result["call_id"]
                'c1'

                ```
        """
        return {
            "type": "function_call",
            "arguments": block.tool_kwargs,
            "call_id": block.tool_call_id,
            "name": block.tool_name,
        }

    # noinspection PyUnresolvedReferences
    content_converters: dict[type, Callable[..., dict[str, Any]]] = {
        DocumentBlock: document.__func__,
        Image: image.__func__,
    }
    """Dispatch table for block types handled via generic dispatch in :class:`ResponsesMessageConverter`.

    Covers ``DocumentBlock`` and ``Image`` only. ``TextChunk``, ``ThinkingBlock``,
    ``ToolCallBlock``, and ``Audio`` are handled by explicit ``isinstance`` checks.
    """


# ---------------------------------------------------------------------------
# Public converters (to OpenAI)
# ---------------------------------------------------------------------------


class ChatMessageConverter:
    """Converts a serapeum :class:`~serapeum.core.llms.Message` into a Chat Completions API dict.

    Use the builder pattern: construct with the message and optional parameters,
    then call :meth:`build` to obtain the final dict.

    The conversion applies the following steps in order:

    1. **Audio reference short-circuit**: if the message is an assistant message
       with ``reference_audio_id`` in ``additional_kwargs``, returns a minimal
       ``{"role": "assistant", "audio": {"id": ...}}`` dict immediately.
    2. **Block processing**: iterates ``message.chunks`` and dispatches to
       :class:`ChatFormat` converters. :class:`~serapeum.core.llms.ThinkingBlock`
       items are silently skipped (not supported by the Chat Completions API).
    3. **Assembly**: builds ``{"role": ..., "content": ..., "tool_calls": ...}``.
    4. **Legacy kwargs merge**: when no ``ToolCallBlock`` chunks are present,
       copies ``tool_calls``/``function_call`` from ``additional_kwargs``.
    5. **Role rewrite**: calls :func:`_rewrite_system_to_developer` for O1 models.
    6. **None stripping**: calls :func:`_strip_none_keys` when ``drop_none=True``.

    Args:
        message: The serapeum message to convert.
        drop_none: When ``True``, keys with ``None`` values are removed from the result.
            Defaults to ``False``.
        model: The target OpenAI model name. Used to determine whether to rewrite
            ``"system"`` → ``"developer"``. Pass ``None`` to skip rewriting.

    Examples:
        - Simple user message
            ```python
            >>> from serapeum.core.llms import Message, MessageRole
            >>> result = ChatMessageConverter(
            ...     Message(role=MessageRole.USER, content="hello")
            ... ).build()
            >>> result
            {'role': 'user', 'content': 'hello'}

            ```
        - Assistant with tool calls
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, ToolCallBlock
            >>> msg = Message(
            ...     role=MessageRole.ASSISTANT,
            ...     chunks=[ToolCallBlock(tool_call_id="c1", tool_name="f", tool_kwargs={})],
            ... )
            >>> result = ChatMessageConverter(msg).build()
            >>> result["content"] is None
            True
            >>> result["tool_calls"][0]["id"]
            'c1'

            ```
        - O1 model rewrites system to developer
            ```python
            >>> from serapeum.core.llms import Message, MessageRole
            >>> result = ChatMessageConverter(
            ...     Message(role=MessageRole.SYSTEM, content="be helpful"),
            ...     model="o3-mini",
            ... ).build()
            >>> result["role"]
            'developer'

            ```

    See Also:
        :class:`ResponsesMessageConverter`: Equivalent converter for the Responses API.
        :func:`to_openai_message_dicts`: Top-level dispatcher that selects this converter.
        :class:`ChatFormat`: Block-level converter namespace used internally.
    """

    def __init__(
        self,
        message: Message,
        *,
        drop_none: bool = False,
        model: str | None = None,
    ) -> None:
        self._message = message
        self._model = model
        self._drop_none = drop_none
        self._content: list[dict[str, Any]] = []
        self._content_txt: str = ""
        self._tool_call_dicts: list[dict[str, Any]] = []

    def build(self) -> ChatCompletionMessageParam:
        """Convert the message and return the Chat Completions API dict.

        Returns:
            A :class:`~openai.types.chat.ChatCompletionMessageParam` dict
            suitable for use as an element of the ``messages`` list in an
            OpenAI chat completion request.

        Raises:
            ValueError: If a content block type is not supported by
                :class:`ChatFormat` and is not a :class:`~serapeum.core.llms.ThinkingBlock`
                or :class:`~serapeum.core.llms.ToolCallBlock`.

        Examples:
            - Build a system message
                ```python
                >>> from serapeum.core.llms import Message, MessageRole
                >>> ChatMessageConverter(
                ...     Message(role=MessageRole.SYSTEM, content="You are helpful.")
                ... ).build()
                {'role': 'system', 'content': 'You are helpful.'}

                ```
        """
        audio_ref = self._try_audio_reference()
        if audio_ref is not None:
            result = audio_ref
        else:
            self._process_blocks()
            result = self._assemble()
            self._merge_legacy_kwargs(result)
        _rewrite_system_to_developer(result, self._model)
        _strip_none_keys(result, self._drop_none)
        return cast(ChatCompletionMessageParam, result)

    def _try_audio_reference(self) -> dict[str, Any] | None:
        """Return a short-circuit dict for assistant messages with a reference audio id.

        When an assistant message's ``additional_kwargs`` contains a
        ``"reference_audio_id"`` key, the Chat Completions API expects a
        minimal ``{"role": "assistant", "audio": {"id": ...}}`` dict instead of
        the normal content dict.  This avoids re-encoding audio that has already
        been produced in a previous turn.

        Returns:
            A dict with ``role`` and ``audio.id`` if the conditions are met;
            ``None`` otherwise.

        Examples:
            - Returns dict for assistant with reference_audio_id
                ```python
                >>> from serapeum.core.llms import Message, MessageRole
                >>> converter = ChatMessageConverter(
                ...     Message(
                ...         role=MessageRole.ASSISTANT,
                ...         content="",
                ...         additional_kwargs={"reference_audio_id": "audio_123"},
                ...     )
                ... )
                >>> result = converter._try_audio_reference()
                >>> result
                {'role': 'assistant', 'audio': {'id': 'audio_123'}}

                ```
            - Returns None for user messages
                ```python
                >>> converter = ChatMessageConverter(
                ...     Message(role=MessageRole.USER, content="hi")
                ... )
                >>> converter._try_audio_reference() is None
                True

                ```
        """
        reference_audio_id = (
            self._message.additional_kwargs.get("reference_audio_id")
            if self._message.role == MessageRole.ASSISTANT
            else None
        )
        result = None
        if reference_audio_id:
            result = {
                "role": self._message.role.value,
                "audio": {"id": reference_audio_id},
            }
        return result

    def _process_blocks(self) -> None:
        """Iterate message chunks and dispatch to :class:`ChatFormat` converters.

        Populates :attr:`_content` (list of block dicts), :attr:`_content_txt`
        (concatenated plain text), and :attr:`_tool_call_dicts` (function call dicts).

        :class:`~serapeum.core.llms.ThinkingBlock` items are silently skipped with
        a ``DEBUG``-level log — the Chat Completions API does not support reasoning.

        Raises:
            ValueError: If a block type is not in :attr:`ChatFormat.content_converters`
                and is not a :class:`~serapeum.core.llms.ThinkingBlock` or
                :class:`~serapeum.core.llms.ToolCallBlock`.
        """
        for block in self._message.chunks:
            if isinstance(block, TextChunk):
                self._content.append(ChatFormat.text(block))
                self._content_txt += block.content
            elif isinstance(block, ThinkingBlock):
                logger.debug(
                    "ThinkingBlock skipped in Chat Completions path (not supported)"
                )
            elif isinstance(block, ToolCallBlock):
                self._tool_call_dicts.append(ChatFormat.tool_call(block))
            else:
                converter = ChatFormat.content_converters.get(type(block))
                if converter:
                    self._content.append(converter(block))
                else:
                    raise ValueError(
                        f"Unsupported content block type: {type(block).__name__}"
                    )

    def _resolve_content(self) -> str | list[dict[str, Any]] | None:
        """Determine the final ``content`` value for the message dict.

        Resolution rules (applied in order):

        1. If ``_content_txt`` is empty **and** the assistant has tool calls →
           return ``None`` (the API requires ``content: null`` in this case).
        2. If the role is **not** ``assistant``, ``tool``, or ``system``, **and**
           the chunks contain non-text blocks → return the full ``_content`` list.
        3. Otherwise → return ``_content_txt`` (the plain-text string).

        Returns:
            A plain-text string, a list of content block dicts, or ``None``.
        """
        has_tool_calls = len(self._tool_call_dicts) > 0
        content: str | list[dict[str, Any]] | None = self._content_txt

        if self._content_txt == "" and _should_null_content(self._message, has_tool_calls):
            content = None
        elif (
            self._message.role.value not in ("assistant", "tool", "system")
            and not all(isinstance(b, TextChunk) for b in self._message.chunks)
        ):
            content = self._content

        return content

    def _assemble(self) -> dict[str, Any]:
        """Construct the base message dict with role, content, and optional tool calls.

        Returns:
            A dict with at minimum ``role`` and ``content`` keys.
            A ``tool_calls`` key is added when :attr:`_tool_call_dicts` is non-empty.
        """
        result: dict[str, Any] = {
            "role": self._message.role.value,
            "content": self._resolve_content(),
        }
        if self._tool_call_dicts:
            result["tool_calls"] = self._tool_call_dicts
        return result

    def _merge_legacy_kwargs(self, result: dict[str, Any]) -> None:
        """Merge legacy tool call kwargs from ``additional_kwargs`` into *result*.

        When a message carries ``tool_calls`` or ``function_call`` in
        ``additional_kwargs`` **and** no :class:`~serapeum.core.llms.ToolCallBlock`
        chunks were found, the entire ``additional_kwargs`` dict is merged into
        *result*. This supports messages that were originally received from the API
        and stored as raw dicts rather than typed blocks.

        ``tool_call_id`` is **always** passed through when present, regardless of
        whether chunk-based tool calls exist.

        Args:
            result: The assembled message dict to mutate in place.
        """
        has_tool_calls = len(self._tool_call_dicts) > 0
        if (
            "tool_calls" in self._message.additional_kwargs
            or "function_call" in self._message.additional_kwargs
        ) and not has_tool_calls:
            result.update(self._message.additional_kwargs)

        if "tool_call_id" in self._message.additional_kwargs:
            result["tool_call_id"] = self._message.additional_kwargs["tool_call_id"]


class ResponsesMessageConverter:
    """Converts a serapeum :class:`~serapeum.core.llms.Message` into a Responses API dict or list.

    Use the builder pattern: construct with the message and optional parameters,
    then call :meth:`build` to obtain the final result.

    The result type varies by message content:

    - A single ``dict`` for standard messages (user, assistant, system/developer).
    - A ``list[dict]`` when the message has tool call chunks, legacy ``tool_calls``
      in ``additional_kwargs``, or when a :class:`~serapeum.core.llms.ThinkingBlock`
      precedes a message dict (reasoning items are prepended).

    Unlike :class:`ChatMessageConverter`, the Responses API **always** uses
    ``"developer"`` instead of ``"system"`` — this rewrite happens unconditionally
    in :meth:`_assemble_message_dict`, independent of the model parameter.

    Args:
        message: The serapeum message to convert.
        drop_none: When ``True``, keys with ``None`` values are removed from
            message dicts. Defaults to ``False``.
        model: Unused in the current implementation (the system→developer rewrite
            is unconditional for the Responses API). Kept for API symmetry with
            :class:`ChatMessageConverter`.

    Examples:
        - Simple user message
            ```python
            >>> from serapeum.core.llms import Message, MessageRole
            >>> ResponsesMessageConverter(
            ...     Message(role=MessageRole.USER, content="hello")
            ... ).build()
            {'role': 'user', 'content': 'hello'}

            ```
        - System message is always rewritten to developer
            ```python
            >>> result = ResponsesMessageConverter(
            ...     Message(role=MessageRole.SYSTEM, content="be helpful")
            ... ).build()
            >>> result["role"]
            'developer'

            ```
        - Tool role message produces a function_call_output dict
            ```python
            >>> from serapeum.core.llms import Message, MessageRole
            >>> result = ResponsesMessageConverter(
            ...     Message(
            ...         role=MessageRole.TOOL,
            ...         content="42",
            ...         additional_kwargs={"tool_call_id": "c1"},
            ...     )
            ... ).build()
            >>> result["type"]
            'function_call_output'
            >>> result["call_id"]
            'c1'

            ```

    See Also:
        :class:`ChatMessageConverter`: Equivalent converter for the Chat Completions API.
        :func:`to_openai_message_dicts`: Top-level dispatcher that selects this converter.
        :class:`ResponsesFormat`: Block-level converter namespace used internally.
    """

    def __init__(
        self,
        message: Message,
        *,
        drop_none: bool = False,
        model: str | None = None,
    ) -> None:
        self._message = message
        self._model = model
        self._drop_none = drop_none
        self._content: list[dict[str, Any]] = []
        self._content_txt: str = ""
        self._tool_call_dicts: list[dict[str, Any]] = []
        self._reasoning: list[dict[str, Any]] = []

    def build(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert the message and return the Responses API representation.

        Returns:
            A single dict for standard messages, or a list of dicts when the
            message has tool calls or :class:`~serapeum.core.llms.ThinkingBlock`
            items that produce reasoning entries.

        Raises:
            ValueError: If the message contains an :class:`~serapeum.core.llms.Audio`
                block (not supported by the Responses API).
            ValueError: If a block type is unsupported and not in
                :attr:`ResponsesFormat.content_converters`.
            ValueError: If a tool-role message lacks ``tool_call_id`` or ``call_id``
                in ``additional_kwargs``.
        """
        self._process_blocks()
        result = self._assemble()
        return result

    def _process_blocks(self) -> None:
        """Iterate message chunks and dispatch to :class:`ResponsesFormat` converters.

        Populates :attr:`_content`, :attr:`_content_txt`, :attr:`_tool_call_dicts`,
        and :attr:`_reasoning`.

        Raises:
            ValueError: If the message contains an :class:`~serapeum.core.llms.Audio` block.
            ValueError: If a block type is not handled.
        """
        for block in self._message.chunks:
            if isinstance(block, TextChunk):
                self._content.append(
                    ResponsesFormat.text(block, self._message.role.value)
                )
                self._content_txt += block.content
            elif isinstance(block, ThinkingBlock):
                item = ResponsesFormat.thinking(block)
                if item is not None:
                    self._reasoning.append(item)
            elif isinstance(block, ToolCallBlock):
                self._tool_call_dicts.append(ResponsesFormat.tool_call(block))
            elif isinstance(block, Audio):
                raise ValueError(
                    "Audio blocks are not supported in the Responses API"
                )
            else:
                converter = ResponsesFormat.content_converters.get(type(block))
                if converter:
                    self._content.append(converter(block))
                else:
                    raise ValueError(
                        f"Unsupported content block type: {type(block).__name__}"
                    )

    def _resolve_content(self) -> str | list[dict[str, Any]] | None:
        """Determine the final ``content`` value for the message dict.

        Resolution rules (applied in order):

        1. If ``_content_txt`` is empty **and** the assistant has tool calls →
           return ``None``.
        2. If the role is ``"system"`` or ``"developer"``, **or** all chunks are
           :class:`~serapeum.core.llms.TextChunk` → return ``_content_txt`` (string).
        3. Otherwise → return the full ``_content`` list.

        Returns:
            A plain-text string, a list of content block dicts, or ``None``.
        """
        has_tool_calls = len(self._tool_call_dicts) > 0
        content: str | list[dict[str, Any]] | None = self._content_txt
        if self._content_txt == "" and _should_null_content(self._message, has_tool_calls):
            content = None
        elif (
            (content is not None and self._message.role.value in ("system", "developer"))
            or all(isinstance(block, TextChunk) for block in self._message.chunks)
        ):
            pass  # content is already the string form
        else:
            content = self._content
        return content

    def _assemble_tool_output(self) -> dict[str, Any]:
        """Construct a ``function_call_output`` dict for tool-role messages.

        Looks up ``tool_call_id`` first, then falls back to ``call_id`` in
        ``additional_kwargs``.

        Returns:
            A dict with ``{"type": "function_call_output", "output": ..., "call_id": ...}``.

        Raises:
            ValueError: If neither ``tool_call_id`` nor ``call_id`` is present in
                ``additional_kwargs``.
        """
        call_id = self._message.additional_kwargs.get(
            "tool_call_id", self._message.additional_kwargs.get("call_id")
        )
        if call_id is None:
            raise ValueError(
                "tool_call_id or call_id is required in additional_kwargs for tool messages"
            )
        return {
            "type": "function_call_output",
            "output": self._content_txt,
            "call_id": call_id,
        }

    def _assemble_message_dict(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Construct a standard message dict, optionally prefixed by reasoning items.

        Always rewrites ``"system"`` → ``"developer"`` (unconditional for the Responses API).
        Applies :func:`_strip_none_keys` when ``drop_none=True``.

        When :attr:`_reasoning` is non-empty, wraps the message in a list:
        ``[reasoning_1, ..., message_dict]``.

        Returns:
            A single message dict, or a list starting with reasoning items followed
            by the message dict.
        """
        message_dict: dict[str, Any] = {
            "role": self._message.role.value,
            "content": self._resolve_content(),
        }
        # Responses API always uses "developer" instead of "system"
        if message_dict.get("role") == "system":
            message_dict["role"] = "developer"
        _strip_none_keys(message_dict, self._drop_none)
        result: dict[str, Any] | list[dict[str, Any]] = (
            [*self._reasoning, message_dict] if self._reasoning else message_dict
        )
        return result

    def _assemble(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Select the appropriate output shape based on message role and content.

        Priority order:

        1. **Chunk tool calls**: :class:`~serapeum.core.llms.ToolCallBlock` chunks
           were found → return ``[*reasoning, *tool_call_dicts]``.
        2. **Legacy tool calls**: ``"tool_calls"`` key in ``additional_kwargs``
           → return ``[*reasoning, *legacy_calls]``.
        3. **Tool role**: message role is ``"tool"`` → return the output of
           :meth:`_assemble_tool_output`.
        4. **Standard message**: all other cases → return output of
           :meth:`_assemble_message_dict`.

        Returns:
            A dict or list of dicts representing the Responses API input item(s).
        """
        if self._tool_call_dicts:
            result: dict[str, Any] | list[dict[str, Any]] = [
                *self._reasoning, *self._tool_call_dicts
            ]
        elif "tool_calls" in self._message.additional_kwargs:
            legacy_calls = [
                tc if isinstance(tc, dict) else tc.model_dump()
                for tc in self._message.additional_kwargs["tool_calls"]
            ]
            result = [*self._reasoning, *legacy_calls]
        elif self._message.role.value == "tool":
            result = self._assemble_tool_output()
        else:
            result = self._assemble_message_dict()
        return result


def to_openai_message_dicts(
    messages: Sequence[Message],
    drop_none: bool = False,
    model: str | None = None,
    is_responses_api: bool = False,
) -> list[ChatCompletionMessageParam] | str:
    """Convert a sequence of serapeum messages to OpenAI API format.

    Selects either the Chat Completions path (via :class:`ChatMessageConverter`) or
    the Responses API path (via :class:`ResponsesMessageConverter`) based on
    *is_responses_api*.

    **Responses API special case**: when the input is a single plain-text user
    message, the function returns the content string directly rather than a list.
    The OpenAI Responses API ``input`` parameter accepts a bare ``str`` for
    simple use-cases like image generation and MCP tool use.

    Args:
        messages: The sequence of serapeum :class:`~serapeum.core.llms.Message`
            objects to convert.
        drop_none: When ``True``, keys with ``None`` values are stripped from each
            message dict. Defaults to ``False``.
        model: The target model name, forwarded to the converter for O1-family
            role rewrites. Pass ``None`` to skip model-specific logic.
        is_responses_api: When ``True``, uses :class:`ResponsesMessageConverter`
            (Responses API format). When ``False`` (default), uses
            :class:`ChatMessageConverter` (Chat Completions format).

    Returns:
        A list of ``ChatCompletionMessageParam`` dicts for the Chat Completions
        path, or for the Responses API path when there are multiple messages or
        non-text content. Returns a plain ``str`` for the Responses API path when
        the input is a single user message with string content.

    Examples:
        - Chat Completions path (default)
            ```python
            >>> from serapeum.core.llms import Message, MessageRole
            >>> msgs = [
            ...     Message(role=MessageRole.USER, content="Hello"),
            ...     Message(role=MessageRole.ASSISTANT, content="Hi"),
            ... ]
            >>> to_openai_message_dicts(msgs)
            [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'}]

            ```
        - Responses API path — single user message returns a bare string
            ```python
            >>> msgs = [Message(role=MessageRole.USER, content="Generate an image")]
            >>> result = to_openai_message_dicts(msgs, is_responses_api=True)
            >>> result
            'Generate an image'

            ```
        - Responses API path — multiple messages return a list
            ```python
            >>> msgs = [
            ...     Message(role=MessageRole.SYSTEM, content="You are helpful."),
            ...     Message(role=MessageRole.USER, content="Hello"),
            ... ]
            >>> result = to_openai_message_dicts(msgs, is_responses_api=True)
            >>> isinstance(result, list)
            True
            >>> result[0]["role"]
            'developer'

            ```
        - O1 model rewrite via model parameter
            ```python
            >>> msgs = [Message(role=MessageRole.SYSTEM, content="You are helpful.")]
            >>> result = to_openai_message_dicts(msgs, model="o3-mini")
            >>> result[0]["role"]
            'developer'

            ```

    See Also:
        :class:`ChatMessageConverter`: Performs the Chat Completions conversion.
        :class:`ResponsesMessageConverter`: Performs the Responses API conversion.
    """
    if is_responses_api:
        final_message_dicts: list[dict[str, Any]] = []
        for message in messages:
            message_dicts = ResponsesMessageConverter(
                message,
                drop_none=drop_none,
                model=model,
            ).build()
            if isinstance(message_dicts, list):
                final_message_dicts.extend(message_dicts)
            else:
                final_message_dicts.append(message_dicts)

        # Single user message → return the content string directly
        if (
            len(final_message_dicts) == 1
            and final_message_dicts[0]["role"] == "user"
            and isinstance(final_message_dicts[0]["content"], str)
        ):
            result: list[ChatCompletionMessageParam] | str = (
                final_message_dicts[0]["content"]
            )
        else:
            result = final_message_dicts
    else:
        result = [
            ChatMessageConverter(message, drop_none=drop_none, model=model).build()
            for message in messages
        ]
    return result


# ---------------------------------------------------------------------------
# "From OpenAI" parsers — typed SDK objects → serapeum types
# ---------------------------------------------------------------------------


class ChatMessageParser:
    """Parses a Chat Completions API :class:`~openai.types.chat.ChatCompletionMessage` into a serapeum :class:`~serapeum.core.llms.Message`.

    Use the builder pattern: construct with the OpenAI message and the list of
    active modalities, then call :meth:`build`.

    Modalities control which content types are extracted:

    - ``"text"`` — enables text extraction from ``content``
    - ``"audio"`` — enables audio extraction from the ``audio`` field

    Tool calls are always extracted when present regardless of modalities.

    Args:
        openai_message: The typed ``ChatCompletionMessage`` returned by the SDK.
        modalities: Active modalities for this request (e.g. ``["text"]`` or
            ``["text", "audio"]``).

    Examples:
        - Parse a text response
            ```python
            >>> # parser = ChatMessageParser(openai_msg, ["text"])  # doctest: +SKIP
            >>> # message = parser.build()
            >>> # message.chunks[0].content
            >>> # 'Hello'

            ```
        - Parse multiple messages at once
            ```python
            >>> # messages = ChatMessageParser.batch(openai_msgs, ["text"])  # doctest: +SKIP

            ```

    See Also:
        :class:`DictMessageParser`: Parses raw dict messages instead of typed SDK objects.
        :class:`ChatMessageConverter`: The reverse direction (serapeum → OpenAI).
    """

    def __init__(self, openai_message: ChatCompletionMessage, modalities: list[str]) -> None:
        self._openai_message = openai_message
        self._modalities = modalities
        self._blocks: list[ContentBlock] = []
        self._additional_kwargs: dict[str, Any] = {}

    def build(self) -> Message:
        """Parse the OpenAI message and return a serapeum :class:`~serapeum.core.llms.Message`.

        Calls :meth:`_extract_text_content`, :meth:`_extract_tool_calls`, and
        :meth:`_extract_audio` in order, then assembles the result.

        Returns:
            A :class:`~serapeum.core.llms.Message` with the extracted blocks and
            ``additional_kwargs`` (includes raw ``tool_calls`` and/or ``reference_audio_id``
            when applicable).
        """
        self._extract_text_content()
        self._extract_tool_calls()
        self._extract_audio()
        return Message(
            role=self._openai_message.role,
            chunks=self._blocks,
            additional_kwargs=self._additional_kwargs,
        )

    def _extract_text_content(self) -> None:
        """Extract ``content`` as a :class:`~serapeum.core.llms.TextChunk` when conditions are met.

        A :class:`~serapeum.core.llms.TextChunk` is appended to :attr:`_blocks` only
        when **both** of the following hold:

        - ``"text"`` is in :attr:`_modalities`
        - ``openai_message.content`` is truthy (non-empty, non-None)

        This handles Azure OpenAI's behaviour of omitting the ``content`` key on
        function-calling messages.
        """
        # NOTE: Azure OpenAI returns function calling messages without a content key
        if "text" in self._modalities and self._openai_message.content:
            self._blocks.append(TextChunk(content=self._openai_message.content or ""))

    def _extract_tool_calls(self) -> None:
        """Extract tool calls into :class:`~serapeum.core.llms.ToolCallBlock` chunks.

        For each tool call in ``openai_message.tool_calls`` where ``function``
        is non-None, appends a :class:`~serapeum.core.llms.ToolCallBlock` to
        :attr:`_blocks`. The raw list of tool call objects is also stored in
        :attr:`_additional_kwargs` under ``"tool_calls"``.
        """
        if self._openai_message.tool_calls:
            tool_calls: list[ChatCompletionMessageToolCall] = self._openai_message.tool_calls
            for tool_call in tool_calls:
                if tool_call.function:
                    self._blocks.append(
                        ToolCallBlock(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.function.name or "",
                            tool_kwargs=tool_call.function.arguments or {},
                        )
                    )
            self._additional_kwargs.update(tool_calls=tool_calls)

    def _extract_audio(self) -> None:
        """Extract audio data into an :class:`~serapeum.core.llms.Audio` block.

        Appends an :class:`~serapeum.core.llms.Audio` block to :attr:`_blocks` and
        stores the ``reference_audio_id`` in :attr:`_additional_kwargs` when
        **both** of the following hold:

        - ``openai_message.audio`` is non-None
        - ``"audio"`` is in :attr:`_modalities`

        The audio format is hard-coded to ``"mp3"`` (the format returned by
        OpenAI's audio modality endpoint).
        """
        if self._openai_message.audio and "audio" in self._modalities:
            reference_audio_id = self._openai_message.audio.id
            audio_data = self._openai_message.audio.data
            self._additional_kwargs["reference_audio_id"] = reference_audio_id
            self._blocks.append(Audio(content=audio_data, format="mp3"))

    @classmethod
    def batch(cls, messages: Sequence[ChatCompletionMessage], modalities: list[str]) -> list[Message]:
        """Parse a sequence of OpenAI messages into serapeum messages.

        Convenience classmethod that applies :meth:`build` to each message.

        Args:
            messages: The sequence of ``ChatCompletionMessage`` objects to parse.
            modalities: Active modalities forwarded to each :class:`ChatMessageParser`
                instance (e.g. ``["text"]`` or ``["text", "audio"]``).

        Returns:
            A list of :class:`~serapeum.core.llms.Message` objects, one per input.

        Examples:
            - Parse a batch of messages (skipped — requires SDK objects)
                ```python
                >>> # messages = ChatMessageParser.batch(openai_msgs, ["text"])  # doctest: +SKIP
                >>> # len(messages)
                >>> # 2

                ```
        """
        return [cls(m, modalities).build() for m in messages]


# ---------------------------------------------------------------------------
# "From OpenAI" parsers — raw dicts → serapeum types
# ---------------------------------------------------------------------------


class DictMessageParser:
    """Parses a raw OpenAI message dict into a serapeum :class:`~serapeum.core.llms.Message`.

    Handles the raw dict format used in Responses API round-trips and legacy
    function-calling storage (where messages are stored as plain dicts rather
    than typed SDK objects).

    Content blocks in the ``"content"`` list are dispatched via the
    :attr:`_BLOCK_PARSERS` class-level dispatch table, which maps ``"type"``
    strings to static parser methods. Supported types:

    - ``"text"`` / ``"input_text"`` / ``"output_text"`` → :class:`~serapeum.core.llms.TextChunk`
    - ``"image_url"`` → :class:`~serapeum.core.llms.Image`
    - ``"function_call"`` → :class:`~serapeum.core.llms.ToolCallBlock`

    When ``"content"`` is a string (not a list), it is passed through as the
    ``Message.content`` field without parsing.

    Args:
        message_dict: The raw message dict to parse. Must contain a ``"role"`` key.

    Examples:
        - Parse a simple text message
            ```python
            >>> DictMessageParser({"role": "user", "content": "hello"}).build().content
            'hello'

            ```
        - Parse a message with content blocks
            ```python
            >>> msg = DictMessageParser({
            ...     "role": "user",
            ...     "content": [{"type": "text", "text": "hi"}],
            ... }).build()
            >>> msg.chunks[0].content
            'hi'

            ```
        - Parse multiple messages at once
            ```python
            >>> dicts = [
            ...     {"role": "user", "content": "q"},
            ...     {"role": "assistant", "content": "a"},
            ... ]
            >>> results = DictMessageParser.batch(dicts)
            >>> len(results)
            2

            ```

    See Also:
        :class:`ChatMessageParser`: Parses typed ``ChatCompletionMessage`` objects.
        :class:`ChatMessageConverter`: The reverse direction (serapeum → OpenAI dict).
    """

    _BLOCK_PARSERS: dict[str, Callable[..., ContentBlock]] = {}
    """Dispatch table mapping ``"type"`` strings to static block-parser methods.

    Populated after class definition at module scope. Keys:
    ``"text"``, ``"image_url"``, ``"function_call"``, ``"output_text"``, ``"input_text"``.
    """

    def __init__(self, message_dict: dict[str, Any]) -> None:
        self._message_dict = message_dict
        self._blocks: list[ContentBlock] = []

    def build(self) -> Message:
        """Parse the message dict and return a serapeum :class:`~serapeum.core.llms.Message`.

        When ``"content"`` is a list, delegates to :meth:`_parse_content_blocks`
        and sets ``content=None`` on the resulting :class:`~serapeum.core.llms.Message`
        (blocks are the canonical representation). When ``"content"`` is a string
        or ``None``, passes it through directly.

        Returns:
            A :class:`~serapeum.core.llms.Message` with role, content, blocks, and
            ``additional_kwargs`` (all keys except ``"role"`` and ``"content"``).

        Raises:
            ValueError: If the content list contains a block with an unsupported
                ``"type"`` value (including ``None``).
            KeyError: If the dict does not contain a ``"role"`` key.
        """
        content = self._message_dict.get("content")
        if isinstance(content, list):
            self._parse_content_blocks(content)
            content = None
        additional_kwargs = self._extract_additional_kwargs()
        return Message(
            role=self._message_dict["role"],
            content=content,
            additional_kwargs=additional_kwargs,
            chunks=self._blocks,
        )

    def _parse_content_blocks(self, content: list[dict[str, Any]]) -> None:
        """Parse each element of a content list and append to :attr:`_blocks`.

        Dispatches each element to the appropriate parser via :attr:`_BLOCK_PARSERS`.

        Args:
            content: The list of content block dicts to parse.

        Raises:
            ValueError: If an element's ``"type"`` value is not in
                :attr:`_BLOCK_PARSERS` (including ``None``).
        """
        for elem in content:
            t = elem.get("type")
            parser = self._BLOCK_PARSERS.get(t)  # type: ignore[arg-type]
            if parser:
                self._blocks.append(parser(elem))
            else:
                raise ValueError(f"Unsupported message type: {t}")

    def _extract_additional_kwargs(self) -> dict[str, Any]:
        """Extract ``additional_kwargs`` by copying the dict and removing ``role`` / ``content``.

        Returns:
            A copy of :attr:`_message_dict` with ``"role"`` and ``"content"``
            removed. All remaining keys (e.g. ``function_call``, ``tool_call_id``)
            are preserved.
        """
        additional_kwargs = self._message_dict.copy()
        additional_kwargs.pop("role")
        additional_kwargs.pop("content", None)
        return additional_kwargs

    @classmethod
    def batch(cls, dicts: Sequence[dict[str, Any]]) -> list[Message]:
        """Parse a sequence of message dicts into serapeum messages.

        Args:
            dicts: Sequence of raw message dicts, each with at least a ``"role"`` key.

        Returns:
            A list of :class:`~serapeum.core.llms.Message` objects, one per input.

        Examples:
            - Batch parse
                ```python
                >>> dicts = [
                ...     {"role": "user", "content": "q1"},
                ...     {"role": "assistant", "content": "a1"},
                ... ]
                >>> [m.content for m in DictMessageParser.batch(dicts)]
                ['q1', 'a1']

                ```
        """
        return [cls(d).build() for d in dicts]

    # -- block parsers (populated after class definition) --

    @staticmethod
    def _parse_text(elem: dict[str, Any]) -> TextChunk:
        """Parse a text content block into a :class:`~serapeum.core.llms.TextChunk`.

        Handles ``"text"``, ``"input_text"``, and ``"output_text"`` type tags.

        Args:
            elem: A content block dict containing an optional ``"text"`` key.

        Returns:
            A :class:`~serapeum.core.llms.TextChunk` with the text content, or an
            empty string when the key is absent.

        Examples:
            - Basic text block
                ```python
                >>> DictMessageParser._parse_text({"type": "text", "text": "hello"}).content
                'hello'

                ```
            - Missing text key defaults to empty string
                ```python
                >>> DictMessageParser._parse_text({"type": "text"}).content
                ''

                ```
        """
        return TextChunk(content=elem.get("text", ""))

    @staticmethod
    def _parse_image(elem: dict[str, Any]) -> Image:
        """Parse an image_url content block into an :class:`~serapeum.core.llms.Image`.

        When the URL starts with ``"data:"``, stores it as inline content
        (``Image(content=...)``). Otherwise stores it as a URL (``Image(url=...)``).

        Args:
            elem: A content block dict with an ``"image_url"`` sub-dict containing
                ``"url"`` and optionally ``"detail"``.

        Returns:
            An :class:`~serapeum.core.llms.Image` block with either ``url`` or
            ``content`` set, and optional ``detail``.

        Examples:
            - Image from URL
                ```python
                >>> img = DictMessageParser._parse_image({
                ...     "type": "image_url",
                ...     "image_url": {"url": "https://example.com/img.png", "detail": "high"},
                ... })
                >>> str(img.url)
                'https://example.com/img.png'
                >>> img.detail
                'high'

                ```
        """
        img = elem["image_url"]["url"]
        detail = elem["image_url"].get("detail")
        if img.startswith("data:"):
            result = Image(content=img, detail=detail)
        else:
            result = Image(url=img, detail=detail)
        return result

    @staticmethod
    def _parse_function_call(elem: dict[str, Any]) -> ToolCallBlock:
        """Parse a function_call content block into a :class:`~serapeum.core.llms.ToolCallBlock`.

        Args:
            elem: A content block dict with optional keys ``"call_id"``, ``"name"``,
                and ``"arguments"``.

        Returns:
            A :class:`~serapeum.core.llms.ToolCallBlock` with the parsed fields.
            Missing fields default to ``None`` (``call_id``), ``""`` (``name``),
            or ``{}`` (``arguments``).

        Examples:
            - Parse a function call block
                ```python
                >>> tc = DictMessageParser._parse_function_call({
                ...     "type": "function_call",
                ...     "call_id": "c1",
                ...     "name": "search",
                ...     "arguments": {"q": "test"},
                ... })
                >>> tc.tool_call_id
                'c1'
                >>> tc.tool_name
                'search'

                ```
        """
        return ToolCallBlock(
            tool_call_id=elem.get("call_id"),
            tool_name=elem.get("name", ""),
            tool_kwargs=elem.get("arguments", {}),
        )


DictMessageParser._BLOCK_PARSERS = {
    "text": DictMessageParser._parse_text,
    "image_url": DictMessageParser._parse_image,
    "function_call": DictMessageParser._parse_function_call,
    "output_text": DictMessageParser._parse_text,
    "input_text": DictMessageParser._parse_text,
}


# ---------------------------------------------------------------------------
# LogProb parsers
# ---------------------------------------------------------------------------


class LogProbParser:
    """Namespace of static methods that convert OpenAI logprob types to serapeum :class:`~serapeum.core.llms.LogProb` lists.

    Provides four converters covering both the Chat Completions and legacy
    completion logprob formats:

    - :meth:`from_token` / :meth:`from_tokens` — Chat Completions token-level logprobs
    - :meth:`from_completion` / :meth:`from_completions` — legacy completion-style logprobs

    All methods return empty lists (never raise) when input data is absent.

    Examples:
        - Parse token logprobs from a chat response
            ```python
            >>> # logprobs = LogProbParser.from_tokens(response.choices[0].logprobs.content)  # doctest: +SKIP

            ```
        - Parse completion logprobs from a completion response
            ```python
            >>> # logprobs = LogProbParser.from_completions(response.choices[0].logprobs)  # doctest: +SKIP

            ```

    See Also:
        :class:`~serapeum.core.llms.LogProb`: The serapeum logprob type.
    """

    @staticmethod
    def from_token(openai_token_logprob: ChatCompletionTokenLogprob) -> list[LogProb]:
        """Convert a single Chat Completions token logprob to a list of :class:`~serapeum.core.llms.LogProb`.

        Extracts the ``top_logprobs`` list (the per-token probability distribution)
        into serapeum :class:`~serapeum.core.llms.LogProb` objects. Returns an
        empty list when ``top_logprobs`` is ``None`` or empty.

        Args:
            openai_token_logprob: A single ``ChatCompletionTokenLogprob`` object
                from the OpenAI SDK response.

        Returns:
            A list of :class:`~serapeum.core.llms.LogProb` objects, one per entry
            in ``top_logprobs``. Empty when ``top_logprobs`` is absent.

        Examples:
            - Returns empty list for None top_logprobs
                ```python
                >>> from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
                >>> logprob = ChatCompletionTokenLogprob(token="hi", logprob=-0.5, top_logprobs=[])
                >>> logprob.top_logprobs = None
                >>> LogProbParser.from_token(logprob)
                []

                ```

        See Also:
            :meth:`from_tokens`: Batch version for a sequence of token logprobs.
        """
        result: list[LogProb] = []
        if openai_token_logprob.top_logprobs:
            result = [
                LogProb(token=el.token, logprob=el.logprob, bytes=el.bytes or [])
                for el in openai_token_logprob.top_logprobs
            ]
        return result

    @staticmethod
    def from_tokens(openai_token_logprobs: Sequence[ChatCompletionTokenLogprob]) -> list[list[LogProb]]:
        """Convert a sequence of Chat Completions token logprobs to a nested list.

        Applies :meth:`from_token` to each element, filtering out tokens that
        produce an empty list (i.e. tokens whose ``top_logprobs`` is absent).

        Args:
            openai_token_logprobs: The ``content`` field of a
                ``ChoiceLogprobs`` object from the SDK response.

        Returns:
            A list of lists — one inner list per token that had ``top_logprobs``
            data. Tokens without data are excluded from the result.

        Examples:
            - Filters tokens with no top_logprobs
                ```python
                >>> from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
                >>> logprob = ChatCompletionTokenLogprob(token="hi", logprob=-0.5, top_logprobs=[])
                >>> logprob.top_logprobs = None
                >>> LogProbParser.from_tokens([logprob])
                []

                ```

        See Also:
            :meth:`from_token`: Single-token version.
        """
        result: list[list[LogProb]] = []
        for token_logprob in openai_token_logprobs:
            if logprobs := LogProbParser.from_token(token_logprob):
                result.append(logprobs)
        return result

    @staticmethod
    def from_completion(openai_completion_logprob: dict[str, float]) -> list[LogProb]:
        """Convert a single legacy completion logprob dict to a list of :class:`~serapeum.core.llms.LogProb`.

        The legacy completions API returns logprobs as ``{token: logprob}`` dicts
        rather than typed objects. Bytes are set to ``[]`` since the legacy format
        does not provide byte-level information.

        Args:
            openai_completion_logprob: A dict mapping token strings to their
                log-probability float values.

        Returns:
            A list of :class:`~serapeum.core.llms.LogProb` objects with empty ``bytes``.

        Examples:
            - Convert a token→logprob dict
                ```python
                >>> result = LogProbParser.from_completion({"hello": -0.5, "hi": -1.2})
                >>> len(result)
                2
                >>> result[0].token
                'hello'
                >>> result[0].bytes
                []

                ```
            - Empty dict returns empty list
                ```python
                >>> LogProbParser.from_completion({})
                []

                ```

        See Also:
            :meth:`from_completions`: Batch version for legacy ``Logprobs`` objects.
        """
        return [
            LogProb(token=t, logprob=v, bytes=[])
            for t, v in openai_completion_logprob.items()
        ]

    @staticmethod
    def from_completions(openai_completion_logprobs: Logprobs) -> list[list[LogProb]]:
        """Convert a legacy completion :class:`~openai.types.completion_choice.Logprobs` to a nested list.

        Applies :meth:`from_completion` to each element of ``top_logprobs``.
        Returns an empty list when ``top_logprobs`` is ``None``.

        Args:
            openai_completion_logprobs: A ``Logprobs`` object from the legacy
                completions API response.

        Returns:
            A list of lists — one inner list per position in ``top_logprobs``.

        Examples:
            - Returns empty list for None top_logprobs
                ```python
                >>> from openai.types.completion_choice import Logprobs
                >>> LogProbParser.from_completions(Logprobs(top_logprobs=None))
                []

                ```

        See Also:
            :meth:`from_completion`: Single-position version.
        """
        result: list[list[LogProb]] = []
        if openai_completion_logprobs.top_logprobs:
            result = [
                LogProbParser.from_completion(completion_logprob)
                for completion_logprob in openai_completion_logprobs.top_logprobs
            ]
        return result


# ---------------------------------------------------------------------------
# Streaming tool call accumulator
# ---------------------------------------------------------------------------


class ToolCallAccumulator:
    """Accumulates streaming ``ChoiceDeltaToolCall`` chunks into complete tool calls.

    The OpenAI streaming API delivers tool calls across multiple chunks:

    - The first chunk for a given tool call contains the ``id``, ``name``, and
      the start of ``arguments``
    - Subsequent chunks for the same tool call append to ``arguments``
    - When the model calls multiple tools in one turn, each tool is identified by
      a monotonically increasing ``index`` field

    This class encapsulates the mutable accumulation state.  Call :meth:`update`
    once per streaming chunk, then read :attr:`tool_calls` after the stream ends.

    Examples:
        - Accumulate a single tool call across two chunks
            ```python
            >>> from openai.types.chat.chat_completion_chunk import (
            ...     ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
            ... )
            >>> acc = ToolCallAccumulator()
            >>> acc.update([ChoiceDeltaToolCall(
            ...     index=0, id="c1", type="function",
            ...     function=ChoiceDeltaToolCallFunction(name="search", arguments='{"q":'),
            ... )])
            >>> acc.update([ChoiceDeltaToolCall(
            ...     index=0, id=None, type="function",
            ...     function=ChoiceDeltaToolCallFunction(name="", arguments='"x"}'),
            ... )])
            >>> acc.tool_calls[0].function.arguments
            '{"q":"x"}'

            ```
        - Multiple tool calls (different index values create new entries)
            ```python
            >>> acc = ToolCallAccumulator()
            >>> acc.update([ChoiceDeltaToolCall(
            ...     index=0, id="c1", type="function",
            ...     function=ChoiceDeltaToolCallFunction(name="search", arguments="{}"),
            ... )])
            >>> acc.update([ChoiceDeltaToolCall(
            ...     index=1, id="c2", type="function",
            ...     function=ChoiceDeltaToolCallFunction(name="fetch", arguments="{}"),
            ... )])
            >>> len(acc.tool_calls)
            2

            ```

    See Also:
        :class:`ChatMessageParser`: Uses the accumulated tool calls when parsing a response.
    """

    def __init__(self) -> None:
        self._tool_calls: list[ChoiceDeltaToolCall] = []

    @property
    def tool_calls(self) -> list[ChoiceDeltaToolCall]:
        """The accumulated list of tool calls built from streaming deltas.

        Returns:
            A list of :class:`~openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall`
            objects. The list grows as :meth:`update` is called. Each entry
            represents one complete (or in-progress) tool call.
        """
        return self._tool_calls

    def update(self, tool_calls_delta: list[ChoiceDeltaToolCall] | None) -> None:
        """Merge a streaming delta into the accumulated tool call list.

        Each call to this method processes the first element of *tool_calls_delta*
        (OpenAI emits exactly one delta per chunk):

        - If :attr:`_tool_calls` is empty, appends the delta as a new entry.
        - If the delta's ``index`` differs from the last entry's ``index``,
          appends it as a new tool call (the model is starting a second tool).
        - Otherwise, calls :meth:`_merge_into_existing` to accumulate the delta's
          fields onto the last entry.

        This method intentionally returns ``None`` — the caller reads the
        accumulated state via :attr:`tool_calls`.

        Args:
            tool_calls_delta: The ``tool_calls`` field from a streaming
                ``ChoiceDelta``. ``None`` or empty list is a no-op.
        """
        if tool_calls_delta and len(tool_calls_delta) > 0:
            tc_delta = tool_calls_delta[0]
            if len(self._tool_calls) == 0:
                self._tool_calls.append(tc_delta)
            elif self._tool_calls[-1].index != tc_delta.index:
                self._tool_calls.append(tc_delta)
            else:
                self._merge_into_existing(self._tool_calls[-1], tc_delta)

    @staticmethod
    def _merge_into_existing(existing: ChoiceDeltaToolCall, delta: ChoiceDeltaToolCall) -> None:
        """Merge *delta* fields into *existing*, accumulating string values.

        Initialises ``arguments``, ``name``, and ``id`` to empty strings when
        they are ``None`` on the existing entry (this happens for the initial
        chunk which may not contain all fields).  Then appends the delta's
        ``arguments`` / ``name`` / ``id`` to the existing values.

        Args:
            existing: The tool call entry to mutate in place.
            delta: The incoming chunk delta whose values are appended.
        """
        existing_fn = cast(ChoiceDeltaToolCallFunction, existing.function)
        delta_fn = cast(ChoiceDeltaToolCallFunction, delta.function)

        if existing_fn.arguments is None:
            existing_fn.arguments = ""
        if existing_fn.name is None:
            existing_fn.name = ""
        if existing.id is None:
            existing.id = ""

        existing_fn.arguments += delta_fn.arguments or ""
        existing_fn.name += delta_fn.name or ""
        existing.id += delta.id or ""


# ---------------------------------------------------------------------------
# "From OpenAI" parsers — Responses API output → serapeum types
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tool schema conversion
# ---------------------------------------------------------------------------


def to_openai_tool(
    pydantic_class: Type[BaseModel], description: str | None = None
) -> dict[str, Any]:
    """Convert a Pydantic model class to an OpenAI function tool specification.

    Generates the JSON schema from the Pydantic model (via ``model_json_schema()``)
    and wraps it in the OpenAI tool dict format. The description is taken from
    the schema's ``"description"`` field (derived from the model's docstring) if
    present; otherwise falls back to the *description* argument.

    Args:
        pydantic_class: A Pydantic ``BaseModel`` subclass whose schema defines
            the tool's parameters.
        description: An explicit description for the tool function. Used as a
            fallback when the model class has no docstring. Defaults to ``None``.

    Returns:
        A dict conforming to the OpenAI function tool format::

            {
                "type": "function",
                "function": {
                    "name": "<model class title>",
                    "description": "<description or None>",
                    "parameters": {<JSON schema>},
                },
            }

    Examples:
        - Model with a docstring — docstring becomes the description
            ```python
            >>> from pydantic import BaseModel
            >>> class SearchTool(BaseModel):
            ...     "Search the web."
            ...     query: str
            ...
            >>> result = to_openai_tool(SearchTool)
            >>> result["type"]
            'function'
            >>> result["function"]["name"]
            'SearchTool'
            >>> result["function"]["description"]
            'Search the web.'

            ```
        - No docstring — explicit description is used
            ```python
            >>> class MyTool(BaseModel):
            ...     value: int
            ...
            >>> to_openai_tool(MyTool, description="A tool")["function"]["description"]
            'A tool'

            ```
        - No docstring, no description — description is None
            ```python
            >>> class Bare(BaseModel):
            ...     x: str
            ...
            >>> to_openai_tool(Bare)["function"]["description"] is None
            True

            ```

    See Also:
        :class:`DictMessageParser`: Parses ``function_call`` blocks that reference tool names.
        :class:`ChatFormat`: Converts :class:`~serapeum.core.llms.ToolCallBlock` instances (not schemas).
    """
    schema = pydantic_class.model_json_schema()
    schema_description = schema.get("description", None) or description
    function = {
        "name": schema["title"],
        "description": schema_description,
        "parameters": schema,
    }
    return {"type": "function", "function": function}
