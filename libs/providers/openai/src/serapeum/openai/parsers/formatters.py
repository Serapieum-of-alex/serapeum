"""Outbound formatters — serapeum Message types to OpenAI API request formats.

Classes:

- :class:`ChatFormat` — static block-level converters for the Chat Completions API
- :class:`ResponsesFormat` — static block-level converters for the Responses API
- :class:`ChatMessageConverter` — converts a ``Message`` to a Chat Completions API dict
- :class:`ResponsesMessageConverter` — converts a ``Message`` to a Responses API dict or list
- :func:`to_openai_message_dicts` — top-level dispatcher that selects the correct converter
- :func:`to_openai_tool` — converts a Pydantic model class to an OpenAI function tool spec
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from typing import Any, Sequence, Type, cast

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel

from serapeum.core.llms import (
    Audio,
    ContentBlock,
    DocumentBlock,
    Image,
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
