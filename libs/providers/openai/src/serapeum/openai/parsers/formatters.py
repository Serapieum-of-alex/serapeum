"""Outbound formatters that convert serapeum ``Message`` types into OpenAI API request payloads.

This module provides two parallel conversion pipelines -- one for the Chat Completions API
and one for the Responses API -- plus a top-level dispatcher that selects the correct pipeline.

Block-level converters:
    - :class:`ChatFormat` -- serapeum content blocks to Chat Completions ``content`` array items.
    - :class:`ResponsesFormat` -- serapeum content blocks to Responses API ``content`` array items.

Message-level converters:
    - :class:`ChatMessageConverter` -- full ``Message`` to a Chat Completions ``messages`` element.
    - :class:`ResponsesMessageConverter` -- full ``Message`` to a Responses API ``input`` element
      (may return a single dict or a list of dicts).

Top-level helpers:
    - :func:`to_openai_message_dicts` -- dispatches a message sequence to the correct converter
      based on the ``is_responses_api`` flag.
    - :func:`to_openai_tool` -- converts a Pydantic model class into an OpenAI function tool spec.

Private helpers:
    - :func:`_rewrite_system_to_developer` -- mutates ``"system"`` to ``"developer"`` for O1 models.
    - :func:`_strip_none_keys` -- removes ``None``-valued keys from a dict.
    - :func:`_should_null_content` -- decides whether an assistant message's content should be ``None``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Sequence, Type, cast

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel

from serapeum.core.llms import (
    Audio,
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
    """Rewrite the ``"system"`` role to ``"developer"`` in place for O1-family models.

    OpenAI's O1-family models that support function calling (e.g. ``o3-mini``)
    require the ``"developer"`` role instead of ``"system"``.  This helper
    mutates *message_dict* in place when the model is in
    :data:`~serapeum.openai.data.models.O1_MODELS` but **not** in
    :data:`~serapeum.openai.data.models.O1_MODELS_WITHOUT_FUNCTION_CALLING`.

    The rewrite is skipped when any of the following are true:

    - *model* is ``None``.
    - *model* is not recognised as an O1-family model.
    - *model* is in :data:`O1_MODELS_WITHOUT_FUNCTION_CALLING` (those models
      do not support system prompts at all, e.g. ``o1-mini``, ``o1-preview``).
    - The message's ``"role"`` is not ``"system"``.

    Args:
        message_dict: A mutable message dict containing at least a ``"role"`` key.
            The ``"role"`` value is overwritten in place when conditions are met.
        model: The target OpenAI model identifier (e.g. ``"o3-mini"``).  Pass
            ``None`` to unconditionally skip the rewrite.

    Examples:
        - Rewrite for an O1 model that supports function calling:
            ```python
            >>> d = {"role": "system", "content": "You are helpful."}
            >>> _rewrite_system_to_developer(d, "o3-mini")
            >>> d["role"]
            'developer'

            ```
        - Standard GPT model is left unchanged:
            ```python
            >>> d = {"role": "system", "content": "You are helpful."}
            >>> _rewrite_system_to_developer(d, "gpt-4o")
            >>> d["role"]
            'system'

            ```
        - ``None`` model skips rewriting entirely:
            ```python
            >>> d = {"role": "system", "content": "You are helpful."}
            >>> _rewrite_system_to_developer(d, None)
            >>> d["role"]
            'system'

            ```

    See Also:
        :data:`~serapeum.openai.data.models.O1_MODELS`:
            The full set of O1-family model identifiers.
        :data:`~serapeum.openai.data.models.O1_MODELS_WITHOUT_FUNCTION_CALLING`:
            O1 models that lack function-calling support.
    """
    if (
        model is not None
        and model in O1_MODELS
        and model not in O1_MODELS_WITHOUT_FUNCTION_CALLING
        and message_dict.get("role") == "system"
    ):
        message_dict["role"] = "developer"


def _strip_none_keys(message_dict: dict[str, Any], drop_none: bool) -> None:
    """Remove every ``None``-valued key from *message_dict* when *drop_none* is ``True``.

    Useful for cleaning up message dicts before sending them to the OpenAI API, which
    may reject unexpected ``null`` values depending on the endpoint.  When *drop_none*
    is ``False``, this function is a no-op.

    Args:
        message_dict: A mutable dict to strip ``None``-valued keys from.  Mutated
            in place -- matching keys are removed via ``dict.pop``.
        drop_none: Controls whether stripping is applied.  ``True`` removes all
            keys whose value is ``None``; ``False`` leaves the dict untouched.

    Examples:
        - Strip all ``None`` values when enabled:
            ```python
            >>> d = {"role": "assistant", "content": None, "tool_calls": None}
            >>> _strip_none_keys(d, drop_none=True)
            >>> sorted(d.keys())
            ['role']

            ```
        - Dict is left intact when stripping is disabled:
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
    """Decide whether an assistant message's ``content`` field should be ``None``.

    The OpenAI API requires ``content: null`` for assistant messages that carry
    only tool calls or function calls.  This predicate returns ``True`` when
    **both** conditions are satisfied:

    1. The message role is ``"assistant"``.
    2. At least one tool-call indicator is present:
       - ``"function_call"`` in ``message.additional_kwargs``, **or**
       - ``"tool_calls"`` in ``message.additional_kwargs``, **or**
       - *has_tool_calls* is ``True`` (the caller already detected
         :class:`~serapeum.core.llms.ToolCallBlock` chunks).

    Args:
        message: The serapeum :class:`~serapeum.core.llms.Message` to inspect.
        has_tool_calls: Pre-computed flag indicating whether the message
            contains :class:`~serapeum.core.llms.ToolCallBlock` chunks.
            Passing this avoids redundant iteration over the chunks list.

    Returns:
        ``True`` when the outgoing ``content`` should be set to ``None``;
        ``False`` otherwise.

    Examples:
        - Assistant with ``tool_calls`` in ``additional_kwargs`` is nulled:
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> msg = Message(
            ...     role=MessageRole.ASSISTANT,
            ...     chunks=[TextChunk(content="")],
            ...     additional_kwargs={"tool_calls": [{"id": "c1"}]},
            ... )
            >>> _should_null_content(msg, has_tool_calls=False)
            True

            ```
        - User messages are never nulled, even with tool calls present:
            ```python
            >>> msg = Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])
            >>> _should_null_content(msg, has_tool_calls=True)
            False

            ```
        - Assistant without any tool-call indicator is not nulled:
            ```python
            >>> msg = Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="hello")])
            >>> _should_null_content(msg, has_tool_calls=False)
            False

            ```

    See Also:
        :meth:`ChatMessageConverter._resolve_content`:
            Calls this to decide the Chat Completions ``content`` value.
        :meth:`ResponsesMessageConverter._resolve_content`:
            Calls this to decide the Responses API ``content`` value.
    """
    return message.role == MessageRole.ASSISTANT and (
        "function_call" in message.additional_kwargs
        or "tool_calls" in message.additional_kwargs
        or has_tool_calls
    )


class ChatFormat:
    """Static converters that transform serapeum content blocks into Chat Completions API dicts.

    Each static method accepts a single serapeum content block
    (:class:`~serapeum.core.llms.TextChunk`, :class:`~serapeum.core.llms.Image`,
    :class:`~serapeum.core.llms.Audio`, :class:`~serapeum.core.llms.DocumentBlock`,
    or :class:`~serapeum.core.llms.ToolCallBlock`) and returns a dict that conforms
    to the ``content`` array item schema of the OpenAI Chat Completions API.

    The :attr:`content_converters` class-level dispatch table maps block **types**
    to their converter callables, enabling :class:`ChatMessageConverter` to handle
    block types that are not covered by explicit ``isinstance`` checks.

    Examples:
        - Convert a text chunk to a Chat Completions content item:
            ```python
            >>> from serapeum.core.llms import TextChunk
            >>> ChatFormat.text(TextChunk(content="hello"))
            {'type': 'text', 'text': 'hello'}

            ```
        - Convert an image URL to an ``image_url`` content item:
            ```python
            >>> from serapeum.core.llms import Image
            >>> result = ChatFormat.image(Image(url="https://example.com/img.png"))
            >>> result["type"]
            'image_url'
            >>> result["image_url"]["url"]
            'https://example.com/img.png'

            ```
        - Convert a tool call block to a function call dict:
            ```python
            >>> from serapeum.core.llms import ToolCallBlock
            >>> result = ChatFormat.tool_call(
            ...     ToolCallBlock(tool_call_id="c1", tool_name="search", tool_kwargs={"q": "x"})
            ... )
            >>> result["function"]["name"]
            'search'

            ```

    See Also:
        :class:`ResponsesFormat`:
            Equivalent converter namespace for the Responses API.
        :class:`ChatMessageConverter`:
            Message-level converter that delegates to these block-level converters.
    """

    @staticmethod
    def text(block: TextChunk) -> dict[str, Any]:
        """Convert a :class:`~serapeum.core.llms.TextChunk` into a Chat Completions ``text`` content item.

        Args:
            block: The text chunk whose ``content`` string is placed into
                the returned dict.

        Returns:
            A dict of the form ``{"type": "text", "text": <content>}`` ready
            for inclusion in the ``content`` array of a Chat Completions
            message.

        Examples:
            - Simple text conversion:
                ```python
                >>> from serapeum.core.llms import TextChunk
                >>> ChatFormat.text(TextChunk(content="hello world"))
                {'type': 'text', 'text': 'hello world'}

                ```
            - Empty content is preserved as-is:
                ```python
                >>> from serapeum.core.llms import TextChunk
                >>> ChatFormat.text(TextChunk(content=""))
                {'type': 'text', 'text': ''}

                ```
        """
        return {"type": "text", "text": block.content}

    @staticmethod
    def document(block: DocumentBlock) -> dict[str, Any]:
        """Convert a :class:`~serapeum.core.llms.DocumentBlock` into a Chat Completions ``file`` content item.

        The document's binary content is base64-encoded via
        :meth:`~serapeum.core.llms.DocumentBlock.as_base64` and embedded as a
        ``data:`` URI in the ``file_data`` field.

        Args:
            block: The document block to convert.  Its ``as_base64()`` method must
                return a ``(b64_string, mimetype)`` tuple, and its ``title``
                attribute provides the filename.

        Returns:
            A dict with keys ``"type"`` (``"file"``), ``"filename"``, and
            ``"file_data"`` (a ``data:<mimetype>;base64,...`` URI string).

        Examples:
            - Convert a PDF document block (requires filesystem):
                ```python
                # result = ChatFormat.document(doc_block)
                # result["type"]
                # 'file'
                # result["filename"]
                # 'report.pdf'

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
        """Convert an :class:`~serapeum.core.llms.Image` block into a Chat Completions ``image_url`` content item.

        When the block carries a ``url``, it is used directly.  When the block
        contains inline image data instead (``url`` is ``None``),
        :meth:`~serapeum.core.llms.Image.as_data_uri` is called to produce a
        ``data:`` URI.  The optional ``detail`` field (``"low"``, ``"high"``, or
        ``"auto"``) is included in the inner dict only when it is set on the block.

        Args:
            block: The :class:`~serapeum.core.llms.Image` block to convert.
                Must have either ``url`` or inline ``content`` set.

        Returns:
            A dict of the form ``{"type": "image_url", "image_url": {"url": ..., ...}}``.
            The inner ``image_url`` dict always contains ``"url"`` and optionally
            contains ``"detail"``.

        Examples:
            - Image from a URL without explicit detail:
                ```python
                >>> from serapeum.core.llms import Image
                >>> result = ChatFormat.image(Image(url="https://example.com/img.png"))
                >>> result["image_url"]["url"]
                'https://example.com/img.png'
                >>> "detail" in result["image_url"]
                False

                ```
            - Image from a URL with detail specified:
                ```python
                >>> from serapeum.core.llms import Image
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
        """Convert an :class:`~serapeum.core.llms.Audio` block into a Chat Completions ``input_audio`` content item.

        The audio payload is resolved via
        :meth:`~serapeum.core.llms.Audio.resolve_audio` with ``as_base64=True``,
        then decoded to a UTF-8 string (base64 is ASCII-safe).  The resulting
        dict is suitable for inclusion in the ``content`` array of an
        ``input_audio``-enabled Chat Completions request.

        Args:
            block: The :class:`~serapeum.core.llms.Audio` block to convert.  Its
                ``resolve_audio()`` method must return a file-like object and its
                ``format`` attribute (e.g. ``"mp3"``, ``"wav"``) is included in the
                output dict.

        Returns:
            A dict of the form
            ``{"type": "input_audio", "input_audio": {"data": <b64_str>, "format": <fmt>}}``.

        Examples:
            - Convert an audio block (requires I/O):
                ```python
                # result = ChatFormat.audio(audio_block)
                # result["type"]
                # 'input_audio'
                # result["input_audio"]["format"]
                # 'mp3'

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
        """Convert a :class:`~serapeum.core.llms.ToolCallBlock` into a Chat Completions ``function`` tool-call dict.

        The returned dict is intended for the ``tool_calls`` array of an
        assistant message, **not** the ``content`` array.  It follows the
        Chat Completions schema: ``type`` is ``"function"``, the tool's
        name and arguments live under ``"function"``, and the call identifier
        is at the top level under ``"id"``.

        Args:
            block: The :class:`~serapeum.core.llms.ToolCallBlock` to convert.
                ``tool_call_id``, ``tool_name``, and ``tool_kwargs`` are mapped
                to ``id``, ``function.name``, and ``function.arguments``
                respectively.

        Returns:
            A dict of the form
            ``{"type": "function", "function": {"name": ..., "arguments": ...}, "id": ...}``.

        Examples:
            - Convert a tool call block and inspect the result:
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
                >>> result["function"]["arguments"]
                {'q': 'x'}

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
    """Dispatch table mapping serapeum block types to their static converter callables.

    :class:`ChatMessageConverter` uses this table as a fallback for block types
    that are not covered by explicit ``isinstance`` checks (i.e. types other than
    :class:`~serapeum.core.llms.TextChunk`, :class:`~serapeum.core.llms.ThinkingBlock`,
    and :class:`~serapeum.core.llms.ToolCallBlock`).  The ``.__func__`` accessor
    is needed because the methods are ``@staticmethod`` descriptors at class-definition
    time.
    """


class ResponsesFormat:
    """Static converters that transform serapeum content blocks into Responses API dicts.

    The Responses API schema differs from Chat Completions in several ways:

    - Text blocks use role-dependent type tags: ``"input_text"`` for user
      messages and ``"output_text"`` for assistant/system messages.
    - Images use a flat ``"image_url"`` string (not a nested object) and
      always include ``"detail"`` (defaulting to ``"auto"``).
    - Thinking blocks map to ``"reasoning"`` items with a ``"summary"`` list.
    - Tool calls use ``"function_call"`` type with ``"call_id"`` (not
      ``"function"`` type with ``"id"`` as in Chat Completions).

    The :attr:`content_converters` dispatch table covers only
    :class:`~serapeum.core.llms.DocumentBlock` and :class:`~serapeum.core.llms.Image`.
    :class:`~serapeum.core.llms.TextChunk`, :class:`~serapeum.core.llms.ThinkingBlock`,
    :class:`~serapeum.core.llms.ToolCallBlock`, and :class:`~serapeum.core.llms.Audio`
    are handled by explicit ``isinstance`` checks in :class:`ResponsesMessageConverter`.

    Examples:
        - User text block produces ``input_text``:
            ```python
            >>> from serapeum.core.llms import TextChunk
            >>> ResponsesFormat.text(TextChunk(content="hello"), "user")
            {'type': 'input_text', 'text': 'hello'}

            ```
        - Assistant text block produces ``output_text``:
            ```python
            >>> from serapeum.core.llms import TextChunk
            >>> ResponsesFormat.text(TextChunk(content="hello"), "assistant")
            {'type': 'output_text', 'text': 'hello'}

            ```
        - Image conversion includes ``detail`` defaulting to ``"auto"``:
            ```python
            >>> from serapeum.core.llms import Image
            >>> result = ResponsesFormat.image(Image(url="https://example.com/img.png"))
            >>> result["detail"]
            'auto'

            ```

    See Also:
        :class:`ChatFormat`:
            Equivalent converter namespace for the Chat Completions API.
        :class:`ResponsesMessageConverter`:
            Message-level converter that delegates to these block-level converters.
    """

    @staticmethod
    def text(block: TextChunk, role: str) -> dict[str, Any]:
        """Convert a :class:`~serapeum.core.llms.TextChunk` into a Responses API text content item.

        The type tag varies by role: ``"input_text"`` for ``"user"`` messages
        and ``"output_text"`` for all other roles (``"assistant"``, ``"system"``,
        ``"developer"``, etc.).

        Args:
            block: The :class:`~serapeum.core.llms.TextChunk` whose ``content``
                string is placed into the returned dict.
            role: The message role string that determines whether the type tag
                is ``"input_text"`` or ``"output_text"``.

        Returns:
            A dict of the form ``{"type": <tag>, "text": <content>}`` where
            ``<tag>`` is ``"input_text"`` when *role* is ``"user"`` and
            ``"output_text"`` otherwise.

        Examples:
            - User role produces ``input_text``:
                ```python
                >>> from serapeum.core.llms import TextChunk
                >>> ResponsesFormat.text(TextChunk(content="Hi"), "user")
                {'type': 'input_text', 'text': 'Hi'}

                ```
            - Assistant role produces ``output_text``:
                ```python
                >>> from serapeum.core.llms import TextChunk
                >>> ResponsesFormat.text(TextChunk(content="Hi"), "assistant")
                {'type': 'output_text', 'text': 'Hi'}

                ```
            - System role also produces ``output_text``:
                ```python
                >>> from serapeum.core.llms import TextChunk
                >>> ResponsesFormat.text(TextChunk(content="Be helpful"), "system")["type"]
                'output_text'

                ```
        """
        text_type = "input_text" if role == "user" else "output_text"
        return {"type": text_type, "text": block.content}

    @staticmethod
    def document(block: DocumentBlock) -> dict[str, Any]:
        """Convert a :class:`~serapeum.core.llms.DocumentBlock` into a Responses API ``input_file`` item.

        The document's binary content is base64-encoded via
        :meth:`~serapeum.core.llms.DocumentBlock.as_base64` and embedded as a
        ``data:`` URI.  The Responses API uses ``"input_file"`` as the type tag
        (versus ``"file"`` in Chat Completions).

        Args:
            block: The :class:`~serapeum.core.llms.DocumentBlock` to convert.
                Its ``as_base64()`` method must return a ``(b64_string, mimetype)``
                tuple, and its ``title`` attribute provides the filename.

        Returns:
            A dict with keys ``"type"`` (``"input_file"``), ``"filename"``, and
            ``"file_data"`` (a ``data:<mimetype>;base64,...`` URI string).

        Examples:
            - Convert a document block (requires filesystem):
                ```python
                # result = ResponsesFormat.document(doc_block)
                # result["type"]
                # 'input_file'
                # result["filename"]
                # 'notes.pdf'

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
        """Convert an :class:`~serapeum.core.llms.Image` block into a Responses API ``input_image`` item.

        Unlike the Chat Completions path (:meth:`ChatFormat.image`), the Responses
        API uses a flat ``"image_url"`` **string** rather than a nested object, and
        the ``"detail"`` field is **always** present (defaults to ``"auto"`` when
        not set on the block).

        When the block carries a ``url``, it is stringified directly.  When the
        block contains inline image data instead (``url`` is ``None``),
        :meth:`~serapeum.core.llms.Image.as_data_uri` produces a ``data:`` URI.

        Args:
            block: The :class:`~serapeum.core.llms.Image` block to convert.
                Must have either ``url`` or inline ``content`` set.

        Returns:
            A dict of the form
            ``{"type": "input_image", "image_url": <url_str>, "detail": <detail>}``.

        Examples:
            - Image from a URL with default detail:
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
            - Image with explicit ``"high"`` detail:
                ```python
                >>> from serapeum.core.llms import Image
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
        """Convert a :class:`~serapeum.core.llms.ThinkingBlock` into a Responses API ``reasoning`` item.

        A valid ``reasoning`` item is produced only when **both** of the
        following conditions are met:

        1. ``block.content`` is truthy (non-empty, non-``None``).
        2. ``block.additional_information`` contains an ``"id"`` key (the
           reasoning item identifier assigned by the API).

        When either condition fails, ``None`` is returned and the caller
        (typically :class:`ResponsesMessageConverter`) silently drops the item.

        Args:
            block: The :class:`~serapeum.core.llms.ThinkingBlock` to convert.
                Its ``content`` becomes the ``summary_text`` and its
                ``additional_information["id"]`` becomes the reasoning ``id``.

        Returns:
            A dict of the form
            ``{"type": "reasoning", "id": ..., "summary": [{"type": "summary_text", "text": ...}]}``
            when the block has sufficient data, or ``None`` otherwise.

        Examples:
            - Valid thinking block with content and id:
                ```python
                >>> from serapeum.core.llms import ThinkingBlock
                >>> result = ResponsesFormat.thinking(
                ...     ThinkingBlock(content="step 1", additional_information={"id": "r1"})
                ... )
                >>> result["type"]
                'reasoning'
                >>> result["id"]
                'r1'
                >>> result["summary"][0]["text"]
                'step 1'

                ```
            - Missing content returns ``None``:
                ```python
                >>> from serapeum.core.llms import ThinkingBlock
                >>> ResponsesFormat.thinking(
                ...     ThinkingBlock(content=None, additional_information={"id": "r1"})
                ... ) is None
                True

                ```
            - Missing ``"id"`` in additional_information returns ``None``:
                ```python
                >>> from serapeum.core.llms import ThinkingBlock
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
                "summary": [{"type": "summary_text", "text": block.content or ""}],
            }
        return None

    @staticmethod
    def tool_call(block: ToolCallBlock) -> dict[str, Any]:
        """Convert a :class:`~serapeum.core.llms.ToolCallBlock` into a Responses API ``function_call`` item.

        The Responses API schema differs from Chat Completions: the type tag is
        ``"function_call"`` (not ``"function"``), the call identifier lives under
        ``"call_id"`` (not ``"id"``), and ``"arguments"`` and ``"name"`` are at the
        top level (not nested under a ``"function"`` key).

        Args:
            block: The :class:`~serapeum.core.llms.ToolCallBlock` to convert.
                ``tool_call_id`` maps to ``call_id``, ``tool_name`` maps to
                ``name``, and ``tool_kwargs`` maps to ``arguments``.

        Returns:
            A dict of the form
            ``{"type": "function_call", "arguments": ..., "call_id": ..., "name": ...}``.

        Examples:
            - Convert a tool call block and inspect the result:
                ```python
                >>> from serapeum.core.llms import ToolCallBlock
                >>> result = ResponsesFormat.tool_call(
                ...     ToolCallBlock(tool_call_id="c1", tool_name="search", tool_kwargs={"q": "x"})
                ... )
                >>> result["type"]
                'function_call'
                >>> result["call_id"]
                'c1'
                >>> result["name"]
                'search'
                >>> result["arguments"]
                {'q': 'x'}

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
    """Dispatch table mapping serapeum block types to their static converter callables.

    Only :class:`~serapeum.core.llms.DocumentBlock` and
    :class:`~serapeum.core.llms.Image` are included here.
    :class:`~serapeum.core.llms.TextChunk`, :class:`~serapeum.core.llms.ThinkingBlock`,
    :class:`~serapeum.core.llms.ToolCallBlock`, and :class:`~serapeum.core.llms.Audio`
    are handled by explicit ``isinstance`` checks in
    :meth:`ResponsesMessageConverter._process_blocks`.
    """


# ---------------------------------------------------------------------------
# Public converters (to OpenAI)
# ---------------------------------------------------------------------------


class ChatMessageConverter:
    """Convert a serapeum :class:`~serapeum.core.llms.Message` into a Chat Completions API message dict.

    Follows the builder pattern: construct an instance with the source message and
    optional parameters, then call :meth:`build` to produce the final
    :class:`~openai.types.chat.ChatCompletionMessageParam` dict.

    The conversion pipeline runs the following steps in order:

    1. **Audio reference short-circuit** -- if the message is an assistant message
       whose ``additional_kwargs`` contains ``"reference_audio_id"``, a minimal
       ``{"role": "assistant", "audio": {"id": ...}}`` dict is returned immediately.
    2. **Block processing** -- each chunk in ``message.chunks`` is dispatched to a
       :class:`ChatFormat` converter.  :class:`~serapeum.core.llms.ThinkingBlock`
       items are silently skipped (Chat Completions does not support reasoning).
    3. **Assembly** -- ``role``, ``content``, and optionally ``tool_calls`` are
       assembled into a dict.
    4. **Legacy kwargs merge** -- when no :class:`~serapeum.core.llms.ToolCallBlock`
       chunks were found, ``tool_calls`` / ``function_call`` from
       ``additional_kwargs`` are copied into the result dict.
    5. **Role rewrite** -- :func:`_rewrite_system_to_developer` rewrites
       ``"system"`` to ``"developer"`` for O1-family models.
    6. **None stripping** -- :func:`_strip_none_keys` removes ``None``-valued keys
       when ``drop_none=True``.

    Args:
        message: The serapeum :class:`~serapeum.core.llms.Message` to convert.
        drop_none: When ``True``, keys whose value is ``None`` are removed from
            the final dict.  Defaults to ``False``.
        model: The target OpenAI model identifier.  Used to decide whether to
            rewrite ``"system"`` to ``"developer"`` for O1-family models.
            Pass ``None`` to skip model-specific rewrites.

    Examples:
        - Convert a simple user message:
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> result = ChatMessageConverter(
            ...     Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")])
            ... ).build()
            >>> result["role"]
            'user'
            >>> result["content"]
            'hello'

            ```
        - Assistant message with tool calls nulls content automatically:
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
        - O1 model rewrites ``"system"`` to ``"developer"``:
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> result = ChatMessageConverter(
            ...     Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="be helpful")]),
            ...     model="o3-mini",
            ... ).build()
            >>> result["role"]
            'developer'

            ```

    See Also:
        :class:`ResponsesMessageConverter`:
            Equivalent converter for the Responses API.
        :func:`to_openai_message_dicts`:
            Top-level dispatcher that selects the correct converter.
        :class:`ChatFormat`:
            Block-level converter namespace used internally.
    """

    def __init__(
        self,
        message: Message,
        *,
        drop_none: bool = False,
        model: str | None = None,
    ) -> None:
        """Initialize ChatMessageConverter."""
        self._message = message
        self._model = model
        self._drop_none = drop_none
        self._content: list[dict[str, Any]] = []
        self._content_txt: str = ""
        self._tool_call_dicts: list[dict[str, Any]] = []

    def build(self) -> ChatCompletionMessageParam:
        """Execute the full conversion pipeline and return the Chat Completions message dict.

        Runs the audio-reference short-circuit check, block processing, assembly,
        legacy kwargs merge, role rewrite, and ``None``-stripping steps described
        in the class docstring.

        Returns:
            A :class:`~openai.types.chat.ChatCompletionMessageParam` dict ready
            to be included in the ``messages`` list of an OpenAI chat completion
            request.

        Raises:
            ValueError: If a content block type is not recognised by
                :attr:`ChatFormat.content_converters` and is not a
                :class:`~serapeum.core.llms.ThinkingBlock` or
                :class:`~serapeum.core.llms.ToolCallBlock`.

        Examples:
            - Build a system message dict:
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, TextChunk
                >>> ChatMessageConverter(
                ...     Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="You are helpful.")])
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
        """Short-circuit for assistant messages that reference a previously generated audio.

        When an assistant message's ``additional_kwargs`` contains
        ``"reference_audio_id"``, the Chat Completions API expects a minimal
        ``{"role": "assistant", "audio": {"id": ...}}`` dict instead of the
        normal ``content``-based dict.  This avoids re-encoding audio that was
        produced in an earlier conversation turn.

        Returns:
            A dict with ``"role"`` and ``"audio"`` keys when the conditions
            are met; ``None`` otherwise (normal processing continues).

        Examples:
            - Assistant with a ``reference_audio_id`` produces a short-circuit dict:
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, TextChunk
                >>> converter = ChatMessageConverter(
                ...     Message(
                ...         role=MessageRole.ASSISTANT,
                ...         chunks=[TextChunk(content="")],
                ...         additional_kwargs={"reference_audio_id": "audio_123"},
                ...     )
                ... )
                >>> result = converter._try_audio_reference()
                >>> result["audio"]["id"]
                'audio_123'

                ```
            - User messages always return ``None``:
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, TextChunk
                >>> converter = ChatMessageConverter(
                ...     Message(role=MessageRole.USER, chunks=[TextChunk(content="hi")])
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
        """Walk ``message.chunks`` and dispatch each block to the appropriate :class:`ChatFormat` converter.

        Populates three internal accumulators:

        - :attr:`_content` -- a list of content-item dicts for the ``content`` array.
        - :attr:`_content_txt` -- the concatenated plain-text string (used when the
          message can be represented as a simple string).
        - :attr:`_tool_call_dicts` -- a list of function-call dicts for the
          ``tool_calls`` array.

        :class:`~serapeum.core.llms.ThinkingBlock` items are silently skipped with
        a ``DEBUG``-level log message because the Chat Completions API does not
        support reasoning items.

        Raises:
            ValueError: If a block's type is not found in
                :attr:`ChatFormat.content_converters` and is not a
                :class:`~serapeum.core.llms.ThinkingBlock` or
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
        """Determine the final ``content`` value for the outgoing message dict.

        The Chat Completions API accepts ``content`` as a plain string, a list
        of content-item dicts, or ``null``.  This method picks the appropriate
        form using the following rules (applied in order):

        1. If :attr:`_content_txt` is empty **and** :func:`_should_null_content`
           returns ``True`` -- return ``None`` (the API requires ``content: null``
           for tool-call-only assistant messages).
        2. If the role is **not** ``"assistant"``, ``"tool"``, or ``"system"``
           **and** the message contains non-text blocks (images, documents, etc.)
           -- return the full :attr:`_content` list.
        3. Otherwise -- return :attr:`_content_txt` as a plain string.

        Returns:
            A plain-text string, a list of content-item dicts, or ``None``.
        """
        has_tool_calls = len(self._tool_call_dicts) > 0
        content: str | list[dict[str, Any]] | None = self._content_txt

        if self._content_txt == "" and _should_null_content(
            self._message, has_tool_calls
        ):
            content = None
        elif self._message.role.value not in (
            "assistant",
            "tool",
            "system",
        ) and not all(isinstance(b, TextChunk) for b in self._message.chunks):
            content = self._content

        return content

    def _assemble(self) -> dict[str, Any]:
        """Construct the base Chat Completions message dict from accumulated state.

        Combines ``role`` (from the source message), ``content`` (resolved by
        :meth:`_resolve_content`), and optionally ``tool_calls`` (when
        :attr:`_tool_call_dicts` is non-empty) into a single dict.

        Returns:
            A dict with at minimum ``"role"`` and ``"content"`` keys.  A
            ``"tool_calls"`` key is added when tool-call blocks were found.
        """
        result: dict[str, Any] = {
            "role": self._message.role.value,
            "content": self._resolve_content(),
        }
        if self._tool_call_dicts:
            result["tool_calls"] = self._tool_call_dicts
        return result

    def _merge_legacy_kwargs(self, result: dict[str, Any]) -> None:
        """Copy legacy tool-call information from ``additional_kwargs`` into *result*.

        This handles messages that were originally received from the OpenAI API
        and stored with ``tool_calls`` or ``function_call`` in
        ``additional_kwargs`` rather than as typed
        :class:`~serapeum.core.llms.ToolCallBlock` chunks.  The merge is
        performed **only** when no chunk-based tool calls were detected (to
        avoid duplicating information).

        The ``tool_call_id`` key is **always** passed through when present,
        regardless of whether chunk-based tool calls exist -- it is needed for
        ``tool``-role response messages.

        Args:
            result: The assembled message dict to mutate in place.  Keys from
                ``additional_kwargs`` are added via ``dict.update``.
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
    """Convert a serapeum :class:`~serapeum.core.llms.Message` into a Responses API input item.

    Follows the builder pattern: construct an instance with the source message and
    optional parameters, then call :meth:`build` to produce the result.

    The return type of :meth:`build` depends on the message content:

    - A **single dict** for standard messages (user, assistant, system/developer).
    - A **list of dicts** when the message contains
      :class:`~serapeum.core.llms.ToolCallBlock` chunks, legacy ``tool_calls`` in
      ``additional_kwargs``, or :class:`~serapeum.core.llms.ThinkingBlock` items
      that produce reasoning entries (which are prepended to the list).

    Unlike :class:`ChatMessageConverter`, the Responses API **always** uses
    ``"developer"`` instead of ``"system"`` -- this rewrite is unconditional in
    :meth:`_assemble_message_dict` and does not depend on the *model* parameter.

    Args:
        message: The serapeum :class:`~serapeum.core.llms.Message` to convert.
        drop_none: When ``True``, keys whose value is ``None`` are removed from
            each generated dict.  Defaults to ``False``.
        model: Currently unused (the ``system`` to ``developer`` rewrite is
            unconditional for the Responses API).  Retained for API symmetry
            with :class:`ChatMessageConverter`.

    Examples:
        - Convert a simple user message:
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> ResponsesMessageConverter(
            ...     Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")])
            ... ).build()
            {'role': 'user', 'content': 'hello'}

            ```
        - System messages are unconditionally rewritten to ``"developer"``:
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> result = ResponsesMessageConverter(
            ...     Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="be helpful")])
            ... ).build()
            >>> result["role"]
            'developer'

            ```
        - Tool-role message produces a ``function_call_output`` dict:
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> result = ResponsesMessageConverter(
            ...     Message(
            ...         role=MessageRole.TOOL,
            ...         chunks=[TextChunk(content="42")],
            ...         additional_kwargs={"tool_call_id": "c1"},
            ...     )
            ... ).build()
            >>> result["type"]
            'function_call_output'
            >>> result["call_id"]
            'c1'
            >>> result["output"]
            '42'

            ```

    See Also:
        :class:`ChatMessageConverter`:
            Equivalent converter for the Chat Completions API.
        :func:`to_openai_message_dicts`:
            Top-level dispatcher that selects the correct converter.
        :class:`ResponsesFormat`:
            Block-level converter namespace used internally.
    """

    def __init__(
        self,
        message: Message,
        *,
        drop_none: bool = False,
        model: str | None = None,
    ) -> None:
        """Initialize ResponsesMessageConverter."""
        self._message = message
        self._model = model
        self._drop_none = drop_none
        self._content: list[dict[str, Any]] = []
        self._content_txt: str = ""
        self._tool_call_dicts: list[dict[str, Any]] = []
        self._reasoning: list[dict[str, Any]] = []

    def build(self) -> dict[str, Any] | list[dict[str, Any]]:
        """Execute the conversion pipeline and return the Responses API input item(s).

        Runs block processing (via :meth:`_process_blocks`) followed by assembly
        (via :meth:`_assemble`) to produce either a single dict or a list of
        dicts suitable for the Responses API ``input`` parameter.

        Returns:
            A single dict for standard messages (user, assistant, developer),
            or a list of dicts when the message produces tool-call items
            and/or reasoning items.

        Raises:
            ValueError: If the message contains an
                :class:`~serapeum.core.llms.Audio` block (audio is not
                supported by the Responses API).
            ValueError: If a content block type is not recognised by
                :attr:`ResponsesFormat.content_converters` and is not
                explicitly handled.
            ValueError: If a ``"tool"``-role message lacks both
                ``"tool_call_id"`` and ``"call_id"`` in ``additional_kwargs``.
        """
        self._process_blocks()
        result = self._assemble()
        return result

    def _process_blocks(self) -> None:
        """Walk ``message.chunks`` and dispatch each block to the appropriate :class:`ResponsesFormat` converter.

        Populates four internal accumulators:

        - :attr:`_content` -- a list of content-item dicts for the ``content`` array.
        - :attr:`_content_txt` -- the concatenated plain-text string.
        - :attr:`_tool_call_dicts` -- a list of ``function_call`` dicts.
        - :attr:`_reasoning` -- a list of ``reasoning`` dicts from
          :class:`~serapeum.core.llms.ThinkingBlock` chunks.

        Raises:
            ValueError: If the message contains an
                :class:`~serapeum.core.llms.Audio` block (the Responses API
                does not support audio input).
            ValueError: If a block's type is not found in
                :attr:`ResponsesFormat.content_converters` and is not
                explicitly handled.
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
                raise ValueError("Audio blocks are not supported in the Responses API")
            else:
                converter = ResponsesFormat.content_converters.get(type(block))
                if converter:
                    self._content.append(converter(block))
                else:
                    raise ValueError(
                        f"Unsupported content block type: {type(block).__name__}"
                    )

    def _resolve_content(self) -> str | list[dict[str, Any]] | None:
        """Determine the final ``content`` value for the outgoing Responses API message dict.

        The Responses API accepts ``content`` as a plain string, a list of
        content-item dicts, or ``null``.  This method picks the appropriate
        form using the following rules (applied in order):

        1. If :attr:`_content_txt` is empty **and** :func:`_should_null_content`
           returns ``True`` -- return ``None``.
        2. If the role is ``"system"`` or ``"developer"``, **or** every chunk is a
           :class:`~serapeum.core.llms.TextChunk` -- return :attr:`_content_txt`
           as a plain string.
        3. Otherwise -- return the full :attr:`_content` list (the message
           contains non-text blocks such as images or documents).

        Returns:
            A plain-text string, a list of content-item dicts, or ``None``.
        """
        has_tool_calls = len(self._tool_call_dicts) > 0
        content: str | list[dict[str, Any]] | None = self._content_txt
        if self._content_txt == "" and _should_null_content(
            self._message, has_tool_calls
        ):
            content = None
        elif (
            content is not None and self._message.role.value in ("system", "developer")
        ) or all(isinstance(block, TextChunk) for block in self._message.chunks):
            pass  # content is already the string form
        else:
            content = self._content
        return content

    def _assemble_tool_output(self) -> dict[str, Any]:
        """Build a ``function_call_output`` dict for ``"tool"``-role messages.

        The Responses API expects tool results as a ``function_call_output`` item
        rather than a message dict.  The ``call_id`` is resolved by looking up
        ``"tool_call_id"`` first, then falling back to ``"call_id"`` in the
        message's ``additional_kwargs``.

        Returns:
            A dict of the form
            ``{"type": "function_call_output", "output": <text>, "call_id": <id>}``.

        Raises:
            ValueError: If neither ``"tool_call_id"`` nor ``"call_id"`` is found
                in ``additional_kwargs``.
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
        """Construct a standard Responses API message dict, optionally prefixed by reasoning items.

        The ``"system"`` role is unconditionally rewritten to ``"developer"``
        (required by the Responses API regardless of model).
        :func:`_strip_none_keys` is applied when ``drop_none=True``.

        When :attr:`_reasoning` contains reasoning items (from
        :class:`~serapeum.core.llms.ThinkingBlock` chunks), the result is a
        list: ``[reasoning_1, ..., reasoning_N, message_dict]``.  Otherwise a
        single message dict is returned.

        Returns:
            A single message dict, or a list whose first elements are reasoning
            dicts and whose last element is the message dict.
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
        """Select the correct Responses API output shape based on message role and accumulated state.

        The selection follows this priority order:

        1. **Chunk-based tool calls** -- :class:`~serapeum.core.llms.ToolCallBlock`
           chunks were found: return ``[*reasoning, *tool_call_dicts]``.
        2. **Legacy tool calls** -- ``"tool_calls"`` key in ``additional_kwargs``
           (no chunk-based calls): return ``[*reasoning, *legacy_call_dicts]``.
        3. **Tool role** -- message role is ``"tool"``: delegate to
           :meth:`_assemble_tool_output` for a ``function_call_output`` dict.
        4. **Standard message** -- all other cases: delegate to
           :meth:`_assemble_message_dict`.

        Returns:
            A single dict or a list of dicts representing one or more Responses
            API ``input`` items.
        """
        if self._tool_call_dicts:
            result: dict[str, Any] | list[dict[str, Any]] = [
                *self._reasoning,
                *self._tool_call_dicts,
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
    """Convert a sequence of serapeum messages into the OpenAI API format.

    This is the main entry point for outbound message formatting.  It selects
    either the **Chat Completions** pipeline (via :class:`ChatMessageConverter`)
    or the **Responses API** pipeline (via :class:`ResponsesMessageConverter`)
    based on the *is_responses_api* flag.

    **Responses API string shortcut**: when the input consists of a single
    plain-text user message, the function returns the content as a bare ``str``
    instead of a list.  The OpenAI Responses API ``input`` parameter accepts a
    raw string for simple use-cases like image generation and MCP tool use.

    Args:
        messages: The sequence of serapeum
            :class:`~serapeum.core.llms.Message` objects to convert.
        drop_none: When ``True``, keys whose value is ``None`` are removed
            from each generated dict.  Defaults to ``False``.
        model: The target OpenAI model identifier, forwarded to the converter
            for O1-family role rewrites.  Pass ``None`` to skip
            model-specific logic.
        is_responses_api: When ``True``, uses the Responses API converter.
            When ``False`` (the default), uses the Chat Completions converter.

    Returns:
        For the **Chat Completions** path: a list of
        :class:`~openai.types.chat.ChatCompletionMessageParam` dicts.

        For the **Responses API** path: a list of dicts when there are
        multiple messages or non-text content, or a plain ``str`` when the
        input is a single user message with string content.

    Examples:
        - Chat Completions path (default) produces a list of message dicts:
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> msgs = [
            ...     Message(role=MessageRole.USER, chunks=[TextChunk(content="Hello")]),
            ...     Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="Hi")]),
            ... ]
            >>> result = to_openai_message_dicts(msgs)
            >>> result[0]["role"]
            'user'
            >>> result[1]["content"]
            'Hi'

            ```
        - Responses API path with a single user message returns a bare string:
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> msgs = [Message(role=MessageRole.USER, chunks=[TextChunk(content="Generate an image")])]
            >>> to_openai_message_dicts(msgs, is_responses_api=True)
            'Generate an image'

            ```
        - Responses API path with multiple messages returns a list:
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> msgs = [
            ...     Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="You are helpful.")]),
            ...     Message(role=MessageRole.USER, chunks=[TextChunk(content="Hello")]),
            ... ]
            >>> result = to_openai_message_dicts(msgs, is_responses_api=True)
            >>> result[0]["role"]
            'developer'
            >>> result[1]["content"]
            'Hello'

            ```
        - O1-family model rewrites ``"system"`` to ``"developer"`` automatically:
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> msgs = [Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="You are helpful.")])]
            >>> result = to_openai_message_dicts(msgs, model="o3-mini")
            >>> result[0]["role"]
            'developer'

            ```

    See Also:
        :class:`ChatMessageConverter`:
            Performs the Chat Completions conversion.
        :class:`ResponsesMessageConverter`:
            Performs the Responses API conversion.
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
            result: list[ChatCompletionMessageParam] | str = final_message_dicts[0][
                "content"
            ]
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
    """Convert a Pydantic model class into an OpenAI function tool specification dict.

    Generates a JSON schema from the Pydantic model via ``model_json_schema()``
    and wraps it in the OpenAI tool dict format expected by the ``tools``
    parameter of the Chat Completions and Responses APIs.

    The tool **description** is resolved in this order:

    1. The ``"description"`` key in the generated JSON schema (populated from
       the model class's docstring).
    2. The explicit *description* argument (used as a fallback).
    3. ``None`` if neither source provides a description.

    Args:
        pydantic_class: A :class:`~pydantic.BaseModel` subclass whose fields
            define the tool's input parameters.
        description: An explicit description string.  Used only when the model
            class has no docstring.  Defaults to ``None``.

    Returns:
        A dict conforming to the OpenAI function tool format::

            {
                "type": "function",
                "function": {
                    "name": "<model class title>",
                    "description": "<resolved description or None>",
                    "parameters": {<full JSON schema>},
                },
            }

    Examples:
        - Model with a docstring uses the docstring as the description:
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
            >>> "query" in result["function"]["parameters"]["properties"]
            True

            ```
        - Model without a docstring falls back to the explicit description:
            ```python
            >>> from pydantic import BaseModel
            >>> class MyTool(BaseModel):
            ...     value: int
            ...
            >>> to_openai_tool(MyTool, description="A tool")["function"]["description"]
            'A tool'

            ```
        - No docstring and no explicit description results in ``None``:
            ```python
            >>> from pydantic import BaseModel
            >>> class Bare(BaseModel):
            ...     x: str
            ...
            >>> to_openai_tool(Bare)["function"]["description"] is None
            True

            ```

    See Also:
        :class:`ChatFormat`:
            Converts :class:`~serapeum.core.llms.ToolCallBlock` instances
            (tool *calls*, not tool *schemas*).
        :class:`ResponsesFormat`:
            Converts :class:`~serapeum.core.llms.ToolCallBlock` instances for
            the Responses API.
    """
    schema = pydantic_class.model_json_schema()
    schema_description = schema.get("description", None) or description
    function = {
        "name": schema["title"],
        "description": schema_description,
        "parameters": schema,
    }
    return {"type": "function", "function": function}
