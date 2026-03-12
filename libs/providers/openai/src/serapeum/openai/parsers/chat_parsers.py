"""Inbound Chat Completions API parsers that convert OpenAI response types to serapeum types.

This module handles the *inbound* direction: parsing data received from the OpenAI
Chat Completions API (both typed SDK objects and raw dicts) into serapeum's internal
:class:`~serapeum.core.llms.Message` and :class:`~serapeum.core.llms.LogProb` types.

Parsers:
    - :class:`ChatMessageParser` -- converts a typed
      :class:`~openai.types.chat.chat_completion_message.ChatCompletionMessage` into a
      serapeum :class:`~serapeum.core.llms.Message`, handling text, tool calls, and audio.
    - :class:`DictMessageParser` -- converts a raw message dict (as used in Responses API
      round-trips and legacy storage) into a serapeum :class:`~serapeum.core.llms.Message`.
    - :class:`LogProbParser` -- converts OpenAI logprob types (both Chat Completions
      token logprobs and legacy completion logprobs) into nested lists of
      :class:`~serapeum.core.llms.LogProb`.

Streaming accumulator:
    - :class:`ToolCallAccumulator` -- incrementally reassembles streaming
      ``ChoiceDeltaToolCall`` chunks into complete tool-call objects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Sequence, cast

from openai.types.chat import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
from openai.types.completion_choice import Logprobs

from serapeum.core.llms import (
    Audio,
    ContentBlock,
    Image,
    LogProb,
    Message,
    TextChunk,
    ToolCallBlock,
)

OpenAIToolCall = ChatCompletionMessageToolCall | ChoiceDeltaToolCall


class ChatMessageParser:
    """Parse a typed ``ChatCompletionMessage`` from the OpenAI SDK into a serapeum ``Message``.

    Follows the builder pattern: construct an instance with the OpenAI message
    object and the list of active modalities, then call :meth:`build` to produce
    the serapeum :class:`~serapeum.core.llms.Message`.

    **Modalities** control which content types are extracted:

    - ``"text"`` -- enables text extraction from the ``content`` field.
    - ``"audio"`` -- enables audio extraction from the ``audio`` field.

    Tool calls are **always** extracted when present, regardless of modalities.

    Args:
        openai_message: The typed
            :class:`~openai.types.chat.chat_completion_message.ChatCompletionMessage`
            returned by the OpenAI SDK.
        modalities: A list of active modality strings for this request
            (e.g. ``["text"]`` or ``["text", "audio"]``).

    Examples:
        - Parse a text-only response (requires a live SDK object):
            ```python
            # parser = ChatMessageParser(openai_msg, ["text"])
            # message = parser.build()
            # message.role
            # 'assistant'
            # message.chunks[0].content
            # 'Hello'

            ```
        - Batch-parse multiple messages at once (requires SDK objects):
            ```python
            # messages = ChatMessageParser.batch(openai_msgs, ["text"])
            # len(messages)
            # 3

            ```

    See Also:
        :class:`DictMessageParser`:
            Parses raw message dicts instead of typed SDK objects.
        :class:`~serapeum.openai.parsers.formatters.ChatMessageConverter`:
            The reverse direction -- serapeum ``Message`` to OpenAI API dict.
    """

    def __init__(
        self, openai_message: ChatCompletionMessage, modalities: list[str]
    ) -> None:
        """Initialize ChatMessageParser."""
        self._openai_message = openai_message
        self._modalities = modalities
        self._blocks: list[ContentBlock] = []
        self._additional_kwargs: dict[str, Any] = {}

    def build(self) -> Message:
        """Execute all extraction steps and assemble the final serapeum ``Message``.

        Runs three extraction passes in order:

        1. :meth:`_extract_text_content` -- text from ``content`` (if ``"text"``
           modality is active).
        2. :meth:`_extract_tool_calls` -- tool calls (always extracted).
        3. :meth:`_extract_audio` -- audio data (if ``"audio"`` modality is active).

        Returns:
            A :class:`~serapeum.core.llms.Message` whose ``role`` matches the
            OpenAI message, ``chunks`` contains the extracted content blocks,
            and ``additional_kwargs`` carries raw SDK objects (``tool_calls``
            list and/or ``reference_audio_id``) for downstream use.
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
        """Extract the ``content`` string as a :class:`~serapeum.core.llms.TextChunk`.

        A :class:`~serapeum.core.llms.TextChunk` is appended to :attr:`_blocks`
        only when **both** conditions are satisfied:

        1. ``"text"`` is in :attr:`_modalities`.
        2. ``openai_message.content`` is truthy (non-empty, non-``None``).

        The truthy check also handles Azure OpenAI's behaviour of omitting the
        ``content`` key entirely on function-calling messages.
        """
        # NOTE: Azure OpenAI returns function calling messages without a content key
        if "text" in self._modalities and self._openai_message.content:
            self._blocks.append(TextChunk(content=self._openai_message.content or ""))

    def _extract_tool_calls(self) -> None:
        """Extract tool calls into :class:`~serapeum.core.llms.ToolCallBlock` chunks.

        For each ``ChatCompletionMessageToolCall`` in
        ``openai_message.tool_calls`` whose ``function`` attribute is non-``None``,
        a :class:`~serapeum.core.llms.ToolCallBlock` is appended to
        :attr:`_blocks`.  The raw list of SDK tool-call objects is also stored
        in :attr:`_additional_kwargs` under the ``"tool_calls"`` key so that
        downstream code (e.g. :class:`~serapeum.openai.parsers.formatters.ChatMessageConverter`)
        can pass them through without re-parsing.
        """
        if self._openai_message.tool_calls:
            tool_calls: list[ChatCompletionMessageToolCall] = (
                self._openai_message.tool_calls
            )
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
        """Extract audio data into an :class:`~serapeum.core.llms.Audio` content block.

        An :class:`~serapeum.core.llms.Audio` block is appended to :attr:`_blocks`
        and the ``reference_audio_id`` is stored in :attr:`_additional_kwargs`
        when **both** conditions are met:

        1. ``openai_message.audio`` is non-``None`` (the response includes audio).
        2. ``"audio"`` is in :attr:`_modalities` (the caller requested audio).

        The audio format is hard-coded to ``"mp3"`` because that is the format
        returned by OpenAI's audio modality endpoint.
        """
        if self._openai_message.audio and "audio" in self._modalities:
            reference_audio_id = self._openai_message.audio.id
            audio_data = self._openai_message.audio.data
            self._additional_kwargs["reference_audio_id"] = reference_audio_id
            self._blocks.append(Audio(content=audio_data, format="mp3"))

    @classmethod
    def batch(
        cls, messages: Sequence[ChatCompletionMessage], modalities: list[str]
    ) -> list[Message]:
        """Parse a sequence of ``ChatCompletionMessage`` objects into serapeum messages.

        Convenience classmethod that constructs a :class:`ChatMessageParser` for
        each input message and calls :meth:`build` on it.

        Args:
            messages: The sequence of
                :class:`~openai.types.chat.chat_completion_message.ChatCompletionMessage`
                objects to parse.
            modalities: Active modality strings forwarded to each
                :class:`ChatMessageParser` instance (e.g. ``["text"]`` or
                ``["text", "audio"]``).

        Returns:
            A list of :class:`~serapeum.core.llms.Message` objects, one per
            input message, in the same order.

        Examples:
            - Batch-parse messages (requires SDK objects):
                ```python
                # messages = ChatMessageParser.batch(openai_msgs, ["text"])
                # len(messages)
                # 2
                # messages[0].role
                # 'assistant'

                ```
        """
        return [cls(m, modalities).build() for m in messages]


class DictMessageParser:
    """Parse a raw OpenAI message dict into a serapeum :class:`~serapeum.core.llms.Message`.

    This parser handles the raw dict format used in Responses API round-trips,
    legacy function-calling storage, and any other context where messages are
    plain dicts rather than typed SDK objects.

    When ``"content"`` is a **list**, each element is dispatched through the
    :attr:`_BLOCK_PARSERS` class-level dispatch table, which maps ``"type"``
    strings to static parser methods.  Supported type tags:

    - ``"text"`` / ``"input_text"`` / ``"output_text"`` --
      :class:`~serapeum.core.llms.TextChunk`
    - ``"image_url"`` -- :class:`~serapeum.core.llms.Image`
    - ``"function_call"`` -- :class:`~serapeum.core.llms.ToolCallBlock`

    When ``"content"`` is a **string**, it is wrapped in a
    :class:`~serapeum.core.llms.TextChunk` and placed in ``chunks``.

    Args:
        message_dict: The raw message dict to parse.  Must contain at least a
            ``"role"`` key.

    Examples:
        - Parse a simple text message and access its content:
            ```python
            >>> DictMessageParser({"role": "user", "content": "hello"}).build().content
            'hello'

            ```
        - Parse a message with structured content blocks:
            ```python
            >>> msg = DictMessageParser({
            ...     "role": "user",
            ...     "content": [{"type": "text", "text": "hi"}],
            ... }).build()
            >>> msg.chunks[0].content
            'hi'

            ```
        - Batch-parse multiple dicts at once:
            ```python
            >>> dicts = [
            ...     {"role": "user", "content": "q"},
            ...     {"role": "assistant", "content": "a"},
            ... ]
            >>> results = DictMessageParser.batch(dicts)
            >>> len(results)
            2
            >>> results[0].content
            'q'

            ```

    See Also:
        :class:`ChatMessageParser`:
            Parses typed ``ChatCompletionMessage`` SDK objects instead of dicts.
        :class:`~serapeum.openai.parsers.formatters.ChatMessageConverter`:
            The reverse direction -- serapeum ``Message`` to OpenAI API dict.
    """

    _BLOCK_PARSERS: dict[str, Callable[..., ContentBlock]] = {}
    """Dispatch table mapping content-block ``"type"`` strings to static parser methods.

    Populated after the class definition at module scope.  Keys:
    ``"text"``, ``"input_text"``, ``"output_text"``, ``"image_url"``, ``"function_call"``.
    Each value is a ``Callable[[dict], ContentBlock]`` that parses a single
    content-block dict into the corresponding serapeum block type.
    """

    def __init__(self, message_dict: dict[str, Any]) -> None:
        """Initialize DictMessageParser."""
        self._message_dict = message_dict
        self._blocks: list[ContentBlock] = []

    def build(self) -> Message:
        """Parse the raw message dict and return a serapeum :class:`~serapeum.core.llms.Message`.

        Handles two content shapes:

        - **List content** -- each element is dispatched through
          :meth:`_parse_content_blocks`; the resulting blocks are stored in
          ``chunks`` and ``content`` is derived from any text blocks.
        - **String / None content** -- the value is wrapped in a
          :class:`~serapeum.core.llms.TextChunk` (if non-``None``) and placed
          in ``chunks``.

        All dict keys other than ``"role"`` and ``"content"`` are passed through
        in ``additional_kwargs``.

        Returns:
            A :class:`~serapeum.core.llms.Message` with ``role``, ``chunks``,
            and ``additional_kwargs`` populated from the source dict.

        Raises:
            ValueError: If the ``"content"`` list contains a block whose
                ``"type"`` value is not recognised by :attr:`_BLOCK_PARSERS`.
            KeyError: If the dict does not contain a ``"role"`` key.
        """
        content = self._message_dict.get("content")
        if isinstance(content, list):
            self._parse_content_blocks(content)
        elif content is not None:
            self._blocks.insert(0, TextChunk(content=content))
        additional_kwargs = self._extract_additional_kwargs()
        return Message(
            role=self._message_dict["role"],
            additional_kwargs=additional_kwargs,
            chunks=self._blocks,
        )

    def _parse_content_blocks(self, content: list[dict[str, Any]]) -> None:
        """Iterate a content-block list and append parsed blocks to :attr:`_blocks`.

        Each element's ``"type"`` key is looked up in :attr:`_BLOCK_PARSERS` to
        find the correct parser function.  The parser converts the raw dict into
        a serapeum :class:`~serapeum.core.llms.ContentBlock` subclass.

        Args:
            content: A list of content-block dicts, each containing at least a
                ``"type"`` key.

        Raises:
            ValueError: If any element's ``"type"`` value is not found in
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
        """Build the ``additional_kwargs`` dict from the source message dict.

        Creates a shallow copy of :attr:`_message_dict` and removes the
        ``"role"`` and ``"content"`` keys (which are handled separately).
        All remaining keys (e.g. ``"function_call"``, ``"tool_call_id"``,
        ``"call_id"``) are preserved for downstream use.

        Returns:
            A dict containing every key from the source message dict except
            ``"role"`` and ``"content"``.
        """
        additional_kwargs = self._message_dict.copy()
        additional_kwargs.pop("role")
        additional_kwargs.pop("content", None)
        return additional_kwargs

    @classmethod
    def batch(cls, dicts: Sequence[dict[str, Any]]) -> list[Message]:
        """Parse a sequence of raw message dicts into serapeum messages.

        Convenience classmethod that constructs a :class:`DictMessageParser`
        for each input dict and calls :meth:`build` on it.

        Args:
            dicts: A sequence of raw message dicts.  Each dict must contain
                at least a ``"role"`` key.

        Returns:
            A list of :class:`~serapeum.core.llms.Message` objects, one per
            input dict, in the same order.

        Examples:
            - Batch-parse two messages and inspect content:
                ```python
                >>> dicts = [
                ...     {"role": "user", "content": "q1"},
                ...     {"role": "assistant", "content": "a1"},
                ... ]
                >>> messages = DictMessageParser.batch(dicts)
                >>> messages[0].content
                'q1'
                >>> messages[1].content
                'a1'

                ```
        """
        return [cls(d).build() for d in dicts]

    # -- block parsers (populated after class definition) --

    @staticmethod
    def _parse_text(elem: dict[str, Any]) -> TextChunk:
        """Parse a text content-block dict into a :class:`~serapeum.core.llms.TextChunk`.

        Handles the ``"text"``, ``"input_text"``, and ``"output_text"`` type
        tags uniformly -- all three are stored in the same ``"text"`` key.

        Args:
            elem: A content-block dict.  The ``"text"`` key provides the string
                content; if absent, defaults to ``""``.

        Returns:
            A :class:`~serapeum.core.llms.TextChunk` containing the extracted
            text, or an empty string when the ``"text"`` key is missing.

        Examples:
            - Standard text block:
                ```python
                >>> DictMessageParser._parse_text({"type": "text", "text": "hello"}).content
                'hello'

                ```
            - Missing ``"text"`` key defaults to empty string:
                ```python
                >>> DictMessageParser._parse_text({"type": "text"}).content
                ''

                ```
        """
        return TextChunk(content=elem.get("text", ""))

    @staticmethod
    def _parse_image(elem: dict[str, Any]) -> Image:
        """Parse an ``image_url`` content-block dict into a serapeum :class:`~serapeum.core.llms.Image`.

        The URL is examined to determine storage mode:

        - URLs starting with ``"data:"`` are treated as **inline** base64 data
          and stored in ``Image(content=...)``.
        - All other URLs are stored as remote references in ``Image(url=...)``.

        The optional ``"detail"`` field (``"low"``, ``"high"``, ``"auto"``) is
        preserved when present.

        Args:
            elem: A content-block dict with an ``"image_url"`` sub-dict that
                contains ``"url"`` (required) and optionally ``"detail"``.

        Returns:
            An :class:`~serapeum.core.llms.Image` block with either ``url`` or
            ``content`` set, plus optional ``detail``.

        Examples:
            - Parse an image from a remote URL with detail:
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
        """Parse a ``function_call`` content-block dict into a :class:`~serapeum.core.llms.ToolCallBlock`.

        Maps the Responses API ``function_call`` schema to serapeum's
        :class:`~serapeum.core.llms.ToolCallBlock` fields:

        - ``"call_id"`` --> ``tool_call_id``
        - ``"name"`` --> ``tool_name``
        - ``"arguments"`` --> ``tool_kwargs``

        All keys are optional in the input dict; missing values default to
        ``None`` (for ``call_id``), ``""`` (for ``name``), or ``{}`` (for
        ``arguments``).

        Args:
            elem: A content-block dict with optional keys ``"call_id"``,
                ``"name"``, and ``"arguments"``.

        Returns:
            A :class:`~serapeum.core.llms.ToolCallBlock` populated from the
            input dict.

        Examples:
            - Parse a complete function call block:
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
                >>> tc.tool_kwargs
                {'q': 'test'}

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


class LogProbParser:
    """Static converters that transform OpenAI logprob types into serapeum :class:`~serapeum.core.llms.LogProb` lists.

    Provides four converters covering both API flavours:

    **Chat Completions token logprobs** (typed SDK objects):
        - :meth:`from_token` -- single ``ChatCompletionTokenLogprob`` to
          ``list[LogProb]``.
        - :meth:`from_tokens` -- sequence of token logprobs to
          ``list[list[LogProb]]``.

    **Legacy completion logprobs** (plain dicts):
        - :meth:`from_completion` -- single ``{token: logprob}`` dict to
          ``list[LogProb]``.
        - :meth:`from_completions` -- ``Logprobs`` object to
          ``list[list[LogProb]]``.

    All methods return empty lists (and never raise) when input data is absent
    or ``None``.

    Examples:
        - Parse Chat Completions token logprobs (requires a live response):
            ```python
            # logprobs = LogProbParser.from_tokens(
            #     response.choices[0].logprobs.content
            # )
            # len(logprobs)
            # 5

            ```
        - Parse a legacy completion logprob dict directly:
            ```python
            >>> result = LogProbParser.from_completion({"hello": -0.5, "hi": -1.2})
            >>> result[0].token
            'hello'
            >>> result[0].logprob
            -0.5

            ```

    See Also:
        :class:`~serapeum.core.llms.LogProb`:
            The serapeum logprob data model.
    """

    @staticmethod
    def from_token(openai_token_logprob: ChatCompletionTokenLogprob) -> list[LogProb]:
        """Convert a single Chat Completions token logprob into a list of serapeum :class:`~serapeum.core.llms.LogProb`.

        Extracts the ``top_logprobs`` field (the per-position probability
        distribution over candidate tokens) and maps each entry to a serapeum
        :class:`~serapeum.core.llms.LogProb`.  Returns an empty list when
        ``top_logprobs`` is ``None`` or empty.

        Args:
            openai_token_logprob: A single
                :class:`~openai.types.chat.chat_completion_token_logprob.ChatCompletionTokenLogprob`
                object from the OpenAI SDK response.

        Returns:
            A list of :class:`~serapeum.core.llms.LogProb` objects, one per
            entry in ``top_logprobs``.  Empty when ``top_logprobs`` is absent.

        Examples:
            - ``None`` top_logprobs yields an empty list:
                ```python
                >>> from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
                >>> logprob = ChatCompletionTokenLogprob(token="hi", logprob=-0.5, top_logprobs=[])
                >>> logprob.top_logprobs = None
                >>> LogProbParser.from_token(logprob)
                []

                ```

        See Also:
            :meth:`from_tokens`:
                Batch version that processes a sequence of token logprobs.
        """
        result: list[LogProb] = []
        if openai_token_logprob.top_logprobs:
            result = [
                LogProb(token=el.token, logprob=el.logprob, bytes=el.bytes or [])
                for el in openai_token_logprob.top_logprobs
            ]
        return result

    @staticmethod
    def from_tokens(
        openai_token_logprobs: Sequence[ChatCompletionTokenLogprob],
    ) -> list[list[LogProb]]:
        """Convert a sequence of Chat Completions token logprobs into a nested list.

        Applies :meth:`from_token` to each element and collects the results,
        **filtering out** tokens whose ``top_logprobs`` is ``None`` or empty
        (these produce empty inner lists that are excluded from the output).

        Args:
            openai_token_logprobs: The ``content`` field of a
                ``ChoiceLogprobs`` object from the SDK response -- a sequence
                of :class:`~openai.types.chat.chat_completion_token_logprob.ChatCompletionTokenLogprob`
                objects.

        Returns:
            A list of lists.  Each inner list contains
            :class:`~serapeum.core.llms.LogProb` objects for one token
            position.  Positions whose ``top_logprobs`` was absent are
            excluded entirely.

        Examples:
            - Tokens without ``top_logprobs`` are excluded:
                ```python
                >>> from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob
                >>> logprob = ChatCompletionTokenLogprob(token="hi", logprob=-0.5, top_logprobs=[])
                >>> logprob.top_logprobs = None
                >>> LogProbParser.from_tokens([logprob])
                []

                ```

        See Also:
            :meth:`from_token`:
                Single-token version.
        """
        result: list[list[LogProb]] = []
        for token_logprob in openai_token_logprobs:
            if logprobs := LogProbParser.from_token(token_logprob):
                result.append(logprobs)
        return result

    @staticmethod
    def from_completion(openai_completion_logprob: dict[str, float]) -> list[LogProb]:
        """Convert a single legacy completion logprob dict into a list of serapeum :class:`~serapeum.core.llms.LogProb`.

        The legacy completions API returns logprobs as plain ``{token: logprob}``
        dicts rather than typed SDK objects.  Each entry is converted to a
        :class:`~serapeum.core.llms.LogProb` with ``bytes`` set to ``[]`` (the
        legacy format does not provide byte-level information).

        Args:
            openai_completion_logprob: A dict mapping token strings to their
                log-probability float values.

        Returns:
            A list of :class:`~serapeum.core.llms.LogProb` objects, one per
            dict entry.  Empty when the input dict is empty.

        Examples:
            - Convert a token-to-logprob dict and inspect the results:
                ```python
                >>> result = LogProbParser.from_completion({"hello": -0.5, "hi": -1.2})
                >>> len(result)
                2
                >>> result[0].token
                'hello'
                >>> result[0].logprob
                -0.5
                >>> result[0].bytes
                []

                ```
            - Empty dict yields an empty list:
                ```python
                >>> LogProbParser.from_completion({})
                []

                ```

        See Also:
            :meth:`from_completions`:
                Batch version for legacy ``Logprobs`` objects.
        """
        return [
            LogProb(token=t, logprob=v, bytes=[])
            for t, v in openai_completion_logprob.items()
        ]

    @staticmethod
    def from_completions(openai_completion_logprobs: Logprobs) -> list[list[LogProb]]:
        """Convert a legacy :class:`~openai.types.completion_choice.Logprobs` object into a nested list.

        Applies :meth:`from_completion` to each element of ``top_logprobs``,
        producing one inner list per token position.  Returns an empty list
        when ``top_logprobs`` is ``None``.

        Args:
            openai_completion_logprobs: A
                :class:`~openai.types.completion_choice.Logprobs` object from
                the legacy completions API response.

        Returns:
            A list of lists.  Each inner list contains
            :class:`~serapeum.core.llms.LogProb` objects for one token
            position.  Empty when ``top_logprobs`` is ``None``.

        Examples:
            - ``None`` top_logprobs yields an empty list:
                ```python
                >>> from openai.types.completion_choice import Logprobs
                >>> LogProbParser.from_completions(Logprobs(top_logprobs=None))
                []

                ```

        See Also:
            :meth:`from_completion`:
                Single-position version for a single ``{token: logprob}`` dict.
        """
        result: list[list[LogProb]] = []
        if openai_completion_logprobs.top_logprobs:
            result = [
                LogProbParser.from_completion(completion_logprob)
                for completion_logprob in openai_completion_logprobs.top_logprobs
            ]
        return result


class ToolCallAccumulator:
    """Incrementally reassemble streaming ``ChoiceDeltaToolCall`` chunks into complete tool calls.

    The OpenAI streaming API delivers tool calls across multiple chunks:

    - The **first** chunk for a given tool call carries the ``id``, ``name``,
      and the beginning of ``arguments``.
    - **Subsequent** chunks for the same tool call append additional characters
      to ``arguments`` (and sometimes to ``name`` / ``id``).
    - When the model invokes **multiple** tools in a single turn, each tool is
      identified by a monotonically increasing ``index`` field.

    This class encapsulates the mutable accumulation state.  Feed each
    streaming chunk to :meth:`update`, then read the final results from the
    :attr:`tool_calls` property after the stream ends.

    Examples:
        - Accumulate a single tool call across two streaming chunks:
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
            >>> acc.tool_calls[0].function.name
            'search'

            ```
        - Multiple tool calls with different ``index`` values create separate entries:
            ```python
            >>> from openai.types.chat.chat_completion_chunk import (
            ...     ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
            ... )
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
            >>> acc.tool_calls[1].function.name
            'fetch'

            ```

    See Also:
        :class:`ChatMessageParser`:
            Uses accumulated tool calls when parsing a complete response.
    """

    def __init__(self) -> None:
        """Initialize ToolCallAccumulator."""
        self._tool_calls: list[ChoiceDeltaToolCall] = []

    @property
    def tool_calls(self) -> list[ChoiceDeltaToolCall]:
        """The accumulated tool calls reassembled from streaming deltas.

        The list grows as :meth:`update` is called.  Each entry is a
        :class:`~openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall`
        whose ``function.arguments``, ``function.name``, and ``id`` fields
        contain the concatenated strings from all chunks seen so far.

        Returns:
            A mutable list of
            :class:`~openai.types.chat.chat_completion_chunk.ChoiceDeltaToolCall`
            objects.  Each entry represents one complete (or in-progress)
            tool call.
        """
        return self._tool_calls

    def update(self, tool_calls_delta: list[ChoiceDeltaToolCall] | None) -> None:
        """Merge a streaming tool-call delta into the accumulated state.

        Each call processes the **first** element of *tool_calls_delta* (the
        OpenAI streaming API emits exactly one delta per chunk):

        - If :attr:`_tool_calls` is empty -- the delta is appended as a new
          entry (first tool call seen).
        - If the delta's ``index`` differs from the last entry's ``index`` --
          the delta is appended as a new tool call (the model is starting an
          additional tool invocation).
        - Otherwise -- :meth:`_merge_into_existing` concatenates the delta's
          ``arguments``, ``name``, and ``id`` onto the existing entry.

        This method intentionally returns ``None``; the caller reads the
        accumulated state via the :attr:`tool_calls` property.

        Args:
            tool_calls_delta: The ``tool_calls`` field from a streaming
                ``ChoiceDelta``.  ``None`` or an empty list is a no-op.
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
    def _merge_into_existing(
        existing: ChoiceDeltaToolCall, delta: ChoiceDeltaToolCall
    ) -> None:
        """Merge *delta*'s string fields into *existing* by concatenation.

        Fields that are ``None`` on the *existing* entry are initialised to
        empty strings before concatenation (this handles the initial chunk
        which may not populate all fields).  The three accumulated fields are:

        - ``function.arguments`` -- the JSON arguments string.
        - ``function.name`` -- the function name (usually complete after the
          first chunk, but the API does not guarantee this).
        - ``id`` -- the tool-call identifier.

        Args:
            existing: The in-progress tool-call entry to mutate in place.
            delta: The incoming streaming delta whose string values are
                appended to *existing*.
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
