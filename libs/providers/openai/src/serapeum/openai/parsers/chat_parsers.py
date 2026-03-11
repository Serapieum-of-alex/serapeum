"""Inbound Chat Completions API parsers — OpenAI response types to serapeum types.

Classes:

- :class:`ChatMessageParser` — typed ``ChatCompletionMessage`` → serapeum ``Message``
- :class:`DictMessageParser` — raw message dict → serapeum ``Message``
- :class:`LogProbParser` — OpenAI logprob types → :class:`~serapeum.core.llms.LogProb` lists
- :class:`ToolCallAccumulator` — streaming ``ChoiceDeltaToolCall`` chunks → accumulated tool calls
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
