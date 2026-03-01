"""Core data models for LLM interactions (messages, chunks, responses, metadata)."""

from __future__ import annotations

import base64
from collections.abc import Sequence as ABCSequence
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Generator, Iterator, Literal
from urllib.parse import urlparse

import requests
from filetype import guess as filetype_guess
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from serapeum.core.configs.defaults import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS

ImageType = str | BytesIO


class MessageRole(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"
    CHATBOT = "chatbot"
    MODEL = "model"


class Chunk(BaseModel):
    """Base content chunk (text, image, audio)."""

    content: bytes | str | None = None
    path: FilePath | None = None
    url: AnyUrl | str | None = None


class TextChunk(Chunk):
    """Plain text chunk."""

    type: Literal["text"] = "text"


class Image(Chunk):
    """Image chunk supporting inline bytes, file paths, or URLs."""

    type: Literal["image"] = "image"
    image_mimetype: str | None = None
    detail: str | None = None
    # Accept base64 payload provided by callers; mapped into content during validation
    base64: bytes | str | None = None

    @field_validator("url", mode="after")  # type: ignore[misc]
    @classmethod
    def url_str_to_any_url(cls, url: str | AnyUrl) -> AnyUrl:
        """Store the url as Anyurl."""
        if isinstance(url, AnyUrl):
            return url
        return AnyUrl(url=url)

    @model_validator(mode="after")  # type: ignore[misc]
    def to_base64(self) -> Self:
        """Store the image as base64 and guess the mimetype when possible.

        In case the model was built passing image data but without a mimetype,
        we try to guess it using the filetype library. To avoid resource-intense
        operations, we won't load the path or the URL to guess the mimetype.
        """
        # If explicit base64 is provided and content is missing, use it as content
        if not self.content and self.base64 is not None:
            self.content = (
                self.base64
                if isinstance(self.base64, bytes)
                else self.base64.encode("utf-8")
            )

        if not self.content:
            return self

        decoded_img: bytes
        try:
            # Check if image is already base64 encoded
            decoded_img = base64.b64decode(self.content)
        except Exception:
            # Not base64 - encode it
            if isinstance(self.content, str):
                content_bytes = self.content.encode()
            elif isinstance(self.content, bytes):
                content_bytes = self.content
            else:
                return self  # None case
            decoded_img = content_bytes
            self.content = base64.b64encode(content_bytes)

        self._guess_mimetype(decoded_img)
        return self

    def _guess_mimetype(self, img_data: bytes) -> None:
        if not self.image_mimetype:
            guess = filetype_guess(img_data)
            self.image_mimetype = guess.mime if guess else None

    def resolve_image(self, as_base64: bool = False) -> BytesIO:
        """Resolve an image such that PIL can read it.

        Args:
            as_base64 (bool): whether the resolved image should be returned as base64-encoded bytes
        """
        return resolve_binary(
            raw_bytes=self.content,
            path=self.path,
            url=str(self.url) if self.url else None,
            as_base64=as_base64,
        )


class Audio(Chunk):
    """Audio chunk supporting inline bytes, file paths, or URLs."""

    type: Literal["audio"] = "audio"
    format: str | None = None

    @field_validator("url", mode="after")  # type: ignore[misc]
    @classmethod
    def url_str_to_any_url(cls, url: str | AnyUrl) -> AnyUrl:
        """Store the url as Anyurl."""
        if isinstance(url, AnyUrl):
            return url
        return AnyUrl(url=url)

    @model_validator(mode="after")  # type: ignore[misc]
    def to_base64(self) -> Self:
        """Store the audio as base64 and guess the mimetype when possible.

        In case the model was built passing audio data but without a mimetype,
        we try to guess it using the filetype library. To avoid resource-intense
        operations, we won't load the path or the URL to guess the mimetype.
        """
        if not self.content:
            return self

        decoded_audio: bytes
        try:
            # Check if audio is already base64 encoded
            decoded_audio = base64.b64decode(self.content)
        except Exception:
            # Not base64 - encode it
            if isinstance(self.content, str):
                content_bytes = self.content.encode()
            elif isinstance(self.content, bytes):
                content_bytes = self.content
            else:
                return self  # None case
            decoded_audio = content_bytes
            self.content = base64.b64encode(content_bytes)

        self._guess_format(decoded_audio)

        return self

    def _guess_format(self, audio_data: bytes) -> None:
        if not self.format:
            guess = filetype_guess(audio_data)
            self.format = guess.extension if guess else None

    def resolve_audio(self, as_base64: bool = False) -> BytesIO:
        """Resolve an audio such that PIL can read it.

        Args:
            as_base64 (bool): whether the resolved audio should be returned as base64-encoded bytes
        """
        return resolve_binary(
            raw_bytes=self.content,
            path=self.path,
            url=str(self.url) if self.url else None,
            as_base64=as_base64,
        )


ChunkType = Annotated[TextChunk | Image | Audio, Field(discriminator="type")]


class Message(BaseModel):
    """Message."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole = MessageRole.USER
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    chunks: list[ChunkType] = Field(default_factory=list)

    def __init__(self, /, content: Any | None = None, **data: Any) -> None:
        """Constructor.

        If content was passed and contained text, store a single TextChunk.
        If content was passed and it was a list, assume it's a list of content chunks and store it.
        """
        if content is not None:
            if isinstance(content, str):
                data["chunks"] = [TextChunk(content=content)]
            elif isinstance(content, list):
                data["chunks"] = content

        super().__init__(**data)

    @property
    def content(self) -> str | None:
        """Content.

        Returns:
            The cumulative content of all TextBlocks in the message.
        """
        texts = [b.content for b in self.chunks if isinstance(b, TextChunk)]
        result = (
            None if not texts else (texts[0] if len(texts) == 1 else "\n".join(texts))
        )

        return result

    @content.setter
    def content(self, content: str) -> None:
        """Set text content.

        Raises:
            ValueError: if chunks contains more than a block, or a block that's not TextChunk.
        """
        if not self.chunks:
            self.chunks = [TextChunk(content=content)]
        elif len(self.chunks) == 1 and isinstance(self.chunks[0], TextChunk):
            self.chunks = [TextChunk(content=content)]
        else:
            raise ValueError(
                "Message contains multiple chunks, use 'Message.chunks' instead."
            )

    def __str__(self) -> str:
        """Return a human-readable representation of the message."""
        return f"{self.role.value}: {self.content}"

    @classmethod
    def from_str(
        cls,
        content: str,
        role: MessageRole | str = MessageRole.USER,
        **kwargs: Any,
    ) -> Self:
        if isinstance(role, str):
            role = MessageRole(role)
        return cls(role=role, chunks=[TextChunk(content=content)], **kwargs)

    def _recursive_serialization(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            value.model_rebuild()  # ensures all fields are initialized and serializable
            return value.model_dump()
        if isinstance(value, dict):
            return {
                key: self._recursive_serialization(value)
                for key, value in value.items()
                # if value is not None
            }
        if isinstance(value, list):
            return [self._recursive_serialization(item) for item in value]
        return value

    @field_serializer("additional_kwargs", check_fields=False)  # type: ignore[misc]
    def serialize_additional_kwargs(self, value: Any, _info: Any) -> Any:
        return self._recursive_serialization(value)


class MessageList(BaseModel, ABCSequence):
    """A collection of Message objects with helper methods."""

    messages: list[Message] = Field(default_factory=list)

    def __iter__(self) -> Iterator[Message]:
        """Iterate through contained messages in order."""
        return iter(self.messages)

    def __len__(self) -> int:
        """Return the number of messages in the list."""
        return len(self.messages)

    def __getitem__(self, index: int | slice) -> Message | MessageList:
        """Retrieve a message or slice of messages."""
        if isinstance(index, slice):
            return MessageList(messages=self.messages[index])
        return self.messages[index]

    def to_prompt(self) -> str:
        """Convert messages to a prompt string."""
        string_messages = []
        for message in self.messages:
            role = message.role
            content = message.content
            string_message = f"{role.value}: {content}"

            additional_kwargs = message.additional_kwargs
            if additional_kwargs:
                string_message += f"\n{additional_kwargs}"
            string_messages.append(string_message)

        string_messages.append(f"{MessageRole.ASSISTANT.value}: ")
        return "\n".join(string_messages)

    def filter_by_role(self, role: MessageRole) -> "MessageList":
        """Return messages with a specific role."""
        return MessageList(messages=[m for m in self.messages if m.role == role])

    def append(self, message: Message) -> None:
        """Add a message to the collection."""
        self.messages.append(message)

    @classmethod
    def from_list(cls, messages: list[Message]) -> "MessageList":
        """Create from a standard list."""
        return cls(messages=messages)

    @classmethod
    def from_str(cls, prompt: str) -> "MessageList":
        """Create from a string prompt."""
        return cls(messages=[Message(role=MessageRole.USER, content=prompt)])


class LikelihoodScore(BaseModel):
    """LikelihoodScore of a token.

    The log probability information for a token generated by the model.

    Attributes:
        token(str):
            the actual text token (string).
        next_token_log_prob(float):
            The logarithmic probability score (float) indicating how likely the model thought this token was the
            correct next token.
        bytes(list[int]):
            The byte representation of the token as a list of integers
    """

    token: str = Field(default_factory=str)
    next_token_log_prob: float = Field(default_factory=float)
    bytes: list[int] = Field(default_factory=list)


class BaseResponse(BaseModel):
    """Base response."""

    raw: Any | None = None
    likelihood_score: list[list[LikelihoodScore]] | None = None
    additional_kwargs: dict = Field(default_factory=dict)
    delta: str | None = None


class ChatResponse(BaseResponse):
    """Chat response."""

    message: Message

    def __str__(self) -> str:
        """Return the assistant message as a string."""
        return str(self.message)

    def to_completion_response(self) -> CompletionResponse:
        """Convert a chat response to a completion response."""
        return CompletionResponse(
            text=self.message.content or "",
            additional_kwargs=self.message.additional_kwargs,
            raw=self.raw,
            delta=self.delta,
        )

    @staticmethod
    def stream_to_completion_response(
        chat_response_gen: ChatResponseGen,
    ) -> CompletionResponseGen:
        """Convert a chat response stream to completion response stream.

        Args:
            chat_response_gen: Generator yielding ChatResponse objects

        Yields:
            CompletionResponse objects converted from each ChatResponse
        """

        def gen() -> CompletionResponseGen:
            for response in chat_response_gen:
                yield response.to_completion_response()

        return gen()

    @staticmethod
    def astream_to_completion_response(
        chat_response_gen: ChatResponseAsyncGen,
    ) -> CompletionResponseAsyncGen:
        """Convert an async chat response stream to completion response stream.

        Args:
            chat_response_gen: Async generator yielding ChatResponse objects

        Yields:
            CompletionResponse objects converted from each ChatResponse
        """

        async def gen() -> CompletionResponseAsyncGen:
            async for response in chat_response_gen:
                yield response.to_completion_response()

        return gen()


ChatResponseGen = Generator[ChatResponse, None, None]
ChatResponseAsyncGen = AsyncGenerator[ChatResponse, None]


class CompletionResponse(BaseResponse):
    """Response from a text completion (non-chat) LLM call.

    Represents the output of a single completion request, with optional
    streaming support via the ``delta`` field.  The class is a Pydantic
    model; all fields are validated on construction.

    Attributes:
        text: Full text content of the response.  When streaming, this
            contains the cumulative text received so far.
        raw: Raw JSON payload returned by the provider, if available.
            Useful for accessing provider-specific metadata not otherwise
            surfaced.
        likelihood_score: Per-token log-probability scores returned by
            the provider.  Outer list corresponds to choices; inner list
            to tokens within each choice.  ``None`` when not provided.
        additional_kwargs: Arbitrary provider-specific metadata such as
            token-usage counters or finish reason.  Defaults to ``{}``.
        delta: Incremental text fragment that arrived in the latest
            streaming chunk.  ``None`` when not streaming.

    Examples:
        - Non-streaming completion
            ```python
            >>> from serapeum.core.llms import CompletionResponse
            >>> resp = CompletionResponse(text="The answer is 42.")
            >>> str(resp)
            'The answer is 42.'
            >>> resp.delta is None
            True

            ```
        - Streaming token with a delta
            ```python
            >>> chunk = CompletionResponse(text="The answer is 42.", delta=" 42.")
            >>> chunk.delta
            ' 42.'

            ```
        - Attaching raw provider payload and metadata
            ```python
            >>> resp = CompletionResponse(
            ...     text="Hello",
            ...     raw={"model": "llama3", "usage": {"prompt_tokens": 5}},
            ...     additional_kwargs={"finish_reason": "stop"},
            ... )
            >>> resp.raw["model"]
            'llama3'
            >>> resp.additional_kwargs["finish_reason"]
            'stop'

            ```

    See Also:
        ChatResponse: Chat-style counterpart; convert via :meth:`to_chat_response`.
        BaseResponse: Parent model supplying ``raw``, ``likelihood_score``,
            ``additional_kwargs``, and ``delta``.
    """

    text: str

    def __str__(self) -> str:
        """Return the text content of the completion response.

        Returns:
            The ``text`` field as a plain string.

        Examples:
            - Standard string conversion
                ```python
                >>> from serapeum.core.llms import CompletionResponse
                >>> str(CompletionResponse(text="Hello, world!"))
                'Hello, world!'

                ```
            - Empty text
                ```python
                >>> str(CompletionResponse(text=""))
                ''

                ```
        """
        return self.text

    def to_chat_response(self) -> ChatResponse:
        """Convert this completion response to a :class:`ChatResponse`.

        Creates a :class:`ChatResponse` whose embedded :class:`Message`
        carries the completion text as its content and the ASSISTANT role.

        Note:
            * ``additional_kwargs`` is forwarded to the **message** (i.e.
              ``ChatResponse.message.additional_kwargs``), not to the top-level
              :class:`ChatResponse`.
            * ``delta`` and ``likelihood_score`` are **not** propagated; the
              resulting :class:`ChatResponse` will have ``delta=None`` and
              ``likelihood_score=None``.

        Returns:
            A :class:`ChatResponse` with an ASSISTANT-role :class:`Message`
            whose content equals ``self.text`` and with ``raw`` copied from
            this response.

        Examples:
            - Basic conversion
                ```python
                >>> from serapeum.core.llms import CompletionResponse, MessageRole
                >>> chat = CompletionResponse(text="Hi there!").to_chat_response()
                >>> chat.message.role
                <MessageRole.ASSISTANT: 'assistant'>
                >>> chat.message.content
                'Hi there!'

                ```
            - ``additional_kwargs`` lands on the message, not on the response
                ```python
                >>> cr = CompletionResponse(text="ok", additional_kwargs={"tokens": 7})
                >>> chat = cr.to_chat_response()
                >>> chat.message.additional_kwargs
                {'tokens': 7}
                >>> chat.additional_kwargs
                {}

                ```
            - ``delta`` is not carried forward
                ```python
                >>> CompletionResponse(text="partial", delta="rtial").to_chat_response().delta is None
                True

                ```
            - Round-trip back to CompletionResponse
                ```python
                >>> cr = CompletionResponse(text="ping", raw={"id": "x"})
                >>> cr.to_chat_response().to_completion_response().text
                'ping'

                ```

        See Also:
            ChatResponse.to_completion_response: Inverse conversion.
            stream_to_chat_response: Streaming variant that also propagates ``delta``.
        """
        return ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT,
                content=self.text,
                additional_kwargs=self.additional_kwargs,
            ),
            raw=self.raw,
        )

    @staticmethod
    def stream_to_chat_response(
        completion_response_gen: CompletionResponseGen,
    ) -> ChatResponseGen:
        """Convert a synchronous completion-response stream to a chat-response stream.

        Lazily maps each :class:`CompletionResponse` yielded by
        *completion_response_gen* to a :class:`ChatResponse` with an
        ASSISTANT-role message.  Unlike :meth:`to_chat_response`, this method
        **does** propagate ``delta`` and ``raw`` to each yielded
        :class:`ChatResponse`.

        Args:
            completion_response_gen: Synchronous generator yielding
                :class:`CompletionResponse` objects, typically from a streaming
                LLM call.

        Returns:
            A synchronous generator yielding :class:`ChatResponse` objects,
            one per input item, in arrival order.

        Examples:
            - Consuming a two-token stream
                ```python
                >>> from serapeum.core.llms import CompletionResponse
                >>> def tokens():
                ...     yield CompletionResponse(text="Hello", delta="Hello")
                ...     yield CompletionResponse(text="Hello world", delta=" world")
                >>> results = list(CompletionResponse.stream_to_chat_response(tokens()))
                >>> [r.message.content for r in results]
                ['Hello', 'Hello world']
                >>> [r.delta for r in results]
                ['Hello', ' world']

                ```
            - Empty stream produces no output
                ```python
                >>> list(CompletionResponse.stream_to_chat_response(iter([])))
                []

                ```

        See Also:
            astream_to_chat_response: Async counterpart.
            to_chat_response: Single-response (non-streaming) conversion.
            ChatResponse.stream_to_completion_response: Inverse conversion.
        """

        def gen() -> ChatResponseGen:
            for response in completion_response_gen:
                yield ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=response.text,
                        additional_kwargs=response.additional_kwargs,
                    ),
                    delta=response.delta,
                    raw=response.raw,
                )

        return gen()

    @staticmethod
    def astream_to_chat_response(
        completion_response_gen: CompletionResponseAsyncGen,
    ) -> ChatResponseAsyncGen:
        """Convert an async completion-response stream to an async chat-response stream.

        Asynchronously maps each :class:`CompletionResponse` from
        *completion_response_gen* to a :class:`ChatResponse` with an
        ASSISTANT-role message, propagating ``delta``, ``raw``, and
        ``additional_kwargs`` to each result.

        Args:
            completion_response_gen: Async generator yielding
                :class:`CompletionResponse` objects, typically from an async
                streaming LLM call.

        Returns:
            An async generator yielding :class:`ChatResponse` objects, one per
            input item, in arrival order.

        Examples:
            - Iterating an async token stream
                ```python
                >>> import asyncio
                >>> from serapeum.core.llms import CompletionResponse
                >>> async def run():
                ...     async def tokens():
                ...         yield CompletionResponse(text="Hi", delta="Hi")
                ...         yield CompletionResponse(text="Hi there", delta=" there")
                ...     results = []
                ...     async for chat in CompletionResponse.astream_to_chat_response(tokens()):
                ...         results.append((chat.message.content, chat.delta))
                ...     return results
                >>> asyncio.run(run())  # doctest: +SKIP
                [('Hi', 'Hi'), ('Hi there', ' there')]

                ```
            - Empty async stream produces no output
                ```python
                >>> async def empty(): return; yield  # doctest: +SKIP
                >>> # async for item in CompletionResponse.astream_to_chat_response(empty()): ...

                ```

        See Also:
            stream_to_chat_response: Synchronous counterpart.
            to_chat_response: Single-response (non-streaming) conversion.
            ChatResponse.astream_to_completion_response: Inverse conversion.
        """

        async def gen() -> ChatResponseAsyncGen:
            async for response in completion_response_gen:
                yield ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=response.text,
                        additional_kwargs=response.additional_kwargs,
                    ),
                    delta=response.delta,
                    raw=response.raw,
                )

        return gen()



CompletionResponseGen = Generator[CompletionResponse, None, None]
CompletionResponseAsyncGen = AsyncGenerator[CompletionResponse, None]


class Metadata(BaseModel):
    """Provider and model capabilities and defaults metadata."""

    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            "Total number of tokens the model can be input and output for one response."
        ),
    )
    num_output: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Number of tokens the model can output when generating a response.",
    )
    is_chat_model: bool = Field(
        default=False,
        description=(
            "Set True if the model exposes a chat interface (i.e. can be passed a"
            " sequence of messages, rather than text), like OpenAI's"
            " /v1/chat/completions endpoint."
        ),
    )
    is_function_calling_model: bool = Field(
        default=False,
        description=(
            "Set True if the model supports function calling messages, similar to"
            " OpenAI's function calling API. For example, converting 'Email Anya to"
            " see if she wants to get coffee next Friday' to a function call like"
            " `send_email(to: string, body: string)`."
        ),
    )
    model_name: str = Field(
        default="unknown",
        description=(
            "The model's name used for logging, testing, and sanity checking. For some"
            " models this can be automatically discerned. For other models, like"
            " locally loaded models, this must be manually specified."
        ),
    )
    system_role: MessageRole = Field(
        default=MessageRole.SYSTEM,
        description="The role this specific LLM provider"
        "expects for system prompt. E.g. 'SYSTEM' for OpenAI, 'CHATBOT' for Cohere",
    )


def resolve_binary(
    raw_bytes: bytes | None = None,
    path: str | Path | None = None,
    url: str | None = None,
    as_base64: bool = False,
) -> BytesIO:
    """Resolve binary data from various sources into a BytesIO object.

    Args:
        raw_bytes:
            Raw bytes data
        path:
            File path to read bytes from
        url:
            URL to fetch bytes from
        as_base64:
            Whether to base64 encode the output bytes

    Returns:
        BytesIO object containing the binary data

    Raises:
        ValueError: If no valid source is provided
    """
    # Handle raw bytes input
    if raw_bytes is not None:
        try:
            # Try to decode the bytes as base64
            decoded_bytes = base64.b64decode(raw_bytes)
        except Exception:
            # the bytes are already raw binary data
            decoded_bytes = raw_bytes

        if as_base64:
            # Re-encode the decoded bytes to base64
            return BytesIO(base64.b64encode(decoded_bytes))

        # Return the decoded binary data
        buffer = BytesIO(decoded_bytes)

    # Handle file path input
    elif path is not None:
        path = Path(path) if isinstance(path, str) else path

        # Read file content as bytes
        data = path.read_bytes()

        # Check if the caller wants base64-encoded output
        if as_base64:
            # Encode the file bytes to base64 and wrap in BytesIO
            return BytesIO(base64.b64encode(data))

        # Create a BytesIO buffer containing the raw file bytes
        buffer = BytesIO(data)

    # Handle URL input
    elif url is not None:
        # Parse the URL to extract its components (scheme, path, etc.)
        parsed_url = urlparse(url)

        # Special handling for data: URLs (embedded data in the URL itself)
        if parsed_url.scheme == "data":
            # Data URL format: data:[<mediatype>][;base64],<data>
            # The path attribute contains everything after "data:"
            data_part = parsed_url.path

            # Validate that the data URL has the required comma separator
            if "," not in data_part:
                raise ValueError("Invalid data URL format: missing comma separator")

            # Split the data URL into metadata (mediatype and encoding) and actual data
            # Only split on the first comma to preserve commas in the data portion
            metadata, url_data = data_part.split(",", 1)

            # Check if the metadata indicates base64 encoding
            is_base64_encoded = metadata.endswith(";base64")

            # Handle base64-encoded data URLs
            if is_base64_encoded:
                # Decode the base64 string from the URL to get raw binary data
                decoded_data = base64.b64decode(url_data)

                # Check if the caller wants base64-encoded output
                if as_base64:
                    # Re-encode to base64 and wrap in BytesIO
                    return BytesIO(base64.b64encode(decoded_data))
                else:
                    # Return the decoded binary data wrapped in BytesIO
                    return BytesIO(decoded_data)

            # Handle plain text data URLs (not base64-encoded)
            else:
                # Check if the caller wants base64-encoded output
                if as_base64:
                    # Convert the text to UTF-8 bytes, then encode to base64
                    return BytesIO(base64.b64encode(url_data.encode("utf-8")))
                else:
                    # Convert the text to UTF-8 bytes and wrap in BytesIO
                    return BytesIO(url_data.encode("utf-8"))

        # Handle HTTP(S) URLs - fetch data from the network
        # Create empty headers dict (placeholder for future authentication/headers)
        headers = {}

        # Make HTTP GET request to fetch the data from the URL
        response = requests.get(url, headers=headers)

        # Raise an exception if the request failed (4xx or 5xx status codes)
        response.raise_for_status()

        # Check if the caller wants base64-encoded output
        if as_base64:
            # Encode the response content to base64 and wrap in BytesIO
            return BytesIO(base64.b64encode(response.content))

        # Create a BytesIO buffer containing the raw response content
        buffer = BytesIO(response.content)

    # Branch 4: No valid source provided
    else:
        # Raise an error if none of the input parameters were provided
        raise ValueError("No valid source provided to resolve binary data!")

    # Return the buffer created in one of the branches above
    return buffer
