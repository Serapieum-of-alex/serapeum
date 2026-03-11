"""Core data models for LLM interactions (messages, chunks, responses, metadata)."""

from __future__ import annotations

import base64
from collections.abc import Sequence as ABCSequence
from enum import Enum
from io import BytesIO, IOBase
from binascii import Error as BinasciiError
from pathlib import Path
from typing import Annotated, Any, AsyncGenerator, Generator, Iterator, Literal
from urllib.parse import urlparse

import requests
from filetype import guess as filetype_guess
from filetype import get_type
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    field_serializer,
    field_validator,
    model_validator,
    ValidationError
)
from typing_extensions import Self

from serapeum.core.configs.defaults import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from serapeum.core.utils.schemas import parse_partial_json

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
        result = url if isinstance(url, AnyUrl) else AnyUrl(url=url)
        return result

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

        if self.content:
            decoded_img: bytes | None = None
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
                    content_bytes = None

                if content_bytes is not None:
                    decoded_img = content_bytes
                    self.content = base64.b64encode(content_bytes)

            if decoded_img is not None:
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

    def as_data_uri(self) -> str:
        """Return a ``data:<mimetype>;base64,<data>`` URI for this image."""
        img_bytes = self.resolve_image(as_base64=True).read()
        img_str = img_bytes.decode("utf-8")
        return f"data:{self.image_mimetype};base64,{img_str}"


class Audio(Chunk):
    """Audio chunk supporting inline bytes, file paths, or URLs."""

    type: Literal["audio"] = "audio"
    format: str | None = None

    @field_validator("url", mode="after")  # type: ignore[misc]
    @classmethod
    def url_str_to_any_url(cls, url: str | AnyUrl) -> AnyUrl:
        """Store the url as Anyurl."""
        result = url if isinstance(url, AnyUrl) else AnyUrl(url=url)
        return result

    @model_validator(mode="after")  # type: ignore[misc]
    def to_base64(self) -> Self:
        """Store the audio as base64 and guess the mimetype when possible.

        In case the model was built passing audio data but without a mimetype,
        we try to guess it using the filetype library. To avoid resource-intense
        operations, we won't load the path or the URL to guess the mimetype.
        """
        if self.content:
            decoded_audio: bytes | None = None
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
                    content_bytes = None

                if content_bytes is not None:
                    decoded_audio = content_bytes
                    self.content = base64.b64encode(content_bytes)

            if decoded_audio is not None:
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


class DocumentBlock(BaseModel):
    """A representation of a document to directly pass to the LLM."""

    type: Literal["document"] = "document"
    data: bytes | IOBase | None = None
    path: FilePath | str | None = None
    url: str | None = None
    title: str | None = None
    document_mimetype: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def document_validation(self) -> Self:
        self.document_mimetype = self.document_mimetype or self._guess_mimetype()

        if not self.title:
            self.title = "input_document"

        if self.data and isinstance(self.data, bytes):
            try:
                base64.b64decode(self.data, validate=True)
            except BinasciiError:
                self.data = base64.b64encode(self.data)

        return self

    @field_serializer("data")
    def serialize_data(self, data: bytes | IOBase | None) -> bytes | None:
        """Serialize the data field."""
        if isinstance(data, bytes):
            result = data
        elif isinstance(data, IOBase):
            data.seek(0)
            result = data.read()
        else:
            result = None
        return result

    def resolve_document(self) -> IOBase:
        """
        Resolve a document such that it is represented by a BufferIO object.
        """
        data_buffer = (
            self.data
            if isinstance(self.data, IOBase)
            else resolve_binary(
                raw_bytes=self.data,
                path=self.path,
                url=str(self.url) if self.url else None,
                as_base64=False,
            )
        )
        # Check size by seeking to end and getting position
        data_buffer.seek(0, 2)  # Seek to end
        size = data_buffer.tell()
        data_buffer.seek(0)  # Reset to beginning

        if size == 0:
            raise ValueError("resolve_document returned zero bytes")
        return data_buffer

    def _get_b64_string(self, data_buffer: IOBase) -> str:
        """
        Get base64-encoded string from a IOBase buffer.
        """
        data = data_buffer.read()
        return base64.b64encode(data).decode("utf-8")

    def _get_b64_bytes(self, data_buffer: IOBase) -> bytes:
        """
        Get base64-encoded bytes from a IOBase buffer.
        """
        data = data_buffer.read()
        return base64.b64encode(data)

    def guess_format(self) -> str | None:
        path = self.path or self.url
        result = Path(str(path)).suffix.replace(".", "") if path else None
        return result

    def _guess_mimetype(self) -> str | None:
        if self.data:
            guess = filetype_guess(self.data)
            result = str(guess.mime) if guess else None
        else:
            suffix = self.guess_format()
            if suffix:
                guess = get_type(ext=suffix)
                result = str(guess.mime) if guess else None
            else:
                result = None
        return result

    def as_base64(self) -> tuple[str, str]:
        """Return ``(base64_string, mimetype)`` for this document."""
        if not self.data:
            file_buffer = self.resolve_document()
            b64_string = self._get_b64_string(file_buffer)
        else:
            b64_string = self.data.decode("utf-8")
        mimetype = self._guess_mimetype()
        return b64_string, mimetype


class ToolCallArguments(BaseModel):
    """Represents a concrete tool choice and its arguments.

    This Pydantic model captures the selection of a tool (by id and name) and the
    keyword arguments that should be passed to it at execution time. It is typically
    produced by an LLM during function-calling or constructed programmatically before
    dispatching to an executor.

    Notes:
    - The ``tool_kwargs`` field uses a validator that replaces non-dictionary inputs
      with an empty dictionary instead of raising a validation error. This keeps
      downstream execution resilient to imperfect upstream outputs.

    Args:
        tool_id (str):
            An identifier for the tool call (e.g., provider-specific id).
        tool_name (str):
            The name of the tool to execute.
        tool_kwargs (dict[str, Any]):
            Keyword arguments for the tool. If a non-dict value is supplied, it is coerced to an empty dict by
            validation.

    Returns:
        ToolCallArguments: A validated instance describing the tool call.

    Raises:
        pydantic.ValidationError: If required fields are missing or have incompatible
            types that cannot be coerced. Note that ``tool_kwargs`` specifically
            coerces non-dict values to ``{}`` instead of raising.

    Examples:
        - Typical usage: construct a selection and access its fields
            ```python
            >>> from serapeum.core.base.llms.types import ToolCallArguments
            >>> sel = ToolCallArguments(tool_id="abc123", tool_name="echo", tool_kwargs={"text": "hi"})
            >>> (sel.tool_name, sel.tool_kwargs["text"])
            ('echo', 'hi')

            ```

        - Non-dict ``tool_kwargs`` are replaced with an empty dict
            ```python
            >>> from serapeum.core.base.llms.types import ToolCallArguments
            >>> sel = ToolCallArguments(tool_id="id-1", tool_name="echo", tool_kwargs="not-a-dict")
            >>> sel.tool_kwargs
            {}

            ```

        - Missing required fields raise a ValidationError
            ```python
            >>> from pydantic import ValidationError
            >>> from serapeum.core.base.llms.types import ToolCallArguments
            >>> try:
            ...     ToolCallArguments(tool_id="only-id", tool_kwargs={})  # missing tool_name
            ... except ValidationError as e:
            ...     print(e.error_count(), "validation error")
            1 validation error

            ```

    See Also:
        - serapeum.core.tools.invoke.ToolExecutor.execute_with_selection: Execute a selection synchronously.
        - serapeum.core.tools.invoke.ToolExecutor.execute_async_with_selection: Execute a selection asynchronously.
    """

    tool_id: str = Field(description="Tool ID to select.")
    tool_name: str = Field(description="Tool name to select.")
    tool_kwargs: dict[str, Any] = Field(description="Keyword arguments for the tool.")

    @field_validator("tool_kwargs", mode="wrap")
    @classmethod
    def ignore_non_dict_arguments(cls, v: Any, handler: Any) -> dict[str, Any]:
        try:
            return handler(v)
        except ValidationError:
            return handler({})


class ToolCallBlock(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str | None = Field(
        default=None, description="ID of the tool call, if provided"
    )
    tool_name: str = Field(description="Name of the called tool")
    tool_kwargs: dict[str, Any] | str = Field(
        default_factory=dict,  # type: ignore
        description="Arguments provided to the tool, if available",
    )

    @property
    def parsed_kwargs(self) -> dict[str, Any]:
        """Return tool_kwargs as a dict, parsing JSON strings on demand."""
        if isinstance(self.tool_kwargs, dict):
            result = self.tool_kwargs
        else:
            try:
                result = parse_partial_json(self.tool_kwargs)
            except (ValueError, TypeError):
                result = {}
        return result

    def get_arguments(self) -> ToolCallArguments:
        return ToolCallArguments(
            tool_id=self.tool_call_id or "",
            tool_name=self.tool_name,
            tool_kwargs=self.parsed_kwargs,
        )

class ThinkingBlock(BaseModel):
    """A representation of the content streamed from reasoning/thinking processes by LLMs"""

    type: Literal["thinking"] = "thinking"
    content: str | None = Field(
        description="Content of the reasoning/thinking process, if available",
        default=None,
    )
    num_tokens: int | None = Field(
        description="Number of token used for reasoning/thinking, if available",
        default=None,
    )
    additional_information: dict[str, Any] = Field(
        description="Additional information related to the thinking/reasoning process, if available",
        default_factory=dict,
    )


ChunkType = Annotated[
    TextChunk | Image | Audio | DocumentBlock | ToolCallBlock | ThinkingBlock,
    Field(discriminator="type")
]


class Message(BaseModel):
    """Message."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole = MessageRole.USER
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    chunks: list[ChunkType] = Field(default_factory=list)

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

    @property
    def tool_calls(self) -> list[ToolCallBlock]:
        """Tool calls contained in this message.

        Returns:
            All ToolCallBlock entries from chunks, in order.
        """
        return [b for b in self.chunks if isinstance(b, ToolCallBlock)]

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


    def _recursive_serialization(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            value.model_rebuild()  # ensures all fields are initialized and serializable
            result = value.model_dump()
        elif isinstance(value, dict):
            result = {
                key: self._recursive_serialization(value)
                for key, value in value.items()
            }
        elif isinstance(value, list):
            result = [self._recursive_serialization(item) for item in value]
        else:
            result = value
        return result

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
        result = (
            MessageList(messages=self.messages[index])
            if isinstance(index, slice)
            else self.messages[index]
        )
        return result

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
    def from_str(cls, prompt: str) -> "MessageList":
        """Create from a string prompt."""
        return cls(messages=[Message(chunks=[TextChunk(content=prompt)])])


class LogProb(BaseModel):
    """LikelihoodScore of a token.

    The log probability information for a token generated by the model.

    Attributes:
        token(str):
            the actual text token (string).
        logprob(float):
            The logarithmic probability score (float) indicating how likely the model thought this token was the
            correct next token.
        bytes(list[int]):
            The byte representation of the token as a list of integers
    """

    token: str = Field(default_factory=str)
    logprob: float = Field(default_factory=float)
    bytes: list[int] = Field(default_factory=list)


class BaseResponse(BaseModel):
    """Base response."""

    raw: Any | None = None
    likelihood_score: list[list[LogProb]] | None = None
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

    def force_single_tool_call(self) -> None:
        """Mutate a response to include at most a single tool call.

        Ollama may return multiple tool calls within a single assistant message. Some
        consumers require a single call at a time. This helper trims the list to the
        first occurrence in-place.

        Args:
            response (ChatResponse):
                Parsed chat response whose ``message.chunks`` may contain multiple
                ToolCallBlock entries.

        Returns:
            None: The function mutates ``response`` and returns nothing.

        Examples:
            - Truncate multiple tool calls to one
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, ChatResponse, ToolCallBlock
                >>> r = ChatResponse(message=Message(
                ...     role=MessageRole.ASSISTANT,
                ...     chunks=[
                ...         ToolCallBlock(tool_name="a", tool_kwargs={}),
                ...         ToolCallBlock(tool_name="b", tool_kwargs={}),
                ...     ],
                ... ))
                >>> r.force_single_tool_call()
                >>> len(r.message.tool_calls)
                1

                ```
            - No-op when there are no tool calls
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, ChatResponse
                >>> r = ChatResponse(message=Message(role=MessageRole.ASSISTANT, chunks=[TextChunk(content="hi")]))
                >>> force_single_tool_call(r)
                >>> r.message.tool_calls
                []

                ```
            - Single tool call is left as-is
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, ChatResponse, ToolCallBlock
                >>> r = ChatResponse(message=Message(
                ...     role=MessageRole.ASSISTANT,
                ...     chunks=[ToolCallBlock(tool_name="a", tool_kwargs={})],
                ... ))
                >>> force_single_tool_call(r)
                >>> r.message.tool_calls[0].tool_name
                'a'

                ```
        """
        tool_calls = [
            block for block in self.message.chunks if isinstance(block, ToolCallBlock)
        ]
        if len(tool_calls) > 1:
            self.message.chunks = [
                  block
                  for block in self.message.chunks
                  if not isinstance(block, ToolCallBlock)
              ] + [tool_calls[0]]


ChatResponseGen = Generator[ChatResponse, None, None]
ChatResponseAsyncGen = AsyncGenerator[ChatResponse, None]

CompletionResponseGen = Generator["CompletionResponse", None, None]
CompletionResponseAsyncGen = AsyncGenerator["CompletionResponse", None]


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
                chunks=[TextChunk(content=self.text)],
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
                        chunks=[TextChunk(content=response.text)],
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
                        chunks=[TextChunk(content=response.text)],
                        additional_kwargs=response.additional_kwargs,
                    ),
                    delta=response.delta,
                    raw=response.raw,
                )

        return gen()


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
    # Each branch resolves to raw bytes; as_base64 encoding is applied once at the end.
    if raw_bytes is not None:
        try:
            resolved = base64.b64decode(raw_bytes)
        except Exception:
            resolved = raw_bytes

    elif path is not None:
        path = Path(path) if isinstance(path, str) else path
        resolved = path.read_bytes()

    elif url is not None:
        parsed_url = urlparse(url)

        if parsed_url.scheme == "data":
            data_part = parsed_url.path
            if "," not in data_part:
                raise ValueError("Invalid data URL format: missing comma separator")

            metadata, url_data = data_part.split(",", 1)
            is_base64_encoded = metadata.endswith(";base64")

            if is_base64_encoded:
                resolved = base64.b64decode(url_data)
            else:
                resolved = url_data.encode("utf-8")
        else:
            # HTTP(S) URLs
            response = requests.get(url, headers={})
            response.raise_for_status()
            resolved = response.content

    else:
        raise ValueError("No valid source provided to resolve binary data!")

    buffer = BytesIO(base64.b64encode(resolved) if as_base64 else resolved)
    return buffer



ContentBlock = Annotated[
    TextChunk | Image | Audio | ThinkingBlock | ToolCallBlock,
    Field(discriminator="block_type"),
]
