from __future__ import annotations

import base64
from enum import Enum
from io import BytesIO
from typing import (
    Annotated,
    Any,
    AsyncGenerator,
    Generator,
    List,
    Literal,
    Optional,
    Union,
)

import filetype
from typing_extensions import Self

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
from serapeum.core.configs.defaults import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from serapeum.core.utils.base import resolve_binary


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
    content: bytes | str | None = None
    path: Optional[FilePath | None] = None
    url: Optional[AnyUrl | str | None] = None


class TextChunk(Chunk):
    type: Literal["text"] = "text"


class Image(Chunk):
    type: Literal["image"] = "image"
    image_mimetype: str | None = None
    detail: str | None = None

    @field_validator("url", mode="after")
    @classmethod
    def url_str_to_any_url(cls, url: str | AnyUrl) -> AnyUrl:
        """Store the url as Anyurl."""
        if isinstance(url, AnyUrl):
            return url
        return AnyUrl(url=url)

    @model_validator(mode="after")
    def to_base64(self) -> Self:
        """Store the image as base64 and guess the mimetype when possible.

        In case the model was built passing image data but without a mimetype,
        we try to guess it using the filetype library. To avoid resource-intense
        operations, we won't load the path or the URL to guess the mimetype.
        """
        if not self.content:
            return self

        try:
            # Check if image is already base64 encoded
            decoded_img = base64.b64decode(self.content)
        except Exception:
            decoded_img = self.content
            # Not base64 - encode it
            self.content = base64.b64encode(self.content)

        self._guess_mimetype(decoded_img)
        return self

    def _guess_mimetype(self, img_data: bytes) -> None:
        if not self.image_mimetype:
            guess = filetype.guess(img_data)
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
    type: Literal["audio"] = "audio"
    format: str | None = None

    @field_validator("url", mode="after")
    @classmethod
    def url_str_to_any_url(cls, url: str | AnyUrl) -> AnyUrl:
        """Store the url as Anyurl."""
        if isinstance(url, AnyUrl):
            return url
        return AnyUrl(url=url)

    @model_validator(mode="after")
    def to_base64(self) -> Self:
        """Store the audio as base64 and guess the mimetype when possible.

        In case the model was built passing audio data but without a mimetype,
        we try to guess it using the filetype library. To avoid resource-intense
        operations, we won't load the path or the URL to guess the mimetype.
        """
        if not self.content:
            return self

        try:
            # Check if audio is already base64 encoded
            decoded_audio = base64.b64decode(self.content)
        except Exception:
            decoded_audio = self.content
            # Not base64 - encode it
            self.content = base64.b64encode(self.content)

        self._guess_format(decoded_audio)

        return self

    def _guess_format(self, audio_data: bytes) -> None:
        if not self.format:
            guess = filetype.guess(audio_data)
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


ChunkType = Annotated[
    Union[TextChunk, Image, Audio], Field(discriminator="type")
]


class Message(BaseModel):
    """Message."""

    role: MessageRole = MessageRole.USER
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    chunks: list[ChunkType] = Field(default_factory=list)

    def __init__(self, /, content: Any | None = None, **data: Any) -> None:
        """Keeps backward compatibility with the old `content` field.

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
        """content

        Returns:
            The cumulative content of all TextBlocks in the message.
        """
        texts = [b.content for b in self.chunks if isinstance(b, TextChunk)]
        result = None if not texts else (texts[0] if len(texts) == 1 else "\n".join(texts))

        return result

    @content.setter
    def content(self, content: str) -> None:
        """content

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
        return f"{self.role.value}: {self.content}"

    @classmethod
    def from_str(
        cls,
        content: str,
        role: Union[MessageRole, str] = MessageRole.USER,
        **kwargs: Any,
    ) -> Self:
        if isinstance(role, str):
            role = MessageRole(role)
        return cls(role=role, chunks=[TextChunk(content=content)], **kwargs)

    def _recursive_serialization(self, value: Any) -> Any:
        if isinstance(value, BaseModel):
            value.model_rebuild()  # ensures all fields are initialized and serializable
            return value.model_dump()  # type: ignore
        if isinstance(value, dict):
            return {
                key: self._recursive_serialization(value)
                for key, value in value.items()
                # if value is not None
            }
        if isinstance(value, list):
            return [self._recursive_serialization(item) for item in value]
        return value

    @field_serializer("additional_kwargs", check_fields=False)
    def serialize_additional_kwargs(self, value: Any, _info: Any) -> Any:
        return self._recursive_serialization(value)


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
    bytes: List[int] = Field(default_factory=list)


class BaseResponse(BaseModel):
    """Base response."""
    raw: Optional[Any] = None
    likelihood_score: Optional[List[List[LikelihoodScore]]] = None
    additional_kwargs: dict = Field(default_factory=dict)
    delta: Optional[str] = None


class ChatResponse(BaseResponse):
    """Chat response."""

    message: Message

    def __str__(self) -> str:
        return str(self.message)

    def to_completion_response(self) -> CompletionResponse:
        """Convert a chat response to a completion response."""
        return CompletionResponse(
            text=self.message.content or "",
            additional_kwargs=self.message.additional_kwargs,
            raw=self.raw,
        )


ChatResponseGen = Generator[ChatResponse, None, None]
ChatResponseAsyncGen = AsyncGenerator[ChatResponse, None]


class CompletionResponse(BaseResponse):
    """
    Completion response.

    Fields:
        text: Text content of the response if not streaming, or if streaming,
            the current extent of streamed text.
        additional_kwargs: Additional information on the response(i.e. token
            counts, function calling information).
        raw: Optional raw JSON that was parsed to populate text, if relevant.
        delta: New text that just streamed in (only relevant when streaming).
    """

    text: str

    def __str__(self) -> str:
        return self.text


CompletionResponseGen = Generator[CompletionResponse, None, None]
CompletionResponseAsyncGen = AsyncGenerator[CompletionResponse, None]


class Metadata(BaseModel):
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
