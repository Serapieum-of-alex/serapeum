from abc import abstractmethod
from typing import (
    Any,
    List,
    Sequence,
)

from serapeum.core.base.llms.models import (
    Message,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    Metadata,
    TextChunk,
)

from pydantic import ConfigDict
from serapeum.core.models import SerializableModel


class BaseLLM(SerializableModel):
    """BaseLLM interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    @abstractmethod
    def metadata(self) -> Metadata:
        """LLM metadata.

        Returns:
            Metadata: LLM metadata containing various information about the LLM.
        """

    def convert_chat_messages(self, messages: Sequence[Message]) -> List[Any]:
        """Convert chat messages to an LLM specific message format."""
        converted_messages = []
        for message in messages:
            if isinstance(message.content, str):
                converted_messages.append(message)
            elif isinstance(message.content, List):
                content_string = ""
                for block in message.content:
                    if isinstance(block, TextChunk):
                        content_string += block.content
                    else:
                        raise ValueError("LLM only supports text inputs")
                message.content = content_string
                converted_messages.append(message)
            else:
                raise ValueError(f"Invalid message content: {message.content!s}")

        return converted_messages

    @abstractmethod
    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        ...
    @abstractmethod
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        ...

    @abstractmethod
    def stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        ...

    @abstractmethod
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        ...

    @abstractmethod
    async def achat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponse:
        ...

    @abstractmethod
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        ...

    @abstractmethod
    async def astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        ...
    @abstractmethod
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        ...
