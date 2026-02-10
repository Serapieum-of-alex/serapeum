"""Abstract base interface for concrete LLM backends.

This module defines the BaseLLM protocol that concrete providers must
implement to support chat/completion, streaming, and async variants.
"""

from abc import abstractmethod
from typing import Any, Sequence

from pydantic import ConfigDict

from serapeum.core.base.llms.types import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    Message,
    Metadata,
    TextChunk,
)
from serapeum.core.types import SerializableModel


class BaseLLM(SerializableModel):
    """BaseLLM interface."""

    # Allow subclasses/tests to attach auxiliary attributes (e.g., test doubles)
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    @property
    @abstractmethod
    def metadata(self) -> Metadata:
        """LLM metadata.

        Returns:
            Metadata: LLM metadata containing various information about the LLM.
        """

    def convert_chat_messages(self, messages: Sequence[Message]) -> list[Any]:
        """Convert chat messages to an LLM specific message format."""
        converted_messages = []
        for message in messages:
            if isinstance(message.content, str):
                converted_messages.append(message)
            elif isinstance(message.content, list):
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
        pass

    @abstractmethod
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        pass

    @abstractmethod
    def stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        pass

    @abstractmethod
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        pass

    @abstractmethod
    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        pass

    @abstractmethod
    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        pass

    @abstractmethod
    async def astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        pass

    @abstractmethod
    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        pass
