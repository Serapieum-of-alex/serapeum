"""Mixin for converting chat interface to completion interface.

This module provides a mixin class that implements completion methods by
delegating to chat methods. This is useful for LLM providers that primarily
support a chat interface but need to implement the completion interface as well.
"""

from typing import Any

from serapeum.core.base.llms.types import (
    ChatResponse,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageList,
)

__all__ = ["ChatToCompletionMixin"]


class ChatToCompletionMixin:
    """Mixin that provides completion methods by delegating to chat methods.

    This mixin adds completion method implementations for classes that have
    chat methods. It uses duck typing and assumes the class it's mixed into
    implements the following methods:

    - ``chat(messages, **kwargs) -> ChatResponse``
    - ``stream_chat(messages, **kwargs) -> ChatResponseGen``
    - ``achat(messages, **kwargs) -> ChatResponse``
    - ``astream_chat(messages, **kwargs) -> ChatResponseAsyncGen``

    The mixin provides implementations for:

    - ``complete(prompt, **kwargs) -> CompletionResponse``
    - ``stream_complete(prompt, **kwargs) -> CompletionResponseGen``
    - ``acomplete(prompt, **kwargs) -> CompletionResponse``
    - ``astream_complete(prompt, **kwargs) -> CompletionResponseAsyncGen``

    Examples:
        Basic usage with a custom LLM class:

        ```python
        from serapeum.core.llms import FunctionCallingLLM
        from serapeum.core.llms.abstractions.mixins import ChatToCompletionMixin

        class MyLLM(FunctionCallingLLM, ChatToCompletionMixin):
            def chat(self, messages, **kwargs):
                # Implementation
                return ChatResponse(...)

            def stream_chat(self, messages, **kwargs):
                # Implementation
                yield ChatResponse(...)

            async def achat(self, messages, **kwargs):
                # Implementation
                return ChatResponse(...)

            async def astream_chat(self, messages, **kwargs):
                # Implementation
                yield ChatResponse(...)

            # Completion methods automatically available from mixin
        ```

    See Also:
        - :mod:`serapeum.core.base.llms.utils`: Decorator-based chat-to-completion adapters
        - :class:`serapeum.core.llms.base.LLM`: Base LLM class
    """

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Implement completion by delegating to the chat method.

        Converts the prompt string to a MessageList and calls the chat method,
        then converts the ChatResponse to a CompletionResponse.

        Args:
            prompt: The prompt string to complete.
            formatted: Whether the prompt is already formatted (unused, for compatibility).
            **kwargs: Additional arguments passed to the chat method.

        Returns:
            CompletionResponse: The completion response.

        Examples:
            ```python
            >>> llm = MyLLM()  # Assumes MyLLM uses ChatToCompletionMixin
            >>> response = llm.complete("Hello, world!")
            >>> print(response.text)
            'Hello! How can I assist you today?'
            ```
        """
        messages = MessageList.from_str(prompt)
        chat_response = self.chat(messages, **kwargs)  # type: ignore[attr-defined]
        return chat_response.to_completion_response()

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Implement streaming completion by delegating to the stream_chat method.

        Converts the prompt string to a MessageList and calls the stream_chat method,
        then converts the ChatResponseGen to a CompletionResponseGen.

        Args:
            prompt: The prompt string to complete.
            formatted: Whether the prompt is already formatted (unused, for compatibility).
            **kwargs: Additional arguments passed to the stream_chat method.

        Returns:
            CompletionResponseGen: Generator yielding completion response chunks.

        Examples:
            ```python
            >>> llm = MyLLM()  # Assumes MyLLM uses ChatToCompletionMixin
            >>> for chunk in llm.stream_complete("Tell me a story"):
            ...     print(chunk.delta, end='')
            'Once upon a time...'
            ```
        """
        messages = MessageList.from_str(prompt)
        chat_response_gen = self.stream_chat(messages, **kwargs)  # type: ignore[attr-defined]
        return ChatResponse.stream_to_completion_response(chat_response_gen)

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Implement async completion by delegating to the achat method.

        Converts the prompt string to a MessageList and calls the achat method,
        then converts the ChatResponse to a CompletionResponse.

        Args:
            prompt: The prompt string to complete.
            formatted: Whether the prompt is already formatted (unused, for compatibility).
            **kwargs: Additional arguments passed to the achat method.

        Returns:
            CompletionResponse: The completion response.

        Examples:
            ```python
            >>> llm = MyLLM()  # Assumes MyLLM uses ChatToCompletionMixin
            >>> response = await llm.acomplete("Hello, world!")
            >>> print(response.text)
            'Hello! How can I assist you today?'
            ```
        """
        messages = MessageList.from_str(prompt)
        chat_response = await self.achat(messages, **kwargs)  # type: ignore[attr-defined]
        return chat_response.to_completion_response()

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Implement async streaming completion by delegating to astream_chat.

        Converts the prompt string to a MessageList and calls the astream_chat method,
        then converts the ChatResponseAsyncGen to a CompletionResponseAsyncGen.

        Args:
            prompt: The prompt string to complete.
            formatted: Whether the prompt is already formatted (unused, for compatibility).
            **kwargs: Additional arguments passed to the astream_chat method.

        Returns:
            CompletionResponseAsyncGen: Async generator yielding completion response chunks.

        Examples:
            ```python
            >>> llm = MyLLM()  # Assumes MyLLM uses ChatToCompletionMixin
            >>> async for chunk in await llm.astream_complete("Tell me a story"):
            ...     print(chunk.delta, end='')
            'Once upon a time...'
            ```
        """
        messages = MessageList.from_str(prompt)
        chat_response_gen = await self.astream_chat(messages, **kwargs)  # type: ignore[attr-defined]
        return ChatResponse.astream_to_completion_response(chat_response_gen)
