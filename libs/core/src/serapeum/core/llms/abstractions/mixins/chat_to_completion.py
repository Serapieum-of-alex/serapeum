"""Mixin for converting chat interface to completion interface.

This module provides a mixin class that implements completion methods by
delegating to chat methods. This is useful for LLM providers that primarily
support a chat interface but need to implement the completion interface as well.
"""

from typing import Any, Literal, overload

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

    - ``chat(messages, stream, **kwargs) -> ChatResponse | ChatResponseGen``
    - ``achat(messages, stream, **kwargs) -> ChatResponse | ChatResponseAsyncGen``

    The mixin provides implementations for:

    - ``complete(prompt, *, stream=False, **kwargs)``
    - ``stream_complete(prompt, **kwargs)`` (shim → ``complete(stream=True)``)
    - ``acomplete(prompt, *, stream=False, **kwargs)``
    - ``astream_complete(prompt, **kwargs)`` (shim → ``acomplete(stream=True)``)

    Examples:
        Basic usage with a custom LLM class:

        ```python
        >>> from serapeum.core.llms import FunctionCallingLLM, ChatResponse, Message, MessageRole
        >>> from serapeum.core.llms.abstractions.mixins import ChatToCompletionMixin
        >>>
        >>> class MyLLM(ChatToCompletionMixin, FunctionCallingLLM):
        ...     def chat(self, messages, *, stream=False, **kwargs):
        ...         if stream:
        ...             return self._stream_chat(messages, **kwargs)
        ...         return ChatResponse(
        ...             message=Message(role=MessageRole.ASSISTANT, content="Response")
        ...         )
        ...     def _stream_chat(self, messages, **kwargs):
        ...         yield ChatResponse(
        ...             message=Message(role=MessageRole.ASSISTANT, content="Response"),
        ...             delta="Response"
        ...         )
        ...     async def achat(self, messages, *, stream=False, **kwargs):
        ...         if stream:
        ...             return await self._astream_chat(messages, **kwargs)
        ...         return ChatResponse(
        ...             message=Message(role=MessageRole.ASSISTANT, content="Response")
        ...         )
        ...     async def _astream_chat(self, messages, **kwargs):
        ...         async def gen():
        ...             yield ChatResponse(
        ...                 message=Message(role=MessageRole.ASSISTANT, content="Response"),
        ...                 delta="Response"
        ...             )
        ...         return gen()
        ...     @property
        ...     def metadata(self):
        ...         from serapeum.core.llms import Metadata
        ...         return Metadata(is_chat_model=True, is_function_calling_model=True)
        ...     def _prepare_chat_with_tools(self, tools, **kwargs):
        ...         return {"messages": [], "tools": []}
        >>>
        >>> # Now MyLLM has complete() and other completion methods automatically
        >>> llm = MyLLM(model="example")
        ```

    See Also:
        - :mod:`serapeum.core.base.llms.utils`: Decorator-based chat-to-completion adapters
        - :class:`serapeum.core.llms.base.LLM`: Base LLM class
    """

    @overload
    def complete(
        self,
        prompt: str,
        formatted: bool = ...,
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> CompletionResponse: ...

    @overload
    def complete(
        self,
        prompt: str,
        formatted: bool = ...,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> CompletionResponseGen: ...

    def complete(
        self, prompt: str, formatted: bool = False, *, stream: bool = False, **kwargs: Any
    ) -> CompletionResponse | CompletionResponseGen:
        """Implement completion by delegating to the chat method.

        Converts the prompt string to a MessageList and calls the chat method,
        then converts the response to the appropriate completion type.

        Args:
            prompt: The prompt string to complete.
            formatted: Whether the prompt is already formatted (unused, for compatibility).
            stream: If True, returns a streaming generator instead of a single response.
            **kwargs: Additional arguments passed to the chat method.

        Returns:
            CompletionResponse when stream=False, CompletionResponseGen when stream=True.

        Examples:
            ```python
            >>> # Assuming MyLLM is defined as shown in class docstring
            >>> llm = MyLLM(model="example")
            >>> response = llm.complete("Hello, world!")
            >>> print(response.text)
            'Response'
            >>> for chunk in llm.complete("Tell me a story", stream=True):
            ...     print(chunk.delta, end='')
            'Response'
            ```
        """
        messages = MessageList.from_str(prompt)
        if stream:
            chat_response_gen = self.chat(messages, stream=True, **kwargs)  # type: ignore[attr-defined]
            return ChatResponse.stream_to_completion_response(chat_response_gen)
        chat_response = self.chat(messages, stream=False, **kwargs)  # type: ignore[attr-defined]
        return chat_response.to_completion_response()

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        """Streaming completion — convenience shim for ``complete(stream=True)``.

        Args:
            prompt: The prompt string to complete.
            formatted: Whether the prompt is already formatted (unused, for compatibility).
            **kwargs: Additional arguments passed to the chat method.

        Returns:
            CompletionResponseGen: Generator yielding completion response chunks.

        Examples:
            ```python
            >>> # Assuming MyLLM is defined as shown in class docstring
            >>> llm = MyLLM(model="example")
            >>> for chunk in llm.stream_complete("Tell me a story"):
            ...     print(chunk.delta, end='')
            'Response'
            ```
        """
        return self.complete(prompt, formatted=formatted, stream=True, **kwargs)

    @overload
    async def acomplete(
        self,
        prompt: str,
        formatted: bool = ...,
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> CompletionResponse: ...

    @overload
    async def acomplete(
        self,
        prompt: str,
        formatted: bool = ...,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen: ...

    async def acomplete(
        self, prompt: str, formatted: bool = False, *, stream: bool = False, **kwargs: Any
    ) -> CompletionResponse | CompletionResponseAsyncGen:
        """Implement async completion by delegating to the achat method.

        Converts the prompt string to a MessageList and calls the achat method,
        then converts the response to the appropriate completion type.

        Args:
            prompt: The prompt string to complete.
            formatted: Whether the prompt is already formatted (unused, for compatibility).
            stream: If True, returns an async streaming generator instead of a single response.
            **kwargs: Additional arguments passed to the achat method.

        Returns:
            CompletionResponse when stream=False, CompletionResponseAsyncGen when stream=True.

        Examples:
            ```python
            >>> import asyncio
            >>> # Assuming MyLLM is defined as shown in class docstring
            >>> async def example():
            ...     llm = MyLLM(model="example")
            ...     response = await llm.acomplete("Hello, world!")
            ...     return response.text
            >>> asyncio.run(example())
            'Response'
            >>> async def stream_example():
            ...     llm = MyLLM(model="example")
            ...     stream = await llm.acomplete("Tell me a story", stream=True)
            ...     async for chunk in stream:
            ...         print(chunk.delta, end='')
            >>> asyncio.run(stream_example())  # doctest: +SKIP
            ```
        """
        messages = MessageList.from_str(prompt)
        if stream:
            chat_response_gen = await self.achat(messages, stream=True, **kwargs)  # type: ignore[attr-defined]
            return ChatResponse.astream_to_completion_response(chat_response_gen)
        chat_response = await self.achat(messages, stream=False, **kwargs)  # type: ignore[attr-defined]
        return chat_response.to_completion_response()

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion — convenience shim for ``acomplete(stream=True)``.

        Args:
            prompt: The prompt string to complete.
            formatted: Whether the prompt is already formatted (unused, for compatibility).
            **kwargs: Additional arguments passed to the achat method.

        Returns:
            CompletionResponseAsyncGen: Async generator yielding completion response chunks.

        Examples:
            ```python
            >>> import asyncio
            >>> # Assuming MyLLM is defined as shown in class docstring
            >>> async def example():
            ...     llm = MyLLM(model="example")
            ...     stream = await llm.astream_complete("Tell me a story")
            ...     async for chunk in stream:
            ...         print(chunk.delta, end='')
            >>> asyncio.run(example())
            'Response'
            ```
        """
        return await self.acomplete(prompt, formatted=formatted, stream=True, **kwargs)
