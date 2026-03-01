"""Mixin for converting completion interface to chat interface.

This module provides a mixin class that implements chat methods by
delegating to completion methods. This is useful for LLM providers that
primarily support a completion interface (e.g. llama.cpp) but need to
implement the full chat interface as well.
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal, Sequence, overload

from serapeum.core.base.llms.types import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    Message,
)

__all__ = ["CompletionToChatMixin"]


class CompletionToChatMixin:
    """Mixin that implements chat methods by delegating to completion methods.

    This is the mirror of :class:`ChatToCompletionMixin`. Use this when your
    provider natively exposes a completion API (prompt-in/text-out) but needs
    to satisfy the full :class:`~serapeum.core.llms.base.LLM` interface.

    The class it is mixed into must implement:

    - ``complete(prompt, formatted=False, *, stream=False, **kwargs)`` → ``CompletionResponse | CompletionResponseGen``
    - ``messages_to_prompt`` callable attribute (provided by :class:`~serapeum.core.llms.base.LLM`)

    The mixin provides implementations for:

    - ``chat(messages, *, stream=False, **kwargs)``
    - ``achat(messages, *, stream=False, **kwargs)``
    - ``acomplete(prompt, formatted=False, *, stream=False, **kwargs)`` — async shim over ``complete``

    Examples:
        Basic usage with a completion-first provider:

        ```python
        >>> from serapeum.core.llms import LLM, CompletionResponse, Metadata
        >>> from serapeum.core.llms.abstractions.mixins import CompletionToChatMixin
        >>>
        >>> class MyCompletionLLM(CompletionToChatMixin, LLM):
        ...     @property
        ...     def metadata(self):
        ...         return Metadata(is_chat_model=False)
        ...     def complete(self, prompt, formatted=False, *, stream=False, **kwargs):
        ...         if stream:
        ...             def gen():
        ...                 yield CompletionResponse(text="Hello", delta="Hello")
        ...             return gen()
        ...         return CompletionResponse(text="Hello")
        ...
        >>> llm = MyCompletionLLM(model="example")
        ```

    See Also:
        - :class:`serapeum.core.llms.abstractions.mixins.ChatToCompletionMixin`: Reverse direction
    """

    @overload
    def chat(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> ChatResponse: ...

    @overload
    def chat(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> ChatResponseGen: ...

    def chat(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | ChatResponseGen:
        """Implement chat by delegating to the completion method.

        Converts messages to a prompt string via ``messages_to_prompt``, then
        calls ``complete()`` or ``stream_complete()`` and converts the result
        to the appropriate chat response type.

        Args:
            messages: Sequence of messages to send.
            stream: If True, returns a streaming generator.
            **kwargs: Additional arguments forwarded to the completion method.

        Returns:
            ChatResponse when stream=False, ChatResponseGen when stream=True.
        """
        prompt = self.messages_to_prompt(messages)  # type: ignore[attr-defined]
        if stream:
            completion_gen: CompletionResponseGen = self.complete(  # type: ignore[attr-defined]
                prompt, formatted=True, stream=True, **kwargs
            )
            result: ChatResponse | ChatResponseGen = CompletionResponse.stream_to_chat_response(completion_gen)
        else:
            completion_response: CompletionResponse = self.complete(  # type: ignore[attr-defined]
                prompt, formatted=True, **kwargs
            )
            result = completion_response.to_chat_response()
        return result

    @overload
    async def achat(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> ChatResponse: ...

    @overload
    async def achat(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen: ...

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | ChatResponseAsyncGen:
        """Implement async chat by delegating to the async completion method.

        Converts messages to a prompt string via ``messages_to_prompt``, then
        calls ``acomplete()`` or ``astream_complete()`` and converts the
        result to the appropriate chat response type.

        Args:
            messages: Sequence of messages to send.
            stream: If True, returns an async streaming generator.
            **kwargs: Additional arguments forwarded to the async completion method.

        Returns:
            ChatResponse when stream=False, ChatResponseAsyncGen when stream=True.
        """
        prompt = self.messages_to_prompt(messages)  # type: ignore[attr-defined]
        if stream:
            completion_gen = await self.acomplete(prompt, formatted=True, stream=True, **kwargs)
            result: ChatResponse | ChatResponseAsyncGen = CompletionResponse.astream_to_chat_response(completion_gen)
        else:
            completion_response = await self.acomplete(prompt, formatted=True, **kwargs)
            result = completion_response.to_chat_response()
        return result

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
        """Async completion — offloads synchronous ``complete`` to a thread pool.

        Uses :func:`asyncio.to_thread` so that CPU/GPU-bound inference in the
        synchronous ``complete`` method never blocks the running event loop.

        Args:
            prompt: The prompt string to complete.
            formatted: Whether the prompt is already formatted.
            stream: If True, returns an async streaming generator.
            **kwargs: Additional arguments passed to the completion method.

        Returns:
            CompletionResponse when stream=False, CompletionResponseAsyncGen when stream=True.
        """
        if stream:
            chunks: list[CompletionResponse] = await asyncio.to_thread(
                lambda: list(
                    self.complete(  # type: ignore[attr-defined]
                        prompt, formatted=formatted, stream=True, **kwargs
                    )
                )
            )

            async def gen() -> CompletionResponseAsyncGen:
                for chunk in chunks:
                    yield chunk

            result: CompletionResponse | CompletionResponseAsyncGen = gen()
        else:
            result = await asyncio.to_thread(
                self.complete, prompt, formatted, stream=False, **kwargs  # type: ignore[attr-defined]
            )
        return result