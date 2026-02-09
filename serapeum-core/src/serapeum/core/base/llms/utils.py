"""Helper adapters to bridge chat and completion interfaces for LLM backends."""

import os
from typing import Any, Awaitable, Callable, Sequence

from serapeum.core.base.llms.models import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageList,
    MessageRole,
    Message
)

__all__ = [
    "chat_to_completion_decorator",
    "stream_chat_to_completion_decorator",
    "achat_to_completion_decorator",
    "astream_chat_to_completion_decorator",
    "get_from_param_or_env",
    "acompletion_to_chat_decorator",
    "astream_completion_to_chat_decorator",
    "completion_to_chat_decorator",
    "stream_completion_to_chat_decorator",
]


def chat_to_completion_decorator(
    func: Callable[..., ChatResponse],
) -> Callable[..., CompletionResponse]:
    """Convert a chat function to a completion function."""

    def wrapper(prompt: str, **kwargs: Any) -> CompletionResponse:
        # normalize input
        messages = MessageList.from_str(prompt)
        chat_response = func(messages, **kwargs)
        # normalize output
        return chat_response.to_completion_response()

    return wrapper


def stream_chat_to_completion_decorator(
    func: Callable[..., ChatResponseGen],
) -> Callable[..., CompletionResponseGen]:
    """Convert a streaming chat function to a completion function."""

    def wrapper(prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # normalize input
        messages = MessageList.from_str(prompt)
        chat_response_gen = func(messages, **kwargs)
        # normalize output
        return ChatResponse.stream_to_completion_response(chat_response_gen)

    return wrapper


def achat_to_completion_decorator(
    func: Callable[..., Awaitable[ChatResponse]],
) -> Callable[..., Awaitable[CompletionResponse]]:
    """Convert an async chat function to a completion function."""

    async def wrapper(prompt: str, **kwargs: Any) -> CompletionResponse:
        # normalize input
        messages = MessageList.from_str(prompt)
        chat_response = await func(messages, **kwargs)
        # normalize output
        return chat_response.to_completion_response()

    return wrapper


def astream_chat_to_completion_decorator(
    func: Callable[..., Awaitable[ChatResponseAsyncGen]],
) -> Callable[..., Awaitable[CompletionResponseAsyncGen]]:
    """Convert an async streaming chat function to a completion function."""

    async def wrapper(prompt: str, **kwargs: Any) -> CompletionResponseAsyncGen:
        # normalize input
        messages = MessageList.from_str(prompt)
        chat_response = await func(messages, **kwargs)
        # normalize output
        return ChatResponse.astream_to_completion_response(chat_response)

    return wrapper


def messages_to_prompt(messages: Sequence[Message]) -> str:
    """Convert messages to a prompt string."""
    string_messages = []
    for message in messages:
        role = message.role
        content = message.content
        string_message = f"{role.value}: {content}"

        additional_kwargs = message.additional_kwargs
        if additional_kwargs:
            string_message += f"\n{additional_kwargs}"
        string_messages.append(string_message)

    string_messages.append(f"{MessageRole.ASSISTANT.value}: ")
    return "\n".join(string_messages)


def completion_to_chat_decorator(
    func: Callable[..., CompletionResponse],
) -> Callable[..., ChatResponse]:
    """Convert a completion function to a chat function."""

    def wrapper(messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        # normalize input
        prompt = messages_to_prompt(messages)
        completion_response = func(prompt, **kwargs)
        # normalize output
        return completion_response.to_chat_response()

    return wrapper


def acompletion_to_chat_decorator(
    func: Callable[..., Awaitable[CompletionResponse]],
) -> Callable[..., Awaitable[ChatResponse]]:
    """Convert a completion function to a chat function."""

    async def wrapper(messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        # normalize input
        prompt = messages_to_prompt(messages)
        completion_response = await func(prompt, **kwargs)
        # normalize output
        return completion_response.to_chat_response()

    return wrapper


def astream_completion_to_chat_decorator(
    func: Callable[..., Awaitable[CompletionResponseAsyncGen]],
) -> Callable[..., Awaitable[ChatResponseAsyncGen]]:
    """Convert a completion function to a chat function."""

    async def wrapper(
        messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        # normalize input
        prompt = messages_to_prompt(messages)
        completion_response = await func(prompt, **kwargs)
        # normalize output
        return CompletionResponse.astream_to_chat_response(completion_response)

    return wrapper


def stream_completion_to_chat_decorator(
        func: Callable[..., CompletionResponseGen],
) -> Callable[..., ChatResponseGen]:
    """Convert a completion function to a chat function."""

    def wrapper(messages: Sequence[Message], **kwargs: Any) -> ChatResponseGen:
        # normalize input
        prompt = messages_to_prompt(messages)
        completion_response = func(prompt, **kwargs)
        # normalize output
        return CompletionResponse.stream_to_chat_response(completion_response)

    return wrapper


def get_from_param_or_env(
    key: str,
    param: str | None = None,
    env_key: str | None = None,
    default: str | None = None,
) -> str:
    """Return value from explicit param, environment, or default.

    Resolution order: ``param`` > environment variable ``env_key`` > ``default``.
    Raises ``ValueError`` when no value can be resolved.
    """
    if param is not None:
        return param
    elif env_key and env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {key}, please add an environment variable"
            f" `{env_key}` which contains it, or pass"
            f"  `{key}` as a named parameter."
        )
