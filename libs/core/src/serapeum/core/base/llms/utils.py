"""Helper adapters to bridge chat and completion interfaces for LLM backends."""
from __future__ import annotations
import os

from typing import Sequence, Callable, Any, Awaitable

from serapeum.core.llms import (
    MessageRole,
    Message,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    ChatResponseGen,
    CompletionResponseAsyncGen,
    ChatResponseAsyncGen
)

__all__ = [
    "get_from_param_or_env",
    "messages_to_prompt",
    "completion_to_chat_decorator",
    "stream_completion_to_chat_decorator",
    "acompletion_to_chat_decorator",
    "astream_completion_to_chat_decorator"
]


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