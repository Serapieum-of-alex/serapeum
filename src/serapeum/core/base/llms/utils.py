import os
from typing import Any, Awaitable, Callable, Optional

from serapeum.core.base.llms.models import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageList,
)

__all__ = [
    "chat_to_completion_decorator",
    "stream_chat_to_completion_decorator",
    "achat_to_completion_decorator",
    "astream_chat_to_completion_decorator",
    "get_from_param_or_env",
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


def get_from_param_or_env(
    key: str,
    param: Optional[str] = None,
    env_key: Optional[str] = None,
    default: Optional[str] = None,
) -> str:
    """Get a value from a param or an environment variable."""
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
