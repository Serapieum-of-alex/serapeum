import os
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Iterator, Union
from collections.abc import Sequence as ABCSequence

from serapeum.core.base.llms.models import (
    Message,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)


class MessageList(ABCSequence):
    """A collection of Message objects with helper methods."""

    def __init__(self, messages: Sequence[Message] = None):
        self._messages: List[Message] = list(messages) if messages else []

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __getitem__(self, index: Union[int, slice]) -> Union[Message, "MessageList"]:
        if isinstance(index, slice):
            return MessageList(self._messages[index])
        return self._messages[index]

    def to_prompt(self) -> str:
        """Convert messages to a prompt string."""
        string_messages = []
        for message in self._messages:
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
        return MessageList([m for m in self._messages if m.role == role])

    def append(self, message: Message) -> None:
        """Add a message to the collection."""
        self._messages.append(message)

    @classmethod
    def from_list(cls, messages: List[Message]) -> "MessageList":
        """Create from a standard list."""
        return cls(messages)

    @classmethod
    def from_str(cls, prompt: str) -> "MessageList":
        """Create from a string prompt."""
        return cls([Message(role=MessageRole.USER, content=prompt)])


def stream_chat_response_to_completion_response(
    chat_response_gen: ChatResponseGen,
) -> CompletionResponseGen:
    """Convert a stream chat response to a completion response."""

    def gen() -> CompletionResponseGen:
        for response in chat_response_gen:
            yield response.to_completion_response()

    return gen()


def chat_to_completion_decorator(
    func: Callable[..., ChatResponse]
) -> Callable[..., CompletionResponse]:
    """Convert a chat function to a completion function."""

    def wrapper(prompt: str, **kwargs: Any) -> CompletionResponse:
        # normalize input
        messages = list(MessageList.from_str(prompt))
        chat_response = func(messages, **kwargs)
        # normalize output
        return chat_response.to_completion_response()

    return wrapper


def stream_chat_to_completion_decorator(
    func: Callable[..., ChatResponseGen]
) -> Callable[..., CompletionResponseGen]:
    """Convert a chat function to a completion function."""

    def wrapper(prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # normalize input
        messages = list(MessageList.from_str(prompt))
        chat_response = func(messages, **kwargs)
        # normalize output
        return stream_chat_response_to_completion_response(chat_response)

    return wrapper


def achat_to_completion_decorator(
    func: Callable[..., Awaitable[ChatResponse]]
) -> Callable[..., Awaitable[CompletionResponse]]:
    """Convert a chat function to a completion function."""

    async def wrapper(prompt: str, **kwargs: Any) -> CompletionResponse:
        # normalize input
        messages = list(MessageList.from_str(prompt))
        chat_response = await func(messages, **kwargs)
        # normalize output
        return chat_response.to_completion_response()

    return wrapper


def astream_chat_to_completion_decorator(
    func: Callable[..., Awaitable[ChatResponseAsyncGen]]
) -> Callable[..., Awaitable[CompletionResponseAsyncGen]]:
    """Convert a chat function to a completion function."""

    async def wrapper(prompt: str, **kwargs: Any) -> CompletionResponseAsyncGen:
        # normalize input
        messages = list(MessageList.from_str(prompt))
        chat_response = await func(messages, **kwargs)
        # normalize output
        return astream_chat_response_to_completion_response(chat_response)

    return wrapper


def astream_chat_response_to_completion_response(
    chat_response_gen: ChatResponseAsyncGen,
) -> CompletionResponseAsyncGen:
    """Convert a stream chat response to a completion response."""

    async def gen() -> CompletionResponseAsyncGen:
        async for response in chat_response_gen:
            yield CompletionResponse(
                text=response.message.content or "",
                additional_kwargs=response.message.additional_kwargs,
                delta=response.delta,
                raw=response.raw,
            )

    return gen()


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
