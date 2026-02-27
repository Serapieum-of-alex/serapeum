from typing import Any, Sequence


from serapeum.core.llms import (
    Message,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
)

from serapeum.core.llms import LLM


class CustomLLM(LLM):
    """
    Simple abstract base class for custom LLMs.

    Subclasses must implement the `__init__`, `_complete`,
        `_stream_complete`, and `metadata` methods.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        assert self.messages_to_prompt is not None

        prompt = self.messages_to_prompt(messages)
        completion_response = self.complete(prompt, formatted=True, **kwargs)
        return completion_response.to_chat_response()

    def stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        assert self.messages_to_prompt is not None

        prompt = self.messages_to_prompt(messages)
        completion_response_gen = self.stream_complete(prompt, formatted=True, **kwargs)
        return CompletionResponse.stream_to_chat_response(completion_response_gen)

    async def achat(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> ChatResponse:
        return self.chat(messages, **kwargs)

    async def astream_chat(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        async def gen() -> ChatResponseAsyncGen:
            for message in self.stream_chat(messages, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self.complete(prompt, formatted=formatted, **kwargs)

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def gen() -> CompletionResponseAsyncGen:
            for message in self.stream_complete(prompt, formatted=formatted, **kwargs):
                yield message

        # NOTE: convert generator to async generator
        return gen()

    @classmethod
    def class_name(cls) -> str:
        return "custom_llm"
