from typing import Any, List, Optional, Sequence

import pytest
from pydantic import BaseModel

from serapeum.core.llm.base import (
    LLM,
    astream_response_to_tokens,
    default_completion_to_prompt,
    stream_response_to_tokens,
)
from serapeum.core.base.llms.models import (
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    Message,
    MessageRole,
    Metadata,
)
from serapeum.core.prompts import PromptTemplate, ChatPromptTemplate
from serapeum.core.output_parsers.models import BaseOutputParser


# --------------------
# Test doubles & helpers
# --------------------

class UpperParser(BaseOutputParser):
    """Simple parser that uppercases text and messages.

    - parse: returns uppercased string
    - format: uppercases prompt string
    - format_messages: uppercases message contents (in-place)
    """

    def parse(self, output: str) -> str:
        return (output or "").upper()

    def format(self, query: str) -> str:
        return (query or "").upper()

    def format_messages(self, messages: List[Message]) -> List[Message]:
        for m in messages:
            if m.content is not None:
                m.content = m.content.upper()  # type: ignore
        return messages


class FailingParser(BaseOutputParser):
    """Parser that always raises during parse to test error propagation."""

    def parse(self, output: str) -> str:
        raise ValueError("invalid output")


class CompletionStubLLM(LLM):
    """LLM stub that implements completion endpoints (non-chat model)."""

    @property
    def metadata(self) -> Metadata:
        return Metadata.model_construct(is_chat_model=False)

    # -- sync
    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError()

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        # return uppercase to make transformations visible
        return CompletionResponse(text=prompt.upper(), delta=prompt.upper())

    def stream_chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponseGen:
        raise NotImplementedError()

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        def gen() -> CompletionResponseGen:
            # yield delta pieces deterministically
            yield CompletionResponse(text=prompt, delta=prompt[:1])
            yield CompletionResponse(text=prompt, delta=prompt[1:])
        return gen()

    # -- async
    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError()

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=prompt[::-1], delta=prompt[::-1])

    async def astream_chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponseAsyncGen:
        raise NotImplementedError()

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        async def agen() -> CompletionResponseAsyncGen:
            yield CompletionResponse(text=prompt, delta=prompt[:1])
            yield CompletionResponse(text=prompt, delta=prompt[1:])
        return agen()


class ChatStubLLM(LLM):
    """LLM stub that implements chat endpoints (chat model)."""

    @property
    def metadata(self) -> Metadata:
        return Metadata.model_construct(is_chat_model=True)

    # -- sync
    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=Message(content="pong", role=MessageRole.ASSISTANT))

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError()

    def stream_chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponseGen:
        def gen() -> ChatResponseGen:
            yield ChatResponse(message=Message(content="ok", role=MessageRole.ASSISTANT), delta="o")
            yield ChatResponse(message=Message(content="ok", role=MessageRole.ASSISTANT), delta="k")
        return gen()

    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()

    # -- async
    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=Message(content="pong", role=MessageRole.ASSISTANT))

    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        raise NotImplementedError()

    async def astream_chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponseAsyncGen:
        async def agen() -> ChatResponseAsyncGen:
            yield ChatResponse(message=Message(content="ok", role=MessageRole.ASSISTANT), delta="o")
            yield ChatResponse(message=Message(content="ok", role=MessageRole.ASSISTANT), delta="k")
        return agen()

    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        raise NotImplementedError()


class TestStreamResponseToTokens:

    def test_nominal_and_empty_deltas_completion_response(self):
        """Inputs: two responses with non-empty deltas and then empty/None.
        Expected: tokens reflect exact delta values or empty strings for falsy deltas.
        Checks: list(stream_response_to_tokens(gen())) matches expected sequence.
        """
        def responses() -> CompletionResponseGen:
            yield CompletionResponse(text="Hello", delta="He")
            yield CompletionResponse(text="Hello", delta="llo")
            yield CompletionResponse(text="Hello", delta="")
            yield CompletionResponse(text="Hello", delta=None)

        tokens = list(stream_response_to_tokens(responses()))
        assert tokens == ["He", "llo", "", ""]

    def test_nominal_and_empty_deltas_chat_response(self):
        """Inputs: chat responses with deltas including empty and None.
        Expected: yielded tokens equal the deltas or empty strings when falsy.
        Checks: list(stream_response_to_tokens(gen())) equals expected.
        """
        def responses() -> ChatResponseGen:
            yield ChatResponse(message=Message(content="Hi", role=MessageRole.ASSISTANT), delta="H")
            yield ChatResponse(message=Message(content="Hi", role=MessageRole.ASSISTANT), delta="i")
            yield ChatResponse(message=Message(content="Hi", role=MessageRole.ASSISTANT), delta="")
            yield ChatResponse(message=Message(content="Hi", role=MessageRole.ASSISTANT), delta=None)

        tokens = list(stream_response_to_tokens(responses()))
        assert tokens == ["H", "i", "", ""]


class TestAStreamResponseToTokens:
    @pytest.mark.asyncio
    async def test_async_nominal_and_empty_deltas_completion_response(self):
        """Inputs: async completion responses with normal and falsy deltas.
        Expected: async generator yields the same sequence of tokens, empty for falsy.
        Checks: collected list equals expected sequence.
        """
        async def responses() -> CompletionResponseAsyncGen:
            yield CompletionResponse(text="Hello", delta="He")
            yield CompletionResponse(text="Hello", delta="llo")
            yield CompletionResponse(text="Hello", delta="")
            yield CompletionResponse(text="Hello", delta=None)

        agen = await astream_response_to_tokens(responses())
        assert [t async for t in agen] == ["He", "llo", "", ""]

    @pytest.mark.asyncio
    async def test_async_nominal_and_empty_deltas_chat_response(self):
        """Inputs: async chat responses with deltas including empty and None.
        Expected: tokens are passed through as-is, empty for falsy values.
        Checks: collected list equals expected.
        """
        async def responses() -> ChatResponseAsyncGen:
            yield ChatResponse(message=Message(content="Hi", role=MessageRole.ASSISTANT), delta="H")
            yield ChatResponse(message=Message(content="Hi", role=MessageRole.ASSISTANT), delta="i")
            yield ChatResponse(message=Message(content="Hi", role=MessageRole.ASSISTANT), delta="")
            yield ChatResponse(message=Message(content="Hi", role=MessageRole.ASSISTANT), delta=None)

        agen = await astream_response_to_tokens(responses())
        assert [t async for t in agen] == ["H", "i", "", ""]
