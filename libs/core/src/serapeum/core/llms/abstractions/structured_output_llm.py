"""Structured output wrapper and helpers around the core LLM API."""

from typing import Any, Sequence, Type

from pydantic import BaseModel, Field, SerializeAsAny

from serapeum.core.base.llms.types import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponseGen,
    Message,
    MessageRole,
    Metadata,
)

from serapeum.core.llms.abstractions.mixins import ChatToCompletionMixin
from serapeum.core.llms.base import LLM
from serapeum.core.prompts.base import ChatPromptTemplate


class StructuredOutputLLM(ChatToCompletionMixin, LLM):
    """Wrap an LLM to produce structured Pydantic outputs.

    This adapter delegates to an underlying LLM while exposing the same
    chat/completion interfaces, converting results to and from the configured
    ``output_cls`` model when appropriate.
    """

    llm: SerializeAsAny[LLM]
    output_cls: Type[BaseModel] = Field(
        ..., description="Output class for the structured LLM.", exclude=True
    )

    @classmethod
    def class_name(cls) -> str:
        return "StructuredOutputLLM"

    @property
    def metadata(self) -> Metadata:
        return self.llm.metadata

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """Chat endpoint for LLM."""
        chat_prompt = ChatPromptTemplate(message_templates=messages)

        output = self.llm.parse(
            output_cls=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
        )
        return ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT, content=output.model_dump_json()
            ),
            raw=output,
        )

    def stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        chat_prompt = ChatPromptTemplate(message_templates=messages)

        stream_output = self.llm.stream_parse(
            output_cls=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
        )
        for partial_output in stream_output:
            yield ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT, content=partial_output.json()
                ),
                raw=partial_output,
            )

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream completion endpoint for LLM."""
        raise NotImplementedError("stream_complete is not supported by default.")

    async def achat(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> ChatResponse:
        chat_prompt = ChatPromptTemplate(message_templates=messages)

        output = await self.llm.aparse(
            output_cls=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
        )
        return ChatResponse(
            message=Message(
                role=MessageRole.ASSISTANT, content=output.model_dump_json()
            ),
            raw=output,
        )

    async def astream_chat(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async stream chat endpoint for LLM."""

        async def gen() -> ChatResponseAsyncGen:
            chat_prompt = ChatPromptTemplate(message_templates=messages)

            stream_output = await self.llm.astream_parse(
                output_cls=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
            )
            async for partial_output in stream_output:
                yield ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT, content=partial_output.json()
                    ),
                    raw=partial_output,
                )

        return gen()

    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        """Async stream completion endpoint for LLM."""
        raise NotImplementedError("astream_complete is not supported by default.")
