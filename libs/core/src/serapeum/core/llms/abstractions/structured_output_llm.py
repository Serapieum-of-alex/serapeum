"""Structured output wrapper and helpers around the core LLM API."""
from __future__ import annotations
from typing import Any, Literal, Sequence, Type, overload

from pydantic import BaseModel, Field, SerializeAsAny

from serapeum.core.base.llms.types import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
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

    def _stream_chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponseGen:
        chat_prompt = ChatPromptTemplate(message_templates=messages)
        for partial_output in self.llm.stream_parse(
            schema=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
        ):
            yield ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT, content=partial_output.json()
                ),
                raw=partial_output,
            )

    def chat(
        self, messages: Sequence[Message], *, stream: bool = False, **kwargs: Any
    ) -> ChatResponse | ChatResponseGen:
        """Chat endpoint for LLM."""
        if stream:
            result = self._stream_chat(messages, **kwargs)
        else:
            chat_prompt = ChatPromptTemplate(message_templates=messages)
            output = self.llm.parse(
                schema=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
            )
            result = ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT, content=output.model_dump_json()
                ),
                raw=output,
            )
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

    async def _astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        chat_prompt = ChatPromptTemplate(message_templates=messages)
        async for partial_output in await self.llm.astream_parse(
            schema=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
        ):
            yield ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT, content=partial_output.json()
                ),
                raw=partial_output,
            )

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | ChatResponseAsyncGen:
        """Async chat endpoint for LLM."""
        if stream:
            result = self._astream_chat(messages, **kwargs)
        else:
            chat_prompt = ChatPromptTemplate(message_templates=messages)
            output = await self.llm.aparse(
                schema=self.output_cls, prompt=chat_prompt, llm_kwargs=kwargs
            )
            result = ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT, content=output.model_dump_json()
                ),
                raw=output,
            )
        return result
