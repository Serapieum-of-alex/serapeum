from __future__ import annotations

import re
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Literal,
    TypeVar,
    overload,
)

from pydantic import BaseModel

from serapeum.core.base.llms.types import Message
from serapeum.core.types import StructuredOutputMode
from serapeum.core.llms import FlexibleModel
from serapeum.core.prompts import PromptTemplate
from serapeum.openai.data.models import is_json_schema_supported
from serapeum.core.llms.orchestrators.utils import process_streaming_content_incremental
from serapeum.core.llms import LLM


Model = TypeVar("Model", bound=BaseModel)


class StructuredOutput(LLM):
    """Native JSON-schema structured output support for the OpenAI chat-completions API.

    Overrides ``parse`` / ``aparse`` so that models supporting the
    ``response_format`` JSON-schema feature use it directly, falling back to
    the base function-calling approach otherwise.

    The concrete class must provide:
    - ``model: str``
    - ``structured_output_mode``
    - ``chat()`` / ``achat()``
    - ``_extend_messages()``  (from ``LLM`` base)
    """

    @staticmethod
    def _prepare_schema(
        llm_kwargs: dict[str, Any] | None, schema: type[Model]
    ) -> dict[str, Any]:
        from openai.resources.chat.completions.completions import (
            _type_to_response_format,
        )

        llm_kwargs = llm_kwargs or {}
        response_format = _type_to_response_format(schema)
        if isinstance(response_format, dict):
            json_schema = response_format.get("json_schema")
            if isinstance(json_schema, dict) and "name" in json_schema:
                json_schema["name"] = re.sub(
                    r"[^a-zA-Z0-9_-]", "_", str(json_schema["name"])
                )
        llm_kwargs["response_format"] = response_format
        if "tool_choice" in llm_kwargs:
            del llm_kwargs["tool_choice"]
        return llm_kwargs

    @staticmethod
    def _ensure_tool_choice(llm_kwargs: dict[str, Any]) -> None:
        """Set ``tool_choice`` to ``"required"`` if not already specified."""
        if "tool_choice" not in llm_kwargs:
            llm_kwargs["tool_choice"] = "required"

    def _should_use_structure_outputs(self) -> bool:
        return (
            self.structured_output_mode == StructuredOutputMode.DEFAULT
            and is_json_schema_supported(self.model)
        )

    def _prepare_json_schema_call(
        self,
        schema: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any],
        **prompt_args: Any,
    ) -> tuple[list[Message], dict[str, Any]]:
        """Build messages and llm_kwargs for a native JSON-schema call."""
        messages = self._extend_messages(prompt.format_messages(**prompt_args))
        llm_kwargs = self._prepare_schema(llm_kwargs, schema)
        return messages, llm_kwargs

    @overload
    def parse(
        self, schema: type[Model], prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ..., *, stream: Literal[False] = ..., **prompt_args: Any,
    ) -> Model: ...

    @overload
    def parse(
        self, schema: type[Model], prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ..., *, stream: Literal[True], **prompt_args: Any,
    ) -> Generator[Model | FlexibleModel | None, None, None]: ...

    def parse(
        self,
        schema: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        **prompt_args: Any,
    ) -> Model | Generator[Model | FlexibleModel | None, None, None]:
        llm_kwargs = llm_kwargs or {}

        if stream:
            result: Model | Generator[Model | FlexibleModel | None, None, None] = (
                self._parse_stream_call(schema, prompt, llm_kwargs, **prompt_args)
            )
        elif self._should_use_structure_outputs():
            messages, llm_kwargs = self._prepare_json_schema_call(
                schema, prompt, llm_kwargs, **prompt_args
            )
            response = self.chat(messages, **llm_kwargs)
            result = schema.model_validate_json(str(response.message.content))
        else:
            self._ensure_tool_choice(llm_kwargs)
            result = super().parse(
                schema, prompt, llm_kwargs=llm_kwargs, **prompt_args
            )
        return result

    def _parse_stream_call(
        self,
        schema: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> Generator[
        Model | list[Model] | FlexibleModel | list[FlexibleModel] | None, None, None
    ]:
        if self._should_use_structure_outputs():
            messages, llm_kwargs = self._prepare_json_schema_call(
                schema, prompt, llm_kwargs or {}, **prompt_args
            )
            curr = None
            for response in self.chat(stream=True, messages=messages, **llm_kwargs):
                curr = process_streaming_content_incremental(response, schema, curr)
                yield curr
        else:
            self._ensure_tool_choice(llm_kwargs)
            yield from super().stream_parse(
                schema, prompt, llm_kwargs=llm_kwargs, **prompt_args
            )

    @overload
    async def aparse(
        self, schema: type[Model], prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ..., *, stream: Literal[False] = ..., **prompt_args: Any,
    ) -> Model: ...

    @overload
    async def aparse(
        self, schema: type[Model], prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ..., *, stream: Literal[True], **prompt_args: Any,
    ) -> AsyncGenerator[Model | FlexibleModel, None]: ...

    async def aparse(
        self,
        schema: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        **prompt_args: Any,
    ) -> Model | AsyncGenerator[Model | FlexibleModel, None]:
        llm_kwargs = llm_kwargs or {}

        if stream:
            result: Model | AsyncGenerator[Model | FlexibleModel, None] = (
                await self._parse_astream_call(
                    schema, prompt, llm_kwargs, **prompt_args
                )
            )
        elif self._should_use_structure_outputs():
            messages, llm_kwargs = self._prepare_json_schema_call(
                schema, prompt, llm_kwargs, **prompt_args
            )
            response = await self.achat(messages, **llm_kwargs)
            result = schema.model_validate_json(str(response.message.content))
        else:
            self._ensure_tool_choice(llm_kwargs)
            result = await super().aparse(
                schema, prompt, llm_kwargs=llm_kwargs, **prompt_args
            )
        return result

    async def _parse_astream_call(
        self,
        schema: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[
        Model | list[Model] | FlexibleModel | list[FlexibleModel] | None, None
    ]:
        if self._should_use_structure_outputs():

            async def gen(
                llm_kwargs=llm_kwargs,
            ) -> AsyncGenerator[
                Model | list[Model] | FlexibleModel | list[FlexibleModel] | None, None
            ]:
                messages, llm_kwargs = self._prepare_json_schema_call(
                    schema, prompt, llm_kwargs or {}, **prompt_args
                )
                curr = None
                async for response in await self.achat(stream=True, messages=messages, **llm_kwargs):
                    curr = process_streaming_content_incremental(
                        response, schema, curr
                    )
                    yield curr

            result = gen()
        else:
            self._ensure_tool_choice(llm_kwargs)
            result = await super()._structured_astream_call(
                schema, prompt, llm_kwargs, **prompt_args
            )
        return result
