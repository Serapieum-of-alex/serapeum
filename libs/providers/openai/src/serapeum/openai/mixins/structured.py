from __future__ import annotations

import re
from typing import (
    Any,
    AsyncGenerator,
    Generator,
    Type,
    TypeVar,
)

from pydantic import BaseModel

from serapeum.core.base.models import PydanticProgramMode
from serapeum.core.llms import FlexibleModel
from serapeum.core.prompts import PromptTemplate
from serapeum.openai.models import is_json_schema_supported

Model = TypeVar("Model", bound=BaseModel)


class OpenAIStructuredOutputMixin:
    """Native JSON-schema structured output support for the OpenAI chat-completions API.

    Overrides ``structured_predict`` / ``astructured_predict`` so that models
    supporting the ``response_format`` JSON-schema feature use it directly,
    falling back to the base function-calling approach otherwise.

    The concrete class must provide:
    - ``model: str``
    - ``pydantic_program_mode``
    - ``chat()`` / ``achat()``
    - ``_extend_messages()``  (from ``LLM`` base)
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_schema(
        llm_kwargs: dict[str, Any] | None, output_cls: Type[Model]
    ) -> dict[str, Any]:
        from openai.resources.chat.completions.completions import (
            _type_to_response_format,
        )

        llm_kwargs = llm_kwargs or {}
        response_format = _type_to_response_format(output_cls)
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

    def _should_use_structure_outputs(self) -> bool:
        return (
            self.pydantic_program_mode == PydanticProgramMode.DEFAULT
            and is_json_schema_supported(self.model)
        )

    # ------------------------------------------------------------------
    # Sync
    # ------------------------------------------------------------------

    def structured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        **prompt_args: Any,
    ) -> Model | Generator[Model | FlexibleModel | None, None]:
        llm_kwargs = llm_kwargs or {}

        if stream:
            result: Model | Generator[Model | FlexibleModel | None, None] = (
                self._structured_stream_call(
                    output_cls, prompt, llm_kwargs, **prompt_args
                )
            )
        elif self._should_use_structure_outputs():
            messages = self._extend_messages(prompt.format_messages(**prompt_args))
            llm_kwargs = self._prepare_schema(llm_kwargs, output_cls)
            response = self.chat(messages, **llm_kwargs)
            result = output_cls.model_validate_json(str(response.message.content))
        else:
            # when uses function calling to extract structured outputs
            # here we force tool_choice to be required
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in llm_kwargs
                else llm_kwargs["tool_choice"]
            )
            result = super().structured_predict(
                output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
            )
        return result

    def _structured_stream_call(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> Generator[
        Model | list[Model] | FlexibleModel | list[FlexibleModel] | None, None
    ]:
        if self._should_use_structure_outputs():
            from serapeum.core.llms.orchestrators.utils import process_streaming_content_incremental

            messages = self._extend_messages(prompt.format_messages(**prompt_args))
            llm_kwargs = self._prepare_schema(llm_kwargs, output_cls)
            curr = None
            for response in self.chat(stream=True, messages=messages, **llm_kwargs):
                curr = process_streaming_content_incremental(
                    response,
                    output_cls,
                    curr
                )
                yield curr
        else:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in llm_kwargs
                else llm_kwargs["tool_choice"]
            )
            yield from super()._structured_stream_call(
                output_cls, prompt, llm_kwargs, **prompt_args
            )

    # ------------------------------------------------------------------
    # Async
    # ------------------------------------------------------------------

    async def astructured_predict(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        **prompt_args: Any,
    ) -> Model | AsyncGenerator[Model | FlexibleModel, None]:
        llm_kwargs = llm_kwargs or {}

        if stream:
            result: Model | AsyncGenerator[Model | FlexibleModel, None] = (
                await self._structured_astream_call(
                    output_cls, prompt, llm_kwargs, **prompt_args
                )
            )
        elif self._should_use_structure_outputs():
            messages = self._extend_messages(prompt.format_messages(**prompt_args))
            llm_kwargs = self._prepare_schema(llm_kwargs, output_cls)
            response = await self.achat(messages, **llm_kwargs)
            result = output_cls.model_validate_json(str(response.message.content))
        else:
            # when uses function calling to extract structured outputs
            # here we force tool_choice to be required
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in llm_kwargs
                else llm_kwargs["tool_choice"]
            )
            result = await super().astructured_predict(
                output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
            )
        return result

    async def _structured_astream_call(
        self,
        output_cls: Type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[
        Model, list[Model] | FlexibleModel | list[FlexibleModel] | None
    ]:
        if self._should_use_structure_outputs():

            async def gen(
                llm_kwargs=llm_kwargs,
            ) -> AsyncGenerator[
                Model, list[Model] | FlexibleModel | list[FlexibleModel] | None
            ]:
                from serapeum.core.llms.orchestrators.utils import process_streaming_content_incremental

                messages = self._extend_messages(prompt.format_messages(**prompt_args))
                llm_kwargs = self._prepare_schema(llm_kwargs, output_cls)
                curr = None
                async for response in await self.achat(stream=True, messages=messages, **llm_kwargs):
                    curr = process_streaming_content_incremental(
                        response, output_cls, curr
                    )
                    yield curr

            result = gen()
        else:
            llm_kwargs["tool_choice"] = (
                "required"
                if "tool_choice" not in llm_kwargs
                else llm_kwargs["tool_choice"]
            )
            result = await super()._structured_astream_call(
                output_cls, prompt, llm_kwargs, **prompt_args
            )
        return result
