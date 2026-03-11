"""OpenAI structured-output mixin for native JSON-schema responses.

This module provides :class:`StructuredOutput`, an abstract :class:`~serapeum.core.llms.LLM`
subclass that overrides ``parse`` / ``aparse`` to leverage OpenAI's native
``response_format`` JSON-schema feature when the target model supports it,
falling back to the base function-calling approach otherwise.

Both synchronous and asynchronous paths support streaming, producing
partially-validated Pydantic model instances as chunks arrive.
"""

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

    The decision is made at call time via :meth:`_should_use_structure_outputs`,
    which inspects both the ``structured_output_mode`` setting and whether the
    current model appears in :data:`~serapeum.openai.data.models.JSON_SCHEMA_MODELS`.

    The concrete class must provide:

    - ``model: str`` -- the OpenAI model identifier (e.g. ``"gpt-4o"``).
    - ``structured_output_mode`` -- a
      :class:`~serapeum.core.types.StructuredOutputMode` enum value.
    - ``chat()`` / ``achat()`` -- synchronous and asynchronous chat methods.
    - ``_extend_messages()`` -- inherited from :class:`~serapeum.core.llms.LLM`.

    Examples:
        - Use parse to get a validated Pydantic model from a prompt:
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.prompts import PromptTemplate
            >>> class City(BaseModel):
            ...     name: str
            ...     country: str
            ...
            >>> # llm = OpenAI(model="gpt-4o")  # doctest: +SKIP
            >>> # city = llm.parse(City, PromptTemplate("Capital of {country}"), country="France")
            >>> # city.name  # doctest: +SKIP
            >>> # city.country  # doctest: +SKIP

            ```

        - Stream partial structured output as it arrives:
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.prompts import PromptTemplate
            >>> class Story(BaseModel):
            ...     title: str
            ...     body: str
            ...
            >>> # llm = OpenAI(model="gpt-4o")  # doctest: +SKIP
            >>> # for partial in llm.parse(Story, PromptTemplate("Write a story"), stream=True):
            >>> #     print(partial.title if partial else "...")  # doctest: +SKIP

            ```

    See Also:
        :class:`~serapeum.core.llms.LLM`: Base class providing the
            function-calling structured output fallback.
        :func:`~serapeum.openai.data.models.is_json_schema_supported`:
            Predicate that determines whether a model can use native JSON-schema
            mode.
    """

    @staticmethod
    def _prepare_schema(
        llm_kwargs: dict[str, Any] | None, schema: type[Model]
    ) -> dict[str, Any]:
        """Build ``llm_kwargs`` with ``response_format`` set for native JSON-schema mode.

        Uses the OpenAI SDK's internal ``_type_to_response_format`` helper to
        convert the Pydantic *schema* into the ``response_format`` payload
        expected by the chat-completions API.  Schema names are sanitised to
        contain only alphanumeric characters, hyphens, and underscores.

        If ``tool_choice`` was previously set in *llm_kwargs* it is removed,
        because ``response_format`` and ``tool_choice`` are mutually exclusive
        parameters in the OpenAI API.

        Args:
            llm_kwargs: Existing keyword arguments destined for the chat call.
                May be ``None``; a new dict is created in that case.
            schema: Pydantic model class whose JSON schema will be embedded in
                the ``response_format`` payload.

        Returns:
            A (possibly new) dict with the ``response_format`` key populated
            and ``tool_choice`` removed.
        """
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
        """Set ``tool_choice`` to ``"required"`` if not already specified.

        Args:
            llm_kwargs: Mutable dict of keyword arguments for the chat call.
                Modified in place.
        """
        if "tool_choice" not in llm_kwargs:
            llm_kwargs["tool_choice"] = "required"

    def _should_use_structure_outputs(self) -> bool:
        """Determine whether to use native JSON-schema structured outputs.

        Returns ``True`` when the instance's ``structured_output_mode`` is
        :attr:`~serapeum.core.types.StructuredOutputMode.DEFAULT` **and** the
        current ``model`` is listed in
        :data:`~serapeum.openai.data.models.JSON_SCHEMA_MODELS`.

        Returns:
            ``True`` if the model supports native JSON-schema mode, ``False``
            otherwise.
        """
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
        """Build messages and ``llm_kwargs`` for a native JSON-schema call.

        Formats the prompt into messages via :meth:`_extend_messages` and
        injects the ``response_format`` payload via :meth:`_prepare_schema`.

        Args:
            schema: Pydantic model class whose JSON schema is sent to the API.
            prompt: Template used to produce the user messages.
            llm_kwargs: Additional keyword arguments for the chat call.
            **prompt_args: Variables interpolated into *prompt*.

        Returns:
            A two-element tuple of ``(messages, llm_kwargs)`` ready for a
            ``chat`` / ``achat`` call.
        """
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
        """Parse LLM output into a validated Pydantic model instance.

        When the model supports native JSON-schema structured output
        (determined by :meth:`_should_use_structure_outputs`), the request
        is sent with ``response_format`` set to the schema's JSON schema.
        Otherwise, the base-class function-calling approach is used.

        Args:
            schema: Pydantic model class describing the expected output.
            prompt: Template that produces the messages sent to the model.
            llm_kwargs: Optional provider-specific arguments forwarded to the
                underlying chat call.
            stream: If ``True``, return a generator that yields progressively
                more-complete partial model instances as tokens arrive.
            **prompt_args: Variables interpolated into *prompt*.

        Returns:
            When ``stream=False``, a fully validated instance of *schema*.
            When ``stream=True``, a generator yielding ``Model``,
            :class:`~serapeum.core.llms.FlexibleModel`, or ``None`` as
            chunks are received.

        Raises:
            ValidationError: If the model's response cannot be validated
                against *schema*.

        Examples:
            - Obtain a structured result synchronously:
                ```python
                >>> from pydantic import BaseModel
                >>> from serapeum.core.prompts import PromptTemplate
                >>> class Sentiment(BaseModel):
                ...     label: str
                ...     score: float
                ...
                >>> # llm = OpenAI(model="gpt-4o")  # doctest: +SKIP
                >>> # result = llm.parse(
                >>> #     Sentiment,
                >>> #     PromptTemplate("Sentiment of: {text}"),
                >>> #     text="I love this!"
                >>> # )  # doctest: +SKIP
                >>> # result.label  # doctest: +SKIP
                >>> # result.score  # doctest: +SKIP

                ```

            - Stream partial results:
                ```python
                >>> from pydantic import BaseModel
                >>> from serapeum.core.prompts import PromptTemplate
                >>> class Summary(BaseModel):
                ...     text: str
                ...
                >>> # llm = OpenAI(model="gpt-4o")  # doctest: +SKIP
                >>> # for partial in llm.parse(
                >>> #     Summary, PromptTemplate("Summarize: {doc}"),
                >>> #     stream=True, doc="..."
                >>> # ):  # doctest: +SKIP
                >>> #     print(partial.text if partial else "")  # doctest: +SKIP

                ```

        See Also:
            :meth:`aparse`: Asynchronous counterpart.
            :meth:`_parse_stream_call`: Internal streaming implementation.
        """
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
        """Synchronous streaming structured-output generator.

        Yields progressively more-complete partial Pydantic model instances.
        Uses native JSON-schema mode when supported; otherwise delegates to
        the base-class ``stream_parse`` with function-calling.

        Args:
            schema: Pydantic model class for the expected output.
            prompt: Template producing the user messages.
            llm_kwargs: Optional provider-specific keyword arguments.
            **prompt_args: Variables interpolated into *prompt*.

        Yields:
            Partial or complete instances of *schema* (or
            :class:`~serapeum.core.llms.FlexibleModel`) as tokens stream in.
            May yield ``None`` before the first parseable chunk arrives.
        """
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
        """Asynchronously parse LLM output into a validated Pydantic model instance.

        Async counterpart of :meth:`parse`.  When the model supports native
        JSON-schema structured output, the request is sent with
        ``response_format``; otherwise, the base-class async function-calling
        path is used.

        Args:
            schema: Pydantic model class describing the expected output.
            prompt: Template that produces the messages sent to the model.
            llm_kwargs: Optional provider-specific arguments forwarded to the
                underlying async chat call.
            stream: If ``True``, return an async generator that yields
                progressively more-complete partial model instances.
            **prompt_args: Variables interpolated into *prompt*.

        Returns:
            When ``stream=False``, a fully validated instance of *schema*.
            When ``stream=True``, an async generator yielding ``Model`` or
            :class:`~serapeum.core.llms.FlexibleModel` instances.

        Raises:
            ValidationError: If the model's response cannot be validated
                against *schema*.

        Examples:
            - Await a structured result:
                ```python
                >>> import asyncio
                >>> from pydantic import BaseModel
                >>> from serapeum.core.prompts import PromptTemplate
                >>> class Entity(BaseModel):
                ...     name: str
                ...     kind: str
                ...
                >>> # async def main():  # doctest: +SKIP
                >>> #     llm = OpenAI(model="gpt-4o")
                >>> #     entity = await llm.aparse(
                >>> #         Entity, PromptTemplate("Extract entity from: {text}"),
                >>> #         text="Paris is a city"
                >>> #     )
                >>> #     print(entity.name, entity.kind)

                ```

            - Stream partial results asynchronously:
                ```python
                >>> import asyncio
                >>> from pydantic import BaseModel
                >>> from serapeum.core.prompts import PromptTemplate
                >>> class Report(BaseModel):
                ...     title: str
                ...     content: str
                ...
                >>> # async def main():  # doctest: +SKIP
                >>> #     llm = OpenAI(model="gpt-4o")
                >>> #     async for partial in await llm.aparse(
                >>> #         Report, PromptTemplate("Report on {topic}"),
                >>> #         stream=True, topic="AI"
                >>> #     ):
                >>> #         print(partial.title if partial else "...")

                ```

        See Also:
            :meth:`parse`: Synchronous counterpart.
            :meth:`_parse_astream_call`: Internal async streaming implementation.
        """
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
        """Asynchronous streaming structured-output generator.

        Returns an async generator that yields progressively more-complete
        partial Pydantic model instances.  Uses native JSON-schema mode when
        supported; otherwise delegates to the base-class async
        ``_structured_astream_call`` with function-calling.

        Args:
            schema: Pydantic model class for the expected output.
            prompt: Template producing the user messages.
            llm_kwargs: Optional provider-specific keyword arguments.
            **prompt_args: Variables interpolated into *prompt*.

        Returns:
            An async generator yielding partial or complete instances of
            *schema* (or :class:`~serapeum.core.llms.FlexibleModel`).
            May yield ``None`` before the first parseable chunk arrives.
        """
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
