from __future__ import annotations

from openai import AzureOpenAI
from openai.types.responses import Response
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Generator,
    Literal,
    Sequence,
    TypeVar,
    cast,
    overload,
)

from serapeum.core.llms import (
    Message,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    Metadata,
    MessageRole,
    ThinkingBlock,
    ToolCallBlock,
    ChatToCompletionMixin,
)
from pydantic import (
    Field,
    PrivateAttr,
    BaseModel,
    model_validator,
)
from serapeum.core.configs.defaults import (
    DEFAULT_TEMPERATURE,
)

from serapeum.core.llms import FunctionCallingLLM
from serapeum.core.tools import ToolCallArguments
from serapeum.core.utils.schemas import parse_partial_json
from serapeum.core.prompts import PromptTemplate
from serapeum.core.llms import FlexibleModel
from serapeum.openai.models import (
    O1_MODELS,
    is_function_calling_model,
    openai_modelname_to_contextsize,
)
from serapeum.openai.converters import (
    ResponsesOutputParser,
    ResponsesStreamAccumulator,
    to_openai_message_dicts,
)
from serapeum.openai.mixins import Client, ModelMetadata
from serapeum.openai.utils import resolve_tool_choice


if TYPE_CHECKING:
    from serapeum.core.tools import BaseTool

Model = TypeVar("Model", bound=BaseModel)


class OpenAIResponses(ModelMetadata, Client, ChatToCompletionMixin, FunctionCallingLLM):
    """
    OpenAI Responses LLM.

    Args:
        model: name of the OpenAI model to use.
        temperature: a float from 0 to 1 controlling randomness in generation; higher will lead to more creative, less deterministic responses.
        max_output_tokens: the maximum number of tokens to generate.
        reasoning_options: Optional dictionary to configure reasoning for O1 models.
                    Corresponds to the 'reasoning' parameter in the OpenAI API.
                    Example: {"effort": "low", "summary": "concise"}
        include: Additional output data to include in the model response.
        instructions: Instructions for the model to follow.
        track_previous_responses: Whether to track previous responses. If true, the LLM class will statefully track previous responses.
        store: Whether to store previous responses in OpenAI's storage.
        built_in_tools: The built-in tools to use for the model to augment responses.
        truncation: Whether to auto-truncate the input if it exceeds the model's context window.
        user: An optional identifier to help track the user's requests for abuse.
        strict: Whether to enforce strict validation of the structured output.
        additional_kwargs: Add additional parameters to OpenAI request body.
        max_retries: How many times to retry the API call if it fails.
        timeout: How long to wait, in seconds, for an API call before failing.
        api_key: Your OpenAI api key
        api_base: The base URL of the API to call
        api_version: the version of the API to call
        default_headers: override the default headers for API requests.
        http_client: pass in your own httpx.Client instance.
        async_http_client: pass in your own httpx.AsyncClient instance.

    Examples:
        `pip install llama-index-llms-openai`

        ```python
        from serapeum.llms.openai import OpenAIResponses

        llm = OpenAIResponses(model="gpt-4o-mini", api_key="sk-...")

        response = llm.complete("Hi, write a short story")
        print(response.text)
        ```
    """

    model: str = Field(description="The OpenAI model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=2.0,
    )
    top_p: float = Field(
        default=1.0,
        description="The top-p value to use during generation.",
        ge=0.0,
        le=1.0,
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    reasoning_options: dict[str, Any] | None = Field(
        default=None,
        description="Optional dictionary to configure reasoning for O1 models. Example: {'effort': 'low', 'summary': 'concise'}",
    )
    include: list[str] | None = Field(
        default=None,
        description="Additional output data to include in the model response.",
    )
    instructions: str | None = Field(
        default=None,
        description="Instructions for the model to follow.",
    )
    track_previous_responses: bool = Field(
        default=False,
        description="Whether to track previous responses. If true, the LLM class will statefully track previous responses.",
    )
    store: bool = Field(
        default=False,
        description="Whether to store previous responses in OpenAI's storage.",
    )
    built_in_tools: list[dict] | None = Field(
        default=None,
        description="The built-in tools to use for the model to augment responses.",
    )
    truncation: str = Field(
        default="disabled",
        description="Whether to auto-truncate the input if it exceeds the model's context window.",
    )
    user: str | None = Field(
        default=None,
        description="An optional identifier to help track the user's requests for abuse.",
    )
    call_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Metadata to include in the API call.",
    )
    additional_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the OpenAI API at inference time.",
    )
    strict: bool = Field(
        default=False,
        description="Whether to enforce strict validation of the structured output.",
    )
    context_window: int | None = Field(
        default=None,
        description="The context window override for the model.",
    )

    _previous_response_id: str | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _inject_response_state(
        cls, data: Any, handler: Any
    ) -> OpenAIResponses:
        """Pop previous_response_id before Pydantic validation and restore after."""
        previous_response_id = None
        if isinstance(data, dict):
            previous_response_id = data.pop("previous_response_id", None)

        instance = handler(data)
        instance._previous_response_id = previous_response_id

        return instance

    @model_validator(mode="after")
    def _validate_model(self) -> OpenAIResponses:
        """Force O1 temperature and sync store flag."""
        if self.model in O1_MODELS:
            self.temperature = 1.0

        if self.track_previous_responses:
            self.store = True

        return self

    @classmethod
    def class_name(cls) -> str:
        return "openai_responses_llm"

    @property
    def metadata(self) -> Metadata:
        return Metadata(
            context_window=self.context_window
            or openai_modelname_to_contextsize(self._get_model_name()),
            num_output=self.max_output_tokens or -1,
            is_chat_model=True,
            is_function_calling_model=is_function_calling_model(
                model=self._get_model_name()
            ),
            model_name=self.model,
        )

    def _is_azure_client(self) -> bool:
        return isinstance(self.client, AzureOpenAI)

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        initial_tools = self.built_in_tools or []
        model_kwargs = {
            "model": self.model,
            "include": self.include,
            "instructions": self.instructions,
            "max_output_tokens": self.max_output_tokens,
            "metadata": self.call_metadata,
            "previous_response_id": self._previous_response_id,
            "store": self.store,
            "temperature": self.temperature,
            "tools": [*initial_tools, *(kwargs.pop("tools", []) or [])],
            "top_p": self.top_p,
            "truncation": self.truncation,
            "user": self.user,
        }

        if self.model in O1_MODELS and self.reasoning_options is not None:
            model_kwargs["reasoning"] = self.reasoning_options

        if self.reasoning_options is not None:
            params_to_exclude_for_reasoning = {
                "top_p",
                "temperature",
                "presence_penalty",
                "frequency_penalty",
            }
            for param in params_to_exclude_for_reasoning:
                model_kwargs.pop(param, None)

        # priority is class args > additional_kwargs > runtime args
        model_kwargs.update(self.additional_kwargs)

        kwargs = kwargs or {}
        model_kwargs.update(kwargs)

        return model_kwargs

    @overload
    def chat(
        self, messages: Sequence[Message], *, stream: Literal[False] = ..., **kwargs: Any,
    ) -> ChatResponse: ...

    @overload
    def chat(
        self, messages: Sequence[Message], *, stream: Literal[True], **kwargs: Any,
    ) -> ChatResponseGen: ...

    def chat(
        self, messages: Sequence[Message], *, stream: bool = False, **kwargs: Any
    ) -> ChatResponse | ChatResponseGen:
        result: ChatResponse | ChatResponseGen = (
            self._stream_chat(messages, **kwargs)
            if stream
            else self._chat(messages, **kwargs)
        )
        return result

    def _chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        kwargs_dict = self._get_model_kwargs(**kwargs)
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        response: Response = self.client.responses.create(
            input=message_dicts,
            stream=False,
            **kwargs_dict,
        )

        if self.track_previous_responses:
            self._previous_response_id = response.id

        chat_response = ResponsesOutputParser(response.output).build()
        chat_response.raw = response
        chat_response.additional_kwargs["usage"] = response.usage
        if hasattr(response.usage.output_tokens_details, "reasoning_tokens"):
            for block in chat_response.message.chunks:
                if isinstance(block, ThinkingBlock):
                    block.num_tokens = (
                        response.usage.output_tokens_details.reasoning_tokens
                    )

        return chat_response

    def _stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        def gen() -> ChatResponseGen:
            accumulator = ResponsesStreamAccumulator(
                track_previous_responses=self.track_previous_responses,
                previous_response_id=self._previous_response_id,
            )

            for event in self.client.responses.create(
                input=message_dicts,
                stream=True,
                **self._get_model_kwargs(**kwargs),
            ):
                blocks, delta = accumulator.update(event)

                if (
                    self.track_previous_responses
                    and accumulator.previous_response_id != self._previous_response_id
                ):
                    self._previous_response_id = accumulator.previous_response_id

                yield ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        chunks=blocks,
                    ),
                    delta=delta,
                    raw=event,
                    additional_kwargs=accumulator.additional_kwargs,
                )

        return gen()

    @overload
    async def achat(
        self, messages: Sequence[Message], *, stream: Literal[False] = ..., **kwargs: Any,
    ) -> ChatResponse: ...

    @overload
    async def achat(
        self, messages: Sequence[Message], *, stream: Literal[True], **kwargs: Any,
    ) -> ChatResponseAsyncGen: ...

    async def achat(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | ChatResponseAsyncGen:
        result: ChatResponse | ChatResponseAsyncGen = (
            await self._astream_chat(messages, **kwargs)
            if stream
            else await self._achat(messages, **kwargs)
        )
        return result

    async def _achat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponse:
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        response: Response = await self.async_client.responses.create(
            input=message_dicts,
            stream=False,
            **self._get_model_kwargs(**kwargs),
        )

        if self.track_previous_responses:
            self._previous_response_id = response.id

        chat_response = ResponsesOutputParser(response.output).build()
        chat_response.raw = response
        chat_response.additional_kwargs["usage"] = response.usage

        return chat_response

    async def _astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        async def gen() -> ChatResponseAsyncGen:
            accumulator = ResponsesStreamAccumulator(
                track_previous_responses=self.track_previous_responses,
                previous_response_id=self._previous_response_id,
            )

            response_stream = await self.async_client.responses.create(
                input=message_dicts,
                stream=True,
                **self._get_model_kwargs(**kwargs),
            )

            async for event in response_stream:
                blocks, delta = accumulator.update(event)

                if (
                    self.track_previous_responses
                    and accumulator.previous_response_id != self._previous_response_id
                ):
                    self._previous_response_id = accumulator.previous_response_id

                yield ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        chunks=blocks,
                    ),
                    delta=delta,
                    raw=event,
                    additional_kwargs=accumulator.additional_kwargs,
                )

        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: str | Message | None = None,
        chat_history: list[Message] | None = None,
        allow_parallel_tool_calls: bool = True,
        tool_required: bool = False,
        tool_choice: str | dict | None = None,
        verbose: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Predict and call the tool."""

        # openai responses api has a slightly different tool spec format
        tool_specs = [
            {
                "type": "function",
                **tool.metadata.to_openai_tool(skip_length_check=True)["function"],
            }
            for tool in tools
        ]

        if strict is not None:
            strict = strict
        else:
            strict = self.strict

        if strict:
            for tool_spec in tool_specs:
                tool_spec["strict"] = True
                tool_spec["parameters"]["additionalProperties"] = False

        if isinstance(user_msg, str):
            user_msg = Message(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": resolve_tool_choice(tool_choice, tool_required)
            if tool_specs
            else None,
            "parallel_tool_calls": allow_parallel_tool_calls,
            **kwargs,
        }

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> list[ToolCallArguments]:
        """Predict and call the tool."""
        tool_calls: list[ToolCallBlock] = [
            block
            for block in response.message.chunks
            if isinstance(block, ToolCallBlock)
        ]

        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            # this should handle both complete and partial jsons
            try:
                argument_dict = parse_partial_json(cast(str, tool_call.tool_kwargs))
            except Exception:
                argument_dict = {}

            tool_selections.append(
                ToolCallArguments(
                    tool_id=tool_call.tool_call_id or "",
                    tool_name=tool_call.tool_name,
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

    @overload
    def structured_predict(
        self, output_cls: type[Model], prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ..., *, stream: Literal[False] = ..., **prompt_args: Any,
    ) -> Model: ...

    @overload
    def structured_predict(
        self, output_cls: type[Model], prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ..., *, stream: Literal[True], **prompt_args: Any,
    ) -> Generator[Model | FlexibleModel, None, None]: ...

    def structured_predict(
        self,
        output_cls: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        **prompt_args: Any,
    ) -> Model | Generator[Model | FlexibleModel, None, None]:
        llm_kwargs = llm_kwargs or {}

        llm_kwargs["tool_choice"] = (
            "required" if "tool_choice" not in llm_kwargs else llm_kwargs["tool_choice"]
        )
        if stream:
            result: Model | Generator[Model | FlexibleModel, None, None] = (
                super().stream_structured_predict(
                    output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
                )
            )
        else:
            result = super().structured_predict(
                output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
            )
        return result

    @overload
    async def astructured_predict(
        self, output_cls: type[Model], prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ..., *, stream: Literal[False] = ..., **prompt_args: Any,
    ) -> Model: ...

    @overload
    async def astructured_predict(
        self, output_cls: type[Model], prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ..., *, stream: Literal[True], **prompt_args: Any,
    ) -> AsyncGenerator[Model | FlexibleModel, None]: ...

    async def astructured_predict(
        self,
        output_cls: type[Model],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        **prompt_args: Any,
    ) -> Model | AsyncGenerator[Model | FlexibleModel, None]:
        llm_kwargs = llm_kwargs or {}

        llm_kwargs["tool_choice"] = (
            "required" if "tool_choice" not in llm_kwargs else llm_kwargs["tool_choice"]
        )
        if stream:
            result: Model | AsyncGenerator[Model | FlexibleModel, None] = (
                await super().astream_structured_predict(
                    output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
                )
            )
        else:
            result = await super().astructured_predict(
                output_cls, prompt, llm_kwargs=llm_kwargs, **prompt_args
            )
        return result
