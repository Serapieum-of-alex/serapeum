from __future__ import annotations
import functools
import base64
from openai import AzureOpenAI
from openai.types.responses import (
    Response,
    ResponseStreamEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFileSearchCallCompletedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseInProgressEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseTextDeltaEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseFileSearchToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseComputerToolCall,
    ResponseReasoningItem,
    ResponseCodeInterpreterToolCall,
    ResponseImageGenCallPartialImageEvent,
    ResponseOutputItemDoneEvent,
)
from openai.types.responses.response_output_item import ImageGenerationCall, McpCall
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Callable,
    Generator,
    Sequence,
    TypeVar,
    cast,
)

from serapeum.core.llms import (
    Message,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    Metadata,
    MessageRole,
    ContentBlock,
    TextChunk,
    Image,
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
from serapeum.openai.converters import to_openai_message_dicts
from serapeum.openai.mixins import OpenAIClientMixin, OpenAIModelMixin
from serapeum.openai.utils import (
    create_retry_decorator,
    resolve_tool_choice,
)


if TYPE_CHECKING:
    from serapeum.core.tools import BaseTool

Model = TypeVar("Model", bound=BaseModel)



def llm_retry_decorator(f: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(f)
    def wrapper(self, *args: Any, **kwargs: Any) -> Any:
        max_retries = getattr(self, "max_retries", 0)
        if max_retries <= 0:
            return f(self, *args, **kwargs)

        retry = create_retry_decorator(
            max_retries=max_retries,
            random_exponential=True,
            stop_after_delay_seconds=60,
            min_seconds=1,
            max_seconds=20,
        )
        return retry(f)(self, *args, **kwargs)

    return wrapper


def force_single_tool_call(response: ChatResponse) -> None:
    tool_calls = [
        block for block in response.message.chunks if isinstance(block, ToolCallBlock)
    ]
    if len(tool_calls) > 1:
        response.message.chunks = [
            block
            for block in response.message.chunks
            if not isinstance(block, ToolCallBlock)
        ] + [tool_calls[0]]


class OpenAIResponses(OpenAIModelMixin, OpenAIClientMixin, ChatToCompletionMixin, FunctionCallingLLM):
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

    def chat(
        self, messages: Sequence[Message], *, stream: bool = False, **kwargs: Any
    ) -> ChatResponse | ChatResponseGen:
        result: ChatResponse | ChatResponseGen = (
            self._stream_chat(messages, **kwargs)
            if stream
            else self._chat(messages, **kwargs)
        )
        return result

    @staticmethod
    def _parse_response_output(output: list[ResponseOutputItem]) -> ChatResponse:
        message = Message(role=MessageRole.ASSISTANT)
        additional_kwargs = {"built_in_tool_calls": []}
        blocks: list[ContentBlock] = []
        for item in output:
            if isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if hasattr(part, "text"):
                        blocks.append(TextChunk(content=part.text))
                    if hasattr(part, "annotations"):
                        additional_kwargs["annotations"] = part.annotations
                    if hasattr(part, "refusal"):
                        additional_kwargs["refusal"] = part.refusal

                message.chunks.extend(blocks)
            elif isinstance(item, ImageGenerationCall):
                # return an Image if there is image generation
                if item.status != "failed":
                    additional_kwargs["built_in_tool_calls"].append(item)
                    if item.result is not None:
                        image_bytes = base64.b64decode(item.result)
                        blocks.append(Image(content=image_bytes))
            elif isinstance(item, ResponseCodeInterpreterToolCall):
                additional_kwargs["built_in_tool_calls"].append(item)
            elif isinstance(item, McpCall):
                additional_kwargs["built_in_tool_calls"].append(item)
            elif isinstance(item, ResponseFileSearchToolCall):
                additional_kwargs["built_in_tool_calls"].append(item)
            elif isinstance(item, ResponseFunctionToolCall):
                message.chunks.append(
                    ToolCallBlock(
                        tool_name=item.name,
                        tool_call_id=item.call_id,
                        tool_kwargs=item.arguments,
                    )
                )
            elif isinstance(item, ResponseFunctionWebSearch):
                additional_kwargs["built_in_tool_calls"].append(item)
            elif isinstance(item, ResponseComputerToolCall):
                additional_kwargs["built_in_tool_calls"].append(item)
            elif isinstance(item, ResponseReasoningItem):
                content: str | None = None

                if item.content:
                    content = "\n".join([i.text for i in item.content])

                if item.summary:
                    if content:
                        content += "\n" + "\n".join([i.text for i in item.summary])
                    else:
                        content = "\n".join([i.text for i in item.summary])
                message.chunks.append(
                    ThinkingBlock(
                        content=content,
                        additional_information=item.model_dump(
                            exclude={"content", "summary"}
                        ),
                    )
                )

        return ChatResponse(message=message, additional_kwargs=additional_kwargs)

    @llm_retry_decorator
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

        chat_response = OpenAIResponses._parse_response_output(response.output)
        chat_response.raw = response
        chat_response.additional_kwargs["usage"] = response.usage
        if hasattr(response.usage.output_tokens_details, "reasoning_tokens"):
            for block in chat_response.message.chunks:
                if isinstance(block, ThinkingBlock):
                    block.num_tokens = (
                        response.usage.output_tokens_details.reasoning_tokens
                    )

        return chat_response

    @staticmethod
    def process_response_event(
        event: ResponseStreamEvent,
        built_in_tool_calls: list[Any],
        additional_kwargs: dict[str, Any],
        current_tool_call: ResponseFunctionToolCall | None,
        track_previous_responses: bool,
        previous_response_id: str | None = None,
    ) -> tuple[
        list[ContentBlock],
        list[Any],
        dict[str, Any],
        ResponseFunctionToolCall | None,
        str | None,
        str,
    ]:
        """
        Process a ResponseStreamEvent and update the state accordingly.

        Args:
            event: The response stream event to process
            built_in_tool_calls: list of built-in tool calls
            additional_kwargs: Additional keyword arguments to include in ChatResponse
            current_tool_call: The currently in-progress tool call, if any
            track_previous_responses: Whether to track previous response IDs
            previous_response_id: Previous response ID if tracking

        Returns:
            A tuple containing the updated state:
            (content, tool_calls, built_in_tool_calls, additional_kwargs, current_tool_call, updated_previous_response_id, delta)
        """
        delta = ""
        updated_previous_response_id = previous_response_id
        # we use blocks instead of content, since now we also support images! :)
        blocks: list[ContentBlock] = []
        if isinstance(event, ResponseCreatedEvent) or isinstance(
            event, ResponseInProgressEvent
        ):
            # Initial events, track the response id
            if track_previous_responses:
                updated_previous_response_id = event.response.id
        elif isinstance(event, ResponseOutputItemAddedEvent):
            # New output item (message, tool call, etc.)
            if isinstance(event.item, ResponseFunctionToolCall):
                current_tool_call = event.item
        elif isinstance(event, ResponseTextDeltaEvent):
            # Text content is being added
            delta = event.delta
            blocks.append(TextChunk(content=delta))
        elif isinstance(event, ResponseImageGenCallPartialImageEvent):
            # Partial image
            if event.partial_image_b64:
                blocks.append(
                    Image(
                        content=base64.b64decode(event.partial_image_b64),
                        detail=f"id_{event.partial_image_index}",
                    )
                )
        elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
            # Function call arguments are being streamed
            if current_tool_call is not None:
                current_tool_call.arguments += event.delta
        elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
            # Function call arguments are complete
            if current_tool_call is not None:
                current_tool_call.arguments = event.arguments
                current_tool_call.status = "completed"
                blocks.append(
                    ToolCallBlock(
                        tool_name=current_tool_call.name,
                        tool_kwargs=current_tool_call.arguments,
                        tool_call_id=current_tool_call.call_id,
                    )
                )

                # clear the current tool call
                current_tool_call = None
        elif isinstance(event, ResponseOutputTextAnnotationAddedEvent):
            # Annotations for the text
            annotations = additional_kwargs.get("annotations", [])
            annotations.append(event.annotation)
            additional_kwargs["annotations"] = annotations
        elif isinstance(event, ResponseFileSearchCallCompletedEvent):
            # File search tool call completed
            built_in_tool_calls.append(event)
        elif isinstance(event, ResponseWebSearchCallCompletedEvent):
            # Web search tool call completed
            built_in_tool_calls.append(event)
        elif isinstance(event, ResponseOutputItemDoneEvent):
            # Reasoning information
            if isinstance(event.item, ResponseReasoningItem):
                content: str | None = None
                if event.item.content:
                    content = "\n".join([i.text for i in event.item.content])
                if event.item.summary:
                    if content:
                        content += "\n" + "\n".join(
                            [i.text for i in event.item.summary]
                        )
                    else:
                        content = "\n".join([i.text for i in event.item.summary])
                blocks.append(
                    ThinkingBlock(
                        content=content,
                        additional_information=event.item.model_dump(
                            exclude={"content", "summary"}
                        ),
                    )
                )
        elif isinstance(event, ResponseCompletedEvent):
            # Response is complete
            if hasattr(event, "response") and hasattr(event.response, "usage"):
                additional_kwargs["usage"] = event.response.usage
            resp = OpenAIResponses._parse_response_output(event.response.output)
            blocks = resp.message.chunks

        return (
            blocks,
            built_in_tool_calls,
            additional_kwargs,
            current_tool_call,
            updated_previous_response_id,
            delta,
        )

    @llm_retry_decorator
    def _stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        def gen() -> ChatResponseGen:
            built_in_tool_calls = []
            additional_kwargs = {"built_in_tool_calls": []}
            current_tool_call: ResponseFunctionToolCall | None = None
            local_previous_response_id = self._previous_response_id

            for event in self.client.responses.create(
                input=message_dicts,
                stream=True,
                **self._get_model_kwargs(**kwargs),
            ):
                # Process the event and update state
                (
                    blocks,
                    built_in_tool_calls,
                    additional_kwargs,
                    current_tool_call,
                    local_previous_response_id,
                    delta,
                ) = OpenAIResponses.process_response_event(
                    event=event,
                    built_in_tool_calls=built_in_tool_calls,
                    additional_kwargs=additional_kwargs,
                    current_tool_call=current_tool_call,
                    track_previous_responses=self.track_previous_responses,
                    previous_response_id=local_previous_response_id,
                )

                if (
                    self.track_previous_responses
                    and local_previous_response_id != self._previous_response_id
                ):
                    self._previous_response_id = local_previous_response_id

                if built_in_tool_calls:
                    additional_kwargs["built_in_tool_calls"] = built_in_tool_calls

                # For any event, yield a ChatResponse with the current state
                yield ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        chunks=blocks,
                    ),
                    delta=delta,
                    raw=event,
                    additional_kwargs=additional_kwargs,
                )

        return gen()

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

    @llm_retry_decorator
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

        chat_response = OpenAIResponses._parse_response_output(response.output)
        chat_response.raw = response
        chat_response.additional_kwargs["usage"] = response.usage

        return chat_response

    @llm_retry_decorator
    async def _astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

        async def gen() -> ChatResponseAsyncGen:
            built_in_tool_calls = []
            additional_kwargs = {"built_in_tool_calls": []}
            current_tool_call: ResponseFunctionToolCall | None = None
            local_previous_response_id = self._previous_response_id

            response_stream = await self.async_client.responses.create(
                input=message_dicts,
                stream=True,
                **self._get_model_kwargs(**kwargs),
            )

            async for event in response_stream:
                # Process the event and update state
                (
                    blocks,
                    built_in_tool_calls,
                    additional_kwargs,
                    current_tool_call,
                    local_previous_response_id,
                    delta,
                ) = OpenAIResponses.process_response_event(
                    event=event,
                    built_in_tool_calls=built_in_tool_calls,
                    additional_kwargs=additional_kwargs,
                    current_tool_call=current_tool_call,
                    track_previous_responses=self.track_previous_responses,
                    previous_response_id=local_previous_response_id,
                )

                if (
                    self.track_previous_responses
                    and local_previous_response_id != self._previous_response_id
                ):
                    self._previous_response_id = local_previous_response_id

                if built_in_tool_calls:
                    additional_kwargs["built_in_tool_calls"] = built_in_tool_calls

                # For any event, yield a ChatResponse with the current state
                yield ChatResponse(
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        chunks=blocks,
                    ),
                    delta=delta,
                    raw=event,
                    additional_kwargs=additional_kwargs,
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
