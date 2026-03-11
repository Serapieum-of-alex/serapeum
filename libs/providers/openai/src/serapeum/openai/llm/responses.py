from __future__ import annotations

import logging

from openai import AzureOpenAI
from openai.types.responses import Response
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Sequence,
    overload,
)

from serapeum.core.llms import (
    Message,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    Metadata,
    MessageRole,
    TextChunk,
    ThinkingBlock,
    ChatToCompletion,
)
from pydantic import (
    Field,
    PrivateAttr,
    model_validator,
)
from serapeum.core.configs.defaults import (
    DEFAULT_TEMPERATURE,
)

from serapeum.core.llms import FunctionCallingLLM
from serapeum.openai.data.models import (
    O1_MODELS,
    is_function_calling_model,
    openai_modelname_to_contextsize,
)
from serapeum.openai.parsers import (
    ResponsesOutputParser,
    ResponsesStreamAccumulator,
    to_openai_message_dicts,
)
from serapeum.openai.llm.base import Client, ModelMetadata, StructuredOutput
from serapeum.openai.retry import is_retryable
from serapeum.openai.utils import resolve_tool_choice
from serapeum.core.retry import retry


if TYPE_CHECKING:
    from serapeum.core.tools import BaseTool

logger = logging.getLogger(__name__)


class Responses(StructuredOutput, ModelMetadata, Client, ChatToCompletion, FunctionCallingLLM):
    """OpenAI Responses API provider.

    Uses the ``/v1/responses`` endpoint, which is required for models such as
    ``o3``, ``o4-mini``, and future reasoning-focused models. Supports
    streaming, built-in tools (file search, web search, code interpreter),
    stateful conversation continuation via ``track_previous_responses``,
    structured output via tool-call forcing, and reasoning-effort control.

    Inherits connection management from :class:`~serapeum.openai.llm.base.Client`,
    model-metadata utilities from
    :class:`~serapeum.openai.llm.base.ModelMetadata`, and the
    chat-to-completion interface bridge from
    :class:`~serapeum.core.llms.ChatToCompletion`.

    Args:
        model: OpenAI model identifier (e.g. ``"o3"``, ``"gpt-4o-mini"``).
        temperature: Sampling temperature in ``[0.0, 2.0]``. Automatically
            forced to ``1.0`` for O1 reasoning models. Omitted from the
            request when ``reasoning_options`` is set. Defaults to
            :data:`~serapeum.core.configs.defaults.DEFAULT_TEMPERATURE`.
        top_p: Nucleus sampling probability mass in ``[0.0, 1.0]``. Omitted
            from the request when ``reasoning_options`` is set. Defaults to
            ``1.0``.
        max_output_tokens: Maximum number of tokens to generate. When ``None``
            the model applies its own default.
        reasoning_options: Dictionary configuring reasoning for O1 models
            (e.g. ``{"effort": "low", "summary": "concise"}``). When set,
            sampling parameters (``temperature``, ``top_p``, etc.) are excluded
            from the request.
        include: List of additional output fields to include in the response
            (e.g. ``["reasoning.encrypted_content"]``).
        instructions: System-level instructions prepended to every request.
        track_previous_responses: When ``True``, the instance stores the
            most-recent response ID and passes it as ``previous_response_id``
            on the next call, enabling stateful multi-turn conversations. Also
            forces ``store=True``.
        store: Whether to persist the response in OpenAI's server-side storage.
            Automatically set to ``True`` when ``track_previous_responses=True``.
        built_in_tools: List of built-in tool configuration dicts to include
            with every request (e.g. ``[{"type": "file_search"}]``).
        truncation: Input truncation strategy when the prompt exceeds the context
            window. ``"disabled"`` (default) raises an error instead of
            truncating.
        user: Optional end-user identifier for abuse-monitoring purposes.
        call_metadata: Arbitrary metadata dict forwarded to the API in the
            ``metadata`` field.
        additional_kwargs: Extra parameters forwarded verbatim to the API request
            body at inference time.
        strict: When ``True``, tool schemas are sent with ``"strict": true``.
            Defaults to ``False``.
        context_window: Manual context-window override in tokens. When ``None``
            the value is inferred from the model name.
        api_key: OpenAI API key. Falls back to the ``OPENAI_API_KEY`` environment
            variable when ``None``.
        api_base: Base URL override (useful for proxies or Azure endpoints).
        api_version: API version string (primarily for Azure OpenAI).
        timeout: Per-request HTTP timeout in seconds. Defaults to ``60.0``.
        default_headers: Additional HTTP headers sent with every request.
        http_client: Custom :class:`httpx.Client` instance for synchronous calls.
        async_http_client: Custom :class:`httpx.AsyncClient` instance for async
            calls.

    Examples:
        - Basic non-streaming completion
            ```python
            >>> from serapeum.openai import Responses
            >>> llm = Responses(model="o3-mini", api_key="sk-test")  # doctest: +SKIP
            >>> resp = llm.complete("Explain recursion briefly")  # doctest: +SKIP
            >>> print(resp.text)  # doctest: +SKIP

            ```
        - Stateful multi-turn conversation
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> llm = Responses(  # doctest: +SKIP
            ...     model="gpt-4o-mini",
            ...     track_previous_responses=True,
            ...     api_key="sk-test",
            ... )
            >>> r1 = llm.chat([  # doctest: +SKIP
            ...     Message(role=MessageRole.USER, chunks=[TextChunk(content="Hello")])
            ... ])
            >>> r2 = llm.chat([  # doctest: +SKIP
            ...     Message(role=MessageRole.USER, chunks=[TextChunk(content="What did I just say?")])
            ... ])

            ```
        - Streaming with a built-in web search tool
            ```python
            >>> llm = Responses(  # doctest: +SKIP
            ...     model="gpt-4o-mini",
            ...     built_in_tools=[{"type": "web_search_preview"}],
            ...     api_key="sk-test",
            ... )
            >>> messages = [  # doctest: +SKIP
            ...     Message(role=MessageRole.USER, chunks=[TextChunk(content="Latest AI news")])
            ... ]
            >>> for chunk in llm.chat(messages, stream=True):  # doctest: +SKIP
            ...     print(chunk.delta, end="", flush=True)

            ```

    See Also:
        Completions: Chat Completions API provider for models such as ``gpt-4o``.
        serapeum.openai.llm.base.Client: SDK client lifecycle management.
        serapeum.openai.converters.ResponsesOutputParser: Parses Responses API
            output items into a :class:`~serapeum.core.llms.ChatResponse`.
        serapeum.openai.converters.ResponsesStreamAccumulator: Accumulates
            streaming Responses API events.
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
    ) -> Responses:
        """Extract ``previous_response_id`` before Pydantic validation.

        A ``mode="wrap"`` validator that pops the ``"previous_response_id"`` key
        from the raw input dict before Pydantic's normal field processing runs
        (since the field is a private attribute, not a declared Pydantic field),
        then restores it onto ``_previous_response_id`` after the instance is
        constructed.

        This allows callers to inject a continuation response ID at construction
        time without Pydantic rejecting the extra key.

        Args:
            data: Raw constructor input (typically a ``dict``).
            handler: Pydantic's next-in-chain validation handler.

        Returns:
            OpenAIResponses: Fully initialised instance with
                ``_previous_response_id`` set from the injected value (or
                ``None`` if not provided).
        """
        previous_response_id = None
        if isinstance(data, dict):
            previous_response_id = data.pop("previous_response_id", None)

        instance = handler(data)
        instance._previous_response_id = previous_response_id

        return instance

    @model_validator(mode="after")
    def _validate_model(self) -> Responses:
        """Enforce O1 temperature and sync ``store`` with ``track_previous_responses``.

        Sets ``temperature`` to ``1.0`` for O1 reasoning models (whose API
        ignores custom temperatures) and forces ``store=True`` whenever
        ``track_previous_responses`` is enabled (server-side storage is required
        for conversation continuation).

        Returns:
            OpenAIResponses: The validated instance (required by Pydantic).
        """
        if self.model in O1_MODELS:
            self.temperature = 1.0

        if self.track_previous_responses:
            self.store = True

        return self

    @classmethod
    def class_name(cls) -> str:
        """Return the canonical provider identifier for this class.

        Returns:
            str: The string ``"openai_responses_llm"``.
        """
        return "openai_responses_llm"

    @property
    def metadata(self) -> Metadata:
        """Return static model metadata for this provider instance.

        Uses :attr:`context_window` when explicitly set; otherwise infers the
        context window from the model name via
        :func:`~serapeum.openai.models.openai_modelname_to_contextsize`.
        Always reports ``is_chat_model=True`` because the Responses API only
        supports chat-style interaction.

        Returns:
            Metadata: Populated metadata object including context window, output
                token cap, capability flags, and model name.
        """
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

    def _should_use_structure_outputs(self) -> bool:
        """Return ``False`` — the Responses API always uses function-calling.

        The Responses API does not support the Chat Completions
        ``response_format`` parameter. All structured output requests go
        through the tool-call forcing path (``tool_choice="required"``).

        Returns:
            bool: Always ``False``.
        """
        return False

    def _is_azure_client(self) -> bool:
        """Return whether the underlying SDK client targets Azure OpenAI.

        Checks whether the synchronous :attr:`client` is an instance of
        :class:`~openai.AzureOpenAI`.

        Returns:
            bool: ``True`` if the provider is configured for Azure OpenAI;
                ``False`` for the standard OpenAI endpoint.
        """
        return isinstance(self.client, AzureOpenAI)

    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Build the keyword-argument dict for a Responses API call.

        Assembles model-level defaults (model, temperature, top_p, reasoning,
        tools, etc.) and merges them with :attr:`additional_kwargs` and any
        per-call overrides. For O1 models with ``reasoning_options`` set, the
        sampling parameters (``temperature``, ``top_p``, ``presence_penalty``,
        ``frequency_penalty``) are excluded from the payload because the API
        rejects them when reasoning is configured.

        Args:
            **kwargs: Per-call overrides merged on top of the model defaults.
                A ``"tools"`` key in *kwargs* is appended to
                :attr:`built_in_tools` rather than replacing it.

        Returns:
            dict[str, Any]: Merged keyword dict ready to be unpacked into the
                OpenAI SDK ``responses.create`` call.
        """
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
        """Send messages to the Responses API and return a chat response.

        Delegates to :meth:`_chat` for non-streaming calls and
        :meth:`_stream_chat` for streaming calls.

        Args:
            messages: Conversation turn list, ordered oldest-first.
            stream: When ``True`` yields incremental
                :class:`~serapeum.core.llms.ChatResponse` objects as events
                arrive; when ``False`` (default) blocks until the full response
                is ready.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Returns:
            ChatResponse | ChatResponseGen: A complete response or a synchronous
                token generator.
        """
        result: ChatResponse | ChatResponseGen = (
            self._stream_chat(messages, **kwargs)
            if stream
            else self._chat(messages, **kwargs)
        )
        return result

    @retry(is_retryable, logger)
    def _chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """Send a non-streaming Responses API request and parse the output.

        Decorated with :func:`~serapeum.core.retry.retry`. Converts
        :class:`~serapeum.core.llms.Message` objects to Responses API message
        dicts, calls ``responses.create``, then parses the ``output`` list
        through :class:`~serapeum.openai.converters.ResponsesOutputParser`.

        When :attr:`track_previous_responses` is enabled, stores the response
        ID in ``_previous_response_id`` for automatic conversation continuation.
        Also copies reasoning token counts from
        ``usage.output_tokens_details`` into any
        :class:`~serapeum.core.llms.ThinkingBlock` chunks in the response.

        Args:
            messages: Conversation history to send to the model.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Returns:
            ChatResponse: Parsed response with ``raw`` set to the SDK
                :class:`~openai.types.responses.Response` object and ``usage``
                stored in ``additional_kwargs``.
        """
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

    @retry(is_retryable, logger)
    def _stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        """Yield streaming Responses API events as chat response chunks.

        Decorated with :func:`~serapeum.core.retry.retry`. Events are
        accumulated by
        :class:`~serapeum.openai.converters.ResponsesStreamAccumulator`. Each
        yielded chunk carries the block list and delta from the accumulator.

        When :attr:`track_previous_responses` is enabled, the instance's
        ``_previous_response_id`` is updated whenever the accumulator receives
        the response ID from the ``response.created`` event.

        Args:
            messages: Conversation history to send to the model.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Yields:
            ChatResponse: Incremental response chunks. ``delta`` is the new
                text fragment; ``additional_kwargs`` mirrors the accumulator's
                current state (tool calls, usage, built-in tool results, etc.).
        """
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
            is_responses_api=True,
        )

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
        """Asynchronously send messages to the Responses API.

        Async counterpart to :meth:`chat`.

        Args:
            messages: Conversation turn list, ordered oldest-first.
            stream: When ``True`` returns a
                :class:`~serapeum.core.llms.ChatResponseAsyncGen` async
                generator; when ``False`` (default) awaits the full response.
            **kwargs: Extra keyword arguments forwarded to the underlying API
                method.

        Returns:
            ChatResponse | ChatResponseAsyncGen: A complete response or an async
                token generator.
        """
        result: ChatResponse | ChatResponseAsyncGen = (
            await self._astream_chat(messages, **kwargs)
            if stream
            else await self._achat(messages, **kwargs)
        )
        return result

    @retry(is_retryable, logger)
    async def _achat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponse:
        """Send a non-streaming async Responses API request.

        Async counterpart to :meth:`_chat`. Decorated with
        :func:`~serapeum.core.retry.retry`.

        Args:
            messages: Conversation history to send to the model.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Returns:
            ChatResponse: Parsed response with ``raw`` set to the SDK
                :class:`~openai.types.responses.Response` object and ``usage``
                stored in ``additional_kwargs``.
        """
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

    @retry(is_retryable, logger, stream=True)
    async def _astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Return an async generator that yields streaming Responses API events.

        Async counterpart to :meth:`_stream_chat`. Decorated with
        :func:`~serapeum.core.retry.retry`. Uses an inner ``gen()`` coroutine
        for safe retry semantics.

        Args:
            messages: Conversation history to send to the model.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Returns:
            ChatResponseAsyncGen: Async generator yielding incremental
                :class:`~serapeum.core.llms.ChatResponse` chunks.
        """
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
        message: str | Message | None = None,
        chat_history: list[Message] | None = None,
        allow_parallel_tool_calls: bool = True,
        tool_required: bool = False,
        tool_choice: str | dict | None = None,
        verbose: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the payload dict for a tool-assisted Responses API call.

        Converts each :class:`~serapeum.core.tools.BaseTool` into the flat tool
        spec format required by the Responses API
        (``{"type": "function", ...}`` rather than the nested
        ``{"type": "function", "function": {...}}`` used by Chat Completions),
        applies ``strict`` mode, and assembles the ``messages`` list.

        Args:
            tools: Tools whose specs are to be included in the request.
            message: Optional user message to append to the conversation.
                Accepts a plain string or a
                :class:`~serapeum.core.llms.Message`.
            chat_history: Prior conversation turns. Defaults to an empty list.
            allow_parallel_tool_calls: When ``True`` (default), passes
                ``parallel_tool_calls=True`` to the API, allowing the model to
                invoke multiple tools in a single turn.
            tool_required: When ``True``, forces ``tool_choice="required"``.
            tool_choice: Explicit tool-choice override. Overrides
                *tool_required*.
            verbose: Unused flag preserved for interface compatibility.
            strict: Override for :attr:`strict`. When ``None``, the instance
                attribute value is used.
            **kwargs: Additional fields merged into the returned payload dict.

        Returns:
            dict[str, Any]: Payload dict with keys ``"messages"``, ``"tools"``,
                ``"tool_choice"``, ``"parallel_tool_calls"``, and any extra
                *kwargs*.
        """

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

        if isinstance(message, str):
            message = Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content=message)],
            )

        messages = chat_history or []
        if message:
            messages.append(message)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": resolve_tool_choice(tool_choice, tool_required)
            if tool_specs
            else None,
            "parallel_tool_calls": allow_parallel_tool_calls,
            **kwargs,
        }


# Backward-compatible alias
OpenAIResponses = Responses
