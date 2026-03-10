from __future__ import annotations

import logging
from json.decoder import JSONDecodeError
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
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    Metadata,
    MessageRole,
    ToolCallBlock,
    TextChunk,
    ChatToCompletion
)
from pydantic import (
    Field,
    model_validator,
    ConfigDict
)

from serapeum.core.configs.defaults import (
    DEFAULT_TEMPERATURE,
)

from serapeum.core.llms import FunctionCallingLLM
from serapeum.core.tools import ToolCallArguments
from serapeum.core.utils.schemas import parse_partial_json
from serapeum.openai.data.models import (
    O1_MODELS,
    is_chat_model,
    is_chatcomp_api_supported,
    is_function_calling_model,
    openai_modelname_to_contextsize,
)
from serapeum.openai.parsers import (
    ChatMessageParser,
    LogProbParser,
    ToolCallAccumulator,
    to_openai_message_dicts,
)
from serapeum.openai.llm.base import Client, ModelMetadata, StructuredOutput
from serapeum.openai.retry import is_retryable
from serapeum.openai.utils import force_single_tool_call, resolve_tool_choice
from serapeum.core.retry import retry
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from serapeum.core.base.llms.utils import (
    completion_to_chat_decorator,
    stream_completion_to_chat_decorator,
    acompletion_to_chat_decorator,
    astream_completion_to_chat_decorator
)

if TYPE_CHECKING:
    from serapeum.core.tools import BaseTool

logger = logging.getLogger(__name__)


class OpenAI(StructuredOutput, ModelMetadata, Client, ChatToCompletion, FunctionCallingLLM):
    """OpenAI Chat Completions and Completions API provider.

    Routes requests to either the Chat Completions API (for chat models such as
    ``gpt-4o`` and ``gpt-4-turbo``) or the legacy Completions API (for instruct
    models such as ``gpt-3.5-turbo-instruct``) based on the model name.
    Streaming, function calling, structured outputs via native JSON-schema, audio
    modalities, and per-token log-probability extraction are all supported.

    Inherits connection management from :class:`~serapeum.openai.llm.base.Client`,
    model-metadata utilities from :class:`~serapeum.openai.llm.base.ModelMetadata`,
    native JSON-schema structured outputs from
    :class:`~serapeum.openai.llm.base.StructuredOutput`, and the
    chat-to-completion interface bridge from
    :class:`~serapeum.core.llms.ChatToCompletion`.

    Args:
        model: OpenAI model identifier (e.g. ``"gpt-4o"``, ``"gpt-4-turbo"``).
            Models that are exclusive to the Responses API (e.g. ``"o3"``) are
            rejected at construction time; use :class:`OpenAIResponses` instead.
        temperature: Sampling temperature in ``[0.0, 2.0]``. Higher values produce
            more creative, less deterministic output. Automatically forced to
            ``1.0`` for O1 reasoning models. Defaults to
            :data:`~serapeum.core.configs.defaults.DEFAULT_TEMPERATURE`.
        max_tokens: Maximum number of tokens to generate. When ``None`` the field
            is omitted from the request (the model applies its own default cap).
            For legacy Completions requests the value is inferred from the model's
            context window minus the prompt token count when possible.
        logprobs: If ``True``, per-token log-probabilities are included in the
            response. Defaults to ``None`` (disabled).
        top_logprobs: Number of top-token log-probability alternatives returned
            per position (range ``0``–``20``). Only used when ``logprobs=True``.
            Defaults to ``0``.
        additional_kwargs: Extra parameters forwarded verbatim to the OpenAI API
            request body (e.g. ``{"frequency_penalty": 0.5}``).
        strict: When ``True``, tool call and structured-output schemas are sent
            with ``"strict": true``, enabling deterministic JSON adherence.
            Defaults to ``False``.
        reasoning_effort: Reasoning budget for O1 models. One of ``"none"``,
            ``"minimal"``, ``"low"``, ``"medium"``, ``"high"``, or ``"xhigh"``.
            Ignored for non-O1 models. Defaults to ``None``.
        modalities: Output modalities requested from the model
            (e.g. ``["text", "audio"]``). Defaults to ``None`` (text only).
        audio_config: Audio-output configuration mapping (voice ID, format, etc.).
            Only relevant when ``"audio"`` appears in ``modalities``.
        api_key: OpenAI API key. Falls back to the ``OPENAI_API_KEY`` environment
            variable when ``None``.
        api_base: Base URL override for the API endpoint (useful for proxies,
            local servers, or Azure-compatible gateways).
        api_version: API version string (primarily used with Azure OpenAI).
        timeout: Per-request HTTP timeout in seconds. Defaults to ``60.0``.
        default_headers: Additional HTTP headers sent with every API request.

    Raises:
        ValueError: If ``model`` is only supported by the Responses API (i.e.
            :func:`~serapeum.openai.models.is_chatcomp_api_supported` returns
            ``False``).

    Examples:
        - Basic chat completion
            ```python
            >>> from serapeum.openai import OpenAI
            >>> from serapeum.core.llms import Message, MessageRole
            >>> llm = OpenAI(model="gpt-4o-mini", api_key="sk-test")  # doctest: +SKIP
            >>> resp = llm.chat([Message(role=MessageRole.USER, content="Say hi")])  # doctest: +SKIP
            >>> isinstance(resp.message.content, str)  # doctest: +SKIP
            True

            ```
        - Streaming text completion
            ```python
            >>> llm = OpenAI(model="gpt-3.5-turbo-instruct", api_key="sk-test")  # doctest: +SKIP
            >>> for chunk in llm.complete("Once upon a time", stream=True):  # doctest: +SKIP
            ...     print(chunk.delta, end="", flush=True)

            ```
        - Structured output prediction
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.prompts import PromptTemplate
            >>> class Summary(BaseModel):
            ...     title: str
            ...     points: list[str]
            >>> llm = OpenAI(model="gpt-4o-mini", api_key="sk-test")  # doctest: +SKIP
            >>> tmpl = PromptTemplate("Summarise: {text}")  # doctest: +SKIP
            >>> out = llm.parse(Summary, tmpl, text="…")  # doctest: +SKIP
            >>> isinstance(out, Summary)  # doctest: +SKIP
            True

            ```

    See Also:
        OpenAIResponses: Responses API provider for models exclusive to that API.
        serapeum.openai.llm.base.Client: OpenAI SDK client lifecycle management.
        serapeum.openai.llm.base.StructuredOutput: Native JSON-schema structured outputs.
        serapeum.openai.llm.base.ModelMetadata: Model name normalisation and tokenizer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: str = Field(description="The OpenAI model to use.")
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use during generation.",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int | None = Field(
        description="The maximum number of tokens to generate.",
        default=None,
        gt=0,
    )
    logprobs: bool | None = Field(
        description="Whether to return logprobs per token.",
        default=None,
    )
    top_logprobs: int = Field(
        description="The number of top token log probs to return.",
        default=0,
        ge=0,
        le=20,
    )
    additional_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the OpenAI API."
    )
    strict: bool = Field(
        default=False,
        description="Whether to use strict mode for invoking tools/using schemas.",
    )
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = Field(
        default=None,
        description="The effort to use for reasoning models.",
    )
    modalities: list[str] | None = Field(
        default=None,
        description="The output modalities to use for the model.",
    )
    audio_config: dict[str, Any] | None = Field(
        default=None,
        description="The audio configuration to use for the model.",
    )

    @model_validator(mode="after")
    def _validate_model(self) -> OpenAI:
        """Force O1 temperature and validate model is Chat Completions-compatible.

        Sets ``temperature`` to ``1.0`` for O1 reasoning models whose API does
        not honour a custom temperature, and rejects any model that requires the
        Responses API rather than the Chat Completions API.

        Returns:
            OpenAI: The validated instance (required by Pydantic).

        Raises:
            ValueError: If the model is not supported by the Chat Completions
                API (e.g. ``"o3"`` or ``"o4"``). Use :class:`OpenAIResponses`
                instead.
        """
        if self.model in O1_MODELS:
            self.temperature = 1.0

        if not is_chatcomp_api_supported(self.model):
            raise ValueError(
                f"Cannot use model {self.model} as it is only supported by the "
                "Responses API. Use the OpenAIResponses class for it."
            )

        return self

    @classmethod
    def class_name(cls) -> str:
        """Return the canonical provider identifier for this class.

        Returns:
            str: The string ``"openai"``.
        """
        return "openai"

    @property
    def metadata(self) -> Metadata:
        """Return static model metadata for this provider instance.

        Resolves context-window size and capability flags from the model name.
        O1 reasoning models always report ``system_role=MessageRole.USER``
        because their API does not accept a dedicated system message.

        Returns:
            Metadata: Populated metadata object with context window, output
                token cap, chat/function-calling capability flags, model name,
                and the appropriate system-message role.
        """
        return Metadata(
            context_window=openai_modelname_to_contextsize(self._get_model_name()),
            num_output=self.max_tokens or -1,
            is_chat_model=is_chat_model(model=self._get_model_name()),
            is_function_calling_model=is_function_calling_model(
                model=self._get_model_name()
            ),
            model_name=self.model,
            system_role=MessageRole.USER
            if self.model in O1_MODELS
            else MessageRole.SYSTEM,
        )

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
        """Send messages to the model and return a chat response.

        Routes to the Chat Completions API or the legacy Completions API
        depending on whether the model is a chat model (determined by
        :meth:`_use_chat_completions`). Pass ``stream=True`` to receive a token
        generator instead of a complete response.

        Args:
            messages: Conversation turn list, ordered oldest-first.
            stream: When ``True`` yields :class:`~serapeum.core.llms.ChatResponse`
                objects progressively as tokens arrive; when ``False`` (default)
                blocks until the full response is ready.
            **kwargs: Extra keyword arguments forwarded to the underlying API
                method. Pass ``use_chat_completions=True|False`` to override the
                automatic API selection for this call.

        Returns:
            ChatResponse | ChatResponseGen: A complete response when
                ``stream=False``, or a synchronous generator of partial responses
                when ``stream=True``.
        """
        if stream:
            if self._use_chat_completions(kwargs):
                result: ChatResponse | ChatResponseGen = self._stream_chat(
                    messages, **kwargs
                )
            else:
                result = stream_completion_to_chat_decorator(self._stream_complete)(
                    messages, **kwargs
                )
        elif self._use_chat_completions(kwargs):
            result = self._chat(messages, **kwargs)
        else:
            result = completion_to_chat_decorator(self._complete)(messages, **kwargs)
        return result

    @overload
    def complete(
        self, prompt: str, formatted: bool = ..., *, stream: Literal[False] = ..., **kwargs: Any,
    ) -> CompletionResponse: ...

    @overload
    def complete(
        self, prompt: str, formatted: bool = ..., *, stream: Literal[True], **kwargs: Any,
    ) -> CompletionResponseGen: ...

    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse | CompletionResponseGen:
        """Generate a text completion for a prompt string.

        Raises immediately if ``"audio"`` is in :attr:`modalities` (audio output
        is only supported via :meth:`chat` / :meth:`achat`). For chat models,
        delegates to the Chat Completions API via the
        :class:`~serapeum.core.llms.ChatToCompletion` mixin; for legacy instruct
        models it calls the Completions endpoint directly.

        Args:
            prompt: The text prompt to complete.
            formatted: Passed through to the base class; indicates whether the
                prompt has already been formatted by a template.
            stream: When ``True`` returns a generator of incremental
                :class:`~serapeum.core.llms.CompletionResponse` objects.
            **kwargs: Additional keyword arguments forwarded to the API call.

        Returns:
            CompletionResponse | CompletionResponseGen: Full response or
                streaming generator.

        Raises:
            ValueError: If ``"audio"`` is in :attr:`modalities`.
        """
        if self.modalities and "audio" in self.modalities:
            raise ValueError(
                "Audio is not supported for completion. Use chat/achat instead."
            )

        if self._use_chat_completions(kwargs):
            # Let the mixin handle chat→completion conversion
            result: CompletionResponse | CompletionResponseGen = super().complete(
                prompt, formatted, stream=stream, **kwargs
            )
        elif stream:
            result = self._stream_complete(prompt, **kwargs)
        else:
            result = self._complete(prompt, **kwargs)
        return result

    def _use_chat_completions(self, kwargs: dict[str, Any]) -> bool:
        """Determine whether to use the Chat Completions API for this call.

        Checks the ``use_chat_completions`` override key in *kwargs* first;
        falls back to the model's ``is_chat_model`` flag from :attr:`metadata`.

        Args:
            kwargs: Runtime keyword arguments that may contain a
                ``"use_chat_completions"`` boolean override.

        Returns:
            bool: ``True`` to route through the Chat Completions endpoint;
                ``False`` to use the legacy Completions endpoint.
        """
        if "use_chat_completions" in kwargs:
            return kwargs["use_chat_completions"]
        return self.metadata.is_chat_model


    def _get_model_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Build the keyword-argument dict for an OpenAI API call.

        Merges model-level defaults (``model``, ``temperature``, ``max_tokens``,
        ``logprobs``) with :attr:`additional_kwargs` and any per-call overrides.
        Applies O1-specific transformations:

        * Renames ``max_tokens`` → ``max_completion_tokens``.
        * Injects ``reasoning_effort`` when set.
        * Strips ``stream_options`` from non-streaming payloads.
        * Adds ``modalities`` and ``audio`` fields when configured.

        Args:
            **kwargs: Per-call overrides merged on top of the model defaults.

        Returns:
            dict[str, Any]: Merged keyword dict ready to be unpacked into the
                OpenAI SDK client call.
        """
        base_kwargs = {"model": self.model, "temperature": self.temperature, **kwargs}
        if self.max_tokens is not None:
            # If max_tokens is None, don't include in the payload:
            # https://platform.openai.com/docs/api-reference/chat
            # https://platform.openai.com/docs/api-reference/completions
            base_kwargs["max_tokens"] = self.max_tokens
        if self.logprobs is not None and self.logprobs is True:
            if self.metadata.is_chat_model:
                base_kwargs["logprobs"] = self.logprobs
                base_kwargs["top_logprobs"] = self.top_logprobs
            else:
                base_kwargs["logprobs"] = self.top_logprobs  # int in this case

        # can't send stream_options to the API when not streaming
        all_kwargs = {**base_kwargs, **self.additional_kwargs}
        if "stream" not in all_kwargs and "stream_options" in all_kwargs:
            del all_kwargs["stream_options"]
        if self.model in O1_MODELS and base_kwargs.get("max_tokens") is not None:
            # O1 models use max_completion_tokens instead of max_tokens
            all_kwargs["max_completion_tokens"] = all_kwargs.get(
                "max_completion_tokens", all_kwargs["max_tokens"]
            )
            all_kwargs.pop("max_tokens", None)
        if self.model in O1_MODELS and self.reasoning_effort is not None:
            # O1 models support reasoning_effort of none, minimal, low, medium, high, xhigh
            all_kwargs["reasoning_effort"] = self.reasoning_effort

        if self.modalities is not None:
            all_kwargs["modalities"] = self.modalities
        if self.audio_config is not None:
            all_kwargs["audio"] = self.audio_config

        return all_kwargs

    @retry(is_retryable, logger)
    def _chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        """Send a non-streaming chat-completions request and return the response.

        Decorated with :func:`~serapeum.core.retry.retry` for automatic retries
        on transient errors. Converts :class:`~serapeum.core.llms.Message` objects
        to OpenAI message dicts via
        :func:`~serapeum.openai.converters.to_openai_message_dicts`, then parses
        the API response through
        :class:`~serapeum.openai.converters.ChatMessageParser`.

        Args:
            messages: Conversation history to send to the model.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Returns:
            ChatResponse: Parsed chat response, optionally including
                log-probability annotations in ``likelihood_score`` and
                token-count statistics in ``additional_kwargs``.
        """
        client = self.client
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
        )

        response = client.chat.completions.create(
            messages=message_dicts,
            stream=False,
            **self._get_model_kwargs(**kwargs),
        )

        openai_message = response.choices[0].message
        message = ChatMessageParser(
            openai_message, modalities=self.modalities or ["text"]
        ).build()
        openai_token_logprobs = response.choices[0].logprobs
        logprobs = None

        if openai_token_logprobs and openai_token_logprobs.content:
            logprobs = LogProbParser.from_tokens(openai_token_logprobs.content)

        return ChatResponse(
            message=message,
            raw=response,
            likelihood_score=logprobs,
            additional_kwargs=self._get_response_token_counts(response),
        )

    @retry(is_retryable, logger)
    def _stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        """Yield streaming chat-completions chunks from the Chat Completions API.

        Decorated with :func:`~serapeum.core.retry.retry`. Each yielded chunk
        carries the cumulative text so far plus the per-step delta. Tool-call
        fragments are merged incrementally by
        :class:`~serapeum.openai.converters.ToolCallAccumulator` so that
        ``additional_kwargs["tool_calls"]`` always reflects the latest accumulated
        state.

        Audio output is not supported in streaming mode.

        Args:
            messages: Conversation history to send to the model.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Yields:
            ChatResponse: Incremental response chunks. ``delta`` holds the new
                text fragment; ``message.chunks`` holds the complete text so far
                as a :class:`~serapeum.core.llms.TextChunk`.

        Raises:
            ValueError: If ``"audio"`` is in :attr:`modalities`.
        """
        if self.modalities and "audio" in self.modalities:
            raise ValueError("Audio is not supported for chat streaming")

        client = self.client
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
        )

        content = ""
        accumulator = ToolCallAccumulator()

        is_function = False
        for response in client.chat.completions.create(
            messages=message_dicts,
            **self._get_model_kwargs(stream=True, **kwargs),
        ):
            blocks = []
            response = response
            if len(response.choices) > 0:
                delta = response.choices[0].delta
            else:
                delta = ChoiceDelta()

            if delta is None:
                continue

            # check if this chunk is the start of a function call
            if delta.tool_calls:
                is_function = True

            # update using deltas
            role = delta.role or MessageRole.ASSISTANT
            content_delta = delta.content or ""
            content += content_delta
            blocks.append(TextChunk(content=content))

            additional_kwargs = {}
            if is_function:
                accumulator.update(delta.tool_calls)
                if accumulator.tool_calls:
                    additional_kwargs["tool_calls"] = accumulator.tool_calls
                    for tool_call in accumulator.tool_calls:
                        if tool_call.function:
                            blocks.append(
                                ToolCallBlock(
                                    tool_call_id=tool_call.id,
                                    tool_kwargs=tool_call.function.arguments or {},
                                    tool_name=tool_call.function.name or "",
                                )
                            )

            yield ChatResponse(
                message=Message(
                    role=role,
                    chunks=blocks,
                    additional_kwargs=additional_kwargs,
                ),
                delta=content_delta,
                raw=response,
                additional_kwargs=self._get_response_token_counts(response),
            )

    @retry(is_retryable, logger)
    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Send a non-streaming Completions API request and return the response.

        Decorated with :func:`~serapeum.core.retry.retry`. Calls
        :meth:`_update_max_tokens` to infer ``max_tokens`` from the context
        window when not set explicitly.

        Args:
            prompt: Prompt string sent to the Completions endpoint.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Returns:
            CompletionResponse: Parsed completion response, optionally including
                log-probability annotations and token-count statistics.
        """
        client = self.client
        all_kwargs = self._get_model_kwargs(**kwargs)
        self._update_max_tokens(all_kwargs, prompt)

        response = client.completions.create(
            prompt=prompt,
            stream=False,
            **all_kwargs,
        )
        text = response.choices[0].text

        openai_completion_logprobs = response.choices[0].logprobs
        logprobs = None
        if openai_completion_logprobs:
            logprobs = LogProbParser.from_completions(openai_completion_logprobs)

        return CompletionResponse(
            text=text,
            raw=response,
            likelihood_score=logprobs,
            additional_kwargs=self._get_response_token_counts(response),
        )

    @retry(is_retryable, logger)
    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Yield streaming Completions API chunks.

        Decorated with :func:`~serapeum.core.retry.retry`. Each chunk carries
        the cumulative text and the new ``delta`` fragment.

        Args:
            prompt: Prompt string sent to the Completions endpoint.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Yields:
            CompletionResponse: Incremental completion chunks.
        """
        client = self.client
        all_kwargs = self._get_model_kwargs(stream=True, **kwargs)
        self._update_max_tokens(all_kwargs, prompt)

        text = ""
        for response in client.completions.create(
            prompt=prompt,
            **all_kwargs,
        ):
            if len(response.choices) > 0:
                delta = response.choices[0].text
                if delta is None:
                    delta = ""
            else:
                delta = ""
            text += delta
            yield CompletionResponse(
                delta=delta,
                text=text,
                raw=response,
                additional_kwargs=self._get_response_token_counts(response),
            )

    def _update_max_tokens(self, all_kwargs: dict[str, Any], prompt: str) -> None:
        """Infer and inject ``max_tokens`` into *all_kwargs* for Completions calls.

        The legacy Completions endpoint requires an explicit ``max_tokens``.
        When :attr:`max_tokens` is unset and a tiktoken tokenizer is available,
        this method counts prompt tokens and sets ``max_tokens`` to the remaining
        capacity within the model's context window.

        If :attr:`max_tokens` is already set or no tokenizer is available, the
        method is a no-op.

        Args:
            all_kwargs: Mutable keyword-argument dict to be sent to the API. The
                ``"max_tokens"`` key is added in-place.
            prompt: The prompt string used to calculate token consumption.

        Raises:
            ValueError: If the prompt exceeds the model's context window and no
                positive ``max_tokens`` budget remains.
        """
        if self.max_tokens is not None or self._tokenizer is None:
            return
        # NOTE: non-chat completion endpoint requires max_tokens to be set
        num_tokens = len(self._tokenizer.encode(prompt))
        max_tokens = self.metadata.context_window - num_tokens
        if max_tokens <= 0:
            raise ValueError(
                f"The prompt has {num_tokens} tokens, which is too long for"
                " the model. Please use a prompt that fits within"
                f" {self.metadata.context_window} tokens."
            )
        all_kwargs["max_tokens"] = max_tokens

    @staticmethod
    def _get_response_token_counts(raw_response: Any) -> dict:
        """Extract token-usage statistics from an API response object.

        Handles both SDK response objects (with a ``.usage`` attribute) and
        legacy plain-dict response representations.

        Args:
            raw_response: The raw value returned by the OpenAI SDK (a
                ``ChatCompletion``, ``Completion``, a streaming chunk, or a
                dict).

        Returns:
            dict: Mapping with keys ``"prompt_tokens"``,
                ``"completion_tokens"``, and ``"total_tokens"``. Returns an
                empty dict when usage information is not present or cannot be
                parsed.
        """
        if hasattr(raw_response, "usage"):
            try:
                prompt_tokens = raw_response.usage.prompt_tokens
                completion_tokens = raw_response.usage.completion_tokens
                total_tokens = raw_response.usage.total_tokens
            except AttributeError:
                return {}
        elif isinstance(raw_response, dict):
            usage = raw_response.get("usage", {})
            # NOTE: other model providers that use the OpenAI client may not report usage
            if usage is None:
                return {}
            # Backwards compatibility with old dict type
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        else:
            return {}

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


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
        """Asynchronously send messages to the model and return a chat response.

        Async counterpart to :meth:`chat`. Routes to the Chat Completions API or
        the legacy Completions API based on :meth:`_use_chat_completions`.

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
        if stream:
            if self._use_chat_completions(kwargs):
                result: ChatResponse | ChatResponseAsyncGen = (
                    await self._astream_chat(messages, **kwargs)
                )
            else:
                result = await astream_completion_to_chat_decorator(
                    self._astream_complete
                )(messages, **kwargs)
        elif self._use_chat_completions(kwargs):
            result = await self._achat(messages, **kwargs)
        else:
            result = await acompletion_to_chat_decorator(self._acomplete)(
                messages, **kwargs
            )
        return result

    @overload
    async def acomplete(
        self, prompt: str, formatted: bool = ..., *, stream: Literal[False] = ..., **kwargs: Any,
    ) -> CompletionResponse: ...

    @overload
    async def acomplete(
        self, prompt: str, formatted: bool = ..., *, stream: Literal[True], **kwargs: Any,
    ) -> CompletionResponseAsyncGen: ...

    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse | CompletionResponseAsyncGen:
        """Asynchronously generate a text completion for a prompt string.

        Async counterpart to :meth:`complete`. Raises immediately if ``"audio"``
        is in :attr:`modalities`.

        Args:
            prompt: The text prompt to complete.
            formatted: Passed through to the base class.
            stream: When ``True`` returns an async generator of incremental
                responses.
            **kwargs: Additional keyword arguments forwarded to the API call.

        Returns:
            CompletionResponse | CompletionResponseAsyncGen: Full response or
                async streaming generator.

        Raises:
            ValueError: If ``"audio"`` is in :attr:`modalities`.
        """
        if self.modalities and "audio" in self.modalities:
            raise ValueError(
                "Audio is not supported for completion. Use chat/achat instead."
            )

        if self._use_chat_completions(kwargs):
            # Let the mixin handle chat→completion conversion
            result: CompletionResponse | CompletionResponseAsyncGen = (
                await super().acomplete(prompt, formatted, stream=stream, **kwargs)
            )
        elif stream:
            result = await self._astream_complete(prompt, **kwargs)
        else:
            result = await self._acomplete(prompt, **kwargs)
        return result

    @retry(is_retryable, logger)
    async def _achat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponse:
        """Send a non-streaming async chat-completions request.

        Async counterpart to :meth:`_chat`. Decorated with
        :func:`~serapeum.core.retry.retry`.

        Args:
            messages: Conversation history to send to the model.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Returns:
            ChatResponse: Parsed chat response with optional log-probability and
                token-count data.
        """
        aclient = self.async_client
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
        )

        response = await aclient.chat.completions.create(
            messages=message_dicts,
            stream=False,
            **self._get_model_kwargs(**kwargs),
        )

        openai_message = response.choices[0].message
        message = ChatMessageParser(
            openai_message, modalities=self.modalities or ["text"]
        ).build()
        openai_token_logprobs = response.choices[0].logprobs

        logprobs = None
        if openai_token_logprobs and openai_token_logprobs.content:
            logprobs = LogProbParser.from_tokens(openai_token_logprobs.content)

        return ChatResponse(
            message=message,
            raw=response,
            likelihood_score=logprobs,
            additional_kwargs=self._get_response_token_counts(response),
        )

    @retry(is_retryable, logger, stream=True)
    async def _astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Return an async generator that yields streaming chat-completion chunks.

        Async counterpart to :meth:`_stream_chat`. Decorated with
        :func:`~serapeum.core.retry.retry`. Uses an inner ``gen()`` coroutine to
        avoid holding a reference to the async generator across retries.

        Audio output is not supported in streaming mode.

        Args:
            messages: Conversation history to send to the model.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Returns:
            ChatResponseAsyncGen: Async generator yielding incremental
                :class:`~serapeum.core.llms.ChatResponse` chunks.

        Raises:
            ValueError: If ``"audio"`` is in :attr:`modalities`.
        """
        if self.modalities and "audio" in self.modalities:
            raise ValueError("Audio is not supported for chat streaming")

        aclient = self.async_client
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
        )

        async def gen() -> ChatResponseAsyncGen:
            content = ""
            accumulator = ToolCallAccumulator()

            is_function = False
            first_chat_chunk = True
            async for response in await aclient.chat.completions.create(
                messages=message_dicts,
                **self._get_model_kwargs(stream=True, **kwargs),
            ):
                blocks = []
                response = response
                if len(response.choices) > 0:
                    # check if the first chunk has neither content nor tool_calls
                    # this happens when 1106 models end up calling multiple tools
                    if (
                        first_chat_chunk
                        and response.choices[0].delta.content is None
                        and response.choices[0].delta.tool_calls is None
                    ):
                        first_chat_chunk = False
                        continue
                    delta = response.choices[0].delta
                else:
                    delta = ChoiceDelta()
                first_chat_chunk = False

                if delta is None:
                    continue

                # check if this chunk is the start of a function call
                if delta.tool_calls:
                    is_function = True

                # update using deltas
                role = delta.role or MessageRole.ASSISTANT
                content_delta = delta.content or ""
                content += content_delta
                blocks.append(TextChunk(content=content))

                additional_kwargs = {}
                if is_function:
                    accumulator.update(delta.tool_calls)
                    if accumulator.tool_calls:
                        additional_kwargs["tool_calls"] = accumulator.tool_calls
                        for tool_call in accumulator.tool_calls:
                            if tool_call.function:
                                blocks.append(
                                    ToolCallBlock(
                                        tool_call_id=tool_call.id,
                                        tool_kwargs=tool_call.function.arguments or {},
                                        tool_name=tool_call.function.name or "",
                                    )
                                )

                yield ChatResponse(
                    message=Message(
                        role=role,
                        chunks=blocks,
                        additional_kwargs=additional_kwargs,
                    ),
                    delta=content_delta,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    @retry(is_retryable, logger)
    async def _acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Send a non-streaming async Completions API request.

        Async counterpart to :meth:`_complete`. Decorated with
        :func:`~serapeum.core.retry.retry`.

        Args:
            prompt: Prompt string sent to the Completions endpoint.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Returns:
            CompletionResponse: Parsed completion response with optional
                log-probability and token-count data.
        """
        aclient = self.async_client
        all_kwargs = self._get_model_kwargs(**kwargs)
        self._update_max_tokens(all_kwargs, prompt)

        response = await aclient.completions.create(
            prompt=prompt,
            stream=False,
            **all_kwargs,
        )

        text = response.choices[0].text
        openai_completion_logprobs = response.choices[0].logprobs
        logprobs = None
        if openai_completion_logprobs:
            logprobs = LogProbParser.from_completions(openai_completion_logprobs)

        return CompletionResponse(
            text=text,
            raw=response,
            likelihood_score=logprobs,
            additional_kwargs=self._get_response_token_counts(response),
        )

    @retry(is_retryable, logger, stream=True)
    async def _astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Return an async generator that yields streaming Completions API chunks.

        Async counterpart to :meth:`_stream_complete`. Decorated with
        :func:`~serapeum.core.retry.retry`. Uses an inner ``gen()`` coroutine
        for safe retry semantics.

        Args:
            prompt: Prompt string sent to the Completions endpoint.
            **kwargs: Extra keyword arguments forwarded to
                :meth:`_get_model_kwargs`.

        Returns:
            CompletionResponseAsyncGen: Async generator yielding incremental
                :class:`~serapeum.core.llms.CompletionResponse` chunks.
        """
        aclient = self.async_client
        all_kwargs = self._get_model_kwargs(stream=True, **kwargs)
        self._update_max_tokens(all_kwargs, prompt)

        async def gen() -> CompletionResponseAsyncGen:
            text = ""
            async for response in await aclient.completions.create(
                prompt=prompt,
                **all_kwargs,
            ):
                if len(response.choices) > 0:
                    delta = response.choices[0].text
                    if delta is None:
                        delta = ""
                else:
                    delta = ""
                text += delta
                yield CompletionResponse(
                    delta=delta,
                    text=text,
                    raw=response,
                    additional_kwargs=self._get_response_token_counts(response),
                )

        return gen()

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: str | Message | None = None,
        chat_history: list[Message] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        tool_required: bool = False,
        tool_choice: str | dict | None = None,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the payload dict for a tool-assisted chat call.

        Converts each :class:`~serapeum.core.tools.BaseTool` into an OpenAI
        tool spec dict, applies ``strict`` mode if requested, and assembles the
        ``messages`` list (appending *user_msg* to *chat_history*).

        Args:
            tools: Tools whose specs are to be included in the request.
            user_msg: Optional user message to append to the conversation.
                Accepts either a plain string (wrapped in a USER
                :class:`~serapeum.core.llms.Message`) or an existing ``Message``
                object.
            chat_history: Prior conversation turns. Defaults to an empty list.
            verbose: Unused flag preserved for interface compatibility.
            allow_parallel_tool_calls: Unused flag preserved for interface
                compatibility (enforcement happens via
                :meth:`_validate_chat_with_tools_response`).
            tool_required: When ``True``, forces the model to invoke at least
                one tool by setting ``tool_choice="required"``.
            tool_choice: Explicit tool-choice override (``"auto"``,
                ``"required"``, ``"none"``, or a function-name dict). Overrides
                *tool_required* when both are provided.
            strict: Override for :attr:`strict`. When ``None``, the instance
                attribute value is used.
            **kwargs: Additional fields merged into the returned payload dict.

        Returns:
            dict[str, Any]: Payload dict with keys ``"messages"``, ``"tools"``,
                ``"tool_choice"``, and any extra *kwargs*.
        """
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ]

        # if strict is passed in, use, else default to the class-level attribute, else default to True`
        if strict is not None:
            strict = strict
        else:
            strict = self.strict

        if self.metadata.is_function_calling_model:
            for tool_spec in tool_specs:
                if tool_spec["type"] == "function":
                    tool_spec["function"]["strict"] = strict
                    # in current openai 1.40.0 it is always false.
                    tool_spec["function"]["parameters"]["additionalProperties"] = False

        if isinstance(user_msg, str):
            user_msg = Message(role=MessageRole.USER, content=user_msg)

        messages = list(chat_history or [])
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
            "tool_choice": resolve_tool_choice(tool_choice, tool_required)
            if tool_specs
            else None,
            **kwargs,
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: Sequence["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Enforce single-tool-call constraint on a chat-with-tools response.

        When *allow_parallel_tool_calls* is ``False``, calls
        :func:`~serapeum.openai.utils.force_single_tool_call` to raise if the
        response contains more than one tool call.

        Args:
            response: The chat response to validate.
            tools: Tools that were available during the call (unused but
                required by the interface).
            allow_parallel_tool_calls: When ``False`` (default), raises if the
                model invoked multiple tools simultaneously.
            **kwargs: Accepted for interface compatibility; not forwarded.

        Returns:
            ChatResponse: The unmodified *response* after validation.

        Raises:
            ValueError: If parallel tool calls are disallowed and the response
                contains more than one tool call.
        """
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> list[ToolCallArguments]:
        """Extract parsed tool-call arguments from a chat response.

        Looks for :class:`~serapeum.core.llms.ToolCallBlock` entries in
        ``response.message.chunks`` first (the modern path). Falls back to
        ``response.message.additional_kwargs["tool_calls"]`` for backward
        compatibility with older response formats.

        Attempts to parse tool-call arguments as JSON using
        :func:`~serapeum.core.utils.schemas.parse_partial_json`; if parsing
        fails, the arguments default to an empty dict.

        Args:
            response: A :class:`~serapeum.core.llms.ChatResponse` that may
                contain tool calls.
            error_on_no_tool_call: When ``True`` (default), raises
                :exc:`ValueError` if no tool calls are found.
            **kwargs: Accepted for interface compatibility; not forwarded.

        Returns:
            list[ToolCallArguments]: One entry per tool call, each carrying the
                ``tool_id``, ``tool_name``, and parsed ``tool_kwargs``.

        Raises:
            ValueError: If *error_on_no_tool_call* is ``True`` and no tool
                calls are present.
            ValueError: If the legacy path encounters a tool type other than
                ``"function"``.
        """
        tool_call_blocks = [
            block
            for block in response.message.chunks
            if isinstance(block, ToolCallBlock)
        ]
        if tool_call_blocks:
            tool_selections = []
            for tool_call in tool_call_blocks:
                # this should handle both complete and partial jsons
                try:
                    if isinstance(tool_call.tool_kwargs, str):
                        argument_dict = parse_partial_json(tool_call.tool_kwargs)
                    else:
                        argument_dict = tool_call.tool_kwargs
                except (ValueError, TypeError, JSONDecodeError):
                    argument_dict = {}

                tool_selections.append(
                    ToolCallArguments(
                        tool_id=tool_call.tool_call_id or "",
                        tool_name=tool_call.tool_name,
                        tool_kwargs=argument_dict,
                    )
                )
            result = tool_selections
        else:
            # Legacy backward-compatible path: read from additional_kwargs.
            legacy_tool_calls = response.message.additional_kwargs.get("tool_calls", [])

            if legacy_tool_calls:
                tool_selections = []
                for tool_call in legacy_tool_calls:
                    if tool_call.type != "function":
                        raise ValueError("Invalid tool type. Unsupported by OpenAI llm")

                    # this should handle both complete and partial jsons
                    try:
                        argument_dict = parse_partial_json(tool_call.function.arguments)
                    except (ValueError, TypeError, JSONDecodeError):
                        argument_dict = {}

                    tool_selections.append(
                        ToolCallArguments(
                            tool_id=tool_call.id,
                            tool_name=tool_call.function.name,
                            tool_kwargs=argument_dict,
                        )
                    )
                result = tool_selections
            elif error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(legacy_tool_calls)} tool calls."
                )
            else:
                result = []

        return result


