"""Ollama LLM implementation providing chat, streaming, and structured output capabilities.

This module implements the Ollama provider for the Serapeum framework, offering
a complete LLM interface with support for:
- Synchronous and asynchronous chat completions
- Streaming responses with delta updates
- Tool/function calling when supported by the model
- Structured outputs using JSON mode and Pydantic validation
- Multi-modal inputs (text and images)
- Automatic client management with event loop handling

The implementation follows the FunctionCallingLLM protocol and integrates with
Ollama's local or remote servers for model inference.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Any, AsyncGenerator, Generator

import ollama as ollama_sdk  # type: ignore[attr-defined]
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from serapeum.core.configs.defaults import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from serapeum.core.llms import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    ChatToCompletionMixin,
    FunctionCallingLLM,
    Image,
    Message,
    MessageList,
    MessageRole,
    Metadata,
    TextChunk,
)

from serapeum.core.llms.orchestrators import StreamingObjectProcessor
from serapeum.core.prompts import PromptTemplate
from serapeum.core.tools import ArgumentCoercer, ToolCallArguments, ToolCallError
from serapeum.core.types import StructuredLLMMode
from serapeum.ollama.client import OllamaClientMixin

if TYPE_CHECKING:
    from serapeum.core.tools.types import BaseTool

DEFAULT_REQUEST_TIMEOUT = 60.0


def get_additional_kwargs(
    response: dict[str, Any], exclude: tuple[str, ...]
) -> dict[str, Any]:
    """Filter out excluded keys from a response dictionary.

    Args:
        response (dict[str, Any]):
            Source dictionary, typically a raw provider response.
        exclude (Tuple[str, ...]):
            Keys that should be omitted from the returned mapping.

    Returns:
        dict[str, Any]:
            A new dictionary containing only entries whose keys are not present in ``exclude``.

    Examples:
        - Keep only non-excluded keys
            ```python
            >>> from serapeum.ollama.llm import get_additional_kwargs  # type: ignore
            >>> get_additional_kwargs({"a": 1, "b": 2, "keep": 3}, ("a", "b"))
            {'keep': 3}

            ```
        - Return all keys when no exclusions are provided
            ```python
            >>> get_additional_kwargs({"x": 10}, tuple())
            {'x': 10}

            ```
    """
    return {k: v for k, v in response.items() if k not in exclude}


def force_single_tool_call(response: ChatResponse) -> None:
    """Mutate a response to include at most a single tool call.

    Ollama may return multiple tool calls within a single assistant message. Some
    consumers require a single call at a time. This helper trims the list to the
    first occurrence in-place.

    Args:
        response (ChatResponse):
            Parsed chat response whose ``message.additional_kwargs['tool_calls']``
            may contain multiple entries.

    Returns:
        None: The function mutates ``response`` and returns nothing.

    Examples:
        - Truncate multiple tool calls to one
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, ChatResponse
            >>> r = ChatResponse(message=Message(
            ...     role=MessageRole.ASSISTANT,
            ...     content="",
            ...     additional_kwargs={
            ...         "tool_calls": [
            ...             {"function": {"name": "a", "arguments": {}}},
            ...             {"function": {"name": "b", "arguments": {}}},
            ...         ]
            ...     },
            ... ))
            >>> force_single_tool_call(r)
            >>> len(r.message.additional_kwargs.get("tool_calls", []))
            1

            ```
        - Leave empty or single tool call lists unchanged
            ```python
            >>> from serapeum.core.llms import Message, MessageRole, ChatResponse
            >>> r = ChatResponse(message=Message(role=MessageRole.ASSISTANT, content=""))
            >>> force_single_tool_call(r)
            >>> r.message.additional_kwargs.get("tool_calls") is None or len(r.message.additional_kwargs.get("tool_calls", [])) == 0
            True

            ```
    """
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if tool_calls and len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


class Ollama(OllamaClientMixin, ChatToCompletionMixin, FunctionCallingLLM):
    """Ollama LLM adapter for chat, streaming, structured output, and tool calling.

    Integrates with a local or remote Ollama server to expose synchronous and
    asynchronous chat, streaming, and structured-output interfaces. The adapter
    implements the ``FunctionCallingLLM`` protocol so it can be composed with
    tool-orchestrating layers from ``serapeum.core``.

    **Local vs Ollama Cloud**

    Without ``api_key`` the class talks to a local Ollama server at
    ``http://localhost:11434``. **To switch to Ollama Cloud, set** ``api_key``
    **— that is the only change required.** When ``api_key`` is provided and
    ``base_url`` is still the local default, ``base_url`` is automatically
    switched to ``https://api.ollama.com``; no manual URL update is needed.
    An explicit non-default ``base_url`` is always preserved so custom remote
    deployments are unaffected. The ``api_key`` value is intentionally
    **excluded from** ``model_dump()`` and ``model_dump_json()`` so it is
    never accidentally serialised to disk or logs.

    **Lazy client initialisation**

    The underlying ``ollama.Client`` and ``ollama.AsyncClient`` instances are
    created on the first ``client`` / ``async_client`` property access, not at
    construction time. For testing you can bypass the network by injecting
    pre-built clients via the constructor::

        Ollama(model="m", client=my_mock_client, async_client=my_async_mock)

    Args:
        model: Ollama model identifier, e.g. ``"llama3.1"`` or
            ``"qwen3-next:80b"``.
        base_url: URL of the Ollama server. Defaults to
            ``"http://localhost:11434"``. Automatically switched to
            ``"https://api.ollama.com"`` when ``api_key`` is provided and
            this value is still the local default.
        api_key: The single switch between local and cloud. When ``None``
            (default), requests go to the local Ollama server. When set,
            requests are routed to Ollama Cloud and ``base_url`` is
            automatically updated. **Not serialised by** ``model_dump()``.
        temperature: Sampling temperature in ``[0.0, 1.0]``. Higher values
            increase creativity; lower values produce more deterministic
            output. Defaults to ``0.75``.
        context_window: Maximum number of tokens in the context window.
            Defaults to ``DEFAULT_CONTEXT_WINDOW``.
        request_timeout: HTTP request timeout in seconds. Defaults to
            ``60.0``.
        prompt_key: Key used when formatting prompt templates. Defaults to
            ``"prompt"``.
        json_mode: When ``True``, sends ``format="json"`` to Ollama so the
            model is constrained to emit valid JSON. Defaults to ``False``.
        additional_kwargs: Extra provider options forwarded to the Ollama
            ``options`` field (e.g. ``{"mirostat": 2}``).
        is_function_calling_model: Whether the chosen model supports tool /
            function calling. Defaults to ``True``.
        keep_alive: How long the model stays loaded in memory after a
            request — a duration string (``"5m"``, ``"1h"``) or float
            seconds. Defaults to ``"5m"``.
        client: Pre-built synchronous ``ollama.Client`` for dependency
            injection or testing. When ``None``, the client is created
            lazily on first access.
        async_client: Pre-built asynchronous ``ollama.AsyncClient`` for
            dependency injection or testing. When ``None``, a client is
            created per event loop on first access.

    Examples:
        - Basic chat via Ollama Cloud
            ```python
            >>> import os
            >>> from serapeum.core.llms import Message, MessageRole
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> llm = Ollama(
            ...     model="qwen3-next:80b",
            ...     api_key=os.environ.get("OLLAMA_API_KEY"),
            ...     temperature=0.0,
            ...     request_timeout=120,
            ... )
            >>> response = llm.chat([Message(role=MessageRole.USER, content="Say 'hello'.")])
            >>> print(response) # doctest: +SKIP
            assistant:

            hello

            ```
        - Supplying api_key automatically switches base_url to Ollama Cloud
            ```python
            >>> import os
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> from serapeum.ollama.client import OLLAMA_CLOUD_BASE_URL
            >>> llm = Ollama(model="qwen3-next:80b", api_key=os.environ.get("OLLAMA_API_KEY"))
            >>> llm.base_url == OLLAMA_CLOUD_BASE_URL
            True

            ```
        - api_key is excluded from model_dump() — it is never serialised
            ```python
            >>> import os
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> llm = Ollama(model="qwen3-next:80b", api_key=os.environ.get("OLLAMA_API_KEY"))
            >>> "api_key" in llm.model_dump()
            False

            ```
        - Inject a mock client for unit tests (no network required)
            ```python
            >>> from unittest.mock import MagicMock
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> mock = MagicMock()
            >>> llm = Ollama(model="m", client=mock)
            >>> llm.client is mock
            True

            ```
        - Stream chat deltas via Ollama Cloud
            ```python
            >>> import os
            >>> from serapeum.core.llms import Message, MessageRole
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> llm = Ollama(
            ...     model="qwen3-next:80b",
            ...     api_key=os.environ.get("OLLAMA_API_KEY"),
            ...     temperature=0.0,
            ...     request_timeout=120,
            ... )
            >>> chunks = list(llm.stream_chat([
            ...     Message(role=MessageRole.USER, content="Say 'hello'.")
            ... ]))
            >>> chunks
            [ChatResponse(raw={'model': 'qwen3-next:80b', ... chunks=[TextChunk(content='', path=None, url=None, type='text')])),
            ChatResponse(raw={'model': 'qwen3-next:80b', ...  chunks=[TextChunk(content='', path=None, url=None, type='text')])),
            ChatResponse(raw={'model': 'qwen3-next:80b', ... chunks=[TextChunk(content='', path=None, url=None, type='text')])),
            ChatResponse(raw={'model': 'qwen3-next:80b', ... chunks=[TextChunk(content='', path=None, url=None, type='text')])),
            ...

            ```
        - Structured output parsed into a Pydantic model
            ```python
            >>> import os
            >>> from pydantic import BaseModel
            >>> from serapeum.core.prompts import PromptTemplate
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> class Capital(BaseModel):
            ...     city: str
            ...     country: str
            >>> llm = Ollama(
            ...     model="qwen3-next:80b",
            ...     api_key=os.environ.get("OLLAMA_API_KEY"),
            ...     temperature=0.0,
            ...     request_timeout=120,
            ... )
            >>> prompt = PromptTemplate("Extract city and country from: {text}")
            >>> result = llm.structured_predict(
            ...     Capital, prompt, text="Paris is the capital of France."
            ... )
            >>> isinstance(result, Capital)
            True

            ```
        - List all models available on the Ollama Cloud server
            ```python
            >>> import os
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> llm = Ollama(model="qwen3-next:80b", api_key=os.environ.get("OLLAMA_API_KEY"))
            >>> models = llm.list_models()
            >>> isinstance(models, list)
            True

            ```
        - Async model listing via Ollama Cloud
            ```python
            >>> import asyncio, os
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> llm = Ollama(model="qwen3-next:80b", api_key=os.environ.get("OLLAMA_API_KEY"))
            >>> async def get_models():
            ...     return await llm.alist_models()
            >>> isinstance(asyncio.run(get_models()), list)
            True

            ```

    See Also:
        OllamaEmbedding: Companion class for generating embeddings with Ollama.
        OllamaClientMixin: Shared connection logic, URL resolution, and client injection.
        chat: Synchronous chat completion.
        stream_chat: Streaming chat yielding incremental deltas.
        achat: Asynchronous chat completion.
        astream_chat: Asynchronous streaming chat.
        structured_predict: Structured output via JSON schema and Pydantic validation.
        list_models: List all models available on the Ollama server.
        alist_models: Async variant of list_models.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = Field(description="The Ollama model to use.")
    temperature: float = Field(
        default=0.75,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    request_timeout: float = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        description="The timeout for making http request to Ollama API server",
    )
    prompt_key: str = Field(
        default="prompt", description="The key to use for the prompt in API calls."
    )
    json_mode: bool = Field(
        default=False,
        description="Whether to use JSON mode for the Ollama API.",
    )
    additional_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model parameters for the Ollama API.",
    )
    is_function_calling_model: bool = Field(
        default=True,
        description="Whether the model is a function calling model.",
    )
    keep_alive: float | str | None = Field(
        default="5m",
        description="controls how long the model will stay loaded into memory following the request(default: 5m)",
    )

    # Track the event loop associated with the async client to avoid
    # reusing a client bound to a closed event loop across tests/runs
    _async_client_loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    def _build_client_kwargs(self) -> dict[str, Any]:
        """Extend base client kwargs with the request timeout for the LLM client."""
        return {**super()._build_client_kwargs(), "timeout": self.request_timeout}

    @classmethod
    def class_name(cls) -> str:
        """Return the registered class name for this provider adapter.

        Returns:
            str: Provider identifier used in registries or logs.

        Examples:
            - Retrieve the class identifier
                ```python
                >>> from serapeum.ollama import Ollama      # type: ignore[attr-defined]
                >>> Ollama.class_name()
                'Ollama'

                ```
        """
        return "Ollama"

    @property
    def metadata(self) -> Metadata:
        """LLM metadata.

        Returns:
            Metadata: Static capabilities such as context window and chat support.

        Examples:
            - Inspect chat model capabilities
                ```python
                >>> from serapeum.ollama import Ollama      # type: ignore[attr-defined]
                >>> Ollama(model="m").metadata.is_chat_model
                True

                ```
        """
        return Metadata(
            context_window=self.context_window,
            num_output=DEFAULT_NUM_OUTPUTS,
            model_name=self.model,
            is_chat_model=True,
            is_function_calling_model=self.is_function_calling_model,
        )

    @property
    def client(self) -> ollama_sdk.Client:      # type: ignore
        """Synchronous Ollama client lazily bound to ``base_url``.

        Returns:
            Client: Underlying Ollama client instance.

        Examples:
            - Lazily create the client on first access
                ```python
                >>> from serapeum.ollama import Ollama      # type: ignore[attr-defined]
                >>> llm = Ollama(model="m", base_url="http://localhost:11434", request_timeout=1.0)
                >>> c = llm.client  # doctest: +SKIP
                >>> type(c).__name__  # doctest: +SKIP
                'Client'
                >>> hasattr(c, "chat")  # doctest: +SKIP
                True

                ```
        """
        if self._client is None:
            self._client = ollama_sdk.Client(**self._build_client_kwargs())      # type: ignore
        return self._client

    def _ensure_async_client(self) -> ollama_sdk.AsyncClient:   # type: ignore
        """Return a per-event-loop AsyncClient, recreating when loop changes or closes.

        This avoids ``Event loop is closed`` errors when test runners (e.g.,
        pytest-asyncio) create and close event loops between invocations.

        Returns:
            AsyncClient: Async client instance associated with the current loop.

        Examples:
            - Re-create the client when the loop changes
                ```python
                >>> import asyncio
                >>> from serapeum.ollama import Ollama      # type: ignore[attr-defined]
                >>> llm = Ollama(model="m")
                >>> c1 = llm._ensure_async_client()  # doctest: +SKIP
                >>> c2 = llm._ensure_async_client()  # doctest: +SKIP
                >>> c1 is c2  # doctest: +SKIP
                True

                ```
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None  # No running loop available in this context

        client_kwargs = self._build_client_kwargs()

        cached_loop = getattr(self, "_async_client_loop", None)
        if self._async_client is None:
            # No client yet: create and bind to current loop (may be None)
            self._async_client = ollama_sdk.AsyncClient(**client_kwargs)        # type: ignore
            self._async_client_loop = current_loop
        else:
            # If no loop recorded yet (e.g., injected client), bind without recreation
            if cached_loop is None:
                self._async_client_loop = current_loop
            # Recreate if the current loop is closed
            elif (
                current_loop is not None
                and hasattr(current_loop, "is_closed")
                and current_loop.is_closed()
            ):
                self._async_client = ollama_sdk.AsyncClient(**client_kwargs)        # type: ignore
                self._async_client_loop = current_loop
            # Or if the cached loop has been closed since creation
            elif hasattr(cached_loop, "is_closed") and cached_loop.is_closed():
                self._async_client = ollama_sdk.AsyncClient(**client_kwargs)        # type: ignore
                self._async_client_loop = current_loop
            else:
                # Reuse existing client even if loop identity differs but both are open
                self._async_client_loop = current_loop

        return self._async_client

    @property
    def async_client(self) -> ollama_sdk.AsyncClient:       # type: ignore
        """Async Ollama client bound to the current asyncio event loop.

        This property lazily creates or reuses an AsyncClient instance, automatically
        handling event loop changes and closures. It's safe to call across different
        async contexts (e.g., multiple pytest-asyncio tests) as it detects closed
        loops and recreates the client as needed.

        Returns:
            The async client instance used for asynchronous operations.

        Examples:
            - Access async client for manual API calls
                ```python
                >>> import asyncio
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> llm = Ollama(model="llama3.1")
                >>> async def check_client():    # doctest: +SKIP
                ...     client = llm.async_client
                ...     return hasattr(client, "chat")
                >>> asyncio.run(check_client())  # Returns True # doctest: +SKIP

                ```

        See Also:
            _ensure_async_client: Ensures the client matches the active event loop.
            client: Synchronous Ollama client property.
        """
        return self._ensure_async_client()

    @property
    def _model_kwargs(self) -> dict[str, Any]:
        """Assemble provider options forwarded under the ``options`` field.

        Returns:
            dict[str, Any]: Merged dictionary where ``additional_kwargs`` override
            base defaults such as ``temperature`` and ``num_ctx``.

        Examples:
            - Merge user-provided options with defaults
                ```python
                >>> from serapeum.ollama import Ollama      # type: ignore[attr-defined]
                >>> llm = Ollama(model="m", additional_kwargs={"mirostat": 2, "temperature": 0.9})
                >>> print(llm._model_kwargs)
                {'temperature': 0.9, 'num_ctx': 3900, 'mirostat': 2}

                ```
        """
        base_kwargs = {
            "temperature": self.temperature,
            "num_ctx": self.context_window,
        }
        return {
            **base_kwargs,
            **self.additional_kwargs,
        }

    @staticmethod
    def _convert_to_ollama_messages(
        messages: MessageList
    ) -> list[dict[str, Any]]:
        """Convert internal MessageList to the Ollama wire format.

        Args:
            messages (MessageList):
                Sequence of messages to be sent to Ollama.

        Returns:
            Dict: A list of dicts compatible with the Ollama chat API (role,
            content, optional images, and tool_calls).

        Raises:
            ValueError: If a content chunk type is unsupported.

        Examples:
            - Text-only conversion
                ```python
                >>> from serapeum.core.llms import Message, MessageList, MessageRole
                >>> from serapeum.ollama import Ollama      # type: ignore[attr-defined]
                >>> llm = Ollama(model="m")
                >>> wire = llm._convert_to_ollama_messages(
                ...     MessageList.from_list([
                ...         Message(role=MessageRole.USER, content="hello"),
                ...     ])
                ... )
                >>> print(wire)
                [{'role': 'user', 'content': 'hello'}]

                ```
        """
        ollama_messages = []
        for message in messages:
            cur_ollama_message = {
                "role": message.role.value,
                "content": "",
            }
            for block in message.chunks:
                if isinstance(block, TextChunk):
                    cur_ollama_message["content"] += block.content
                elif isinstance(block, Image):
                    if "images" not in cur_ollama_message:
                        cur_ollama_message["images"] = []

                    # Prefer an explicit base64 attribute if provided by the caller
                    b64 = getattr(block, "base64", None)
                    if b64 is None:
                        extra = getattr(block, "model_extra", None) or {}
                        b64 = extra.get("base64")

                    if b64 is None:
                        try:
                            b64 = dict(block).get("base64")
                        except Exception:
                            b64 = None

                    if b64 is None:
                        b64 = getattr(block, "__dict__", {}).get("base64")

                    if isinstance(b64, (bytes, str)):
                        base64_str = (
                            b64.decode("utf-8") if isinstance(b64, bytes) else b64
                        )
                    else:
                        # Fall back to resolving image bytes via the helper
                        base64_str = (
                            block.resolve_image(as_base64=True).read().decode("utf-8")
                        )

                    cur_ollama_message["images"].append(base64_str)
                else:
                    raise ValueError(f"Unsupported block type: {type(block)}")

            if "tool_calls" in message.additional_kwargs:
                cur_ollama_message["tool_calls"] = message.additional_kwargs[
                    "tool_calls"
                ]

            ollama_messages.append(cur_ollama_message)

        return ollama_messages

    @staticmethod
    def _get_response_token_counts(raw_response: dict[str, Any]) -> dict[str, Any]:
        """Extract token usage fields from a raw Ollama response.

        Args:
            raw_response (dict):
                Provider response possibly containing token counts.

        Returns:
            dict: Mapping with ``prompt_tokens``, ``completion_tokens``, and
            ``total_tokens`` when available; otherwise an empty dict.

        Examples:
            - Compute totals when both fields are present
                ```python
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> Ollama._get_response_token_counts({"prompt_eval_count": 2, "eval_count": 3})
                {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}

                ```
        """
        token_counts = {}
        try:
            prompt_tokens = raw_response["prompt_eval_count"]
            completion_tokens = raw_response["eval_count"]
            total_tokens = prompt_tokens + completion_tokens
            token_counts = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        except (KeyError, TypeError):
            pass

        return token_counts

    def _prepare_chat_with_tools(
        self,
        tools: list["BaseTool"],
        user_msg: str | Message | None = None,
        chat_history: list[Message] | None = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare a chat payload including tool specifications.

        Args:
            tools (List[BaseTool]): Tools to expose to the model (converted using OpenAI schema).
            user_msg (str | Message | None): Optional user message to append.
            chat_history (list[Message] | None): Optional existing conversation history.
            verbose (bool): Currently unused verbosity flag.
            allow_parallel_tool_calls (bool): Indicator forwarded to validators.
            **kwargs (Any): Reserved for future extensions.

        Returns:
            dict[str, Any]: Dict with ``messages`` and ``tools`` entries suitable for chat calls.

        Examples:
            - Combine history, a new user message, and tool specs
                ```python
                >>> from serapeum.core.llms import Message, MessageRole
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> class T:
                ...     def __init__(self, n):
                ...         class M:
                ...             def to_openai_tool(self, skip_length_check=False):
                ...                 return {"type": "function", "function": {"name": n}}
                ...         self.metadata = M()
                ...
                >>> llm = Ollama(model="m")
                >>> payload = llm._prepare_chat_with_tools([T("t1")], user_msg="hi", chat_history=[Message(role=MessageRole.SYSTEM, content="s")])
                >>> len(payload["messages"])
                2
                >>> payload["messages"][0].role == MessageRole.SYSTEM
                True
                >>> payload["messages"][1].role == MessageRole.USER
                True
                >>> payload["tools"]
                [{'type': 'function', 'function': {'name': 't1'}}]

                ```
        """
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ]

        if isinstance(user_msg, str):
            user_msg = Message(role=MessageRole.USER, content=user_msg)

        messages = list(chat_history or [])
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: list["BaseTool"],
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Validate and normalize a chat-with-tools response.

        If ``allow_parallel_tool_calls`` is ``False``, the response is mutated to
        include at most a single tool call.

        Args:
            response (ChatResponse): Response to validate.
            tools (List[BaseTool]): Tools originally requested (unused, reserved for future checks).
            allow_parallel_tool_calls (bool): Whether multiple tool calls are allowed.
            **kwargs (Any): Reserved for future options.

        Returns:
            ChatResponse: The validated response (possibly mutated in-place).

        Examples:
            - Force single tool call when multiple are present (remove the multiple tool calls and leave only the first)
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, ChatResponse
                >>> llm = Ollama(model="m")
                >>> response = ChatResponse(
                ...     message=Message(
                ...         role=MessageRole.ASSISTANT, content="", additional_kwargs={
                ...             "tool_calls": [
                ...                 {"function": {"name": "a", "arguments": {}}},
                ...                 {"function": {"name": "b", "arguments": {}}},
                ...             ]
                ...         }
                ...     )
                ... )
                >>> validated_response = llm._validate_chat_with_tools_response(
                ...     response,
                ...     tools=[],
                ...     allow_parallel_tool_calls=False,
                ... )
                >>> len(validated_response.message.additional_kwargs.get("tool_calls"))
                1

                ```
        """
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
    ) -> list[ToolCallArguments]:
        """Extract tool call selections from a chat response.

        Args:
            response (ChatResponse): Response potentially containing tool calls.
            error_on_no_tool_call (bool): Whether to raise when no tool calls are present.

        Returns:
            list[ToolCallArguments]: Parsed tool selections (empty when allowed and none present).

        Raises:
            ValueError: When ``error_on_no_tool_call`` is ``True`` and no tool calls exist.

        Examples:
            - Parse a single tool call
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, ChatResponse
                >>> llm = Ollama(model="m")
                >>> r = ChatResponse(
                ...     message=Message(
                ...         role=MessageRole.ASSISTANT,
                ...         content="",
                ...         additional_kwargs={
                ...             "tool_calls": [
                ...                 {"function": {"name": "run", "arguments": {"a": 1}}}
                ...             ]
                ...         },
                ...     )
                ... )
                >>> calls = llm.get_tool_calls_from_response(r)
                >>> (
                ...     calls[0].tool_id,
                ...     calls[0].tool_name,
                ...     calls[0].tool_kwargs["a"],
                ... ) == ("run", "run", 1)
                True

                ```
            - Raise when no tool call is present and errors are enabled
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, ChatResponse
                >>> llm = Ollama(model="m")
                >>> empty = ChatResponse(message=Message(role=MessageRole.ASSISTANT, content=""))
                >>> try:
                ...     _ = llm.get_tool_calls_from_response(empty, error_on_no_tool_call=True)
                ... except ToolCallError as e:
                ...     msg = str(e)
                >>> 'Expected at least one tool call' in msg
                True

                ```
        """
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])
        if not tool_calls or len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ToolCallError(
                    f"Expected at least one tool call, but the LLM response contained "
                    f"{len(tool_calls) if tool_calls else 0} tool calls.",
                    tool_name=None,
                )

            else:
                return []

        tool_selections = []
        coercer = ArgumentCoercer()

        for tool_call in tool_calls:
            # Coerce arguments to proper types (handles JSON strings, type mismatches, etc.)
            raw_arguments = tool_call["function"]["arguments"]
            argument_dict = coercer.coerce(raw_arguments)

            tool_selections.append(
                ToolCallArguments(
                    # tool ids not provided by Ollama
                    tool_id=tool_call["function"]["name"],
                    tool_name=tool_call["function"]["name"],
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

    @staticmethod
    def _build_chat_response(raw: Any) -> ChatResponse:
        """Build a ``ChatResponse`` from a raw (non-streaming) Ollama API response.

        Converts the SDK response object to a plain dict, extracts tool calls and
        token usage, then constructs a typed ``ChatResponse``.

        Args:
            raw: The response object returned by ``ollama.Client.chat`` or
                ``ollama.AsyncClient.chat`` with ``stream=False``.

        Returns:
            ChatResponse: Typed response with message content, role, tool calls,
            and token usage populated in ``raw["usage"]`` when available.

        Examples:
            - Build a response from a minimal raw dict
                ```python
                >>> from serapeum.ollama import Ollama  # type: ignore
                >>> raw = {"message": {"role": "assistant", "content": "Hi"}}
                >>> resp = Ollama._build_chat_response(raw)
                >>> resp.message.content
                'Hi'
                >>> resp.message.additional_kwargs["tool_calls"]
                []

                ```
        """
        raw = dict(raw)
        tool_calls = raw["message"].get("tool_calls") or []
        token_counts = Ollama._get_response_token_counts(raw)
        if token_counts:
            raw["usage"] = token_counts
        return ChatResponse(
            message=Message(
                content=raw["message"]["content"],
                role=raw["message"]["role"],
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=raw,
        )

    def chat(self, messages: MessageList | list[Message], **kwargs: Any) -> ChatResponse:
        """Send a chat request to Ollama and return the assistant message.

        Args:
            messages (MessageList):
                Sequence of chat messages.
            **kwargs (Any):
                Provider-specific overrides such as ``tools`` or ``format``.

        Returns:
            ChatResponse: Parsed response containing the assistant message and optional token usage.

        Examples:
            - Minimal chat against a running Ollama server (requires server and model)
                ```python
                >>> from serapeum.core.llms import Message, MessageRole
                >>> # Ensure `ollama serve` is running and the model is available locally.
                >>> llm = Ollama(model="llama3.1", request_timeout=120)
                >>> resp = llm.chat([Message(role=MessageRole.USER, content="hi")])  # doctest: +SKIP
                >>> print(resp)   # doctest: +SKIP
                Hello! How are you today? Is there something I can help you with or would you like to chat?
                >>> isinstance(resp.message.content, str)   # doctest: +SKIP
                True

                ```
        """
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)
        response_format = kwargs.pop("format", "json" if self.json_mode else None)

        response = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            stream=False,
            format=response_format,
            tools=tools,
            options=self._model_kwargs,
            keep_alive=self.keep_alive,
        )

        return self._build_chat_response(response)

    @staticmethod
    def _parse_tool_call_response(
        tools_dict: dict[str, Any], r: dict[str, Any]
    ) -> ChatResponse:
        """Accumulate streaming content and unique tool calls into a ChatResponse.

        This static method processes individual streaming chunks from Ollama's chat API,
        accumulating text content and de-duplicating tool calls across multiple deltas.
        It maintains state in the tools_dict to track cumulative response text and
        unique tool calls seen so far.

        Args:
            tools_dict: Mutable aggregation state with keys:
                - "response_txt": Accumulated text content
                - "seen_tool_calls": Set of (function_name, arguments) tuples for deduplication
                - "all_tool_calls": List of unique tool call dictionaries
            r: A single streaming chunk from Ollama containing message content and metadata.

        Returns:
            ChatResponse with cumulative message content, the current delta, and all
            unique tool calls accumulated so far.

        Examples:
            - Process streaming chunk with text content
                ```python
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> tools_dict = {"response_txt": "", "seen_tool_calls": set(), "all_tool_calls": []}
                >>> chunk = {"message": {"role": "assistant", "content": "Hello"}}
                >>> response = Ollama._parse_tool_call_response(tools_dict, chunk)
                >>> response.message.content
                'Hello'
                >>> response.delta
                'Hello'

                ```
            - Process chunk with tool calls (deduplicated)
                ```python
                >>> tools_dict = {"response_txt": "", "seen_tool_calls": set(), "all_tool_calls": []}
                >>> chunk1 = {
                ...     "message": {
                ...         "role": "assistant",
                ...         "content": "",
                ...         "tool_calls": [{"function": {"name": "calc", "arguments": {"x": 1}}}]
                ...     }
                ... }
                >>> r1 = Ollama._parse_tool_call_response(tools_dict, chunk1)
                >>> len(tools_dict["all_tool_calls"])
                1
                >>> # Same tool call again - should not duplicate
                >>> r2 = Ollama._parse_tool_call_response(tools_dict, chunk1)
                >>> len(tools_dict["all_tool_calls"])
                1

                ```

        See Also:
            stream_chat: Uses this helper to materialize per-chunk responses.
            astream_chat: Async variant that uses this helper.
        """
        r = dict(r)

        tools_dict["response_txt"] += r["message"]["content"]
        new_tool_calls = [dict(t) for t in (r["message"].get("tool_calls", []) or [])]
        for tool_call in new_tool_calls:
            func_name = str(tool_call["function"]["name"])
            func_args = str(tool_call["function"]["arguments"])
            if (func_name, func_args) not in tools_dict["seen_tool_calls"]:
                tools_dict["seen_tool_calls"].add((func_name, func_args))
                tools_dict["all_tool_calls"].append(tool_call)

        token_counts = Ollama._get_response_token_counts(r)
        if token_counts:
            r["usage"] = token_counts

        return ChatResponse(
            message=Message(
                content=tools_dict["response_txt"],
                role=r["message"]["role"],
                additional_kwargs={"tool_calls": tools_dict["all_tool_calls"]},
            ),
            delta=r["message"]["content"],
            raw=r,
        )

    def stream_chat(self, messages: MessageList | list[Message], **kwargs: Any) -> ChatResponseGen:
        """Stream assistant deltas for a chat request.

        Args:
            messages (MessageList): Sequence of chat messages.
            **kwargs (Any): Provider-specific options such as ``tools`` or ``format``.

        Yields:
            ChatResponse: Incremental responses with ``delta`` and cumulative content.

        Examples:
            - Stream deltas from a real Ollama server (requires server and model)
                ```python
                >>> from serapeum.core.llms import Message, MessageRole
                >>> # Pre-requisites:
                >>> #   1) Start the server: `ollama serve`
                >>> #   2) Pull a model:    `ollama pull llama3.1`
                >>> llm = Ollama(model="llama3.1", request_timeout=180)
                >>> chunks = list(llm.stream_chat([Message(role=MessageRole.USER, content="Say hello succinctly")])) # doctest: +SKIP
                >>> chunks  # doctest: +SKIP
                [ChatResponse(raw={'model': 'llama3.1', 'created_at': '2026-02-15T20:33:53.7035231Z', 'done': False, 'done_reason': None, 'total_duration': None, 'load_duration': None, 'prompt_eval_count': None, 'prompt_eval_duration': None, 'eval_count': None, 'eval_duration': None, 'message': Message(role='assistant', content='Hello', thinking=None, images=None, tool_name=None, tool_calls=None), 'logprobs': None}, likelihood_score=None, additional_kwargs={}, delta='Hello', message=Message(role=<MessageRole.ASSISTANT: 'assistant'>, additional_kwargs={'tool_calls': []}, chunks=[TextChunk(content='Hello', path=None, url=None, type='text')])),
                 ChatResponse(raw={'model': 'llama3.1', 'created_at': '2026-02-15T20:33:53.7201343Z', 'done': False, 'done_reason': None, 'total_duration': None, 'load_duration': None, 'prompt_eval_count': None, 'prompt_eval_duration': None, 'eval_count': None, 'eval_duration': None, 'message': Message(role='assistant', content='!', thinking=None, images=None, tool_name=None, tool_calls=None), 'logprobs': None}, likelihood_score=None, additional_kwargs={}, delta='!', message=Message(role=<MessageRole.ASSISTANT: 'assistant'>, additional_kwargs={'tool_calls': []}, chunks=[TextChunk(content='Hello!', path=None, url=None, type='text')])),
                 ChatResponse(raw={'model': 'llama3.1', 'created_at': '2026-02-15T20:33:53.7350848Z', 'done': True, 'done_reason': 'stop', 'total_duration': 2473382300, 'load_duration': 2159171600, 'prompt_eval_count': 14, 'prompt_eval_duration': 278611400, 'eval_count': 3, 'eval_duration': 29859300, 'message': Message(role='assistant', content='', thinking=None, images=None, tool_name=None, tool_calls=None), 'logprobs': None, 'usage': {'prompt_tokens': 14, 'completion_tokens': 3, 'total_tokens': 17}}, likelihood_score=None, additional_kwargs={}, delta='', message=Message(role=<MessageRole.ASSISTANT: 'assistant'>, additional_kwargs={'tool_calls': []}, chunks=[TextChunk(content='Hello!', path=None, url=None, type='text')]))]
                >>> isinstance(chunks[-1].message.content, str) and len(chunks) >= 1    # doctest: +SKIP
                True

                ```
        """
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)
        response_format = kwargs.pop("format", "json" if self.json_mode else None)

        def gen() -> ChatResponseGen:
            response = self.client.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True,
                format=response_format,
                tools=tools,
                options=self._model_kwargs,
                keep_alive=self.keep_alive,
            )

            tools_dict = {
                "response_txt": "",
                "seen_tool_calls": set(),
                "all_tool_calls": [],
            }

            for r in response:
                if r["message"]["content"] is not None:
                    yield self._parse_tool_call_response(tools_dict, r)

        return gen()

    async def astream_chat(
        self, messages: MessageList | list[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Asynchronously stream assistant deltas for a chat request.

        Async variant of stream_chat that yields ChatResponse chunks as the model
        generates content. Each chunk includes both the current delta and cumulative
        content. Tool calls are de-duplicated across chunks.

        Args:
            messages: Sequence of chat messages forming the conversation context.
            **kwargs: Provider-specific options such as:
                - tools: List of tool specifications for function calling
                - format: Response format (e.g., "json")

        Returns:
            Async generator yielding ChatResponse chunks with incremental deltas
            and cumulative content.

        Examples:
            - Async stream chat responses
                ```python
                >>> import asyncio
                >>> from serapeum.core.llms import Message, MessageRole
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> llm = Ollama(model="llama3.1", request_timeout=120) # doctest: +SKIP
                >>> async def stream_example():
                ...     chunks = []
                ...     async for chunk in await llm.astream_chat([
                ...         Message(role=MessageRole.USER, content="Count to 3")
                ...     ]):
                ...         chunks.append(chunk.delta)
                ...     return len(chunks) > 0
                >>> asyncio.run(stream_example())  # Returns True   # doctest: +SKIP

                ```

        See Also:
            stream_chat: Synchronous streaming variant.
            achat: Non-streaming async chat.
        """
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)
        response_format = kwargs.pop("format", "json" if self.json_mode else None)

        async def gen() -> ChatResponseAsyncGen:
            response = await self.async_client.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True,
                format=response_format,
                tools=tools,
                options=self._model_kwargs,
                keep_alive=self.keep_alive,
            )

            # Some client/mocking setups may return a coroutine that resolves to
            # an async iterator; normalize by awaiting when needed.
            if inspect.iscoroutine(response) and not hasattr(response, "__aiter__"):
                response = await response

            tools_dict = {
                "response_txt": "",
                "seen_tool_calls": set(),
                "all_tool_calls": [],
            }

            async for r in response:
                if r["message"]["content"] is not None:
                    yield self._parse_tool_call_response(tools_dict, r)

        return gen()

    async def achat(self, messages: MessageList | list[Message], **kwargs: Any) -> ChatResponse:
        """Asynchronously send a chat request and return the complete assistant message.

        Async variant of the chat method that sends messages to Ollama and waits
        for the complete response. Unlike astream_chat, this returns a single
        ChatResponse with the full assistant message after generation completes.

        Args:
            messages: Sequence of chat messages forming the conversation context.
            **kwargs: Provider-specific overrides such as:
                - tools: List of tool specifications for function calling
                - format: Response format (e.g., "json")

        Returns:
            ChatResponse containing the complete assistant message, any tool calls,
            and optional token usage statistics.

        Examples:
            - Async chat with minimal setup
                ```python
                >>> import asyncio
                >>> from serapeum.core.llms import Message, MessageRole
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> llm = Ollama(model="llama3.1", request_timeout=120) # doctest: +SKIP
                >>> async def chat_example():   # doctest: +SKIP
                ...     response = await llm.achat([
                ...         Message(role=MessageRole.USER, content="Say hello")
                ...     ])
                ...     return isinstance(response.message.content, str)
                >>> asyncio.run(chat_example())  # Returns True     # doctest: +SKIP

                ```

        See Also:
            chat: Synchronous chat variant.
            astream_chat: Async streaming variant.
        """
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)
        response_format = kwargs.pop("format", "json" if self.json_mode else None)

        response = await self.async_client.chat(
            model=self.model,
            messages=ollama_messages,
            stream=False,
            format=response_format,
            tools=tools,
            options=self._model_kwargs,
            keep_alive=self.keep_alive,
        )

        return self._build_chat_response(response)

    def structured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> BaseModel:
        """Generate structured output conforming to a Pydantic model schema.

        Instructs the Ollama model to emit JSON matching the schema of output_cls,
        then validates and parses the response into a Pydantic instance. When using
        StructuredLLMMode.DEFAULT, this injects the model's JSON schema into the
        format parameter and validates the response content.

        Args:
            output_cls: Target Pydantic model class defining the expected structure.
            prompt: PromptTemplate that will be formatted with prompt_args to create messages.
            llm_kwargs: Additional provider arguments passed to the chat method.
                Defaults to empty dict.
            **prompt_args: Template variables used to format the prompt.

        Returns:
            Instance of output_cls parsed and validated from the model's JSON response.

        Raises:
            ValidationError: If the model's response doesn't match the schema.

        Examples:
            - Extract structured data from unstructured text
                ```python
                >>> from examples.core.structured_outputs.structured_outputs import api_key                >>> import os
                >>> from pydantic import BaseModel, Field
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> class Person(BaseModel):
                ...     name: str = Field(description="Person's full name")
                ...     age: int = Field(description="Person's age in years")
                >>> llm = Ollama(model="llama3.1", api_key=os.environ['OLLAMA_API_KEY'], request_timeout=120)     # doctest: +SKIP
                >>> prompt = PromptTemplate("Extract person info: {text}")  # doctest: +SKIP
                >>> result = llm.structured_predict(    # doctest: +SKIP
                ...     Person,
                ...     prompt,
                ...     text="John Doe is 30 years old"
                ... )
                >>> result  # doctest: +SKIP
                Person(name='John Doe', age=30)

                ```

        See Also:
            astructured_predict: Async variant.
            stream_structured_predict: Streaming counterpart yielding partial models.
        """
        if self.pydantic_program_mode == StructuredLLMMode.DEFAULT:
            llm_kwargs = llm_kwargs or {}
            llm_kwargs["format"] = output_cls.model_json_schema()

            messages = prompt.format_messages(**prompt_args)
            response = self.chat(messages, **llm_kwargs)

            return output_cls.model_validate_json(response.message.content or "")
        else:
            return super().structured_predict(
                output_cls, prompt, llm_kwargs, **prompt_args
            )

    async def astructured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> BaseModel:
        """Asynchronously generate structured output conforming to a Pydantic model schema.

        Async variant of structured_predict. Instructs the Ollama model to emit JSON
        matching the schema of output_cls, then validates and parses the response
        into a Pydantic instance using the async chat interface.

        Args:
            output_cls: Target Pydantic model class defining the expected structure.
            prompt: PromptTemplate that will be formatted with prompt_args to create messages.
            llm_kwargs: Additional provider arguments passed to the achat method.
                Defaults to empty dict.
            **prompt_args: Template variables used to format the prompt.

        Returns:
            Instance of output_cls parsed and validated from the model's JSON response.

        Raises:
            ValidationError: If the model's response doesn't match the schema.

        Examples:
            - Async structured extraction
                ```python
                >>> import asyncio
                >>> from pydantic import BaseModel
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> class City(BaseModel):
                ...     name: str
                ...     country: str
                >>> llm = Ollama(model="llama3.1", request_timeout=120)  # doctest: +SKIP
                >>> async def extract_city():       # doctest: +SKIP
                ...     prompt = PromptTemplate("Extract city: {text}")
                ...     result = await llm.astructured_predict(
                ...         City,
                ...         prompt,
                ...         text="Paris is in France"
                ...     )
                ...     return result.name == "Paris"
                >>> asyncio.run(extract_city())  # Returns True     # doctest: +SKIP

                ```

        See Also:
            structured_predict: Synchronous variant.
            astream_structured_predict: Async streaming variant.
        """
        if self.pydantic_program_mode == StructuredLLMMode.DEFAULT:
            llm_kwargs = llm_kwargs or {}
            llm_kwargs["format"] = output_cls.model_json_schema()

            messages = prompt.format_messages(**prompt_args)
            response = await self.achat(messages, **llm_kwargs)

            return output_cls.model_validate_json(response.message.content or "")
        else:
            return await super().astructured_predict(
                output_cls, prompt, llm_kwargs, **prompt_args
            )

    def stream_structured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> Generator[BaseModel | list[BaseModel], None, None]:
        """Stream incrementally parsed structured objects as the model generates JSON.

        Yields partially complete Pydantic instances as the model streams JSON content,
        allowing early access to structured data before the full response completes.
        Uses StreamingObjectProcessor with flexible mode to handle incomplete JSON.

        Args:
            output_cls: Pydantic model class defining the expected structure.
            prompt: PromptTemplate that will be formatted with prompt_args to create messages.
            llm_kwargs: Additional provider arguments passed to stream_chat.
                Defaults to empty dict.
            **prompt_args: Template variables used to format the prompt.

        Yields:
            Parsed Pydantic instance(s) - either a single BaseModel or list of BaseModel.
            Each yielded value represents the current state of parsing as more JSON arrives.

        Examples:
            - Stream structured data as it's generated
                ```python
                >>> from pydantic import BaseModel
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> class Summary(BaseModel):
                ...     title: str
                ...     points: list[str]
                >>> llm = Ollama(model="llama3.1", request_timeout=120)     # doctest: +SKIP
                >>> prompt = PromptTemplate("Summarize: {text}")
                >>> for obj in llm.stream_structured_predict(   # doctest: +SKIP
                ...     Summary,
                ...     prompt,
                ...     text="Long article text..."
                ... ):
                ...     # obj is a Summary instance, progressively more complete
                ...     print(f"Current title: {obj.title if hasattr(obj, 'title') else 'N/A'}")

                ```

        See Also:
            astream_structured_predict: Asynchronous streaming counterpart.
            structured_predict: Non-streaming variant.
        """
        if self.pydantic_program_mode == StructuredLLMMode.DEFAULT:

            def gen(
                output_cls: type[BaseModel],
                prompt: PromptTemplate,
                llm_kwargs: dict[str, Any] | None,
                prompt_args: dict[str, Any],
            ) -> Generator[BaseModel | list[BaseModel], None, None]:
                llm_kwargs = llm_kwargs or {}
                llm_kwargs["format"] = output_cls.model_json_schema()

                messages = prompt.format_messages(**prompt_args)
                response_gen = self.stream_chat(messages, **llm_kwargs)

                processor = StreamingObjectProcessor(
                    output_cls=output_cls,
                    flexible_mode=True,
                    allow_parallel_tool_calls=False,
                )
                cur_objects = None
                for response in response_gen:
                    try:
                        objects = processor.process(response, cur_objects)

                        cur_objects = (
                            objects if isinstance(objects, list) else [objects]
                        )
                        yield objects
                    except Exception:
                        continue

            return gen(output_cls, prompt, llm_kwargs, prompt_args)
        else:
            return super().stream_structured_predict(  # type: ignore[return-value]
                output_cls, prompt, llm_kwargs, **prompt_args
            )

    async def astream_structured_predict(
        self,
        output_cls: type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[BaseModel | list[BaseModel], None]:
        """Asynchronously stream incrementally parsed structured objects as the model generates JSON.

        Async variant of stream_structured_predict. Yields partially complete Pydantic
        instances as the model streams JSON content, allowing early access to structured
        data before the full response completes. Uses StreamingObjectProcessor with
        flexible mode to handle incomplete JSON.

        Args:
            output_cls: Pydantic model class defining the expected structure.
            prompt: PromptTemplate that will be formatted with prompt_args to create messages.
            llm_kwargs: Additional provider arguments passed to astream_chat.
                Defaults to empty dict.
            **prompt_args: Template variables used to format the prompt.

        Returns:
            Async generator yielding parsed Pydantic instance(s) - either a single BaseModel
            or list of BaseModel. Each yielded value represents the current state of parsing
            as more JSON arrives.

        Examples:
            - Async stream structured data as it's generated
                ```python
                >>> import asyncio
                >>> from pydantic import BaseModel
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> class Analysis(BaseModel):
                ...     sentiment: str
                ...     keywords: list[str]
                >>> llm = Ollama(model="llama3.1", request_timeout=120)     # doctest: +SKIP
                >>> async def stream_analysis():
                ...     prompt = PromptTemplate("Analyze: {text}")
                ...     async for obj in await llm.astream_structured_predict(
                ...         Analysis,
                ...         prompt,
                ...         text="Product review text..."
                ...     ):
                ...         # obj is progressively more complete
                ...         print(f"Sentiment: {obj.sentiment if hasattr(obj, 'sentiment') else 'pending'}")
                >>> asyncio.run(stream_analysis())      # doctest: +SKIP

                ```

        See Also:
            stream_structured_predict: Synchronous streaming counterpart.
            astructured_predict: Non-streaming async variant.
        """
        if self.pydantic_program_mode == StructuredLLMMode.DEFAULT:

            async def gen(
                output_cls: type[BaseModel],
                prompt: PromptTemplate,
                llm_kwargs: dict[str, Any] | None,
                prompt_args: dict[str, Any],
            ) -> AsyncGenerator[BaseModel | list[BaseModel], None]:
                llm_kwargs = llm_kwargs or {}
                llm_kwargs["format"] = output_cls.model_json_schema()

                messages = prompt.format_messages(**prompt_args)
                response_gen = await self.astream_chat(messages, **llm_kwargs)

                processor = StreamingObjectProcessor(
                    output_cls=output_cls,
                    flexible_mode=True,
                    allow_parallel_tool_calls=False,
                )
                cur_objects = None
                async for response in response_gen:
                    try:
                        objects = processor.process(response, cur_objects)

                        cur_objects = (
                            objects if isinstance(objects, list) else [objects]
                        )
                        yield objects
                    except Exception:
                        continue

            return gen(output_cls, prompt, llm_kwargs, prompt_args)
        else:
            # Fall back to non-streaming structured predict
            return await super().astream_structured_predict(  # type: ignore[return-value]
                output_cls, prompt, llm_kwargs, **prompt_args
            )
