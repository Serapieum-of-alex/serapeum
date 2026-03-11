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
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Generator, Literal, overload

import ollama as ollama_sdk  # type: ignore[attr-defined]
from pydantic import BaseModel, Field, PrivateAttr, ConfigDict

from serapeum.core.configs.defaults import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from serapeum.core.llms import (
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    ChatToCompletion,
    FunctionCallingLLM,
    Image,
    Message,
    MessageList,
    MessageRole,
    Metadata,
    TextChunk,
    ToolCallBlock,
)

from serapeum.core.llms.orchestrators import StreamingObjectProcessor
from serapeum.core.prompts import PromptTemplate
from serapeum.core.tools import ArgumentCoercer
from serapeum.core.types import StructuredOutputMode
from serapeum.core.retry import retry
from serapeum.ollama.client import Client
from serapeum.ollama.retry import is_retryable

if TYPE_CHECKING:
    from serapeum.core.tools.types import BaseTool

logger = logging.getLogger(__name__)

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


class Ollama(Client, ChatToCompletion, FunctionCallingLLM):
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

    Attributes:
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
        timeout: HTTP request timeout in seconds. Defaults to
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
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> llm = Ollama(
            ...     model="qwen3-next:80b",
            ...     api_key=os.environ.get("OLLAMA_API_KEY"),
            ...     temperature=0.0,
            ...     timeout=120,
            ... )
            >>> response = llm.chat([Message(role=MessageRole.USER, chunks=[TextChunk(content="Say 'hello'.")])])
            >>> response.message.role
            <MessageRole.ASSISTANT: 'assistant'>
            >>> print("content:", response.message.content)
            content: ...

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
        - Configure additional model parameters and inspect them
            ```python
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> llm = Ollama(
            ...     model="llama3.1",
            ...     additional_kwargs={"mirostat": 2, "top_k": 40},
            ... )
            >>> llm.additional_kwargs
            {'mirostat': 2, 'top_k': 40}
            >>> llm._model_kwargs["mirostat"]
            2

            ```
        - Stream chat deltas and collect incremental content
            ```python
            >>> import os
            >>> from serapeum.core.llms import Message, MessageRole, TextChunk
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> llm = Ollama(
            ...     model="qwen3-next:80b",
            ...     api_key=os.environ.get("OLLAMA_API_KEY"),
            ...     temperature=0.0,
            ...     timeout=120,
            ... )
            >>> chunks = list(llm.chat([  # doctest: +SKIP, +ELLIPSIS
            ...     Message(role=MessageRole.USER, chunks=[TextChunk(content="Say 'hello'.")])
            ... ], stream=True))
            >>> print("delta:", chunks[0].delta)  # first streamed token  # doctest: +SKIP, +ELLIPSIS
            delta: ...
            >>> print("final:", chunks[-1].message.content)  # final accumulated text  # doctest: +SKIP, +ELLIPSIS
            final: ...

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
            ...     model="llama3.1",
            ...     temperature=0.0,
            ...     timeout=120,
            ... )
            >>> prompt = PromptTemplate("Extract city and country from: {text}")
            >>> result = llm.parse(
            ...     Capital, prompt, text="Paris is the capital of France."
            ... )  # doctest: +SKIP, +ELLIPSIS
            >>> print("city:", result.city)  # doctest: +SKIP, +ELLIPSIS
            city: ...
            >>> print("country:", result.country)  # doctest: +SKIP, +ELLIPSIS
            country: ...
            >>> sorted(result.model_dump().keys())  # doctest: +SKIP
            ['city', 'country']

            ```
        - List all models available on the Ollama server
            ```python
            >>> import os
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> llm = Ollama(model="qwen3-next:80b", api_key=os.environ.get("OLLAMA_API_KEY"))
            >>> models = llm.list_models()  # doctest: +SKIP, +ELLIPSIS
            >>> models[:2]  # doctest: +SKIP, +ELLIPSIS
            ['...', '...']

            ```
        - Async model listing via Ollama Cloud
            ```python
            >>> import asyncio, os
            >>> from serapeum.ollama import Ollama  # type: ignore
            >>> llm = Ollama(model="qwen3-next:80b", api_key=os.environ.get("OLLAMA_API_KEY"))
            >>> async def get_models():  # doctest: +SKIP
            ...     return await llm.alist_models()
            >>> models = asyncio.run(get_models())  # doctest: +SKIP, +ELLIPSIS
            >>> models[:2]  # doctest: +SKIP, +ELLIPSIS
            ['...', '...']

            ```

    See Also:
        OllamaEmbedding: Companion class for generating embeddings with Ollama.
        Client: Shared connection logic, URL resolution, and client injection.
        chat: Synchronous chat completion (supports streaming via ``stream=True``).
        achat: Asynchronous chat completion (supports streaming via ``stream=True``).
        parse: Structured output via JSON schema and Pydantic validation.
        list_models: List all models available on the Ollama server.
        alist_models: Async variant of list_models.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
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
    timeout: float = Field(
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
        return {**super()._build_client_kwargs(), "timeout": self.timeout}

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
        """LLM metadata describing model capabilities and configuration.

        Returns:
            Metadata: Static capabilities such as context window and chat support.

        Examples:
            - Inspect model capabilities and configuration
                ```python
                >>> from serapeum.ollama import Ollama      # type: ignore[attr-defined]
                >>> meta = Ollama(model="llama3.1").metadata
                >>> meta.model_name
                'llama3.1'
                >>> meta.is_chat_model
                True
                >>> meta.is_function_calling_model
                True
                >>> meta.context_window
                3900

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
    def client(self) -> ollama_sdk.Client:  # type: ignore
        """Synchronous Ollama client lazily bound to ``base_url``.

        Returns:
            Client: Underlying Ollama client instance.

        Examples:
            - Access the lazily-created sync client and inspect its host
                ```python
                >>> from serapeum.ollama import Ollama      # type: ignore[attr-defined]
                >>> llm = Ollama(model="m", base_url="http://localhost:11434", timeout=1.0)
                >>> c = llm.client  # doctest: +SKIP, +ELLIPSIS
                >>> str(c._client.base_url)  # doctest: +SKIP, +ELLIPSIS
                'http://localhost:11434'

                ```
        """
        if self._client is None:
            self._client = ollama_sdk.Client(**self._build_client_kwargs())  # type: ignore
        return self._client

    def _ensure_async_client(self) -> ollama_sdk.AsyncClient:  # type: ignore
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
            self._async_client = ollama_sdk.AsyncClient(**client_kwargs)  # type: ignore
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
                self._async_client = ollama_sdk.AsyncClient(**client_kwargs)  # type: ignore
                self._async_client_loop = current_loop
            # Or if the cached loop has been closed since creation
            elif hasattr(cached_loop, "is_closed") and cached_loop.is_closed():
                self._async_client = ollama_sdk.AsyncClient(**client_kwargs)  # type: ignore
                self._async_client_loop = current_loop
            else:
                # Reuse existing client even if loop identity differs but both are open
                self._async_client_loop = current_loop

        return self._async_client

    @property
    def async_client(self) -> ollama_sdk.AsyncClient:  # type: ignore
        """Async Ollama client bound to the current asyncio event loop.

        This property lazily creates or reuses an AsyncClient instance, automatically
        handling event loop changes and closures. It's safe to call across different
        async contexts (e.g., multiple pytest-asyncio tests) as it detects closed
        loops and recreates the client as needed.

        Returns:
            The async client instance used for asynchronous operations.

        Examples:
            - Access the async client within an async context
                ```python
                >>> import asyncio
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> llm = Ollama(model="llama3.1")
                >>> async def use_client():  # doctest: +SKIP
                ...     client = llm.async_client
                ...     response = await client.list()
                ...     return [m.model for m in response.models][:2]
                >>> asyncio.run(use_client())  # doctest: +SKIP, +ELLIPSIS
                ['...', '...']

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
    def _convert_to_ollama_messages(messages: MessageList) -> list[dict[str, Any]]:
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
                >>> from serapeum.core.llms import Message, MessageList, MessageRole, TextChunk
                >>> from serapeum.ollama import Ollama      # type: ignore[attr-defined]
                >>> llm = Ollama(model="m")
                >>> wire = llm._convert_to_ollama_messages(
                ...     MessageList(messages=[
                ...         Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")]),
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
        tools: list[BaseTool],
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
                >>> from serapeum.core.llms import Message, MessageRole, TextChunk
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> class T:
                ...     def __init__(self, n):
                ...         class M:
                ...             def to_openai_tool(self, skip_length_check=False):
                ...                 return {"type": "function", "function": {"name": n}}
                ...         self.metadata = M()
                ...
                >>> llm = Ollama(model="m")
                >>> payload = llm._prepare_chat_with_tools(
                ...     [T("t1")],
                ...     user_msg="hi",
                ...     chat_history=[Message(role=MessageRole.SYSTEM, chunks=[TextChunk(content="s")])],
                ... )
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
            user_msg = Message(
                role=MessageRole.USER,
                chunks=[TextChunk(content=user_msg)],
            )

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
            - Trim multiple tool calls down to the first one
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, ChatResponse, ToolCallBlock
                >>> from serapeum.ollama import Ollama  # type: ignore
                >>> llm = Ollama(model="m")
                >>> response = ChatResponse(
                ...     message=Message(
                ...         role=MessageRole.ASSISTANT,
                ...         chunks=[
                ...             ToolCallBlock(tool_name="a", tool_kwargs={}),
                ...             ToolCallBlock(tool_name="b", tool_kwargs={}),
                ...         ],
                ...     )
                ... )
                >>> validated = llm._validate_chat_with_tools_response(
                ...     response, tools=[], allow_parallel_tool_calls=False,
                ... )
                >>> len(validated.message.tool_calls)
                1

                ```
            - Allow parallel tool calls to pass through untrimmed
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, ChatResponse, ToolCallBlock
                >>> from serapeum.ollama import Ollama  # type: ignore
                >>> llm = Ollama(model="m")
                >>> response = ChatResponse(
                ...     message=Message(
                ...         role=MessageRole.ASSISTANT,
                ...         chunks=[
                ...             ToolCallBlock(tool_name="a", tool_kwargs={}),
                ...             ToolCallBlock(tool_name="b", tool_kwargs={}),
                ...         ],
                ...     )
                ... )
                >>> validated = llm._validate_chat_with_tools_response(
                ...     response, tools=[], allow_parallel_tool_calls=True,
                ... )
                >>> [tc.tool_name for tc in validated.message.tool_calls]
                ['a', 'b']

                ```
        """
        if not allow_parallel_tool_calls:
            response.force_single_tool_call()
        return response

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
            - Build a response from a minimal raw dict and explore it
                ```python
                >>> from serapeum.ollama import Ollama  # type: ignore
                >>> raw = {"message": {"role": "assistant", "content": "Hi"}}
                >>> resp = Ollama._build_chat_response(raw)
                >>> resp.message.content
                'Hi'
                >>> resp.message.role
                <MessageRole.ASSISTANT: 'assistant'>
                >>> resp.message.tool_calls
                []

                ```
            - Build a response with token usage from raw provider data
                ```python
                >>> from serapeum.ollama import Ollama  # type: ignore
                >>> raw = {
                ...     "message": {"role": "assistant", "content": "Hello!"},
                ...     "prompt_eval_count": 10,
                ...     "eval_count": 5,
                ... }
                >>> resp = Ollama._build_chat_response(raw)
                >>> resp.raw["usage"]
                {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}

                ```
        """
        raw = dict(raw)
        raw_tool_calls = raw["message"].get("tool_calls") or []
        token_counts = Ollama._get_response_token_counts(raw)
        if token_counts:
            raw["usage"] = token_counts

        chunks: list = []
        content = raw["message"]["content"]
        if content:
            chunks.append(TextChunk(content=content))

        coercer = ArgumentCoercer()
        for tc in raw_tool_calls:
            func = tc["function"]
            chunks.append(
                ToolCallBlock(
                    tool_call_id=func["name"],
                    tool_name=func["name"],
                    tool_kwargs=coercer.coerce(func["arguments"]),
                )
            )

        return ChatResponse(
            message=Message(
                chunks=chunks,
                role=raw["message"]["role"],
            ),
            raw=raw,
        )

    @overload
    def chat(
        self,
        messages: MessageList | list[Message],
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> ChatResponse: ...

    @overload
    def chat(
        self,
        messages: MessageList | list[Message],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> ChatResponseGen: ...

    def chat(
        self,
        messages: MessageList | list[Message],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | ChatResponseGen:
        """Send a chat request to Ollama and return the assistant message.

        Args:
            messages (MessageList):
                Sequence of chat messages.
            stream (bool):
                If ``False`` (default), returns a single ChatResponse with the complete
                message. If ``True``, returns a generator yielding incremental ChatResponse
                chunks with deltas.
            **kwargs (Any):
                Provider-specific overrides such as ``tools`` or ``format``.

        Returns:
            ChatResponse when ``stream=False``, or ChatResponseGen when ``stream=True``.

        Examples:
            - Non-streaming chat — explore the response message
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, TextChunk
                >>> from serapeum.ollama import Ollama  # type: ignore
                >>> llm = Ollama(model="llama3.1", timeout=120)
                >>> resp = llm.chat([  # doctest: +SKIP, +ELLIPSIS
                ...     Message(role=MessageRole.USER, chunks=[TextChunk(content="Say hi")])
                ... ])
                >>> resp.message.role  # doctest: +SKIP
                <MessageRole.ASSISTANT: 'assistant'>
                >>> print("content:", resp.message.content)  # doctest: +SKIP, +ELLIPSIS
                content: ...
                >>> print("model:", resp.raw.get("model"))  # raw provider metadata  # doctest: +SKIP, +ELLIPSIS
                model: ...

                ```
            - Streaming chat — collect deltas and see accumulated text
                ```python
                >>> from serapeum.core.llms import Message, MessageRole, TextChunk
                >>> from serapeum.ollama import Ollama  # type: ignore
                >>> llm = Ollama(model="llama3.1", timeout=180)
                >>> chunks = list(llm.chat(  # doctest: +SKIP, +ELLIPSIS
                ...     [Message(role=MessageRole.USER, chunks=[TextChunk(content="Count to 3")])],
                ...     stream=True,
                ... ))
                >>> print("delta:", chunks[0].delta)  # first streamed token  # doctest: +SKIP, +ELLIPSIS
                delta: ...
                >>> print("final:", chunks[-1].message.content)  # final accumulated  # doctest: +SKIP, +ELLIPSIS
                final: ...

                ```
        """
        result = self._stream_chat(messages, **kwargs) if stream else self._chat(messages, **kwargs)
        return result

    @retry(is_retryable, logger)
    def _chat(
        self, messages: MessageList | list[Message], **kwargs: Any
    ) -> ChatResponse:
        """Internal non-streaming chat implementation."""
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
            chat: Uses this helper to materialize per-chunk responses.
            achat: Async variant that uses this helper.
        """
        r = dict(r)

        tools_dict["response_txt"] += r["message"]["content"]
        new_tool_calls = [dict(t) for t in (r["message"].get("tool_calls", []) or [])]
        coercer = ArgumentCoercer()
        for tool_call in new_tool_calls:
            func_name = str(tool_call["function"]["name"])
            func_args = str(tool_call["function"]["arguments"])
            if (func_name, func_args) not in tools_dict["seen_tool_calls"]:
                tools_dict["seen_tool_calls"].add((func_name, func_args))
                tools_dict["all_tool_calls"].append(
                    ToolCallBlock(
                        tool_call_id=func_name,
                        tool_name=func_name,
                        tool_kwargs=coercer.coerce(tool_call["function"]["arguments"]),
                    )
                )

        token_counts = Ollama._get_response_token_counts(r)
        if token_counts:
            r["usage"] = token_counts

        chunks: list = []
        if tools_dict["response_txt"]:
            chunks.append(TextChunk(content=tools_dict["response_txt"]))
        chunks.extend(tools_dict["all_tool_calls"])

        return ChatResponse(
            message=Message(
                chunks=chunks,
                role=r["message"]["role"],
            ),
            delta=r["message"]["content"],
            raw=r,
        )

    @retry(is_retryable, logger)
    def _stream_chat(
        self, messages: MessageList | list[Message], **kwargs: Any
    ) -> ChatResponseGen:
        """Internal streaming chat implementation."""
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)
        response_format = kwargs.pop("format", "json" if self.json_mode else None)

        tools_dict = {
            "response_txt": "",
            "seen_tool_calls": set(),
            "all_tool_calls": [],
        }

        response = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            stream=True,
            format=response_format,
            tools=tools,
            options=self._model_kwargs,
            keep_alive=self.keep_alive,
        )

        for r in response:
            if r["message"]["content"] is not None:
                yield self._parse_tool_call_response(tools_dict, r)

    @overload
    async def achat(
        self,
        messages: MessageList | list[Message],
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> ChatResponse: ...

    @overload
    async def achat(
        self,
        messages: MessageList | list[Message],
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen: ...

    async def achat(
        self,
        messages: MessageList | list[Message],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> ChatResponse | ChatResponseAsyncGen:
        """Asynchronously send a chat request and return the assistant message.

        Async variant of the chat method that sends messages to Ollama. When
        ``stream=False`` (default), waits for the complete response. When
        ``stream=True``, returns an async generator yielding incremental chunks.

        Args:
            messages: Sequence of chat messages forming the conversation context.
            stream (bool):
                If ``False`` (default), awaits the full response and returns a single
                ChatResponse. If ``True``, returns an async generator yielding
                ChatResponse chunks with deltas.
            **kwargs: Provider-specific overrides such as:
                - tools: List of tool specifications for function calling
                - format: Response format (e.g., "json")

        Returns:
            ChatResponse when ``stream=False``, or ChatResponseAsyncGen when ``stream=True``.

        Examples:
            - Async non-streaming chat — explore the response
                ```python
                >>> import asyncio
                >>> from serapeum.core.llms import Message, MessageRole, TextChunk
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> llm = Ollama(model="llama3.1", timeout=120)  # doctest: +SKIP
                >>> async def chat_example():  # doctest: +SKIP
                ...     response = await llm.achat([
                ...         Message(role=MessageRole.USER, chunks=[TextChunk(content="Say hello")])
                ...     ])
                ...     return response
                >>> resp = asyncio.run(chat_example())  # doctest: +SKIP, +ELLIPSIS
                >>> resp.message.role  # doctest: +SKIP
                <MessageRole.ASSISTANT: 'assistant'>
                >>> print("content:", resp.message.content)  # doctest: +SKIP, +ELLIPSIS
                content: ...

                ```
            - Async streaming chat — collect deltas
                ```python
                >>> import asyncio
                >>> from serapeum.core.llms import Message, MessageRole, TextChunk
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> llm = Ollama(model="llama3.1", timeout=120)  # doctest: +SKIP
                >>> async def stream_example():  # doctest: +SKIP
                ...     deltas = []
                ...     async for chunk in await llm.achat(
                ...         [Message(role=MessageRole.USER, chunks=[TextChunk(content="Count to 3")])],
                ...         stream=True,
                ...     ):
                ...         deltas.append(chunk.delta)
                ...     return deltas
                >>> deltas = asyncio.run(stream_example())  # doctest: +SKIP, +ELLIPSIS
                >>> len(deltas) >= 1  # doctest: +SKIP
                True
                >>> print("text:", "".join(deltas))  # final accumulated text  # doctest: +SKIP, +ELLIPSIS
                text: ...

                ```

        See Also:
            chat: Synchronous variant.
        """
        result = await self._astream_chat(messages, **kwargs) if stream else await self._achat(messages, **kwargs)
        return result

    @retry(is_retryable, logger)
    async def _achat(
        self, messages: MessageList | list[Message], **kwargs: Any
    ) -> ChatResponse:
        """Internal non-streaming async chat implementation."""
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

    @retry(is_retryable, logger, stream=True)
    async def _astream_chat(
        self, messages: MessageList | list[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Internal streaming async chat implementation."""
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)
        response_format = kwargs.pop("format", "json" if self.json_mode else None)

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

        async def gen() -> ChatResponseAsyncGen:
            async for r in response:
                if r["message"]["content"] is not None:
                    yield self._parse_tool_call_response(tools_dict, r)

        return gen()

    @overload
    def parse(
        self,
        schema: type[BaseModel] | Callable[..., Any],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ...,
        *,
        stream: Literal[False] = ...,
        **prompt_args: Any,
    ) -> BaseModel: ...

    @overload
    def parse(
        self,
        schema: type[BaseModel] | Callable[..., Any],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ...,
        *,
        stream: Literal[True],
        **prompt_args: Any,
    ) -> Generator[BaseModel | list[BaseModel], None, None]: ...

    def parse(
        self,
        schema: type[BaseModel] | Callable[..., Any],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        **prompt_args: Any,
    ) -> BaseModel | Generator[BaseModel | list[BaseModel], None, None]:
        """Generate structured output conforming to a Pydantic model schema.

        Instructs the Ollama model to emit JSON matching the schema of output_cls,
        then validates and parses the response into a Pydantic instance. When using
        StructuredOutputMode.DEFAULT, this injects the model's JSON schema into the
        format parameter and validates the response content.

        When ``stream=True``, yields incrementally parsed Pydantic instances as the
        model streams JSON content, using StreamingObjectProcessor with flexible mode
        to handle incomplete JSON fragments.

        Args:
            schema: Target Pydantic model class (or callable) defining the expected
                structure. A callable is accepted when routing through
                ToolOrchestratingLLM (non-DEFAULT modes).
            prompt: PromptTemplate that will be formatted with prompt_args to create
                messages.
            llm_kwargs: Additional provider arguments passed to the chat method.
                Defaults to empty dict.
            stream: If ``False`` (default), returns a single validated instance after
                the full response is received. If ``True``, returns a generator that
                yields partially complete instances as JSON is streamed.
            **prompt_args: Template variables used to format the prompt.

        Returns:
            A validated ``BaseModel`` instance when ``stream=False``, or a
            ``Generator`` yielding ``BaseModel | list[BaseModel]`` when
            ``stream=True``.

        Raises:
            ValidationError: If the model's response doesn't match the schema
                (non-streaming mode only).

        Examples:
            - Extract structured data and explore the parsed fields
                ```python
                >>> import os
                >>> from pydantic import BaseModel, Field
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> class Person(BaseModel):
                ...     name: str = Field(description="Person's full name")
                ...     age: int = Field(description="Person's age in years")
                >>> llm = Ollama(model="llama3.1", timeout=120)     # doctest: +SKIP
                >>> prompt = PromptTemplate("Extract person info: {text}")  # doctest: +SKIP
                >>> result = llm.parse(  # doctest: +SKIP, +ELLIPSIS
                ...     Person,
                ...     prompt,
                ...     text="John Doe is 30 years old",
                ... )
                >>> print("name:", result.name)  # doctest: +SKIP, +ELLIPSIS
                name: ...
                >>> result.age >= 0  # age is a valid integer  # doctest: +SKIP
                True
                >>> sorted(result.model_dump().keys())  # doctest: +SKIP
                ['age', 'name']

                ```

            - Stream structured data and observe incremental updates
                ```python
                >>> from pydantic import BaseModel
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> class Summary(BaseModel):
                ...     title: str
                ...     points: list[str]
                >>> llm = Ollama(model="llama3.1", timeout=120)     # doctest: +SKIP
                >>> prompt = PromptTemplate("Summarize: {text}")  # doctest: +SKIP
                >>> last = None  # doctest: +SKIP
                >>> for obj in llm.parse(  # doctest: +SKIP, +ELLIPSIS
                ...     Summary, prompt, stream=True, text="Long article about AI..."
                ... ):
                ...     last = obj
                >>> print("title:", last.title)  # final parsed title  # doctest: +SKIP, +ELLIPSIS
                title: ...
                >>> len(last.points) >= 0  # doctest: +SKIP
                True

                ```

        See Also:
            aparse: Async variant (non-streaming).
        """
        if self.structured_output_mode == StructuredOutputMode.DEFAULT:
            result = (
                self._stream_parse_default(schema, prompt, llm_kwargs, prompt_args)
                if stream
                else self._parse_default(schema, prompt, llm_kwargs, prompt_args)
            )
        else:
            result = (
                super().stream_parse(schema, prompt, llm_kwargs, **prompt_args)  # type: ignore[return-value]
                if stream
                else super().parse(schema, prompt, llm_kwargs, **prompt_args)
            )
        return result

    def _parse_default(
        self,
        schema: type[BaseModel] | Callable[..., Any],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None,
        prompt_args: dict[str, Any],
    ) -> BaseModel:
        llm_kwargs = llm_kwargs or {}
        llm_kwargs["format"] = schema.model_json_schema()

        # Explicitly remove 'stream' to prevent override of non-streaming behavior
        llm_kwargs.pop("stream", None)

        messages = prompt.format_messages(**prompt_args)
        response = self.chat(messages, **llm_kwargs)
        return schema.model_validate_json(response.message.content or "")

    def _stream_parse_default(
        self,
        schema: type[BaseModel] | Callable[..., Any],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None,
        prompt_args: dict[str, Any],
    ) -> Generator[BaseModel | list[BaseModel], None, None]:
        _llm_kwargs = llm_kwargs or {}
        _llm_kwargs["format"] = schema.model_json_schema()
        messages = prompt.format_messages(**prompt_args)
        processor = StreamingObjectProcessor(
            output_cls=schema,
            flexible_mode=True,
            allow_parallel_tool_calls=False,
        )
        cur_objects = None
        for response in self.chat(messages, stream=True, **_llm_kwargs):
            try:
                objects = processor.process(response, cur_objects)
                cur_objects = objects if isinstance(objects, list) else [objects]
                yield objects
            except Exception:
                continue

    @overload
    async def aparse(
        self,
        schema: type[BaseModel] | Callable[..., Any],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ...,
        *,
        stream: Literal[False] = ...,
        **prompt_args: Any,
    ) -> BaseModel: ...

    @overload
    async def aparse(
        self,
        schema: type[BaseModel] | Callable[..., Any],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = ...,
        *,
        stream: Literal[True],
        **prompt_args: Any,
    ) -> AsyncGenerator[BaseModel | list[BaseModel], None]: ...

    async def aparse(
        self,
        schema: type[BaseModel] | Callable[..., Any],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None = None,
        *,
        stream: bool = False,
        **prompt_args: Any,
    ) -> BaseModel | AsyncGenerator[BaseModel | list[BaseModel], None]:
        """Asynchronously generate structured output conforming to a Pydantic model schema.

        Async variant of parse. Instructs the Ollama model to emit JSON
        matching the schema of output_cls, then validates and parses the response
        into a Pydantic instance using the async chat interface.

        When ``stream=True``, returns an async generator that yields incrementally
        parsed Pydantic instances as the model streams JSON content, using
        StreamingObjectProcessor with flexible mode to handle incomplete JSON fragments.

        Args:
            schema: Target Pydantic model class (or callable) defining the expected
                structure. A callable is accepted when routing through
                ToolOrchestratingLLM (non-DEFAULT modes).
            prompt: PromptTemplate that will be formatted with prompt_args to create
                messages.
            llm_kwargs: Additional provider arguments passed to the achat method.
                Defaults to empty dict.
            stream: If ``False`` (default), awaits the full response and returns a
                single validated instance. If ``True``, returns an async generator
                that yields partially complete instances as JSON is streamed.
            **prompt_args: Template variables used to format the prompt.

        Returns:
            A validated ``BaseModel`` instance when ``stream=False``, or an
            ``AsyncGenerator`` yielding ``BaseModel | list[BaseModel]`` when
            ``stream=True``.

        Raises:
            ValidationError: If the model's response doesn't match the schema
                (non-streaming mode only).

        Examples:
            - Async structured extraction — explore the parsed object
                ```python
                >>> import asyncio
                >>> from pydantic import BaseModel
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> class City(BaseModel):
                ...     name: str
                ...     country: str
                >>> llm = Ollama(model="llama3.1", timeout=120)  # doctest: +SKIP
                >>> async def extract_city():  # doctest: +SKIP
                ...     prompt = PromptTemplate("Extract city: {text}")
                ...     return await llm.aparse(City, prompt, text="Paris is in France")
                >>> result = asyncio.run(extract_city())  # doctest: +SKIP, +ELLIPSIS
                >>> print("name:", result.name)  # doctest: +SKIP, +ELLIPSIS
                name: ...
                >>> print("country:", result.country)  # doctest: +SKIP, +ELLIPSIS
                country: ...
                >>> sorted(result.model_dump().keys())  # doctest: +SKIP
                ['country', 'name']

                ```

            - Async stream structured data and observe incremental fields
                ```python
                >>> import asyncio
                >>> from pydantic import BaseModel
                >>> from serapeum.core.prompts import PromptTemplate
                >>> from serapeum.ollama import Ollama      # type: ignore
                >>> class Analysis(BaseModel):
                ...     sentiment: str
                ...     keywords: list[str]
                >>> llm = Ollama(model="llama3.1", timeout=120)  # doctest: +SKIP
                >>> async def stream_analysis():  # doctest: +SKIP
                ...     prompt = PromptTemplate("Analyze: {text}")
                ...     last = None
                ...     async for obj in await llm.aparse(
                ...         Analysis, prompt, stream=True, text="Great product!"
                ...     ):
                ...         last = obj
                ...     return last
                >>> last = asyncio.run(stream_analysis())  # doctest: +SKIP, +ELLIPSIS
                >>> print("sentiment:", last.sentiment)  # doctest: +SKIP, +ELLIPSIS
                sentiment: ...
                >>> len(last.keywords) >= 0  # doctest: +SKIP
                True

                ```

        See Also:
            parse: Synchronous variant.
        """
        if self.structured_output_mode == StructuredOutputMode.DEFAULT:
            result = (
                self._astream_parse_default(schema, prompt, llm_kwargs, prompt_args)
                if stream
                else await self._aparse_default(schema, prompt, llm_kwargs, prompt_args)
            )
        else:
            result = (
                await super().astream_parse(schema, prompt, llm_kwargs, **prompt_args)  # type: ignore[return-value]
                if stream
                else await super().aparse(schema, prompt, llm_kwargs, **prompt_args)
            )
        return result

    async def _aparse_default(
        self,
        schema: type[BaseModel] | Callable[..., Any],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None,
        prompt_args: dict[str, Any],
    ) -> BaseModel:
        llm_kwargs = llm_kwargs or {}
        llm_kwargs["format"] = schema.model_json_schema()

        # Explicitly remove 'stream' to prevent override of non-streaming behavior
        llm_kwargs.pop("stream", None)

        messages = prompt.format_messages(**prompt_args)
        response = await self.achat(messages, **llm_kwargs)
        return schema.model_validate_json(response.message.content or "")

    async def _astream_parse_default(
        self,
        schema: type[BaseModel] | Callable[..., Any],
        prompt: PromptTemplate,
        llm_kwargs: dict[str, Any] | None,
        prompt_args: dict[str, Any],
    ) -> AsyncGenerator[BaseModel | list[BaseModel], None]:
        _llm_kwargs = llm_kwargs or {}
        _llm_kwargs["format"] = schema.model_json_schema()
        messages = prompt.format_messages(**prompt_args)
        response_gen = await self.achat(messages, stream=True, **_llm_kwargs)
        processor = StreamingObjectProcessor(
            output_cls=schema,
            flexible_mode=True,
            allow_parallel_tool_calls=False,
        )
        cur_objects = None
        async for response in response_gen:
            try:
                objects = processor.process(response, cur_objects)
                cur_objects = objects if isinstance(objects, list) else [objects]
                yield objects
            except Exception:
                continue
