"""Ollama base module."""
from __future__ import annotations
from ollama import Client, AsyncClient
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    Generator,
    AsyncGenerator,
)

from serapeum.core.base.llms.utils import (
    chat_to_completion_decorator,
    achat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    astream_chat_to_completion_decorator,
)
from serapeum.core.base.llms.models import (
    Message,
    MessageList,
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    Metadata,
    MessageRole,
    TextChunk,
    Image,
)
from pydantic import BaseModel, Field, PrivateAttr
from serapeum.core.configs.defaults import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS
from serapeum.core.llm.function_calling import FunctionCallingLLM
from serapeum.core.prompts import PromptTemplate
from serapeum.core.tools import ToolCallArguments
from serapeum.core.models import StructuredLLMMode
from serapeum.core.structured_tools.utils import StreamingObjectProcessor
import asyncio
import inspect

if TYPE_CHECKING:
    from serapeum.core.tools.models import BaseTool

DEFAULT_REQUEST_TIMEOUT = 60.0


def get_additional_kwargs(
    response: Dict[str, Any], exclude: Tuple[str, ...]
) -> Dict[str, Any]:
    """Filter out excluded keys from a response dictionary.

    Args:
        response (Dict[str, Any]):
            Source dictionary, typically a raw provider response.
        exclude (Tuple[str, ...]):
            Keys that should be omitted from the returned mapping.

    Returns:
        Dict[str, Any]:
            A new dictionary containing only entries whose keys are not present in ``exclude``.

    Examples:
        - Keep only non-excluded keys
            ```python
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
            >>> from serapeum.core.base.llms.models import Message, MessageRole, ChatResponse
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
            >>> from serapeum.core.base.llms.models import Message, MessageRole, ChatResponse
            >>> r = ChatResponse(message=Message(role=MessageRole.ASSISTANT, content=""))
            >>> force_single_tool_call(r)
            >>> r.message.additional_kwargs.get("tool_calls") is None or len(r.message.additional_kwargs.get("tool_calls", [])) == 0
            True

            ```
    """
    tool_calls = response.message.additional_kwargs.get("tool_calls", [])
    if len(tool_calls) > 1:
        response.message.additional_kwargs["tool_calls"] = [tool_calls[0]]


class Ollama(FunctionCallingLLM):
    """Ollama.

    This class integrates with the local/remote Ollama server to provide chat,
    streaming, and structured output capabilities. It supports tool/function
    calling and JSON mode formatting when the model allows it.

    Visit https://ollama.com/ to install Ollama and run ``ollama serve`` to
    start the server.

    Args:
        model (str):
            Identifier of the Ollama model to use (e.g., ``"llama3.1:latest"``).
        base_url (str):
            Base URL where the Ollama server is hosted. Defaults to
            ``"http://localhost:11434"``.
        temperature (float):
            Sampling temperature in the range [0.0, 1.0]. Higher values increase
            randomness. Defaults to 0.75.
        context_window (int):
            Maximum context tokens for the model. Defaults to the project
            ``DEFAULT_CONTEXT_WINDOW``.
        request_timeout (float):
            Timeout (seconds) for API calls. Defaults to 60.0.
        prompt_key (str):
            Key used for prompt formatting when applicable. Defaults to ``"prompt"``.
        json_mode (bool):
            Whether to request JSON-formatted responses when supported. Defaults to ``False``.
        additional_kwargs (Dict[str, Any]):
            Extra provider-specific options forwarded under ``options``.
        client (Optional[Client]):
            Pre-constructed synchronous Ollama client. When omitted, the client
            is created lazily from ``base_url`` and ``request_timeout``.
        async_client (Optional[AsyncClient]):
            Pre-constructed asynchronous Ollama client. If omitted, a client is
            created per event loop.
        is_function_calling_model (bool):
            Flag indicating whether the selected model supports tool/function
            calling. Defaults to ``True``.
        keep_alive (Optional[Union[float, str]]):
            Controls how long the model stays loaded following the request
            (e.g., ``"5m"``). When ``None``, provider defaults apply.
        **kwargs (Any):
            Reserved for future extensions and compatibility with the base class.

    Examples:
        - Basic chat using a real Ollama server (requires a running server and a pulled model)
            ```python
            >>> from serapeum.core.base.llms.models import Message, MessageRole
            >>> # Ensure `ollama serve` is running locally and the model is pulled, e.g.:
            >>> #   ollama pull llama3.1
            >>> from serapeum.llms.ollama import Ollama
            >>> llm = Ollama(model="llama3.1", request_timeout=120)
            >>> response = llm.chat([Message(role=MessageRole.USER, content="Say 'pong'.")])  # doctest: +SKIP
            >>> print(response)
            assistant: Pong!

            ```
        - Enabling JSON mode for structured outputs with a real server
            ```python
            >>> from serapeum.core.base.llms.models import Message, MessageRole
            >>> # When json_mode=True, this adapter sets format="json" under the hood.
            >>> llm = Ollama(model="llama3.1", json_mode=True, request_timeout=120)
            >>> response = llm.chat([Message(role=MessageRole.USER, content='Return {"ok": true} as JSON')])  # doctest: +SKIP
            >>> print(response)
            assistant: {"ok":true}

            ```

    See Also:
        - chat: Synchronous chat API.
        - stream_chat: Streaming chat API yielding deltas.
        - structured_predict: Parse pydantic models from model output.
    """

    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted under.",
    )
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
    additional_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model parameters for the Ollama API.",
    )
    is_function_calling_model: bool = Field(
        default=True,
        description="Whether the model is a function calling model.",
    )
    keep_alive: Optional[Union[float, str]] = Field(
        default="5m",
        description="controls how long the model will stay loaded into memory following the request(default: 5m)",
    )

    _client: Optional[Client] = PrivateAttr()
    _async_client: Optional[AsyncClient] = PrivateAttr()

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.75,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        prompt_key: str = "prompt",
        json_mode: bool = False,
        additional_kwargs: Dict[str, Any] = None,
        client: Optional[Client] = None,
        async_client: Optional[AsyncClient] = None,
        is_function_calling_model: bool = True,
        keep_alive: Optional[Union[float, str]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama LLM adapter."""
        if additional_kwargs is None:
            additional_kwargs = {}

        super().__init__(
            model=model,
            base_url=base_url,
            temperature=temperature,
            context_window=context_window,
            request_timeout=request_timeout,
            prompt_key=prompt_key,
            json_mode=json_mode,
            additional_kwargs=additional_kwargs,
            is_function_calling_model=is_function_calling_model,
            keep_alive=keep_alive,
            **kwargs,
        )

        self._client = client
        self._async_client = async_client
        # Track the event loop associated with the async client to avoid
        # reusing a client bound to a closed event loop across tests/runs
        self._async_client_loop = None

        # Cache decorated methods to avoid creating wrappers on every call
        self._complete_fn = chat_to_completion_decorator(self.chat)
        self._acomplete_fn = achat_to_completion_decorator(self.achat)
        self._stream_complete_fn = stream_chat_to_completion_decorator(self.stream_chat)
        self._astream_complete_fn = astream_chat_to_completion_decorator(
            self.astream_chat
        )

    @classmethod
    def class_name(cls) -> str:
        """Return the registered class name for this provider adapter.

        Returns:
            str: Provider identifier used in registries or logs.

        Examples:
            - Retrieve the class identifier
                ```python
                >>> Ollama.class_name()
                'Ollama_llm'

                ```
        """
        return "Ollama_llm"

    @property
    def metadata(self) -> Metadata:
        """LLM metadata.

        Returns:
            Metadata: Static capabilities such as context window and chat support.

        Examples:
            - Inspect chat model capabilities
                ```python
                >>> from serapeum.llms.ollama import Ollama
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
    def client(self) -> Client:
        """Synchronous Ollama client lazily bound to ``base_url``.

        Returns:
            Client: Underlying Ollama client instance.

        Examples:
            - Lazily create the client on first access
                ```python
                >>> from serapeum.llms.ollama import Ollama
                >>> llm = Ollama(model="m", base_url="http://localhost:11434", request_timeout=1.0)
                >>> c = llm.client  # doctest: +ELLIPSIS
                >>> type(c) # doctest: + SKIP
                ollama._client.Client
                >>> hasattr(c, "chat")
                True

                ```
        """
        if self._client is None:
            self._client = Client(host=self.base_url, timeout=self.request_timeout)
        return self._client

    def _ensure_async_client(self) -> AsyncClient:
        """Return a per-event-loop AsyncClient, recreating when loop changes or closes.

        This avoids ``Event loop is closed`` errors when test runners (e.g.,
        pytest-asyncio) create and close event loops between invocations.

        Returns:
            AsyncClient: Async client instance associated with the current loop.

        Examples:
            - Re-create the client when the loop changes
                ```python
                >>> import asyncio
                >>> llm = Ollama(model="m")
                >>> c1 = llm._ensure_async_client()  # doctest: +ELLIPSIS
                >>> c2 = llm._ensure_async_client()  # same loop, same client
                >>> c1 is c2
                True

                ```
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None  # No running loop available in this context

        cached_loop = getattr(self, "_async_client_loop", None)
        if self._async_client is None:
            # No client yet: create and bind to current loop (may be None)
            self._async_client = AsyncClient(
                host=self.base_url, timeout=self.request_timeout
            )
            self._async_client_loop = current_loop  # type: ignore[attr-defined]
        else:
            # If no loop recorded yet (e.g., injected client), bind without recreation
            if cached_loop is None:
                self._async_client_loop = current_loop  # type: ignore[attr-defined]
            # Recreate if the current loop is closed
            elif (
                current_loop is not None
                and hasattr(current_loop, "is_closed")
                and current_loop.is_closed()
            ):
                self._async_client = AsyncClient(
                    host=self.base_url, timeout=self.request_timeout
                )
                self._async_client_loop = current_loop  # type: ignore[attr-defined]
            # Or if the cached loop has been closed since creation
            elif hasattr(cached_loop, "is_closed") and cached_loop.is_closed():
                self._async_client = AsyncClient(
                    host=self.base_url, timeout=self.request_timeout
                )
                self._async_client_loop = current_loop  # type: ignore[attr-defined]
            else:
                # Reuse existing client even if loop identity differs but both are open
                self._async_client_loop = current_loop  # type: ignore[attr-defined]

        return self._async_client

    @property
    def async_client(self) -> AsyncClient:
        """Async Ollama client bound to the current asyncio event loop.

        Returns:
            AsyncClient: The async client instance used for asynchronous calls.

        See Also:
            _ensure_async_client: Ensures the client matches the active event loop.
        """
        return self._ensure_async_client()

    @property
    def _model_kwargs(self) -> Dict[str, Any]:
        """Assemble provider options forwarded under the ``options`` field.

        Returns:
            Dict[str, Any]: Merged dictionary where ``additional_kwargs`` override
            base defaults such as ``temperature`` and ``num_ctx``.

        Examples:
            - Merge user-provided options with defaults
                ```python
                >>> from serapeum.llms.ollama import Ollama
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

    def _convert_to_ollama_messages(
        self, messages: MessageList
    ) -> List[Dict[str, Any]]:
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
                >>> from serapeum.core.base.llms.models import Message, MessageList, MessageRole
                >>> from serapeum.llms.ollama import Ollama
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
    def _get_response_token_counts(raw_response: dict) -> dict:
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
                >>> from serapeum.llms.ollama import Ollama
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
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, Message]] = None,
        chat_history: Optional[List[Message]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Prepare a chat payload including tool specifications.

        Args:
            tools (List[BaseTool]): Tools to expose to the model (converted using OpenAI schema).
            user_msg (Optional[Union[str, Message]]): Optional user message to append.
            chat_history (Optional[List[Message]]): Optional existing conversation history.
            verbose (bool): Currently unused verbosity flag.
            allow_parallel_tool_calls (bool): Indicator forwarded to validators.
            **kwargs (Any): Reserved for future extensions.

        Returns:
            Dict[str, Any]: Dict with ``messages`` and ``tools`` entries suitable for chat calls.

        Examples:
            - Combine history, a new user message, and tool specs
                ```python
                >>> from serapeum.core.base.llms.models import Message, MessageRole
                >>> from serapeum.llms.ollama import Ollama
                >>> class T:
                ...     def __init__(self, n):
                ...         class M:
                ...             def to_openai_tool(self, skip_length_check=False):
                ...                 return {"type": "function", "function": {"name": n}}
                ...         self.metadata = M()
                ...
                >>> llm = Ollama(model="m")
                >>> payload = llm._prepare_chat_with_tools([T("t1")], user_msg="hi", chat_history=[Message(role=MessageRole.SYSTEM, content="s")])
                >>> payload["messages"]
                [Message(role=<MessageRole.SYSTEM: 'system'>, additional_kwargs={}, chunks=[TextChunk(content='s', path=None, url=None, type='text')]),
                 Message(role=<MessageRole.USER: 'user'>, additional_kwargs={}, chunks=[TextChunk(content='hi', path=None, url=None, type='text')])]
                >>> payload["tools"]
                [{'type': 'function', 'function': {'name': 't1'}}]

                ```
        """
        tool_specs = [
            tool.metadata.to_openai_tool(skip_length_check=True) for tool in tools
        ]

        if isinstance(user_msg, str):
            user_msg = Message(role=MessageRole.USER, content=user_msg)

        messages = chat_history or []
        if user_msg:
            messages.append(user_msg)

        return {
            "messages": messages,
            "tools": tool_specs or None,
        }

    def _validate_chat_with_tools_response(
        self,
        response: ChatResponse,
        tools: List["BaseTool"],
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
                >>> from serapeum.core.base.llms.models import Message, MessageRole, ChatResponse
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
                >>> validated_response = llm._validate_chat_with_tools_response(response, tools=[], allow_parallel_tool_calls=False)
                >>> len( validated_response.message.additional_kwargs.get("tool_calls"))
                1

                ```
        """
        if not allow_parallel_tool_calls:
            force_single_tool_call(response)
        return response

    def get_tool_calls_from_response(
        self,
        response: "ChatResponse",
        error_on_no_tool_call: bool = True,
    ) -> List[ToolCallArguments]:
        """Extract tool call selections from a chat response.

        Args:
            response (ChatResponse): Response potentially containing tool calls.
            error_on_no_tool_call (bool): Whether to raise when no tool calls are present.

        Returns:
            List[ToolCallArguments]: Parsed tool selections (empty when allowed and none present).

        Raises:
            ValueError: When ``error_on_no_tool_call`` is ``True`` and no tool calls exist.

        Examples:
            - Parse a single tool call
                ```python
                >>> from serapeum.core.base.llms.models import Message, MessageRole, ChatResponse
                >>> llm = Ollama(model="m")
                >>> r = ChatResponse(message=Message(role=MessageRole.ASSISTANT, content="", additional_kwargs={
                ...     "tool_calls": [{"function": {"name": "run", "arguments": {"a": 1}}}],
                ... }))
                >>> calls = llm.get_tool_calls_from_response(r)
                >>> (calls[0].tool_id, calls[0].tool_name, calls[0].tool_kwargs["a"]) == ("run", "run", 1)
                True

                ```
            - Raise when no tool call is present and errors are enabled
                ```python
                >>> from serapeum.core.base.llms.models import Message, MessageRole, ChatResponse
                >>> llm = Ollama(model="m")
                >>> empty = ChatResponse(message=Message(role=MessageRole.ASSISTANT, content=""))
                >>> try:
                ...     _ = llm.get_tool_calls_from_response(empty, error_on_no_tool_call=True)
                ... except ValueError as e:
                ...     msg = str(e)
                >>> 'Expected at least one tool call' in msg
                True

                ```
        """
        tool_calls = response.message.additional_kwargs.get("tool_calls", [])
        if len(tool_calls) < 1:
            if error_on_no_tool_call:
                raise ValueError(
                    f"Expected at least one tool call, but got {len(tool_calls)} tool calls."
                )
            else:
                return []

        tool_selections = []
        for tool_call in tool_calls:
            argument_dict = tool_call["function"]["arguments"]

            tool_selections.append(
                ToolCallArguments(
                    # tool ids not provided by Ollama
                    tool_id=tool_call["function"]["name"],
                    tool_name=tool_call["function"]["name"],
                    tool_kwargs=argument_dict,
                )
            )

        return tool_selections

    def chat(self, messages: MessageList, **kwargs: Any) -> ChatResponse:
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
                >>> from serapeum.core.base.llms.models import Message, MessageRole
                >>> # Ensure `ollama serve` is running and the model is available locally.
                >>> llm = Ollama(model="llama3.1", request_timeout=120)
                >>> resp = llm.chat([Message(role=MessageRole.USER, content="hi")])  # doctest: +SKIP
                >>> isinstance(resp.message.content, str)  # doctest: +SKIP
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

        response = dict(response)

        tool_calls = response["message"].get("tool_calls", [])
        token_counts = self._get_response_token_counts(response)
        if token_counts:
            response["usage"] = token_counts

        return ChatResponse(
            message=Message(
                content=response["message"]["content"],
                role=response["message"]["role"],
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=response,
        )

    @staticmethod
    def _parse_tool_call_response(tools_dict, r):
        """Accumulate streaming content and unique tool calls into a ChatResponse.

        Args:
            tools_dict (dict): Mutable aggregation state tracking text and tool calls.
            r (dict): A single streaming chunk from the provider.

        Returns:
            ChatResponse: Response with cumulative content and the current delta.

        See Also:
            stream_chat: Uses this helper to materialize per-chunk responses.
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

    def stream_chat(self, messages: MessageList, **kwargs: Any) -> ChatResponseGen:
        """Stream assistant deltas for a chat request.

        Args:
            messages (MessageList): Sequence of chat messages.
            **kwargs (Any): Provider-specific options such as ``tools`` or ``format``.

        Yields:
            ChatResponse: Incremental responses with ``delta`` and cumulative content.

        Examples:
            - Stream deltas from a real Ollama server (requires server and model)
                ```python
                >>> from serapeum.core.base.llms.models import Message, MessageRole
                >>> # Pre-requisites:
                >>> #   1) Start the server: `ollama serve`
                >>> #   2) Pull a model:    `ollama pull llama3.1`
                >>> llm = Ollama(model="llama3.1", request_timeout=180)
                >>> chunks = list(llm.stream_chat([Message(role=MessageRole.USER, content="Say hello succinctly")]))  # doctest: +SKIP
                >>> isinstance(chunks[-1].message.content, str) and len(chunks) >= 1  # doctest: +SKIP
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
        self, messages: MessageList, **kwargs: Any
    ) -> ChatResponseAsyncGen:
        """Asynchronously stream assistant deltas for a chat request.

        Args:
            messages (MessageList): Sequence of chat messages.
            **kwargs (Any): Provider-specific options such as ``tools`` or ``format``.

        Returns:
            ChatResponseAsyncGen: Async generator yielding ``ChatResponse`` chunks.
        """
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)
        format = kwargs.pop("format", "json" if self.json_mode else None)

        async def gen() -> ChatResponseAsyncGen:
            response = await self.async_client.chat(
                model=self.model,
                messages=ollama_messages,
                stream=True,
                format=format,
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

    async def achat(self, messages: MessageList, **kwargs: Any) -> ChatResponse:
        """Asynchronously send a chat request and return the assistant message.

        Args:
            messages (MessageList): Sequence of chat messages.
            **kwargs (Any): Provider-specific overrides such as ``tools`` or ``format``.

        Returns:
            ChatResponse: Parsed response containing the assistant message and optional token usage.
        """
        ollama_messages = self._convert_to_ollama_messages(messages)

        tools = kwargs.pop("tools", None)
        format = kwargs.pop("format", "json" if self.json_mode else None)

        response = await self.async_client.chat(
            model=self.model,
            messages=ollama_messages,
            stream=False,
            format=format,
            tools=tools,
            options=self._model_kwargs,
            keep_alive=self.keep_alive,
        )

        response = dict(response)

        tool_calls = response["message"].get("tool_calls", [])
        token_counts = self._get_response_token_counts(response)
        if token_counts:
            response["usage"] = token_counts

        return ChatResponse(
            message=Message(
                content=response["message"]["content"],
                role=response["message"]["role"],
                additional_kwargs={"tool_calls": tool_calls},
            ),
            raw=response,
        )

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Synchronous completion wrapper over the chat API.

        The implementation delegates to ``chat`` through a compatibility adapter.

        Args:
            prompt (str): Pre-formatted prompt string.
            **kwargs (Any): Provider-specific options.

        Returns:
            CompletionResponse: Textual response compatible with completion helpers.

        See Also:
            acomplete: Asynchronous counterpart.
        """
        return self._complete_fn(prompt, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Asynchronous completion wrapper over the chat API.

        Args:
            prompt (str): Pre-formatted prompt string.
            **kwargs (Any): Provider-specific options.

        Returns:
            CompletionResponse: Textual response compatible with completion helpers.
        """
        return await self._acomplete_fn(prompt, **kwargs)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Stream completion deltas via the chat streaming API.

        Args:
            prompt (str): Pre-formatted prompt string.
            **kwargs (Any): Provider-specific options.

        Yields:
            CompletionResponseGen: Stream yielding completion deltas.
        """
        return self._stream_complete_fn(prompt, **kwargs)

    async def astream_complete(
        self, prompt: str, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        """Asynchronously stream completion deltas via the chat streaming API.

        Args:
            prompt (str): Pre-formatted prompt string.
            **kwargs (Any): Provider-specific options.

        Returns:
            CompletionResponseAsyncGen: Async generator producing completion deltas.
        """
        return await self._astream_complete_fn(prompt, **kwargs)

    def structured_predict(
        self,
        output_cls: Type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> BaseModel:
        """Parse structured output by instructing the model to emit JSON.

        When operating in ``StructuredLLMMode.DEFAULT``, the method injects the
        pydantic JSON schema of ``output_cls`` via the ``format`` argument and
        validates the assistant content using ``output_cls.model_validate_json``.

        Args:
            output_cls (Type[BaseModel]): Target pydantic model for the output.
            prompt (PromptTemplate): Prompt template used to construct messages.
            llm_kwargs (Optional[Dict[str, Any]]): Provider arguments forwarded to ``chat``.
            **prompt_args (Any): Additional template variables.

        Returns:
            BaseModel: Instance of ``output_cls`` parsed from the model output.

        See Also:
            astructured_predict: Async variant.
            stream_structured_predict: Streaming counterpart producing partial values.
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
        output_cls: Type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> BaseModel:
        """Async variant of ``structured_predict``.

        Args:
            output_cls (Type[BaseModel]): Target pydantic model for the output.
            prompt (PromptTemplate): Prompt template used to construct messages.
            llm_kwargs (Optional[Dict[str, Any]]): Provider arguments forwarded to ``achat``.
            **prompt_args (Any): Additional template variables.

        Returns:
            BaseModel: Instance of ``output_cls`` parsed from the model output.
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
        output_cls: Type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> Generator[Union[BaseModel, List[BaseModel]], None, None]:
        """Stream structured objects parsed from chat deltas.

        The method incrementally parses partial JSON content into ``output_cls``
        instances using ``StreamingObjectProcessor`` with flexible mode.

        Args:
            output_cls (Type[BaseModel]): Pydantic model describing the structure.
            prompt (PromptTemplate): Prompt template rendered to messages.
            llm_kwargs (Optional[Dict[str, Any]]): Provider options forwarded to ``stream_chat``.
            **prompt_args (Any): Additional template variables.

        Yields:
            Union[BaseModel, List[BaseModel]]: Parsed model(s) per streamed chunk.

        See Also:
            astream_structured_predict: Asynchronous streaming counterpart.
        """
        if self.pydantic_program_mode == StructuredLLMMode.DEFAULT:

            def gen(
                output_cls: Type[BaseModel],
                prompt: PromptTemplate,
                llm_kwargs: Dict[str, Any],
                prompt_args: Dict[str, Any],
            ) -> Generator[Union[BaseModel, List[BaseModel]], None, None]:
                llm_kwargs = llm_kwargs or {}
                llm_kwargs["format"] = output_cls.model_json_schema()

                messages = prompt.format_messages(**prompt_args)
                response_gen = self.stream_chat(messages, **llm_kwargs)

                cur_objects = None
                for response in response_gen:
                    try:
                        processor = StreamingObjectProcessor(
                            output_cls=output_cls,
                            flexible_mode=True,
                            allow_parallel_tool_calls=False,
                        )
                        objects = processor.process(response, cur_objects)

                        cur_objects = (
                            objects if isinstance(objects, list) else [objects]
                        )
                        yield objects
                    except Exception:
                        continue

            return gen(output_cls, prompt, llm_kwargs, prompt_args)
        else:
            return super().stream_structured_predict(
                output_cls, prompt, llm_kwargs, **prompt_args
            )

    async def astream_structured_predict(
        self,
        output_cls: Type[BaseModel],
        prompt: PromptTemplate,
        llm_kwargs: Optional[Dict[str, Any]] = None,
        **prompt_args: Any,
    ) -> AsyncGenerator[Union[BaseModel, List[BaseModel]], None]:
        """Asynchronously stream structured objects parsed from chat deltas.

        Args:
            output_cls (Type[BaseModel]): Pydantic model describing the structure.
            prompt (PromptTemplate): Prompt template rendered to messages.
            llm_kwargs (Optional[Dict[str, Any]]): Provider options forwarded to ``astream_chat``.
            **prompt_args (Any): Additional template variables.

        Returns:
            AsyncGenerator[Union[BaseModel, List[BaseModel]], None]: Async stream of parsed model(s).

        See Also:
            stream_structured_predict: Synchronous streaming counterpart.
        """
        if self.pydantic_program_mode == StructuredLLMMode.DEFAULT:

            async def gen(
                output_cls: Type[BaseModel],
                prompt: PromptTemplate,
                llm_kwargs: Dict[str, Any],
                prompt_args: Dict[str, Any],
            ) -> AsyncGenerator[Union[BaseModel, List[BaseModel]], None]:
                llm_kwargs = llm_kwargs or {}
                llm_kwargs["format"] = output_cls.model_json_schema()

                messages = prompt.format_messages(**prompt_args)
                response_gen = await self.astream_chat(messages, **llm_kwargs)

                cur_objects = None
                async for response in response_gen:
                    try:
                        processor = StreamingObjectProcessor(
                            output_cls=output_cls,
                            flexible_mode=True,
                            allow_parallel_tool_calls=False,
                        )
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
            return await super().astream_structured_predict(
                output_cls, prompt, llm_kwargs, **prompt_args
            )
