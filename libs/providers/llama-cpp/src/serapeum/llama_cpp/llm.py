"""LlamaCPP provider — local GGUF inference via llama-cpp-python.

Contains the :class:`LlamaCPP` class, a concrete
:class:`~serapeum.core.llms.LLM` implementation that runs quantised GGUF
models on-device using the
`llama-cpp-python <https://github.com/abetlen/llama-cpp-python>`_ backend.

Key capabilities:

- **Model sources**: local path, direct URL download, or HuggingFace Hub.
- **Prompt formatters**: pluggable ``messages_to_prompt`` /
  ``completion_to_prompt`` per model family (Llama 2, Llama 3, …).
- **GPU offloading**: ``n_gpu_layers`` controls layer offloading via
  cuBLAS / Metal / Vulkan.
- **Model caching**: a module-level :class:`weakref.WeakValueDictionary`
  reuses loaded ``Llama`` instances across :class:`LlamaCPP` objects with
  identical model path and kwargs.
- **Async-safe**: :meth:`~LlamaCPP.acomplete` offloads CPU-bound inference
  to a thread pool so the event loop is never blocked.

See Also:
    serapeum.llama_cpp.formatters: Ready-made prompt formatters.
    serapeum.llama_cpp.utils: Internal download helpers.
"""

import asyncio
import json
import threading
import weakref
from pathlib import Path
from typing import Any, Literal, overload

from serapeum.core.llms import (
    LLM,
    CompletionToChatMixin,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    Metadata,
)
from pydantic import Field, PrivateAttr, field_validator, model_validator
from serapeum.core.configs.defaults import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from serapeum.llama_cpp.utils import _fetch_model_file, _fetch_model_file_hf
from serapeum.core.utils.base import get_cache_dir

from llama_cpp import Llama


DEFAULT_LLAMA_CPP_GGUF_MODEL = (
    "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve"
    "/main/llama-2-13b-chat.Q4_0.gguf"
)
DEFAULT_MODEL_VERBOSITY = False

# Module-level model cache.  WeakValues mean the Llama instance is released
# automatically when the last LlamaCPP referencing it is garbage-collected.
_MODEL_CACHE: weakref.WeakValueDictionary[tuple[str, str], Llama] = weakref.WeakValueDictionary()  # type: ignore[valid-type]
_MODEL_CACHE_LOCK = threading.Lock()


class LlamaCPP(CompletionToChatMixin, LLM):  # type: ignore[misc]
    """LlamaCPP LLM — local inference via llama-cpp-python.

    Runs GGUF models locally using the llama-cpp-python backend.  The model is
    loaded (or downloaded) once at construction time.

    ``messages_to_prompt`` and ``completion_to_prompt`` are **required**.
    GGUF models each have a specific chat template; using the wrong one
    produces garbage output.  Pass the formatter that matches the model family
    you are loading.  Ready-made formatters live in
    ``serapeum.llama_cpp.formatters``.

    Warning:
        Construction is **blocking**. Loading a large GGUF file can take
        10-30 seconds. To construct inside an async context without blocking
        the event loop wrap the call in ``asyncio.to_thread``::

            llm = await asyncio.to_thread(LlamaCPP, model_path="...", ...)

    Examples:
        - Load a Llama 3 instruct model from a local path and run a completion
            ```python
            >>> import os
            >>> from serapeum.llama_cpp import LlamaCPP
            >>> from serapeum.llama_cpp.formatters.llama3 import (
            ...     messages_to_prompt_v3_instruct,
            ...     completion_to_prompt_v3_instruct,
            ... )
            >>> model_path = "path/to/llama-3-8b-instruct-v0.1.Q2_K.gguf"
            >>> llm_v3 = LlamaCPP(
            ...     model_path=model_path,
            ...     temperature=0.1,
            ...     max_new_tokens=256,
            ...     context_window=512,
            ...     messages_to_prompt=messages_to_prompt_v3_instruct,
            ...     completion_to_prompt=completion_to_prompt_v3_instruct,
            ... )
            >>> response = llm_v3.complete("Hello, how are you?")
            >>> response.text
            'He, Annaxter...'

            ```
        - Load a Llama 2 / Mistral model from a local path
            ```python
            >>> from serapeum.llama_cpp import LlamaCPP
            >>> from serapeum.llama_cpp.formatters.llama2 import (
            ...     messages_to_prompt,
            ...     completion_to_prompt,
            ... )
            >>> llm_v2 = LlamaCPP(
            ...     model_path="path/to/mistral-7b-instruct-v0.1.Q2_K.gguf",
            ...     temperature=0.1,
            ...     max_new_tokens=256,
            ...     context_window=512,
            ...     messages_to_prompt=messages_to_prompt,
            ...     completion_to_prompt=completion_to_prompt,
            ... )
            >>> response = llm_v2.complete("Hello, how are you?")
            >>> response.text
            " I'm doing well, thank you for asking!"

            ```
    """

    model_url: str | None = Field(
        default=None,
        description="URL of a GGUF model to download and cache locally.",
    )
    model_path: str | None = Field(
        default=None,
        description="Path to a local GGUF model file.",
    )
    hf_model_id: str | None = Field(
        default=None,
        description=(
            "HuggingFace Hub repo ID (e.g. 'TheBloke/Llama-2-13B-chat-GGUF'). "
            "Requires ``pip install huggingface-hub``."
        ),
    )
    hf_filename: str | None = Field(
        default=None,
        description=(
            "Filename within the HuggingFace Hub repo "
            "(e.g. 'llama-2-13b-chat.Q4_0.gguf'). Required when hf_model_id is set."
        ),
    )
    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        ge=0.0,
        le=1.0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    n_gpu_layers: int = Field(
        default=0,
        description=(
            "Number of model layers to offload to GPU. "
            "Set to -1 to offload all layers."
        ),
    )
    stop: list[str] = Field(
        default_factory=list,
        description="Token sequences that stop generation (e.g. ['</s>', '<|eot_id|>']).",
    )
    generate_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )
    verbose: bool = Field(
        default=DEFAULT_MODEL_VERBOSITY,
        description="Whether to print verbose output.",
    )

    _model: Any = PrivateAttr(default=None)
    # Serializes concurrent model calls: llama_cpp releases the GIL during
    # C-level inference, so two asyncio.to_thread calls on the same Llama
    # instance can race and abort().  One lock per LlamaCPP instance is
    # sufficient for the common "one model, many callers" pattern.
    _model_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @field_validator("model_path")
    @classmethod
    def _validate_model_path_exists(cls, v: str | None) -> str | None:
        """Validate that model_path points to an existing file when provided."""
        if v is not None and not Path(v).exists():
            raise ValueError(
                "Provided model path does not exist. "
                "Please check the path or provide a model_url to download."
            )
        return v

    @model_validator(mode="after")
    def _check_model_source(self) -> "LlamaCPP":
        """Ensure the cross-field model source combination is valid."""
        if (
            self.model_path is None
            and self.model_url is None
            and self.hf_model_id is None
        ):
            raise ValueError(
                "One of model_path, model_url, or hf_model_id must be provided. "
                "Set model_path to a local GGUF file, model_url to download one, "
                "or hf_model_id + hf_filename to download from HuggingFace Hub."
            )
        if self.hf_model_id is not None and self.hf_filename is None:
            raise ValueError(
                "hf_filename is required when hf_model_id is provided. "
                "Example: hf_filename='llama-2-13b-chat.Q4_0.gguf'."
            )
        return self

    @model_validator(mode="after")
    def _check_formatters(self) -> "LlamaCPP":
        """Ensure both prompt formatters were explicitly provided by the caller.

        The base LLM silently falls back to a generic lambda when
        messages_to_prompt / completion_to_prompt are omitted, producing no
        instruct template and therefore garbage output from any GGUF instruct
        model.  model_fields_set contains only the fields the caller explicitly
        passed, so omitted fields are detected even if they have a default.
        """
        missing = [
            name
            for name in ("messages_to_prompt", "completion_to_prompt")
            if name not in self.model_fields_set
        ]
        if missing:
            raise ValueError(
                f"LlamaCPP requires explicit prompt formatters: {', '.join(missing)}.\n"
                "Pass a formatter that matches your model's chat template.\n"
                "Ready-made formatters are available in serapeum.llama_cpp.formatters:\n"
                "  Llama 2 / Mistral:  messages_to_prompt, completion_to_prompt\n"
                "  Llama 3:            messages_to_prompt_v3_instruct, completion_to_prompt_v3_instruct"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def _prepare_kwargs(cls, data: Any) -> Any:
        """Merge n_ctx, verbose, and n_gpu_layers defaults into model_kwargs.

        User-supplied model_kwargs take precedence over these defaults.
        """
        if isinstance(data, dict):
            context_window = data.get("context_window", DEFAULT_CONTEXT_WINDOW)
            verbose = data.get("verbose", DEFAULT_MODEL_VERBOSITY)
            n_gpu_layers = data.get("n_gpu_layers", 0)
            model_kwargs = dict(data.get("model_kwargs") or {})
            model_kwargs.setdefault("n_ctx", context_window)
            model_kwargs.setdefault("verbose", verbose)
            model_kwargs.setdefault("n_gpu_layers", n_gpu_layers)
            data = {**data, "model_kwargs": model_kwargs}
        return data

    def model_post_init(self, __context: Any) -> None:
        """Resolve the model path, download if needed, then load the model.

        Called automatically by Pydantic after ``__init__``.  All validation
        has already completed before this method runs; it performs only I/O
        (path resolution, optional download, GGUF loading).

        See Also:
            _resolve_model_path: Locates or downloads the GGUF file.
            _load_model: Loads (or retrieves from cache) the Llama instance.
        """
        model_path = self._resolve_model_path()
        self._model = self._load_model(model_path)

    def _resolve_model_path(self) -> Path:
        """Return the local Path to the GGUF file, downloading it if required.

        Checks :attr:`model_path`, :attr:`hf_model_id`, and :attr:`model_url`
        in that priority order.  Downloads or fetches from HuggingFace Hub
        when a local path is not available.  Sets :attr:`model_path` as a
        side-effect so subsequent reloads skip the network step.

        Returns:
            :class:`~pathlib.Path` pointing to the resolved local GGUF file.

        Raises:
            RuntimeError: If a URL download appears to succeed but the file
                is not present on disk afterwards.

        See Also:
            _load_model: Called immediately after this method in
                :meth:`model_post_init`.
            serapeum.llama_cpp.utils._fetch_model_file: URL download helper.
            serapeum.llama_cpp.utils._fetch_model_file_hf: HuggingFace Hub
                download helper.
        """
        if self.model_path is not None:
            model_path = Path(self.model_path)
        elif self.hf_model_id is not None and self.hf_filename is not None:
            cache_dir = Path(get_cache_dir()) / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)
            model_path = _fetch_model_file_hf(
                self.hf_model_id,
                self.hf_filename,
                cache_dir,
            )
            self.model_path = str(model_path)
        else:
            model_url = self.model_url or DEFAULT_LLAMA_CPP_GGUF_MODEL
            model_path = Path(get_cache_dir()) / "models" / model_url.rsplit("/", 1)[-1]
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                _fetch_model_file(model_url, model_path)
                if not model_path.exists():
                    raise RuntimeError(
                        f"Download appeared to succeed but model not found at {model_path!r}"
                    )
            self.model_path = str(model_path)
        return model_path

    def _load_model(self, model_path: Path) -> Llama:  # type: ignore[valid-type]
        """Return a Llama instance for *model_path*, reusing the cache if possible.

        Uses double-checked locking so threads loading different models do not
        serialise on a single global lock, while still preventing duplicate
        loads of the same model.  The cache key is a ``(path, kwargs_json)``
        tuple so models with different generation settings are kept separate.

        Args:
            model_path: Absolute path to the local GGUF file.

        Returns:
            A :class:`llama_cpp.Llama` instance — either freshly loaded or
            retrieved from the module-level ``_MODEL_CACHE``.

        See Also:
            _resolve_model_path: Resolves the path before this method is called.
            _MODEL_CACHE: Module-level WeakValueDictionary that holds cached
                Llama instances.
        """
        cache_key = (str(model_path), json.dumps(self.model_kwargs, sort_keys=True))

        with _MODEL_CACHE_LOCK:
            result = _MODEL_CACHE.get(cache_key)

        if result is None:
            loaded = Llama(model_path=str(model_path), **self.model_kwargs)  # type: ignore[operator]
            with _MODEL_CACHE_LOCK:
                if _MODEL_CACHE.get(cache_key) is None:
                    _MODEL_CACHE[cache_key] = loaded
                result = _MODEL_CACHE[cache_key]

        return result

    @classmethod
    def class_name(cls) -> str:
        """Return the canonical class identifier used in serialisation.

        Returns:
            The string ``"LlamaCPP"``.
        """
        return "LlamaCPP"

    @property
    def metadata(self) -> Metadata:
        """LLM metadata derived from the loaded model's configuration.

        Returns:
            :class:`~serapeum.core.llms.Metadata` instance with:

            - ``context_window``: effective context size from the loaded model.
            - ``num_output``: :attr:`max_new_tokens` configured for generation.
            - ``model_name``: resolved local path to the GGUF file.

        Examples:
            - Inspect metadata fields of a loaded model
                ```python
                >>> meta = llm.metadata
                >>> meta.model_name
                '/models/llama-3-8b-instruct.Q4_0.gguf'
                >>> meta.num_output
                256
                >>> meta.context_window
                512

                ```

        See Also:
            class_name: Class identifier used for serialisation.
        """
        return Metadata(
            context_window=self._model.context_params.n_ctx,
            num_output=self.max_new_tokens,
            model_name=self.model_path or "unknown",
        )

    def tokenize(self, text: str) -> list[int]:
        """Return the token IDs for *text* using the loaded model's vocabulary.

        Args:
            text: The input string to tokenize.

        Returns:
            List of integer token IDs produced by the model's tokenizer.

        Examples:
            - Tokenize a short string and explore the token IDs
                ```python
                >>> tokens = llm.tokenize("Hello!")
                >>> len(tokens)
                4
                >>> tokens[:3]
                [1, 15043, 29991]

                ```

        See Also:
            count_tokens: Returns the token count instead of the full list.
            _guard_context: Uses token count to validate prompt length.
        """
        return self._model.tokenize(text.encode())  # type: ignore[no-any-return]

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens *text* encodes to.

        Args:
            text: The input string to count tokens for.

        Returns:
            Integer token count for *text*.

        Examples:
            - Count tokens in phrases of different lengths
                ```python
                >>> llm.count_tokens("Hello!")
                4
                >>> llm.count_tokens("A longer sentence has more tokens.")
                9

                ```

        See Also:
            tokenize: Returns the full token ID list.
            _guard_context: Calls this method to check prompt length.
        """
        return len(self.tokenize(text))

    def _guard_context(self, prompt: str) -> None:
        """Raise ValueError if *prompt* exceeds the model's context window.

        Args:
            prompt: Already-formatted prompt string whose token length is
                checked against :attr:`context_window`.

        Raises:
            ValueError: If the token count of *prompt* exceeds
                :attr:`context_window`, reporting the actual count and the
                configured limit.

        Examples:
            - A prompt within the context window raises no error
                ```python
                >>> llm._guard_context("Short prompt.")

                ```
            - A prompt that exceeds the context window raises ValueError
                ```python
                >>> llm._guard_context("word " * 10_000)
                Traceback (most recent call last):
                    ...
                ValueError: Prompt is 10001 tokens but context_window is 512...

                ```

        See Also:
            count_tokens: Used internally to measure the prompt length.
        """
        n = self.count_tokens(prompt)
        if n > self.context_window:
            raise ValueError(
                f"Prompt is {n} tokens but context_window is {self.context_window}. "
                "Shorten the prompt or increase context_window."
            )

    @overload  # type: ignore[override]
    def complete(
        self,
        prompt: str,
        formatted: bool = ...,
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> CompletionResponse: ...

    @overload
    def complete(
        self,
        prompt: str,
        formatted: bool = ...,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> CompletionResponseGen: ...

    def complete(
        self,
        prompt: str,
        formatted: bool = False,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse | CompletionResponseGen:
        """Run text completion, optionally streaming token-by-token.

        Args:
            prompt: The input text to complete.
            formatted: When ``True``, *prompt* is passed to the model as-is.
                When ``False`` (default) it is first wrapped by
                :attr:`completion_to_prompt` to apply the model's chat template.
            stream: When ``True`` returns a :class:`CompletionResponseGen`
                generator that yields one :class:`CompletionResponse` per
                token delta.  When ``False`` (default) returns a single
                :class:`CompletionResponse` with the full completion.
            **kwargs: Additional keyword arguments forwarded to the underlying
                ``Llama.__call__`` (e.g. ``top_p``, ``repeat_penalty``).

        Returns:
            A :class:`CompletionResponse` when ``stream=False``, or a
            :class:`CompletionResponseGen` generator when ``stream=True``.

        Raises:
            ValueError: If *prompt* exceeds :attr:`context_window` tokens.

        Examples:
            - Non-streaming completion — explore the response text and raw output
                ```python
                >>> response = llm.complete("The capital of France is")
                >>> response.text
                ' Paris, the City of Light.'
                >>> response.raw["choices"][0]["finish_reason"]
                'stop'

                ```
            - Streaming completion — iterate over token deltas
                ```python
                >>> gen = llm.complete("Hello", stream=True)
                >>> first = next(gen)
                >>> first.delta
                ' there'
                >>> first.text
                ' there'

                ```

        See Also:
            acomplete: Async variant that offloads inference to a thread pool.
            _complete: Non-streaming implementation.
            _stream_complete: Streaming implementation.
        """
        if not formatted:
            prompt = self.completion_to_prompt(prompt)  # type: ignore[misc]
        result: CompletionResponse | CompletionResponseGen = (
            self._stream_complete(prompt, **kwargs)
            if stream
            else self._complete(prompt, **kwargs)
        )
        return result

    @overload  # type: ignore[override]
    async def acomplete(
        self,
        prompt: str,
        formatted: bool = ...,
        *,
        stream: Literal[False] = ...,
        **kwargs: Any,
    ) -> CompletionResponse: ...

    @overload
    async def acomplete(
        self,
        prompt: str,
        formatted: bool = ...,
        *,
        stream: Literal[True],
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen: ...

    async def acomplete(
        self,
        prompt: str,
        formatted: bool = False,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> CompletionResponse | CompletionResponseAsyncGen:
        """Async text completion — offloads CPU-bound inference to a thread pool.

        Wraps :meth:`complete` in :func:`asyncio.to_thread` so that the
        llama-cpp-python C-level inference call never blocks the running event
        loop.  The streaming variant collects all token chunks in the worker
        thread and re-yields them as an async generator once all chunks are
        ready.

        Args:
            prompt: The input text to complete.
            formatted: When ``True``, *prompt* is passed to the model as-is.
                When ``False`` (default) it is first wrapped by
                :attr:`completion_to_prompt` to apply the model's chat template.
            stream: When ``True`` returns a :class:`CompletionResponseAsyncGen`
                async generator that yields one :class:`CompletionResponse` per
                token delta.  When ``False`` (default) returns a single
                :class:`CompletionResponse` with the full completion.
            **kwargs: Additional keyword arguments forwarded to the underlying
                ``Llama.__call__`` (e.g. ``top_p``, ``repeat_penalty``).

        Returns:
            A :class:`CompletionResponse` when ``stream=False``, or a
            :class:`CompletionResponseAsyncGen` async generator when
            ``stream=True``.

        Raises:
            ValueError: If *prompt* exceeds :attr:`context_window` tokens.

        Examples:
            - Non-streaming async completion — explore the response
                ```python
                >>> import asyncio
                >>> response = asyncio.run(llm.acomplete("Hello"))
                >>> response.text
                ' there! How can I help you today?'

                ```
            - Streaming async completion — collect and inspect chunks
                ```python
                >>> import asyncio
                >>> async def _collect():
                ...     return [c async for c in await llm.acomplete("Hello", stream=True)]
                >>> chunks = asyncio.run(_collect())
                >>> chunks[0].delta
                ' there'
                >>> chunks[-1].text
                ' there! How can I help you today?'

                ```

        See Also:
            complete: Synchronous variant of this method.
            _complete: Non-streaming implementation called in the thread pool.
            _stream_complete: Streaming implementation called in the thread pool.
        """
        if stream:
            chunks: list[CompletionResponse] = await asyncio.to_thread(
                lambda: list(self.complete(prompt, formatted=formatted, stream=True, **kwargs))
            )

            async def gen() -> CompletionResponseAsyncGen:
                for chunk in chunks:
                    yield chunk

            result: CompletionResponse | CompletionResponseAsyncGen = gen()
        else:
            result = await asyncio.to_thread(self.complete, prompt, formatted, stream=False, **kwargs)  # type: ignore[arg-type]
        return result

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Run a single non-streaming inference pass and return the full completion.

        Args:
            prompt: Already-formatted prompt string to send to the model.
            **kwargs: Additional keyword arguments forwarded to the underlying
                ``Llama.__call__`` (e.g. ``top_p``, ``repeat_penalty``).

        Returns:
            :class:`CompletionResponse` with the full generated text in
            ``.text`` and the raw llama-cpp-python response dict in ``.raw``.

        Raises:
            ValueError: If *prompt* exceeds :attr:`context_window` tokens
                (checked via :meth:`_guard_context`).

        See Also:
            _stream_complete: Streaming variant.
            _guard_context: Context-window overflow check.
        """
        self._guard_context(prompt)
        call_kwargs = {
            **self.generate_kwargs,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "stream": False,
            **kwargs,
        }
        call_kwargs.setdefault("stop", self.stop or None)
        with self._model_lock:
            response = self._model(prompt=prompt, **call_kwargs)
        return CompletionResponse(text=response["choices"][0]["text"], raw=response)

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Run a streaming inference pass and yield one response per token delta.

        The underlying ``Llama.__call__`` is called with ``stream=True`` inside
        a local generator so the model lock is held for the entire streaming
        session, preventing concurrent calls from corrupting state.

        Args:
            prompt: Already-formatted prompt string to send to the model.
            **kwargs: Additional keyword arguments forwarded to the underlying
                ``Llama.__call__``.

        Yields:
            :class:`CompletionResponse` objects — one per generated token —
            where ``.delta`` contains the incremental text and ``.text`` the
            cumulative completion so far.

        Raises:
            ValueError: If *prompt* exceeds :attr:`context_window` tokens
                (checked via :meth:`_guard_context`).

        See Also:
            _complete: Non-streaming variant.
            _guard_context: Context-window overflow check.
        """
        self._guard_context(prompt)
        call_kwargs = {
            **self.generate_kwargs,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "stream": True,
            **kwargs,
        }
        call_kwargs.setdefault("stop", self.stop or None)

        def gen() -> CompletionResponseGen:
            text = ""
            with self._model_lock:
                for response in self._model(prompt=prompt, **call_kwargs):
                    delta = response["choices"][0]["text"]
                    text += delta
                    yield CompletionResponse(delta=delta, text=text, raw=response)

        return gen()
