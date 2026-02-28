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
from pydantic import Field, PrivateAttr, model_validator
from serapeum.core.configs.defaults import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from serapeum.llama_cpp.utils import _fetch_model_file
from serapeum.core.utils.base import get_cache_dir

from llama_cpp import Llama


DEFAULT_LLAMA_CPP_GGUF_MODEL = (
    "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve"
    "/main/llama-2-13b-chat.Q4_0.gguf"
)
DEFAULT_MODEL_VERBOSITY = False

# Module-level model cache.  WeakValues mean the Llama instance is released
# automatically when the last LlamaCPP referencing it is garbage-collected.
_MODEL_CACHE: weakref.WeakValueDictionary[str, Llama] = weakref.WeakValueDictionary()
_MODEL_CACHE_LOCK = threading.Lock()


class LlamaCPP(CompletionToChatMixin, LLM):
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
        Install llama-cpp-python following instructions:
        https://github.com/abetlen/llama-cpp-python

        Then ``pip install serapeum-llama-cpp``

        Llama 3 / Mistral instruct model:

        ```python
        from serapeum.llama_cpp import LlamaCPP
        from serapeum.llama_cpp.formatters import (
            messages_to_prompt_v3_instruct,
            completion_to_prompt_v3_instruct,
        )

        llm = LlamaCPP(
            model_path="/models/llama-3-8b-instruct.Q4_0.gguf",
            temperature=0.1,
            max_new_tokens=256,
            context_window=8192,
            messages_to_prompt=messages_to_prompt_v3_instruct,
            completion_to_prompt=completion_to_prompt_v3_instruct,
        )
        response = llm.complete("Hello, how are you?")
        print(str(response))
        ```

        Llama 2 model:

        ```python
        from serapeum.llama_cpp import LlamaCPP
        from serapeum.llama_cpp.formatters import (
            messages_to_prompt,
            completion_to_prompt,
        )

        llm = LlamaCPP(
            model_path="/models/llama-2-13b-chat.Q4_0.gguf",
            temperature=0.1,
            max_new_tokens=256,
            context_window=4096,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
        )
        response = llm.complete("Hello, how are you?")
        print(str(response))
        ```
    """

    model_url: str | None = Field(
        default=None,
        description="The URL llama-cpp model to download and use."
    )
    model_path: str | None = Field(
        default=None,
        description="The path to the llama-cpp model to use."
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

    @model_validator(mode="after")
    def _check_model_source(self) -> "LlamaCPP":
        """Ensure at least one of model_path or model_url is provided."""
        if self.model_path is None and self.model_url is None:
            raise ValueError(
                "Either model_path or model_url must be provided. "
                "Set model_path to a local GGUF file, or model_url to download one."
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
        """Validate formatters, resolve the model path, and load the model."""
        # Fail fast if the caller omitted a formatter.  The base LLM silently
        # falls back to a generic lambda that produces no instruct template,
        # which causes garbage output from any GGUF instruct model.
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

        # Resolve the model path from whichever source was provided.
        if self.model_path is not None:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise ValueError(
                    "Provided model path does not exist. "
                    "Please check the path or provide a model_url to download."
                )
        elif self.hf_model_id is not None:
            # hf_filename is guaranteed non-None by _check_model_source.
            cache_dir = Path(get_cache_dir()) / "models"
            cache_dir.mkdir(parents=True, exist_ok=True)
            model_path = _fetch_model_file_hf(
                self.hf_model_id,
                self.hf_filename,  # type: ignore[arg-type]
                cache_dir,
            )
            self.model_path = str(model_path)
        else:
            cache_dir = Path(get_cache_dir())
            model_url = self.model_url or DEFAULT_LLAMA_CPP_GGUF_MODEL
            model_path = cache_dir / "models" / model_url.rsplit("/", 1)[-1]
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                _fetch_model_file(model_url, model_path)
                if not model_path.exists():
                    raise RuntimeError(
                        f"Download appeared to succeed but model not found at {model_path!r}"
                    )
            self.model_path = str(model_path)

        # Check the module-level cache before loading.  Double-checked locking:
        # load outside the lock so threads with different models don't serialise.
        cache_key = (str(model_path), json.dumps(self.model_kwargs, sort_keys=True))
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                self._model = cached
                return

        model = Llama(model_path=str(model_path), **self.model_kwargs)

        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                # Another thread loaded the same model while we were loading.
                self._model = cached
                return
            _MODEL_CACHE[cache_key] = model

        self._model = model

    @classmethod
    def class_name(cls) -> str:
        return "LlamaCPP"

    @property
    def metadata(self) -> Metadata:
        """LLM metadata."""
        return Metadata(
            context_window=self._model.context_params.n_ctx,
            num_output=self.max_new_tokens,
            model_name=self.model_path,
        )

    def tokenize(self, text: str) -> list[int]:
        """Return the token IDs for *text* using the loaded model's vocabulary."""
        return self._model.tokenize(text.encode())

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens *text* encodes to."""
        return len(self.tokenize(text))

    def _guard_context(self, prompt: str) -> None:
        """Raise ValueError if *prompt* exceeds the model's context window."""
        n = self.count_tokens(prompt)
        if n > self.context_window:
            raise ValueError(
                f"Prompt is {n} tokens but context_window is {self.context_window}. "
                "Shorten the prompt or increase context_window."
            )

    @overload
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
        if not formatted:
            prompt = self.completion_to_prompt(prompt)
        result: CompletionResponse | CompletionResponseGen = (
            self._stream_complete(prompt, **kwargs)
            if stream
            else self._complete(prompt, **kwargs)
        )
        return result

    @overload
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
        """Async completion: offloads CPU-bound llama.cpp inference to a thread pool.

        Overrides the mixin's thin sync shim so inference does not block the
        running event loop. The streaming variant collects all chunks in the
        worker thread, then re-yields them as an async generator.
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
            result = await asyncio.to_thread(self.complete, prompt, formatted, stream=False, **kwargs)
        return result

    def _complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        self._guard_context(prompt)
        call_kwargs = {
            **self.generate_kwargs,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "stream": False,
            **kwargs,
        }
        call_kwargs.setdefault("stop", self.stop or None)
        response = self._model(prompt=prompt, **call_kwargs)
        return CompletionResponse(text=response["choices"][0]["text"], raw=response)

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        self._guard_context(prompt)
        call_kwargs = {
            **self.generate_kwargs,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "stream": True,
            **kwargs,
        }
        call_kwargs.setdefault("stop", self.stop or None)
        response_iter = self._model(prompt=prompt, **call_kwargs)

        def gen() -> CompletionResponseGen:
            text = ""
            for response in response_iter:
                delta = response["choices"][0]["text"]
                text += delta
                yield CompletionResponse(delta=delta, text=text, raw=response)

        return gen()
