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
_MODEL_CACHE: weakref.WeakValueDictionary[tuple[str, str], Llama] = weakref.WeakValueDictionary()
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
        from serapeum.llama_cpp.formatters.llama3 import (
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
        from serapeum.llama_cpp.formatters.llama2 import (
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
        """Download if needed, then load the model."""
        model_path = self._resolve_model_path()
        self._model = self._load_model(model_path)

    def _resolve_model_path(self) -> Path:
        """Return the local Path to the GGUF file, downloading it if required."""
        if self.model_path is not None:
            model_path = Path(self.model_path)
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

    def _load_model(self, model_path: Path) -> Llama:
        """Return a Llama instance for *model_path*, reusing the cache if possible.

        Uses double-checked locking so threads loading different models do not
        serialise on a single lock, while still preventing duplicate loads of
        the same model.
        """
        cache_key = (str(model_path), json.dumps(self.model_kwargs, sort_keys=True))

        with _MODEL_CACHE_LOCK:
            result = _MODEL_CACHE.get(cache_key)

        if result is None:
            loaded = Llama(model_path=str(model_path), **self.model_kwargs)
            with _MODEL_CACHE_LOCK:
                if _MODEL_CACHE.get(cache_key) is None:
                    _MODEL_CACHE[cache_key] = loaded
                result = _MODEL_CACHE[cache_key]

        return result

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
        with self._model_lock:
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

        def gen() -> CompletionResponseGen:
            text = ""
            with self._model_lock:
                for response in self._model(prompt=prompt, **call_kwargs):
                    delta = response["choices"][0]["text"]
                    text += delta
                    yield CompletionResponse(delta=delta, text=text, raw=response)

        return gen()
