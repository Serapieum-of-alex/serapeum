import asyncio
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
        """Merge n_ctx/verbose defaults into model_kwargs before field validation.

        User-supplied model_kwargs take precedence over these defaults.
        """
        if isinstance(data, dict):
            context_window = data.get("context_window", DEFAULT_CONTEXT_WINDOW)
            verbose = data.get("verbose", DEFAULT_MODEL_VERBOSITY)
            model_kwargs = dict(data.get("model_kwargs") or {})
            model_kwargs.setdefault("n_ctx", context_window)
            model_kwargs.setdefault("verbose", verbose)
            data = {**data, "model_kwargs": model_kwargs}
        return data

    def model_post_init(self, __context: Any) -> None:
        """Load or download the Llama model after all fields are validated."""
        # The LLM base silently falls back to a generic adapter when
        # messages_to_prompt / completion_to_prompt are not provided.  That
        # generic adapter produces no instruct template, so the output of any
        # GGUF instruct model will be garbage.  We use model_fields_set (the
        # set of fields the caller explicitly passed) to detect omissions and
        # fail fast with an actionable message instead of silently producing
        # bad output.
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

        # check if model is cached
        if self.model_path is not None:
            model_path = Path(self.model_path)
            if not model_path.exists():
                raise ValueError(
                    "Provided model path does not exist. "
                    "Please check the path or provide a model_url to download."
                )
            model = Llama(model_path=str(model_path), **self.model_kwargs)
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

            model = Llama(model_path=str(model_path), **self.model_kwargs)
            self.model_path = str(model_path)

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
        call_kwargs = {
            **self.generate_kwargs,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "stream": False,
            **kwargs,
        }
        response = self._model(prompt=prompt, **call_kwargs)
        return CompletionResponse(text=response["choices"][0]["text"], raw=response)

    def _stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        call_kwargs = {
            **self.generate_kwargs,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "stream": True,
            **kwargs,
        }
        response_iter = self._model(prompt=prompt, **call_kwargs)

        def gen() -> CompletionResponseGen:
            text = ""
            for response in response_iter:
                delta = response["choices"][0]["text"]
                text += delta
                yield CompletionResponse(delta=delta, text=text, raw=response)

        return gen()
