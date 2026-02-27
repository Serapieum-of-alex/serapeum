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
from serapeum.llama_cpp.utils import _download_url
from serapeum.core.utils.base import get_cache_dir

from llama_cpp import Llama

DEFAULT_LLAMA_CPP_GGML_MODEL = (
    "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGML/resolve"
    "/main/llama-2-13b-chat.ggmlv3.q4_0.bin"
)
DEFAULT_LLAMA_CPP_GGUF_MODEL = (
    "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve"
    "/main/llama-2-13b-chat.Q4_0.gguf"
)
DEFAULT_LLAMA_CPP_MODEL_VERBOSITY = True


class LlamaCPP(CompletionToChatMixin, LLM):
    r"""
    LlamaCPP LLM.

    Examples:
        Install llama-cpp-python following instructions:
        https://github.com/abetlen/llama-cpp-python

        Then `pip install serapeum-llama-cpp`

        ```python
        from serapeum.llama_cpp import LlamaCPP

        def messages_to_prompt(messages):
            prompt = ""
            for message in messages:
                if message.role == 'system':
                prompt += f"<|system|>\n{message.content}</s>\n"
                elif message.role == 'user':
                prompt += f"<|user|>\n{message.content}</s>\n"
                elif message.role == 'assistant':
                prompt += f"<|assistant|>\n{message.content}</s>\n"

            # ensure we start with a system prompt, insert blank if needed
            if not prompt.startswith("<|system|>\n"):
                prompt = "<|system|>\n</s>\n" + prompt

            # add final assistant prompt
            prompt = prompt + "<|assistant|>\n"

            return prompt

        def completion_to_prompt(completion):
            return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"

        model_url = "https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q4_0.gguf"

        llm = LlamaCPP(
            model_url=model_url,
            model_path=None,
            temperature=0.1,
            max_new_tokens=256,
            context_window=3900,
            generate_kwargs={},
            model_kwargs={"n_gpu_layers": -1},  # if compiled to use GPU
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            verbose=True,
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
        default=DEFAULT_LLAMA_CPP_MODEL_VERBOSITY,
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
            verbose = data.get("verbose", DEFAULT_LLAMA_CPP_MODEL_VERBOSITY)
            model_kwargs = dict(data.get("model_kwargs") or {})
            model_kwargs.setdefault("n_ctx", context_window)
            model_kwargs.setdefault("verbose", verbose)
            data = {**data, "model_kwargs": model_kwargs}
        return data

    def model_post_init(self, __context: Any) -> None:
        """Load or download the Llama model after all fields are validated."""
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
                _download_url(model_url, model_path)
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
        if stream:
            return self._stream_complete(prompt, **kwargs)
        return self._complete(prompt, **kwargs)

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

            return gen()

        return await asyncio.to_thread(self.complete, prompt, formatted, stream=False, **kwargs)

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
