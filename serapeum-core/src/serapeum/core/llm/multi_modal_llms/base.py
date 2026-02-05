from abc import abstractmethod
from typing import Any, List, Optional, Sequence, Union

from serapeum.core.base.llms.models import (
    Message,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    Image,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
# from serapeum.core.callbacks import CallbackManager
from serapeum.core.configs.defaults import DEFAULT_CONTEXT_WINDOW, DEFAULT_NUM_OUTPUTS, DEFAULT_NUM_INPUT_FILES
# from serapeum.core.instrumentation import DispatcherSpanMixin
# from serapeum.core.llms.callbacks import llm_chat_callback, llm_completion_callback
from serapeum.core.schema import ImageNode


class MultiModalLLMMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=("pydantic_model_",))
    context_window: Optional[int] = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description=(
            "Total number of tokens the model can be input when generating a response."
        ),
    )
    num_output: Optional[int] = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="Number of tokens the model can output when generating a response.",
    )
    num_input_files: Optional[int] = Field(
        default=DEFAULT_NUM_INPUT_FILES,
        description="Number of input files the model can take when generating a response.",
    )
    is_function_calling_model: Optional[bool] = Field(
        default=False,
        # SEE: https://openai.com/blog/function-calling-and-other-api-updates
        description=(
            "Set True if the model supports function calling messages, similar to"
            " OpenAI's function calling API. For example, converting 'Email Anya to"
            " see if she wants to get coffee next Friday' to a function call like"
            " `send_email(to: string, body: string)`."
        ),
    )
    model_name: str = Field(
        default="unknown",
        description=(
            "The model's name used for logging, testing, and sanity checking. For some"
            " models this can be automatically discerned. For other models, like"
            " locally loaded models, this must be manually specified."
        ),
    )

    is_chat_model: bool = Field(
        default=False,
        description=(
            "Set True if the model exposes a chat interface (i.e. can be passed a"
            " sequence of messages, rather than text), like OpenAI's"
            " /v1/chat/completions endpoint."
        ),
    )


class MultiModalLLM:
    """Multi-Modal LLM interface."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Help static checkers understand this class hierarchy
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def metadata(self) -> MultiModalLLMMetadata:
        """Multi-Modal LLM metadata."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, Image]],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    def stream_complete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, Image]],
        **kwargs: Any,
    ) -> CompletionResponseGen:
        """Streaming completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> ChatResponse:
        """Chat endpoint for Multi-Modal LLM."""

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> ChatResponseGen:
        """Stream chat endpoint for Multi-Modal LLM."""

    # ===== Async Endpoints =====

    @abstractmethod
    async def acomplete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, Image]],
        **kwargs: Any,
    ) -> CompletionResponse:
        """Async completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def astream_complete(
        self,
        prompt: str,
        image_documents: List[Union[ImageNode, Image]],
        **kwargs: Any,
    ) -> CompletionResponseAsyncGen:
        """Async streaming completion endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def achat(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> ChatResponse:
        """Async chat endpoint for Multi-Modal LLM."""

    @abstractmethod
    async def astream_chat(
        self,
        messages: Sequence[Message],
        **kwargs: Any,
    ) -> ChatResponseAsyncGen:
        """Async streaming chat endpoint for Multi-Modal LLM."""