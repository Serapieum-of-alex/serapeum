"""Base classes and implementations for prompt templates used by LLMs."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, WithJsonSchema
from typing_extensions import Annotated

from serapeum.core.base.llms.models import ChunkType, Message, MessageList, TextChunk
from serapeum.core.output_parsers import BaseParser
from serapeum.core.prompts.models import PromptType
from serapeum.core.prompts.utils import format_string, get_template_vars

AnnotatedCallable = Annotated[
    Callable,
    WithJsonSchema({"type": "string"}),
    WithJsonSchema({"type": "string"}),
    PlainSerializer(lambda x: f"{x.__module__}.{x.__name__}", return_type=str),
]


class BasePromptTemplate(BaseModel, ABC):  # type: ignore[no-redef]
    """Abstract base class for prompt templates.

    Subclasses must implement string and chat formatting helpers so the same
    prompt can be used with both completion and chat models.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    metadata: Dict[str, Any]
    template_vars: List[str]
    kwargs: Dict[str, str]
    output_parser: Optional[BaseParser]
    template_var_mappings: Optional[Dict[str, Any]] = Field(
        default_factory=dict,  # type: ignore
        description="Template variable mappings (Optional).",
    )
    function_mappings: Optional[Dict[str, AnnotatedCallable]] = Field(
        default_factory=dict,  # type: ignore
        description=(
            "Function mappings (Optional). This is a mapping from template "
            "variable names to functions that take in the current kwargs and "
            "return a string."
        ),
    )

    def _map_template_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """For keys in template_var_mappings, swap in the right keys."""
        template_var_mappings = self.template_var_mappings or {}
        return {template_var_mappings.get(k, k): v for k, v in kwargs.items()}

    def _map_function_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        For keys in function_mappings, compute values and combine w/ kwargs.

        Users can pass in functions instead of fixed values as format variables.
        For each function, we call the function with the current kwargs,
        get back the value, and then use that value in the template
        for the corresponding format variable.

        """
        function_mappings = self.function_mappings or {}
        # first generate the values for the functions
        new_kwargs = {}
        for k, v in function_mappings.items():
            new_kwargs[k] = v(**kwargs)

        # then, add the fixed variables only if not in new_kwargs already
        # (implying that function mapping will override fixed variables)
        for k, v in kwargs.items():
            if k not in new_kwargs:
                new_kwargs[k] = v

        return new_kwargs

    def _map_all_vars(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map both template and function variables.

        We (1) first call function mappings to compute functions,
        and then (2) call the template_var_mappings.

        """
        # map function
        new_kwargs = self._map_function_vars(kwargs)
        # map template vars (to point to existing format vars in string template)
        return self._map_template_vars(new_kwargs)

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Render the template to a single string."""

    @abstractmethod
    def format_messages(self, **kwargs: Any) -> List[Message]:
        """Render the template into a list of chat messages."""

    @abstractmethod
    def get_template(self) -> str:
        """Return the raw template string used by this prompt."""


class PromptTemplate(BasePromptTemplate):  # type: ignore[no-redef]
    """Prompt template for string-based LLM prompts."""

    template: str

    def __init__(
        self,
        template: str,
        prompt_type: str = PromptType.CUSTOM,
        output_parser: Optional[BaseParser] = None,
        metadata: Optional[Dict[str, Any]] = None,
        template_var_mappings: Optional[Dict[str, Any]] = None,
        function_mappings: Optional[Dict[str, Callable]] = None,
        **kwargs: Any,
    ) -> None:
        """Create a plain-text prompt template.

        Args:
            template: The raw template string (e.g., "Hello {name}").
            prompt_type: Logical type used for analytics/telemetry.
            output_parser: Optional output parser bound to this template.
            metadata: Optional metadata dictionary.
            template_var_mappings: Optional variable remapping for formatting.
            function_mappings: Optional mapping of template vars to callables.
            **kwargs: Default values for template variables.
        """
        if metadata is None:
            metadata = {}
        metadata["prompt_type"] = prompt_type

        template_vars = get_template_vars(template)

        super().__init__(
            template=template,
            template_vars=template_vars,
            kwargs=kwargs,
            metadata=metadata,
            output_parser=output_parser,
            template_var_mappings=template_var_mappings,
            function_mappings=function_mappings,
        )

    def format(
        self,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        **kwargs: Any,
    ) -> str:
        """Format the prompt into a string."""
        all_kwargs = {
            **self.kwargs,
            **kwargs,
        }

        mapped_all_kwargs = self._map_all_vars(all_kwargs)
        prompt = format_string(self.template, **mapped_all_kwargs)

        if self.output_parser is not None:
            prompt = self.output_parser.format(prompt)

        if completion_to_prompt is not None:
            prompt = completion_to_prompt(prompt)

        return prompt

    def format_messages(self, **kwargs: Any) -> List[Message]:
        """Format the prompt into a list of chat messages."""
        prompt = self.format(**kwargs)
        return list(MessageList.from_str(prompt))

    def get_template(self) -> str:
        return self.template


class ChatPromptTemplate(BasePromptTemplate):  # type: ignore[no-redef]
    """Prompt template for chat-based LLM prompts."""

    message_templates: List[Message]

    def __init__(
        self,
        message_templates: Sequence[Message],
        prompt_type: str = PromptType.CUSTOM,
        output_parser: Optional[BaseParser] = None,
        metadata: Optional[Dict[str, Any]] = None,
        template_var_mappings: Optional[Dict[str, Any]] = None,
        function_mappings: Optional[Dict[str, Callable]] = None,
        **kwargs: Any,
    ):
        """Create a chat-style prompt template.

        Args:
            message_templates: Sequence of message templates to render.
            prompt_type: Logical type used for analytics/telemetry.
            output_parser: Optional output parser bound to this template.
            metadata: Optional metadata dictionary.
            template_var_mappings: Optional variable remapping for formatting.
            function_mappings: Optional mapping of template vars to callables.
            **kwargs: Default values for template variables.
        """
        if metadata is None:
            metadata = {}
        metadata["prompt_type"] = prompt_type

        template_vars = []
        for message_template in message_templates:
            template_vars.extend(get_template_vars(message_template.content or ""))

        super().__init__(
            message_templates=message_templates,
            kwargs=kwargs,
            metadata=metadata,
            output_parser=output_parser,
            template_vars=template_vars,
            template_var_mappings=template_var_mappings,
            function_mappings=function_mappings,
        )

    @classmethod
    def from_messages(
        cls,
        message_templates: Union[List[Tuple[str, str]], List[Message]],
        **kwargs: Any,
    ) -> "ChatPromptTemplate":
        """From messages."""
        if isinstance(message_templates[0], tuple):
            message_templates = [
                Message.from_str(role=role, content=content)  # type: ignore[arg-type]
                for role, content in message_templates
            ]
        return cls(message_templates=message_templates, **kwargs)  # type: ignore[arg-type]

    def format(
        self,
        messages_to_prompt: Optional[Callable[[Sequence[Message]], str]] = None,
        **kwargs: Any,
    ) -> str:
        messages = self.format_messages(**kwargs)

        if messages_to_prompt is not None:
            return messages_to_prompt(messages)

        return MessageList(messages=messages).to_prompt()

    def format_messages(self, **kwargs: Any) -> List[Message]:
        """Format the prompt into a list of chat messages."""
        all_kwargs = {
            **self.kwargs,
            **kwargs,
        }
        mapped_all_kwargs = self._map_all_vars(all_kwargs)

        messages: List[Message] = []
        for message_template in self.message_templates:
            # Handle messages with multiple chunks
            if message_template.chunks:
                formatted_blocks: List[ChunkType] = []
                for block in message_template.chunks:
                    if isinstance(block, TextChunk):
                        template_vars = get_template_vars(block.content)
                        relevant_kwargs = {
                            k: v
                            for k, v in mapped_all_kwargs.items()
                            if k in template_vars
                        }
                        formatted_text = format_string(block.content, **relevant_kwargs)
                        formatted_blocks.append(TextChunk(content=formatted_text))
                    else:
                        # For non-text chunks (like images), keep them as is
                        # TODO: can images be formatted as variables?
                        formatted_blocks.append(block)

                message = message_template.model_copy()
                message.chunks = formatted_blocks
                messages.append(message)
            else:
                # Handle empty messages (if any)
                messages.append(message_template.model_copy())

        if self.output_parser is not None:
            messages = self.output_parser.format_messages(messages)

        return messages

    def get_template(self) -> str:
        return MessageList(messages=self.message_templates).to_prompt()
