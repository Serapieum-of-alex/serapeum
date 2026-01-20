"""Utilities for structured tools."""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Sequence,
    Type,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from serapeum.core.models import StructuredLLMMode
from serapeum.core.output_parsers import PydanticParser

if TYPE_CHECKING:
    from serapeum.core.base.llms.models import ChatResponse
    from serapeum.core.llm.base import LLM
    from serapeum.core.llm.function_calling import FunctionCallingLLM
    from serapeum.core.prompts.base import BasePromptTemplate
    from serapeum.core.structured_tools.models import BasePydanticLLM
    from serapeum.core.tools.models import ToolCallArguments

Model = TypeVar("Model", bound=BaseModel)

_logger = logging.getLogger(__name__)


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    @classmethod
    def create(cls, model: Type[Model]) -> Type[Model]:
        """Create a flexible version of the model that allows any fields."""
        return create_model(
            f"Flexible{model.__name__}",
            __base__=cls,
            **{field: (Any | None, None) for field in model.model_fields},
        )


def get_program_for_llm(
    output_cls: Type[BaseModel],
    prompt: BasePromptTemplate,
    llm: LLM,
    pydantic_program_mode: StructuredLLMMode = StructuredLLMMode.DEFAULT,
    **kwargs: Any,
) -> BasePydanticLLM:
    if pydantic_program_mode == StructuredLLMMode.DEFAULT:
        if llm.metadata.is_function_calling_model:
            from serapeum.core.structured_tools.tools_llm import ToolOrchestratingLLM

            return ToolOrchestratingLLM(
                output_cls=output_cls,
                llm=llm,
                prompt=prompt,
                **kwargs,
            )
        else:
            from serapeum.core.structured_tools.text_completion_llm import (
                TextCompletionLLM,
            )

            return TextCompletionLLM(
                output_parser=PydanticParser(output_cls=output_cls),
                llm=llm,
                prompt=prompt,
                **kwargs,
            )
    elif pydantic_program_mode == StructuredLLMMode.FUNCTION:
        from serapeum.core.structured_tools.tools_llm import ToolOrchestratingLLM

        return ToolOrchestratingLLM(
            output_cls=output_cls,
            llm=llm,
            prompt=prompt,
            **kwargs,
        )

    elif pydantic_program_mode == StructuredLLMMode.LLM:
        from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM

        return TextCompletionLLM(
            output_parser=PydanticParser(output_cls=output_cls),
            llm=llm,
            prompt=prompt,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported pydantic program mode: {pydantic_program_mode}")


def _repair_incomplete_json(json_str: str) -> str:
    """Attempt to repair incomplete JSON strings.

    Args:
        json_str (str): Potentially incomplete JSON string

    Returns:
        str: Repaired JSON string
    """
    if not json_str.strip():
        return "{}"

    # Add missing quotes
    quote_count = json_str.count('"')
    if quote_count % 2 == 1:
        json_str += '"'

    # Add missing braces
    brace_count = json_str.count("{") - json_str.count("}")
    if brace_count > 0:
        json_str += "}" * brace_count

    return json_str


class StreamingObjectProcessor:
    """Processes streaming chat responses into structured Pydantic objects.

    This processor handles incremental parsing of streaming responses, with support
    for flexible schemas, multiple tool calls, and progressive object accumulation.
    """

    __slots__ = (
        "_output_cls",
        "_parsing_cls",
        "_flexible_mode",
        "_allow_parallel",
        "_llm",
    )

    def __init__(
        self,
        output_cls: Type[BaseModel],
        flexible_mode: bool = True,
        allow_parallel_tool_calls: bool = False,
        llm: FunctionCallingLLM | None = None,
    ) -> None:
        """Initialize the streaming object processor.

        Args:
            output_cls: Target Pydantic model class for output
            flexible_mode: Use flexible schema allowing partial fields during parsing
            allow_parallel_tool_calls: Return all objects vs only the first
            llm: LLM instance for extracting tool calls from responses
        """
        self._output_cls = output_cls
        self._flexible_mode = flexible_mode
        self._allow_parallel = allow_parallel_tool_calls
        self._llm = llm

        # Cache parsing class to avoid recreating on each call
        self._parsing_cls = (
            FlexibleModel.create(output_cls) if flexible_mode else output_cls
        )

    def process(
        self,
        chat_response: ChatResponse,
        cur_objects: Sequence[BaseModel] | None = None,
    ) -> BaseModel | list[BaseModel]:
        """Process a streaming chat response into structured objects.

        Args:
            chat_response: The chat response to process
            cur_objects: Previously accumulated objects from earlier chunks

        Returns:
            Single object or list of objects based on allow_parallel_tool_calls
        """
        # Extract and parse arguments
        args = self._extract_args(chat_response)
        parsed = self._parse_objects(args, cur_objects)

        # Select and finalize objects
        selected = self._select_best(parsed, cur_objects)
        finalized = self._finalize(selected)

        return self._format_output(finalized)

    def _extract_args(self, chat_response: ChatResponse) -> list[Any]:
        """Extract output arguments from chat response."""
        tool_calls_data = chat_response.message.additional_kwargs.get("tool_calls")

        if not tool_calls_data:
            # return the args in the message content
            return [chat_response.message.content]

        if not self._llm:
            raise ValueError("LLM is required to extract tool calls from response")

        if not isinstance(tool_calls_data, list):
            # return an instance of the flexible class
            return [self._parsing_cls()]

        # get the tool calls from the response
        tool_calls: list[ToolCallArguments] = self._llm.get_tool_calls_from_response(
            chat_response, error_on_no_tool_call=False
        )

        return (
            [call.tool_kwargs for call in tool_calls]
            if tool_calls
            else [self._parsing_cls()]
        )

    def _parse_objects(
        self,
        args: list[Any],
        fallback: Sequence[BaseModel] | None,
    ) -> list[BaseModel]:
        """Parse arguments into Pydantic objects with error recovery."""
        parsed: list[BaseModel] = []

        for args_set in args:
            obj = self._parse_single(args_set)
            if obj is not None:
                parsed.append(obj)

        result = (
            parsed if parsed else list(fallback) if fallback else [self._parsing_cls()]
        )

        return result

    def _parse_single(self, args: Any) -> BaseModel | None:
        """Parse a single set of arguments into a Pydantic object."""
        result = None
        # Try direct validation
        try:
            result = self._parsing_cls.model_validate(args)
        except (ValidationError, ValueError):
            # Try JSON repair for string arguments
            if isinstance(args, str):
                try:
                    repaired = _repair_incomplete_json(args)
                    result = self._parsing_cls.model_validate_json(repaired)
                except (ValidationError, ValueError) as e:
                    _logger.debug(f"Validation error during streaming: {e}")

        return result

    def _select_best(
        self,
        new_objects: list[BaseModel],
        cur_objects: Sequence[BaseModel] | None,
    ) -> list[BaseModel]:
        """Select object set with more valid fields."""

        return (
            new_objects
            if cur_objects is None
            or num_valid_fields(new_objects) >= num_valid_fields(cur_objects)
            else list(cur_objects)
        )

    def _finalize(self, objects: list[BaseModel]) -> list[BaseModel]:
        """Convert flexible objects to target schema if applicable."""
        if not self._flexible_mode:
            result = objects
        else:
            finalized: list[BaseModel] = []
            for obj in objects:
                try:
                    converted = self._output_cls.model_validate(
                        obj.model_dump(exclude_unset=True)
                    )
                    finalized.append(converted)
                except ValidationError:
                    # Keep the flexible object as-is when strict validation fails;
                    # callers may choose to coerce it if needed.
                    finalized.append(obj)
            result = finalized

        return result

    def _format_output(
        self, objects: list[BaseModel]
    ) -> BaseModel | list[BaseModel]:
        """Format output based on parallel tool calls setting."""
        if self._allow_parallel:
            result = objects
        else:
            if len(objects) > 1:
                _logger.warning(
                    "Multiple outputs found, returning first one. "
                    "Set allow_parallel_tool_calls=True to return all outputs."
                )
            result = objects[0] if objects else objects
        return result


def num_valid_fields(
    obj: BaseModel | Sequence[BaseModel] | dict[str, BaseModel],
) -> int:
    """
    Recursively count the number of fields in a Pydantic object (including nested objects) that aren't None.

    Args:
        obj (Any): A Pydantic model instance or any other object.

    Returns:
        int: The number of fields that have non-None values.
    """
    if isinstance(obj, BaseModel):
        count = 0
        for value in obj.__dict__.values():
            if isinstance(value, (list, tuple)):
                count += sum(num_valid_fields(item) for item in value)
            elif isinstance(value, dict):
                count += sum(num_valid_fields(item) for item in value.values())
            elif isinstance(value, BaseModel):
                count += num_valid_fields(value)
            elif value is not None:
                count += 1
        return count
    elif isinstance(obj, (list, tuple)):
        return sum(num_valid_fields(item) for item in obj)
    elif isinstance(obj, dict):
        return sum(num_valid_fields(item) for item in obj.values())
    else:
        return 1 if obj is not None else 0
