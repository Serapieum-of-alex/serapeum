"""Output parser interfaces and concrete implementations.

This module defines the minimal interfaces for transforming raw LLM text into
typed Python objects (e.g., Pydantic models) and utilities for formatting
prompts with schema hints.
"""

import json
from abc import ABC, abstractmethod
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Type,
    Union,
)

from pydantic import BaseModel, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import CoreSchema, core_schema

from serapeum.core.base.llms.models import Message, MessageRole, TextChunk
from serapeum.core.models import Model
from serapeum.core.output_parsers.utils import extract_json_str

TokenGen = Generator[str, None, None]
TokenAsyncGen = AsyncGenerator[str, None]
RESPONSE_TEXT_TYPE = Union[BaseModel, str, TokenGen, TokenAsyncGen]


PYDANTIC_FORMAT_TMPL = """
Here's a JSON schema to follow:
{schema}

Output a valid JSON object but do not repeat the schema.
"""


class BaseOutputParser(ABC):
    """Abstract interface for parsing and formatting LLM outputs.

    Subclasses must implement :meth:`parse` to turn raw text into a target
    Python object. Optionally override :meth:`format` and
    :meth:`format_messages` to inject schema hints or guidance into prompts or
    messages before sending them to an LLM.
    """

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse a raw text output into a structured Python object."""
        pass

    def format(self, query: str) -> str:
        """Optionally augment the prompt string prior to completion."""
        return query

    def _format_message(self, message: Message) -> Message:
        """Apply :meth:`format` to the appropriate message text chunk."""
        text_blocks: list[tuple[int, TextChunk]] = [
            (idx, block)
            for idx, block in enumerate(message.chunks)
            if isinstance(block, TextChunk)
        ]

        # add text to the last text block, or add a new text block
        format_text = ""
        if text_blocks:
            format_idx = text_blocks[-1][0]
            format_text = text_blocks[-1][1].content

            if format_idx != -1:
                # this should always be a text block
                assert isinstance(message.chunks[format_idx], TextChunk)
                message.chunks[format_idx].content = self.format(format_text)  # type: ignore
        else:
            message.chunks.append(TextChunk(content=self.format(format_text)))

        return message

    def format_messages(self, messages: List[Message]) -> List[Message]:
        """Optionally augment a list of chat messages prior to chat calls."""
        if messages:
            if messages[0].role == MessageRole.SYSTEM:
                # get text from the last text chunks
                messages[0] = self._format_message(messages[0])
            else:
                messages[-1] = self._format_message(messages[-1])

        return messages

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Type[Any], handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Return a permissive core schema for arbitrary parser instances."""
        return core_schema.any_schema()

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> Dict[str, Any]:
        """Resolve JSON schema references for Pydantic integration."""
        json_schema = handler(core_schema)
        return handler.resolve_ref_schema(json_schema)


class PydanticOutputParser(BaseOutputParser, Generic[Model]):
    """Parse JSON text into a Pydantic model and provide schema formatting.

    This parser injects a compact JSON schema into the prompt (optional) and
    extracts the first JSON object from the model output, validating it against
    the provided ``output_cls``.
    """

    def __init__(
        self,
        output_cls: Type[Model],
        excluded_schema_keys_from_format: Optional[List] = None,
        pydantic_format_tmpl: str = PYDANTIC_FORMAT_TMPL,
    ) -> None:
        """Initialize the parser with a target Pydantic model and options."""
        self._output_cls = output_cls
        self._excluded_schema_keys_from_format = excluded_schema_keys_from_format or []
        self._pydantic_format_tmpl = pydantic_format_tmpl

    @property
    def output_cls(self) -> Type[Model]:
        """Return the target Pydantic model class."""
        return self._output_cls

    @property
    def format_string(self) -> str:
        """Return the schema hint string with JSON braces escaped."""
        return self.get_format_string(escape_json=True)

    def get_format_string(self, escape_json: bool = True) -> str:
        schema_dict = self._output_cls.model_json_schema()
        for key in self._excluded_schema_keys_from_format:
            del schema_dict[key]

        schema_str = json.dumps(schema_dict)
        output_str = self._pydantic_format_tmpl.format(schema=schema_str)
        if escape_json:
            return output_str.replace("{", "{{").replace("}", "}}")
        else:
            return output_str

    def parse(self, text: str) -> Any:
        json_str = extract_json_str(text)
        return self._output_cls.model_validate_json(json_str)

    def format(self, query: str) -> str:
        """Append an escaped JSON schema hint to the prompt string."""
        return query + "\n\n" + self.get_format_string(escape_json=True)
