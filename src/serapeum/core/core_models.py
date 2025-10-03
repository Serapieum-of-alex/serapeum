from abc import ABC, abstractmethod

from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Type,
    Union,
)

from serapeum.core.base.llms.models import Message, MessageRole, TextChunk
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from pydantic_core import CoreSchema, core_schema


TokenGen = Generator[str, None, None]
TokenAsyncGen = AsyncGenerator[str, None]
RESPONSE_TEXT_TYPE = Union[BaseModel, str, TokenGen, TokenAsyncGen]


class BaseOutputParser(ABC):
    """Output parser class."""

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""

    def format(self, query: str) -> str:
        """Format a query with structured output formatting instructions."""
        return query

    def _format_message(self, message: Message) -> Message:
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
        """Format a list of messages with structured output formatting instructions."""
        # NOTE: apply output parser to either the first message if it's a system message
        #       or the last message
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
        return core_schema.any_schema()

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> Dict[str, Any]:
        json_schema = handler(core_schema)
        return handler.resolve_ref_schema(json_schema)

