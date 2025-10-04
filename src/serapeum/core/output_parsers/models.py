from abc import ABC, abstractmethod
import json
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Type,
    Union,
    Generic,
    Optional
)
from serapeum.core.output_parsers.utils import extract_json_str
from serapeum.core.base.llms.models import Message, MessageRole, TextChunk
from serapeum.core.models import Model
from pydantic import (
    BaseModel,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
)
from pydantic_core import CoreSchema, core_schema


TokenGen = Generator[str, None, None]
TokenAsyncGen = AsyncGenerator[str, None]
RESPONSE_TEXT_TYPE = Union[BaseModel, str, TokenGen, TokenAsyncGen]


PYDANTIC_FORMAT_TMPL = """
Here's a JSON schema to follow:
{schema}

Output a valid JSON object but do not repeat the schema.
"""


class BaseOutputParser(ABC):

    @abstractmethod
    def parse(self, output: str) -> Any:
        ...

    def format(self, query: str) -> str:
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


class PydanticOutputParser(BaseOutputParser, Generic[Model]):

    def __init__(
        self,
        output_cls: Type[Model],
        excluded_schema_keys_from_format: Optional[List] = None,
        pydantic_format_tmpl: str = PYDANTIC_FORMAT_TMPL,
    ) -> None:
        """Init params."""
        self._output_cls = output_cls
        self._excluded_schema_keys_from_format = excluded_schema_keys_from_format or []
        self._pydantic_format_tmpl = pydantic_format_tmpl

    @property
    def output_cls(self) -> Type[Model]:
        return self._output_cls

    @property
    def format_string(self) -> str:
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
        return query + "\n\n" + self.get_format_string(escape_json=True)