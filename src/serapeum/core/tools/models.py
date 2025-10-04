import asyncio
import json
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
from serapeum.core.base.llms.models import ChunkType, TextChunk
from pydantic import BaseModel


class DefaultToolFnSchema(BaseModel):
    """Default tool function Schema."""

    input: str


@dataclass
class ToolMetadata:
    description: str
    name: Optional[str] = None
    fn_schema: Optional[Type[BaseModel]] = DefaultToolFnSchema
    return_direct: bool = False

    def get_parameters_dict(self) -> dict:
        if self.fn_schema is None:
            parameters = {
                "type": "object",
                "properties": {
                    "input": {"title": "input query string", "type": "string"},
                },
                "required": ["input"],
            }
        else:
            parameters = self.fn_schema.model_json_schema()
            parameters = {
                k: v
                for k, v in parameters.items()
                if k in ["type", "properties", "required", "definitions", "$defs"]
            }
        return parameters

    @property
    def fn_schema_str(self) -> str:
        """Get fn schema as string."""
        if self.fn_schema is None:
            raise ValueError("fn_schema is None.")
        parameters = self.get_parameters_dict()
        return json.dumps(parameters, ensure_ascii=False)

    def get_name(self) -> str:
        """Get name."""
        if self.name is None:
            raise ValueError("name is None.")
        return self.name

    def to_openai_tool(self, skip_length_check: bool = False) -> Dict[str, Any]:
        if not skip_length_check and len(self.description) > 1024:
            raise ValueError(
                "Tool description exceeds maximum length of 1024 characters. "
                "Please shorten your description or move it to the prompt."
            )
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters_dict(),
            },
        }


class ToolOutput(BaseModel):

    chunks: List[ChunkType]
    tool_name: str
    raw_input: Dict[str, Any]
    raw_output: Any
    is_error: bool = False

    def __init__(
        self,
        tool_name: str,
        content: Optional[str] = None,
        chunks: Optional[List[ChunkType]] = None,
        raw_input: Optional[Dict[str, Any]] = None,
        raw_output: Optional[Any] = None,
        is_error: bool = False,
    ):
        if content and chunks:
            raise ValueError("Cannot provide both content and chunks.")
        if content:
            chunks = [TextChunk(content=content)]
        elif chunks:
            pass
        else:
            chunks = []

        super().__init__(
            tool_name=tool_name,
            chunks=chunks,
            raw_input=raw_input,
            raw_output=raw_output,
            is_error=is_error,
        )

    @property
    def content(self) -> str:
        """Get the content of the tool output."""
        return "\n".join(
            [chunk.content for chunk in self.chunks if isinstance(chunk, TextChunk)]
        )

    @content.setter
    def content(self, content: str) -> None:
        """Set the content of the tool output."""
        self.chunks = [TextChunk(content=content)]

    def __str__(self) -> str:
        """String."""
        return self.content


class BaseTool:

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        pass

    @abstractmethod
    def __call__(self, input_values: Any) -> ToolOutput:
        pass


class AsyncBaseTool(BaseTool):
    """
    Base-level tool class that is backwards compatible with the old tool spec but also
    supports async.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, input_values: Any) -> ToolOutput:
        """
        This is the method that should be implemented by the tool developer.
        """

    @abstractmethod
    async def acall(self, input_values: Any) -> ToolOutput:
        """
        This is the async version of the call method.
        Should also be implemented by the tool developer as an
        async-compatible implementation.
        """


class BaseToolAsyncAdapter(AsyncBaseTool):
    """
    Adapter class that allows a synchronous tool to be used as an async tool.
    """

    def __init__(self, tool: BaseTool):
        self.base_tool = tool

    @property
    def metadata(self) -> ToolMetadata:
        return self.base_tool.metadata

    def call(self, input_values: Any) -> ToolOutput:
        return self.base_tool(input_values)

    async def acall(self, input_values: Any) -> ToolOutput:
        return await asyncio.to_thread(self.call, input_values)


def adapt_to_async_tool(tool: BaseTool) -> AsyncBaseTool:
    """
    Converts a synchronous tool to an async tool.
    """
    if isinstance(tool, AsyncBaseTool):
        return tool
    else:
        return BaseToolAsyncAdapter(tool)
