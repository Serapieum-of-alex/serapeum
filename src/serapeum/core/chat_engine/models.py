import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generator, List, Optional, Dict, Any

from serapeum.core.base.llms.models import Message
from serapeum.core.base.response.models import Response, StreamingResponse
from serapeum.core.schemas.models import NodeWithScore
from serapeum.core.tools import ToolOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def is_function(message: Message) -> bool:
    """Utility for Message responses from OpenAI models."""
    return (
        "tool_calls" in message.additional_kwargs
        and len(message.additional_kwargs["tool_calls"]) > 0
    )


@dataclass
class AgentChatResponse:
    """Agent chat response."""

    response: str = ""
    sources: List[ToolOutput] = field(default_factory=list)
    source_nodes: List[NodeWithScore] = field(default_factory=list)
    is_dummy_stream: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def set_source_nodes(self) -> None:
        if self.sources and not self.source_nodes:
            for tool_output in self.sources:
                if isinstance(tool_output.raw_output, (Response, StreamingResponse)):
                    self.source_nodes.extend(tool_output.raw_output.source_nodes)

    def __post_init__(self) -> None:
        self.set_source_nodes()

    def __str__(self) -> str:
        return self.response

    @property
    def response_gen(self) -> Generator[str, None, None]:
        """Used for fake streaming, i.e. with tool outputs."""
        if not self.is_dummy_stream:
            raise ValueError(
                "response_gen is only available for streaming responses. "
                "Set is_dummy_stream=True if you still want a generator."
            )

        for token in self.response.split(" "):
            yield token + " "
            time.sleep(0.1)

    async def async_response_gen(self) -> AsyncGenerator[str, None]:
        """Used for fake streaming, i.e. with tool outputs."""
        if not self.is_dummy_stream:
            raise ValueError(
                "response_gen is only available for streaming responses. "
                "Set is_dummy_stream=True if you still want a generator."
            )

        for token in self.response.split(" "):
            yield token + " "
            await asyncio.sleep(0.1)