import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generator, List, Optional, Dict, Any
from serapeum.core.tools import ToolOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@dataclass
class AgentChatResponse:
    """Agent chat response."""

    response: str = ""
    sources: List[ToolOutput] = field(default_factory=list)
    is_dummy_stream: bool = False
    metadata: Optional[Dict[str, Any]] = None

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