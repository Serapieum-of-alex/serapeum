import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Generator, List, Optional, Dict, Any, cast, Union
from serapeum.core.tools import ToolOutput
from pydantic import (
    BaseModel,
)


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

    def _parse_tool_outputs(
        self,
        allow_parallel_tool_calls: bool = False,
    ) -> Union[BaseModel, List[BaseModel]]:
        """Parse tool outputs from an agent chat response.

        Extracts structured objects (Pydantic models) from an agent/LLM tool-call
        response. When parallel tool calls are disabled, only the first output is
        returned. When enabled, all outputs are returned as a list.

        Args:
            agent_response: An object that contains a ``sources`` attribute which is
                an iterable of items, each having a ``raw_output`` attribute that is
                a Pydantic ``BaseModel`` instance.
            allow_parallel_tool_calls (bool, optional): If True, return all parsed
                models as a list. If False, return only the first model (and log a
                warning if there are multiple). Defaults to False.

        Returns:
            Union[BaseModel, List[BaseModel]]: A single model (when
            ``allow_parallel_tool_calls`` is False) or a list of models (when it is
            True).

        Raises:
            IndexError: If ``agent_response.sources`` is empty and parallel tool
                calls are disabled (attempts to access the first element).
            AttributeError: If the expected attributes (``sources`` on
                ``agent_response`` or ``raw_output`` on a source) are missing.
            TypeError: If a ``raw_output`` item is not a Pydantic ``BaseModel``.

        Warns:
            Logs a warning when multiple outputs are present but
            ``allow_parallel_tool_calls`` is False.

        Examples:
        - Parse a single tool output (parallel calls disabled).
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.structured_tools.tools_llm import _parse_tool_outputs
            >>>
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>>
            >>> class Source:
            ...     def __init__(self, raw_output):
            ...         self.raw_output = raw_output
            >>> class Response:
            ...     def __init__(self, sources):
            ...         self.sources = sources
            >>>
            >>> resp = Response([Source(Person(name='Alice', age=30))])
            >>> result = _parse_tool_outputs(resp, allow_parallel_tool_calls=False)
            >>> (result.name, result.age)
            ('Alice', 30)

            ```
        - Parse multiple outputs with parallel calls enabled.
            ```python
            >>> p1 = Person(name='Bob', age=25)
            >>> p2 = Person(name='Charlie', age=35)
            >>> resp = Response([Source(p1), Source(p2)])
            >>> results = _parse_tool_outputs(resp, allow_parallel_tool_calls=True)
            >>> [r.name for r in results]
            ['Bob', 'Charlie']

            ```
        - Multiple outputs with parallel calls disabled returns the first one and logs a warning.
            ```python
            >>> result = _parse_tool_outputs(resp, allow_parallel_tool_calls=False)
            >>> result.name
            'Bob'

            ```

        See Also:
            - ToolOrchestratingLLM.__call__: Main execution method using this parser
            - ToolOrchestratingLLM.stream_call: Streaming variant that yields partial models
        """
        outputs = [cast(BaseModel, s.raw_output) for s in self.sources]
        if allow_parallel_tool_calls:
            val = outputs
        else:
            if len(outputs) > 1:
                logger.warning(
                    "Multiple outputs found, returning first one. "
                    "If you want to return all outputs, set allow_parallel_tool_calls=True."
                )

            val = outputs[0]

        return val