import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union, cast

from pydantic import BaseModel

from serapeum.core.tools import ToolOutput

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


@dataclass
class AgentChatResponse:
    """Container for an agent's chat response, tool outputs, and metadata.

    This dataclass is returned by chat/LLM components to represent a single
    response turn. It can optionally hold outputs produced by tools (functions)
    the agent invoked, and it can simulate token-by-token streaming for simple
    UI rendering via the ``response_gen`` and ``async_response_gen`` helpers.

    Args:
        response (str):
            The final textual response produced by the agent/LLM.
        sources (List[ToolOutput]):
            A list of tool call results associated with this response. Each item may
            contain a structured ``raw_output`` (for example, a Pydantic ``BaseModel``)
            and/or textual content.
        is_dummy_stream (bool):
            When True, enables the fake streaming helpers that yield tokens from ``response``
            one by one.
        metadata (Optional[Dict[str, Any]]):
            Optional extra information attached to this response (e.g., model info, timing,
            custom flags).

    Examples:
    - Basic usage and string conversion
        ```python
        >>> from serapeum.core.chat.models import AgentChatResponse
        >>> r = AgentChatResponse(response="Hello, world!")
        >>> str(r)
        'Hello, world!'

        ```
    - Simulate token streaming for simple UIs
        ```python
        >>> r = AgentChatResponse(response="hello world", is_dummy_stream=True)
        >>> list(r.response_gen)
        ['hello ', 'world ']

        ```

    See Also:
        - serapeum.core.tools.models.ToolOutput: Structured tool output container.
        - AgentChatResponse._parse_tool_outputs: Helper to retrieve structured models from ``sources``.
    """

    response: str = ""
    sources: List[ToolOutput] = field(default_factory=list)
    is_dummy_stream: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        """Return the textual response as the string representation.

        Returns:
            str: The value of the ``response`` field.

        Examples:
        - Convert to string
            ```python
            >>> AgentChatResponse(response="OK").__str__()
            'OK'

            ```
        - Using Python's built-in str()
            ```python
            >>> str(AgentChatResponse(response="Hi"))
            'Hi'

            ```
        """
        return self.response

    @property
    def response_gen(self) -> Generator[str, None, None]:
        """Yield tokens from ``response`` to simulate streaming.

        When ``is_dummy_stream`` is True, this property returns a generator that
        yields each token from ``response`` (split on spaces), each suffixed with
        a single space. This is intended for simple UI streaming demos and tests.

        Returns:
            Generator[str, None, None]: A generator that yields tokens with trailing spaces.

        Raises:
            ValueError: If ``is_dummy_stream`` is False.

        Examples:
        - Typical usage
            ```python
            >>> r = AgentChatResponse(response="hello world", is_dummy_stream=True)
            >>> list(r.response_gen)
            ['hello ', 'world ']

            ```
        - Attempting to access when streaming is disabled
            ```python
            >>> try:
            ...     _ = list(AgentChatResponse(response="hi").response_gen)
            ... except Exception as e:
            ...     type(e).__name__
            'ValueError'

            ```
        """
        if not self.is_dummy_stream:
            raise ValueError(
                "response_gen is only available for streaming responses. \n"
                "Set is_dummy_stream=True if you still want a generator."
            )

        for token in self.response.split(" "):
            yield token + " "
            time.sleep(0.1)

    async def async_response_gen(self) -> AsyncGenerator[str, None]:
        """Asynchronously yield tokens from ``response`` to simulate streaming.

        This coroutine returns an async generator that yields each token from
        ``response`` (split on spaces), each suffixed with a single space. A
        short ``asyncio.sleep`` is used between tokens to emulate streaming.

        Returns:
            AsyncGenerator[str, None]: An async generator yielding tokens with trailing spaces.

        Raises:
            ValueError: If ``is_dummy_stream`` is False.

        Examples:
        - Consume with ``asyncio.run``
            ```python
            >>> import asyncio
            >>> async def collect():
            ...     r = AgentChatResponse(response="foo bar", is_dummy_stream=True)
            ...     return [t async for t in r.async_response_gen()]
            >>> asyncio.run(collect())
            ['foo ', 'bar ']

            ```
        - Attempting to access when streaming is disabled
            ```python
            >>> import asyncio
            >>> async def try_access():
            ...     try:
            ...         async for _ in AgentChatResponse(response="x").async_response_gen():
            ...             pass
            ...     except Exception as e:
            ...         return type(e).__name__
            >>> asyncio.run(try_access())
            'ValueError'

            ```
        """
        if not self.is_dummy_stream:
            raise ValueError(
                "response_gen is only available for streaming responses. "
                "Set is_dummy_stream=True if you still want a generator."
            )

        for token in self.response.split(" "):
            yield token + " "
            await asyncio.sleep(0.1)

    def parse_tool_outputs(
        self,
        allow_parallel_tool_calls: bool = False,
    ) -> Union[BaseModel, List[BaseModel]]:
        """Parse structured tool outputs from ``sources``.

        Extracts the Pydantic models placed in ``ToolOutput.raw_output`` and
        returns either the single model (default) or a list of models when
        parallel tool calls are enabled.

        Args:
            allow_parallel_tool_calls (bool):
                If True, return all parsed models as a list. If False, return only
                the first model (and log a warning if there are multiple). Defaults to False.

        Returns:
            Union[BaseModel, List[BaseModel]]: The parsed model(s).

        Raises:
            IndexError: If ``sources`` is empty and ``allow_parallel_tool_calls`` is False.

        Examples:
        - Single tool output (default behavior returns the first model)
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.chat.models import AgentChatResponse
            >>> from serapeum.core.tools import ToolOutput
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> resp = AgentChatResponse(
            ...     sources=[ToolOutput(tool_name="people", raw_output=Person(name="Alice", age=30))]
            ... )
            >>> result = resp.parse_tool_outputs()
            >>> (result.name, result.age)
            ('Alice', 30)

            ```
        - Multiple outputs with parallel calls enabled
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.chat.models import AgentChatResponse
            >>> from serapeum.core.tools import ToolOutput
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> p1 = Person(name="Bob", age=25)
            >>> p2 = Person(name="Carol", age=35)
            >>> resp = AgentChatResponse(
            ...     sources=[
            ...         ToolOutput(tool_name="people", raw_output=p1),
            ...         ToolOutput(tool_name="people", raw_output=p2),
            ...     ]
            ... )
            >>> [p.name for p in resp.parse_tool_outputs(allow_parallel_tool_calls=True)]
            ['Bob', 'Carol']

            ```
        - Multiple outputs with parallel calls disabled returns the first one (warning is logged)
            ```python
            >>> first = resp.parse_tool_outputs(allow_parallel_tool_calls=False)
            >>> first.name
            'Bob'

            ```
        - Empty sources result in ``IndexError`` when parallel calls are disabled
            ```python
            >>> try:
            ...     AgentChatResponse().parse_tool_outputs()
            ... except Exception as e:
            ...     type(e).__name__
            'IndexError'

            ```

        See Also:
            - serapeum.core.tools.models.ToolOutput: Container used in ``sources``.
        """
        outputs = [cast(BaseModel, s.raw_output) for s in self.sources]
        if allow_parallel_tool_calls:
            val = outputs
        else:
            if len(outputs) > 1:
                logger.warning(
                    "Multiple outputs found, returning first one. \n"
                    "If you want to return all outputs, set allow_parallel_tool_calls=True."
                )

            val = outputs[0]

        return val
