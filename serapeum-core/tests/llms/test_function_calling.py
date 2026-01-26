"""Tests for FunctionCallingLLM."""

from typing import Any, AsyncGenerator, Coroutine, Dict, List, Optional, Sequence, Union

import pytest
from pydantic import BaseModel, Field

from serapeum.core.base.llms.models import (
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    Message,
    Metadata,
)
from serapeum.core.llm.function_calling import FunctionCallingLLM
from serapeum.core.structured_tools.tools_llm import CallableTool
from serapeum.core.tools.models import BaseTool, ToolCallArguments


class MockFunctionCallingLLM(FunctionCallingLLM):
    """Mock FunctionCallingLLM for testing."""

    def __init__(self, tool_selection: List[ToolCallArguments]):
        super().__init__()
        self._tool_selection = tool_selection

    async def achat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> Coroutine[Any, Any, ChatResponse]:
        return ChatResponse(message=Message(role="user", content=""))

    def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Coroutine[Any, Any, CompletionResponse]:
        pass

    def astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> Coroutine[Any, Any, AsyncGenerator[ChatResponse, None]]:
        pass

    def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Coroutine[Any, Any, AsyncGenerator[CompletionResponse, None]]:
        pass

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=Message(role="user", content=""))

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        pass

    def stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        pass

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> ChatResponseGen:
        pass

    @property
    def metadata(self) -> Metadata:
        return Metadata(is_function_calling_model=True)

    def _prepare_chat_with_tools(
        self,
        tools: Sequence["BaseTool"],
        user_msg: Optional[Union[str, Message]] = None,
        chat_history: Optional[List[Message]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return {"messages": []}

    def get_tool_calls_from_response(
        self,
        response: ChatResponse,
        error_on_no_tool_call: bool = True,
        **kwargs: Any,
    ) -> List[ToolCallArguments]:
        return self._tool_selection


class Person(BaseModel):
    """Person model."""

    name: str = Field(description="Person name")


@pytest.fixture()
def person_tool() -> CallableTool:
    """Person tool."""
    return CallableTool.from_model(Person)


@pytest.fixture()
def person_tool_selection(person_tool: CallableTool) -> ToolCallArguments:
    """Person tool selection."""
    return ToolCallArguments(
        tool_id="",
        tool_name=person_tool.metadata.name,
        tool_kwargs={},
    )


def test_predict_and_call(
    person_tool: CallableTool, person_tool_selection: ToolCallArguments
) -> None:
    """Test predict_and_call will return ToolOutput with error rather than raising one."""
    llm = MockFunctionCallingLLM([person_tool_selection])
    response = llm.predict_and_call(tools=[person_tool])
    assert all(tool_output.is_error for tool_output in response.sources)


def test_predict_and_call_throws_if_error_on_tool(
    person_tool: CallableTool, person_tool_selection: ToolCallArguments
) -> None:
    """Test predict_and_call will raise an error."""
    llm = MockFunctionCallingLLM([person_tool_selection])
    with pytest.raises(ValueError):
        llm.predict_and_call(tools=[person_tool], error_on_tool_error=True)


@pytest.mark.asyncio()
async def test_apredict_and_call(
    person_tool: CallableTool, person_tool_selection: ToolCallArguments
) -> None:
    """Test apredict_and_call will return ToolOutput with error rather than raising one."""
    llm = MockFunctionCallingLLM([person_tool_selection])
    response = await llm.apredict_and_call(tools=[person_tool])
    assert all(tool_output.is_error for tool_output in response.sources)


@pytest.mark.asyncio()
async def test_apredict_and_call_throws_if_error_on_tool(
    person_tool: CallableTool, person_tool_selection: ToolCallArguments
) -> None:
    """Test apredict_and_call will raise an error."""
    llm = MockFunctionCallingLLM([person_tool_selection])
    with pytest.raises(ValueError):
        await llm.apredict_and_call(tools=[person_tool], error_on_tool_error=True)
