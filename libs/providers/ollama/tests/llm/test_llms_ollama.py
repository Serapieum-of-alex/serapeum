"""Tests for Ollama LLM integration and related functionality."""
from __future__ import annotations
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from ollama import ChatResponse
from ollama import Message as OllamaMessage
from pydantic import BaseModel

from serapeum.core.llms import BaseLLM, Message
from serapeum.core.tools import CallableTool
from serapeum.ollama import Ollama
from ..models import client

response_dict = {
    "model": "llama3.1:latest",
    "created_at": "2025-09-20T21:52:10.0615697Z",
    "done": True,
    "done_reason": "stop",
    "total_duration": 794134100,
    "load_duration": 214719400,
    "prompt_eval_count": 195,
    "prompt_eval_duration": 4125100,
    "eval_count": 24,
    "eval_duration": 573800600,
    "message": dict(
        role="assistant",
        content="hello",
        thinking=None,
        images=None,
        tool_name=None,
        tool_calls=None,
    ),
}
normal_response = ChatResponse(**response_dict)
response_with_tool_dict = deepcopy(response_dict)
response_with_tool_dict["message"]["tool_calls"] = [
    OllamaMessage.ToolCall(
        function=OllamaMessage.ToolCall.Function(
            name="generate_song", arguments={"artist": "The Beatles", "name": "Hello!"}
        )
    )
]

response_for_tool_call = ChatResponse(**response_with_tool_dict)


class Song(BaseModel):
    """A song with name and artist."""

    name: str
    artist: str


def generate_song(name: str, artist: str) -> Song:
    """Generates a song with provided name and artist."""
    return Song(name=name, artist=artist)


tool = CallableTool.from_function(func=generate_song)


def test_embedding_class() -> None:
    """Test Ollama class inheritance structure.

    Checks: Ollama inherits from BaseLLM.
    """
    names_of_base_classes = [b.__name__ for b in Ollama.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes


@pytest.mark.mock
@patch.object(Ollama, "client")
def test_ollama_chat(mock_ollama_client, model_name) -> None:
    """Test chat method with mock client.

    Inputs: mock_ollama_client with chat method returning normal_response.
    Expected: llm.chat returns a non-empty response string.
    """
    mock_ollama_client.chat = MagicMock(return_value=normal_response)
    llm = Ollama(model=model_name, request_timeout=80)
    response = llm.chat([Message(role="user", content="Hello!")])
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.mock
@patch.object(Ollama, "client")
def test_ollama_complete(mock_ollama_client, model_name) -> None:
    """Test complete method with mock client.

    Inputs: mock_ollama_client with chat method returning normal_response.
    Expected: llm.complete returns a non-empty response string.
    """
    mock_ollama_client.chat = MagicMock(return_value=normal_response)
    llm = Ollama(model=model_name, request_timeout=80)
    response = llm.complete("Hello!")
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_ollama_stream_chat(model_name) -> None:
    """Test stream_chat method with real client.

    Inputs: model_name and user message.
    Expected: Each streamed response is non-empty and has a delta.
    """
    llm = Ollama(model=model_name, request_timeout=100)
    response = llm.stream_chat([Message(role="user", content="Hello!")])
    for r in response:
        assert r is not None
        assert r.delta is not None
        assert str(r).strip() != ""


@pytest.mark.e2e
@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_ollama_stream_complete(model_name) -> None:
    """Test stream_complete method with real client.

    Inputs: model_name and prompt string.
    Expected: Each streamed response is non-empty and has a delta.
    """
    llm = Ollama(model=model_name)
    response = llm.stream_complete("Hello!")
    for r in response:
        assert r is not None
        assert r.delta is not None
        assert str(r).strip() != ""


@pytest.mark.e2e
@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio()
async def test_ollama_async_chat(model_name) -> None:
    """Test async chat method with real client.

    Inputs: model_name and user message.
    Expected: Response is non-empty string.
    """
    llm = Ollama(model=model_name)
    response = await llm.achat([Message(role="user", content="Hello!")])
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.e2e
@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio()
async def test_ollama_async_complete(model_name) -> None:
    """Test async complete method with real client.

    Inputs: model_name and prompt string.
    Expected: Response is non-empty string.
    """
    llm = Ollama(model=model_name)
    response = await llm.acomplete("Hello!")
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.e2e
@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio()
async def test_ollama_async_stream_chat(model_name) -> None:
    """Test async stream_chat method with real client.

    Inputs: model_name and user message.
    Expected: Each streamed response is non-empty and has a delta.
    """
    llm = Ollama(model=model_name, request_timeout=80)
    response = await llm.astream_chat([Message(role="user", content="Hello!")])
    async for r in response:
        assert r is not None
        assert r.delta is not None
        assert str(r).strip() != ""


@pytest.mark.e2e
@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio()
async def test_ollama_async_stream_complete(model_name) -> None:
    """Test async stream_complete method with real client.

    Inputs: model_name and prompt string.
    Expected: Each streamed response is non-empty and has a delta.
    """
    llm = Ollama(model=model_name)
    response = await llm.astream_complete("Hello!")
    async for r in response:
        assert r is not None
        assert r.delta is not None
        assert str(r).strip() != ""


@pytest.mark.mock
@patch.object(Ollama, "client")
def test_chat_with_tools(mock_ollama_client, model_name) -> None:
    """Test chat_with_tools method with mock client.

    Inputs: mock_ollama_client with chat returning response_for_tool_call.
    Expected: Tool call is detected and tool result is correct type.
    """
    mock_ollama_client.chat = MagicMock(return_value=response_for_tool_call)
    llm = Ollama(model=model_name, request_timeout=80)
    response = llm.chat_with_tools([tool], user_msg="Hello!")
    tool_calls = llm.get_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == tool.metadata.name

    tool_result = tool(**tool_calls[0].tool_kwargs)
    assert tool_result.raw_output is not None
    assert isinstance(tool_result.raw_output, Song)


@pytest.mark.mock
@pytest.mark.asyncio()
@patch.object(Ollama, "async_client", new_callable=PropertyMock)
async def test_async_chat_with_tools(mock_ollama_async_prop, model_name) -> None:
    """Test async chat_with_tools method with mock client.

    Inputs: mock_ollama_async_client with chat returning response_for_tool_call.
    Expected: Tool call is detected and tool result is correct type.
    """
    mock_ollama_async_client = MagicMock()
    mock_ollama_async_client.chat = AsyncMock(return_value=response_for_tool_call)
    mock_ollama_async_prop.return_value = mock_ollama_async_client
    llm = Ollama(model=model_name)
    response = await llm.achat_with_tools([tool], user_msg="Hello!")
    tool_calls = llm.get_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == tool.metadata.name

    tool_result = tool(**tool_calls[0].tool_kwargs)
    assert tool_result.raw_output is not None
    assert isinstance(tool_result.raw_output, Song)
