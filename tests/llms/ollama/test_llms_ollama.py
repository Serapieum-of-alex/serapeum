import pytest
import os

from pydantic import BaseModel
from serapeum.core.base.llms.base import BaseLLM
from serapeum.core.llm import Message
from serapeum.core.tools import CallableTool
from serapeum.llms.ollama import Ollama
from unittest.mock import patch, AsyncMock, MagicMock, PropertyMock
from ollama import Client
from copy import deepcopy
from ollama import Message as OllamaMessage
from ollama import ChatResponse
response_dict = {
    'model': 'llama3.1:latest',
    'created_at': '2025-09-20T21:52:10.0615697Z',
    'done': True,
    'done_reason': 'stop',
    'total_duration': 794134100,
    'load_duration': 214719400,
    'prompt_eval_count': 195,
    'prompt_eval_duration': 4125100,
    'eval_count': 24,
    'eval_duration': 573800600,
    'message': dict(
        role='assistant',
        content='hello',
        thinking=None,
        images=None,
        tool_name=None,
        tool_calls=None
    )
}
normal_response = ChatResponse(**response_dict)
response_with_tool_dict = deepcopy(response_dict)
response_with_tool_dict["message"]["tool_calls"] = [
    OllamaMessage.ToolCall(
        function=OllamaMessage.ToolCall.Function(
            name='generate_song',
            arguments={
                'artist': 'The Beatles',
                'name': 'Hello!'
            }
        )
    )
]

response_for_tool_call = ChatResponse(**response_with_tool_dict)


test_model = os.environ.get("OLLAMA_TEST_MODEL", "llama3.1:latest")
try:
    client = Client()
    models = client.list()

    model_found = False
    for model in models["models"]:
        if model["model"] == test_model:
            model_found = True
            break

    if not model_found:
        client = None
except Exception:
    client = None


class Song(BaseModel):
    """A song with name and artist."""

    name: str
    artist: str


def generate_song(name: str, artist: str) -> Song:
    """Generates a song with provided name and artist."""
    return Song(name=name, artist=artist)


tool = CallableTool.from_function(func=generate_song)


def test_embedding_class() -> None:
    names_of_base_classes = [b.__name__ for b in Ollama.__mro__]
    assert BaseLLM.__name__ in names_of_base_classes

@pytest.mark.mock
@patch.object(Ollama, "client")
def test_ollama_chat(mock_ollama_client) -> None:
    mock_ollama_client.chat = MagicMock(return_value=normal_response)
    llm = Ollama(model=test_model, request_timeout=80)
    response = llm.chat([Message(role="user", content="Hello!")])
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.mock
@patch.object(Ollama, "client")
def test_ollama_complete(mock_ollama_client) -> None:
    mock_ollama_client.chat = MagicMock(return_value=normal_response)
    llm = Ollama(model=test_model, request_timeout=80)
    response = llm.complete("Hello!")
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_ollama_stream_chat() -> None:
    llm = Ollama(model=test_model, request_timeout=100)
    response = llm.stream_chat([Message(role="user", content="Hello!")])
    for r in response:
        assert r is not None
        assert r.delta is not None
        assert str(r).strip() != ""


@pytest.mark.e2e
@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
def test_ollama_stream_complete() -> None:
    llm = Ollama(model=test_model)
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
async def test_ollama_async_chat() -> None:
    llm = Ollama(model=test_model)
    response = await llm.achat([Message(role="user", content="Hello!")])
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.e2e
@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio()
async def test_ollama_async_complete() -> None:
    llm = Ollama(model=test_model)
    response = await llm.acomplete("Hello!")
    assert response is not None
    assert str(response).strip() != ""


@pytest.mark.e2e
@pytest.mark.skipif(
    client is None, reason="Ollama client is not available or test model is missing"
)
@pytest.mark.asyncio()
async def test_ollama_async_stream_chat() -> None:
    llm = Ollama(model=test_model, request_timeout=80)
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
async def test_ollama_async_stream_complete() -> None:
    llm = Ollama(model=test_model)
    response = await llm.astream_complete("Hello!")
    async for r in response:
        assert r is not None
        assert r.delta is not None
        assert str(r).strip() != ""


@pytest.mark.mock
@patch.object(Ollama, "client")
def test_chat_with_tools(mock_ollama_client) -> None:
    mock_ollama_client.chat = MagicMock(return_value=response_for_tool_call)
    llm = Ollama(model=test_model, request_timeout=80)
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
async def test_async_chat_with_tools(mock_ollama_async_prop) -> None:
    mock_ollama_async_client = MagicMock()
    mock_ollama_async_client.chat = AsyncMock(return_value=response_for_tool_call)
    mock_ollama_async_prop.return_value = mock_ollama_async_client
    llm = Ollama(model=test_model)
    response = await llm.achat_with_tools([tool], user_msg="Hello!")
    tool_calls = llm.get_tool_calls_from_response(response)
    assert len(tool_calls) == 1
    assert tool_calls[0].tool_name == tool.metadata.name

    tool_result = tool(**tool_calls[0].tool_kwargs)
    assert tool_result.raw_output is not None
    assert isinstance(tool_result.raw_output, Song)
