"""Test Tool Orchestrating LLM."""

from typing import Any, List, Optional, Union
from unittest.mock import MagicMock

import pytest
from tests.models import MOCK_ALBUM, MOCK_ALBUM_2, MockAlbum

from serapeum.core.chat.types import AgentChatResponse
from serapeum.core.llms import Message, Metadata, ToolOrchestratingLLM
from serapeum.core.tools import ToolOutput
from serapeum.core.tools.types import BaseTool


def _get_mock_album_response(
    allow_parallel_tool_calls: bool = False,
) -> AgentChatResponse:
    """Get mock album."""
    if allow_parallel_tool_calls:
        albums = [MOCK_ALBUM, MOCK_ALBUM_2]
    else:
        albums = [MOCK_ALBUM]

    tool_outputs = [
        ToolOutput(
            content=str(a),
            tool_name="tool_output",
            raw_input={},
            raw_output=a,
        )
        for a in albums
    ]

    # return tool outputs
    return AgentChatResponse(
        response="output",
        sources=tool_outputs,
    )


class MockLLM(MagicMock):
    """Mock LLM that returns predefined responses."""

    def predict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, Message]] = None,
        chat_history: Optional[List[Message]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
        """Predict and call the tool."""
        return _get_mock_album_response(
            allow_parallel_tool_calls=allow_parallel_tool_calls
        )

    async def apredict_and_call(
        self,
        tools: List["BaseTool"],
        user_msg: Optional[Union[str, Message]] = None,
        chat_history: Optional[List[Message]] = None,
        verbose: bool = False,
        allow_parallel_tool_calls: bool = False,
        **kwargs: Any,
    ) -> "AgentChatResponse":
        """Predict and call the tool."""
        return _get_mock_album_response(
            allow_parallel_tool_calls=allow_parallel_tool_calls
        )

    @property
    def metadata(self) -> Metadata:
        return Metadata(is_function_calling_model=True)


class TestToolOrchestratingLLM:
    """Tests for ToolOrchestratingLLM."""

    def test_tools_llm(self) -> None:
        """Test Function program."""
        prompt_template_str = """This is a test album with {topic}"""
        llm = MockLLM()
        # from serapeum.ollama import Ollama
        # llm = Ollama(model="llama3.1:latest", request_timeout=80)
        tools_llm = ToolOrchestratingLLM(
            schema=MockAlbum,
            prompt=prompt_template_str,
            llm=llm,
        )
        obj_output = tools_llm(topic="songs")
        assert isinstance(obj_output, MockAlbum)
        assert obj_output.title == "hello"
        assert obj_output.artist == "world"
        assert obj_output.songs[0].title == "song1"
        assert obj_output.songs[1].title == "song2"

    def test_tools_llm_multiple(self) -> None:
        """Test Function program multiple."""
        llm = MockLLM()
        # llm = Ollama(model="llama3.1:latest", request_timeout=80)
        prompt_template_str = """This is a test album with {topic}"""
        tools_llm = ToolOrchestratingLLM(
            schema=MockAlbum,
            prompt=prompt_template_str,
            llm=llm,
            allow_parallel_tool_calls=True,
        )
        obj_outputs = tools_llm(topic="songs")
        assert isinstance(obj_outputs, list)
        assert len(obj_outputs) == 2
        assert isinstance(obj_outputs[0], MockAlbum)
        assert isinstance(obj_outputs[1], MockAlbum)
        # test second output
        assert obj_outputs[1].title == "hello2"
        assert obj_outputs[1].artist == "world2"
        assert obj_outputs[1].songs[0].title == "song3"
        assert obj_outputs[1].songs[1].title == "song4"

    @pytest.mark.asyncio()
    async def test_async(self) -> None:
        """Test async function program."""
        llm = MockLLM()
        # llm = Ollama(model="llama3.1:latest", request_timeout=80)
        # same as above but async
        prompt_template_str = """This is a test album with {topic}"""
        tools_llm = ToolOrchestratingLLM(
            schema=MockAlbum,
            prompt=prompt_template_str,
            llm=llm,
        )
        obj_output = await tools_llm.acall(topic="songs")
        assert isinstance(obj_output, MockAlbum)
        assert obj_output.title == "hello"
        assert obj_output.artist == "world"
        assert obj_output.songs[0].title == "song1"
        assert obj_output.songs[1].title == "song2"
