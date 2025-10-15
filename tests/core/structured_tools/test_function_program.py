"""Test LLM program."""

from unittest.mock import MagicMock, patch
import pytest

from serapeum.core.base.llms.models import (
    Message,
    Metadata,
)
from pydantic import BaseModel
from typing import List, Optional, Union, Any
from serapeum.core.tools.models import BaseTool
from serapeum.core.chat.models import AgentChatResponse
from serapeum.core.tools import ToolOutput
from serapeum.core.structured_tools import ToolOrchestratingLLM
from serapeum.core.structured_tools.function_program import _parse_tool_outputs


class MockSong(BaseModel):
    """Mock Song class."""

    title: str


class MockAlbum(BaseModel):
    title: str
    artist: str
    songs: List[MockSong]


MOCK_ALBUM = MockAlbum(
    title="hello",
    artist="world",
    songs=[MockSong(title="song1"), MockSong(title="song2")],
)

MOCK_ALBUM_2 = MockAlbum(
    title="hello2",
    artist="world2",
    songs=[MockSong(title="song3"), MockSong(title="song4")],
)


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


class TestParseToolOutputs:
    """Test class for _parse_tool_outputs function."""

    def test_parse_tool_outputs_single_output(self):
        """Test parsing tool outputs with a single output.

        Input:
            AgentChatResponse with one tool output
        Expected:
            Returns the single BaseModel output
        Check:
            Result is not a list and is the first output
        """
        tool_output = ToolOutput(
            content="test",
            tool_name="test_tool",
            raw_input={},
            raw_output=MOCK_ALBUM,
        )
        agent_response = AgentChatResponse(
            response="test response",
            sources=[tool_output],
        )

        result = _parse_tool_outputs(agent_response, allow_parallel_tool_calls=False)

        assert result == MOCK_ALBUM
        assert not isinstance(result, list)

    def test_parse_tool_outputs_multiple_outputs_parallel_disabled(self):
        """Test parsing multiple outputs when parallel calls are disabled.

        Input: AgentChatResponse with multiple tool outputs, parallel=False
        Expected: Returns only the first output and logs a warning
        Check: Result is first output, not a list
        """
        tool_outputs = [
            ToolOutput(
                content="test1",
                tool_name="test_tool",
                raw_input={},
                raw_output=MOCK_ALBUM,
            ),
            ToolOutput(
                content="test2",
                tool_name="test_tool",
                raw_input={},
                raw_output=MOCK_ALBUM_2,
            ),
        ]
        agent_response = AgentChatResponse(
            response="test response",
            sources=tool_outputs,
        )

        with patch("serapeum.core.structured_tools.function_program._logger") as mock_logger:
            result = _parse_tool_outputs(agent_response, allow_parallel_tool_calls=False)

        assert result == MOCK_ALBUM
        assert not isinstance(result, list)
        mock_logger.warning.assert_called_once()

    def test_parse_tool_outputs_multiple_outputs_parallel_enabled(self):
        """Test parsing multiple outputs when parallel calls are enabled.

        Input: AgentChatResponse with multiple tool outputs, parallel=True
        Expected: Returns list of all outputs
        Check: Result is a list with all outputs
        """
        tool_outputs = [
            ToolOutput(
                content="test1",
                tool_name="test_tool",
                raw_input={},
                raw_output=MOCK_ALBUM,
            ),
            ToolOutput(
                content="test2",
                tool_name="test_tool",
                raw_input={},
                raw_output=MOCK_ALBUM_2,
            ),
        ]
        agent_response = AgentChatResponse(
            response="test response",
            sources=tool_outputs,
        )

        result = _parse_tool_outputs(agent_response, allow_parallel_tool_calls=True)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == MOCK_ALBUM
        assert result[1] == MOCK_ALBUM_2

    def test_parse_tool_outputs_empty_sources(self):
        """Test parsing tool outputs with empty sources list.

        Input: AgentChatResponse with no sources
        Expected: Raises IndexError when accessing first output
        Check: IndexError is raised
        """
        agent_response = AgentChatResponse(
            response="test response",
            sources=[],
        )

        with pytest.raises(IndexError):
            _parse_tool_outputs(agent_response, allow_parallel_tool_calls=False)

class TestToolOrchestratingLLM:
    def test_function_program(self) -> None:
        """Test Function program."""
        prompt_template_str = """This is a test album with {topic}"""
        tools_llm = ToolOrchestratingLLM.from_defaults(
            output_cls=MockAlbum,
            prompt=prompt_template_str,
            llm=MockLLM(),
        )
        obj_output = tools_llm(topic="songs")
        assert isinstance(obj_output, MockAlbum)
        assert obj_output.title == "hello"
        assert obj_output.artist == "world"
        assert obj_output.songs[0].title == "song1"
        assert obj_output.songs[1].title == "song2"

    def test_function_program_multiple(self) -> None:
        """Test Function program multiple."""
        prompt_template_str = """This is a test album with {topic}"""
        tools_llm = ToolOrchestratingLLM.from_defaults(
            output_cls=MockAlbum,
            prompt=prompt_template_str,
            llm=MockLLM(),
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
        # same as above but async
        prompt_template_str = """This is a test album with {topic}"""
        tools_llm = ToolOrchestratingLLM.from_defaults(
            output_cls=MockAlbum,
            prompt=prompt_template_str,
            llm=MockLLM(),
        )
        obj_output = await tools_llm.acall(topic="songs")
        assert isinstance(obj_output, MockAlbum)
        assert obj_output.title == "hello"
        assert obj_output.artist == "world"
        assert obj_output.songs[0].title == "song1"
        assert obj_output.songs[1].title == "song2"
