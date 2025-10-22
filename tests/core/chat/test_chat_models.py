import pytest
from unittest.mock import patch

from serapeum.core.chat.models import AgentChatResponse
from serapeum.core.tools import ToolOutput

from tests.core.models import MOCK_ALBUM, MOCK_ALBUM_2


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

        result = agent_response._parse_tool_outputs(allow_parallel_tool_calls=False)

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

        with patch("serapeum.core.chat.models.logger") as mock_logger:
            result = agent_response._parse_tool_outputs(allow_parallel_tool_calls=False)

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

        result = agent_response._parse_tool_outputs(allow_parallel_tool_calls=True)

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
            agent_response._parse_tool_outputs(allow_parallel_tool_calls=False)