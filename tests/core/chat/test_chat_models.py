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

        result = agent_response.parse_tool_outputs(allow_parallel_tool_calls=False)

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
            result = agent_response.parse_tool_outputs(allow_parallel_tool_calls=False)

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

        result = agent_response.parse_tool_outputs(allow_parallel_tool_calls=True)

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
            agent_response.parse_tool_outputs(allow_parallel_tool_calls=False)


class TestAgentChatResponseStr:
    """Test class for AgentChatResponse.__str__ method.

    Scenarios covered:
    - Returns the underlying response string.
    - Handles empty responses.
    - Preserves unicode characters.
    """

    def test_str_returns_response(self):
        """Input: AgentChatResponse(response="hello").

        Expected: "hello" is returned by str(instance).
        Check: str(r) equals the provided response.
        """
        r = AgentChatResponse(response="hello")
        assert str(r) == "hello"

    def test_str_empty_response(self):
        """Input: AgentChatResponse() with default empty response.

        Expected: Empty string is returned by str(instance).
        Check: str(r) == "".
        """
        r = AgentChatResponse()
        assert str(r) == ""

    def test_str_unicode_characters(self):
        """Input: AgentChatResponse(response with unicode).

        Expected: Unicode preserved in __str__ output.
        Check: str(r) equals original unicode string.
        """
        text = "こんにちは 世界"
        r = AgentChatResponse(response=text)
        assert str(r) == text


class TestAgentChatResponseResponseGen:
    """Test class for AgentChatResponse.response_gen property.

    Scenarios covered:
    - Typical streaming of space-separated tokens with trailing spaces.
    - Error when streaming disabled (is_dummy_stream=False).
    - Empty response yields a single space token (due to split(" ")).
    - Multiple spaces preserve empty tokens between words.
    """

    def test_response_gen_typical_tokens(self, monkeypatch):
        """Input: response="hello world", is_dummy_stream=True; patch time.sleep to 0.

        Expected: Tokens ['hello ', 'world '].
        Check: list(response_gen) equals expected token sequence.
        """
        # Speed up the generator by removing real sleep
        monkeypatch.setattr("serapeum.core.chat.models.time.sleep", lambda _x: None)
        r = AgentChatResponse(response="hello world", is_dummy_stream=True)
        assert list(r.response_gen) == ["hello ", "world "]

    def test_response_gen_disabled_raises(self):
        """Input: response="hi", is_dummy_stream=False.

        Expected: Iterating response_gen raises ValueError (generator body executes on iteration).
        Check: pytest.raises(ValueError) while converting generator to list.
        """
        with pytest.raises(ValueError):
            _ = list(AgentChatResponse(response="hi").response_gen)

    def test_response_gen_empty_response_yields_single_space(self, monkeypatch):
        """Input: response="", is_dummy_stream=True; patch time.sleep to 0.

        Expected: Because "".split(" ") == [''], generator yields [' '].
        Check: Exactly one token containing a single space.
        """
        monkeypatch.setattr("serapeum.core.chat.models.time.sleep", lambda _x: None)
        r = AgentChatResponse(response="", is_dummy_stream=True)
        assert list(r.response_gen) == [" "]

    def test_response_gen_multiple_spaces_preserves_empty_tokens(self, monkeypatch):
        """Input: response="a  b" (two spaces), is_dummy_stream=True; patch sleep to 0.

        Expected: split(" ") preserves empty tokens -> ['a', '', 'b'] => ['a ', ' ', 'b '].
        Check: Generator output matches expected with empty middle token as a single space.
        """
        monkeypatch.setattr("serapeum.core.chat.models.time.sleep", lambda _x: None)
        r = AgentChatResponse(response="a  b", is_dummy_stream=True)
        assert list(r.response_gen) == ["a ", " ", "b "]


class TestAgentChatResponseAsyncResponseGen:
    """Test class for AgentChatResponse.async_response_gen coroutine.

    Scenarios covered:
    - Typical async streaming with patched asyncio.sleep.
    - Error when streaming disabled (is_dummy_stream=False).
    - Multiple spaces preserve empty tokens between words.
    """

    @pytest.mark.asyncio
    async def test_async_response_gen_typical_tokens(self, monkeypatch):
        """Input: response="foo bar", is_dummy_stream=True; patch asyncio.sleep to no-op.

        Expected: Tokens ['foo ', 'bar '].
        Check: List collected from async generator equals expected.
        """
        async def _fast_sleep(_x):
            return None
        monkeypatch.setattr("serapeum.core.chat.models.asyncio.sleep", _fast_sleep)
        r = AgentChatResponse(response="foo bar", is_dummy_stream=True)
        out = [t async for t in r.async_response_gen()]
        assert out == ["foo ", "bar "]

    @pytest.mark.asyncio
    async def test_async_response_gen_disabled_raises(self):
        """Input: response="x", is_dummy_stream=False.

        Expected: Iterating async_response_gen raises ValueError immediately.
        Check: pytest.raises(ValueError) around async for loop.
        """
        r = AgentChatResponse(response="x", is_dummy_stream=False)
        with pytest.raises(ValueError):
            async for _ in r.async_response_gen():  # type: ignore[misc]
                pass

    @pytest.mark.asyncio
    async def test_async_response_gen_multiple_spaces_preserves_empty_tokens(self, monkeypatch):
        """Input: response="a  b" (two spaces), is_dummy_stream=True; patch asyncio.sleep to no-op.

        Expected: ['a ', ' ', 'b '] due to the explicit separator in split(" ").
        Check: Collected list matches expected sequence with the empty middle token.
        """
        async def _fast_sleep(_x):
            return None
        monkeypatch.setattr("serapeum.core.chat.models.asyncio.sleep", _fast_sleep)
        r = AgentChatResponse(response="a  b", is_dummy_stream=True)
        out = [t async for t in r.async_response_gen()]
        assert out == ["a ", " ", "b "]


class TestAgentChatResponseInit:
    """Test class for AgentChatResponse initialization and defaults.

    Scenarios covered:
    - Default field values when not provided.
    - Metadata preservation.
    """

    def test_defaults(self):
        """Input: Construct AgentChatResponse() without arguments.

        Expected: response=="", sources==[], is_dummy_stream is False, metadata is None.
        Check: Field values equal their documented defaults.
        """
        r = AgentChatResponse()
        assert r.response == ""
        assert isinstance(r.sources, list) and r.sources == []
        assert r.is_dummy_stream is False
        assert r.metadata is None

    def test_metadata_preserved(self):
        """Input: Provide metadata dict at construction time.

        Expected: metadata attribute references the same structure.
        Check: Keys and values preserved exactly.
        """
        meta = {"model": "gpt", "tags": ["a", "b"]}
        r = AgentChatResponse(response="ok", metadata=meta)
        assert r.metadata == meta
