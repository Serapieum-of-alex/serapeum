"""Tests for ChatToCompletionMixin."""

import pytest

from serapeum.core.base.llms.types import (
    ChatResponse,
    Message,
    MessageList,
    MessageRole,
)
from serapeum.core.llms.abstractions.mixins import ChatToCompletionMixin


class MockLLM(ChatToCompletionMixin):
    """Mock LLM that uses ChatToCompletionMixin for testing.

    This mock implements only the chat methods. The completion methods
    are provided by the mixin.
    """

    def __init__(self, response_text: str = "OK"):
        """Initialize with a response text to return."""
        self.response_text = response_text
        self.last_messages = None

    def chat(self, messages: MessageList, **kwargs):
        """Mock chat implementation."""
        self.last_messages = messages
        return ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content=self.response_text)
        )

    def stream_chat(self, messages: MessageList, **kwargs):
        """Mock streaming chat implementation."""
        self.last_messages = messages
        # Yield two chunks
        yield ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content="O"), delta="O"
        )
        yield ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content="OK"), delta="K"
        )

    async def achat(self, messages: MessageList, **kwargs):
        """Mock async chat implementation."""
        self.last_messages = messages
        return ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content=self.response_text)
        )

    async def astream_chat(self, messages: MessageList, **kwargs):
        """Mock async streaming chat implementation."""
        self.last_messages = messages

        async def gen():
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="O"), delta="O"
            )
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="OK"), delta="K"
            )

        return gen()


class TestChatToCompletionMixin:
    """Test ChatToCompletionMixin functionality."""

    def test_complete_delegates_to_chat(self):
        """Test that complete() delegates to chat().

        Inputs: Call complete() with a prompt string.
        Expected: The mixin converts it to MessageList, calls chat(), returns CompletionResponse.
        Checks: Response text matches chat response, messages were converted correctly.
        """
        llm = MockLLM(response_text="Hello!")
        response = llm.complete("Test prompt")

        # Check response
        assert response.text == "Hello!"

        # Check that messages were converted correctly
        assert llm.last_messages is not None
        assert len(llm.last_messages) == 1
        assert llm.last_messages[0].role == MessageRole.USER
        assert llm.last_messages[0].content == "Test prompt"

    def test_stream_complete_delegates_to_stream_chat(self):
        """Test that stream_complete() delegates to stream_chat().

        Inputs: Call stream_complete() with a prompt string.
        Expected: Returns a generator yielding CompletionResponse chunks.
        Checks: Chunks match the chat stream chunks.
        """
        llm = MockLLM()
        chunks = list(llm.stream_complete("Stream test"))

        # Check we got chunks
        assert len(chunks) == 2
        assert chunks[0].text == "O"
        assert chunks[0].delta == "O"
        assert chunks[1].text == "OK"
        assert chunks[1].delta == "K"

        # Check messages were converted
        assert llm.last_messages is not None
        assert len(llm.last_messages) == 1
        assert llm.last_messages[0].content == "Stream test"

    @pytest.mark.asyncio
    async def test_acomplete_delegates_to_achat(self):
        """Test that acomplete() delegates to achat().

        Inputs: Call acomplete() with a prompt string.
        Expected: Returns CompletionResponse from async chat.
        Checks: Response text matches, messages converted correctly.
        """
        llm = MockLLM(response_text="Async response")
        response = await llm.acomplete("Async test")

        assert response.text == "Async response"

        assert llm.last_messages is not None
        assert len(llm.last_messages) == 1
        assert llm.last_messages[0].content == "Async test"

    @pytest.mark.asyncio
    async def test_astream_complete_delegates_to_astream_chat(self):
        """Test that astream_complete() delegates to astream_chat().

        Inputs: Call astream_complete() with a prompt string.
        Expected: Returns async generator yielding CompletionResponse chunks.
        Checks: Chunks match async chat stream.
        """
        llm = MockLLM()
        gen = await llm.astream_complete("Async stream test")

        chunks = []
        async for chunk in gen:
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].text == "O"
        assert chunks[0].delta == "O"
        assert chunks[1].text == "OK"
        assert chunks[1].delta == "K"

        assert llm.last_messages is not None
        assert llm.last_messages[0].content == "Async stream test"

    def test_formatted_parameter_accepted(self):
        """Test that the formatted parameter is accepted but not used.

        Inputs: Call complete() with formatted=True.
        Expected: Works without error (parameter is for compatibility).
        Checks: Response is correct.
        """
        llm = MockLLM(response_text="Formatted")
        response = llm.complete("Test", formatted=True)

        assert response.text == "Formatted"

    def test_kwargs_passed_through(self):
        """Test that additional kwargs are passed through to chat methods.

        Inputs: Call complete() with extra kwargs.
        Expected: The kwargs are passed to the underlying chat method.
        Checks: This is tested by ensuring no errors occur with extra params.
        """

        class KwargsCapturingLLM(ChatToCompletionMixin):
            def __init__(self):
                self.captured_kwargs = {}

            def chat(self, messages, **kwargs):
                self.captured_kwargs = kwargs
                return ChatResponse(
                    message=Message(role=MessageRole.ASSISTANT, content="OK")
                )

        llm = KwargsCapturingLLM()
        llm.complete("Test", temperature=0.7, max_tokens=100)

        assert llm.captured_kwargs == {"temperature": 0.7, "max_tokens": 100}
