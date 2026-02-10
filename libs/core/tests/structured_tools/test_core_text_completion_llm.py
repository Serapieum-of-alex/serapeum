"""Test LLM program."""

import json
from unittest.mock import MagicMock

from pydantic import BaseModel

from serapeum.core.base.llms.types import (
    ChatResponse,
    CompletionResponse,
    Message,
    MessageRole,
    Metadata,
)
from serapeum.core.output_parsers import PydanticParser
from serapeum.core.prompts import ChatPromptTemplate
from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM


class MockLLM(MagicMock):
    """Mock LLM for simulating completion responses."""

    def complete(self, prompt: str) -> CompletionResponse:
        test_object = {"hello": "world"}
        text = json.dumps(test_object)
        return CompletionResponse(text=text)

    @property
    def metadata(self) -> Metadata:
        return Metadata()


class MockChatLLM(MagicMock):
    """Mock chat LLM for simulating chat responses."""

    def chat(self, prompt: str) -> ChatResponse:
        test_object = {"hello": "chat"}
        text = json.dumps(test_object)
        return ChatResponse(message=Message(role=MessageRole.ASSISTANT, content=text))

    @property
    def metadata(self) -> Metadata:
        metadata = Metadata()
        metadata.is_chat_model = True
        return metadata


class ModelTest(BaseModel):
    """Pydantic model for test output."""

    __test__ = False
    hello: str


class TestTextCompletionLLM:
    """Tests for TextCompletionLLM."""

    def test_text_completion_llm_core(self) -> None:
        """Test LLM program."""
        output_parser = PydanticParser(output_cls=ModelTest)
        text_llm = TextCompletionLLM(
            output_parser=output_parser,
            prompt="This is a test prompt with a {test_input}.",
            llm=MockLLM(),
        )

        obj_output = text_llm(test_input="hello")
        assert isinstance(obj_output, ModelTest)
        assert obj_output.hello == "world"

    def test_text_llm_with_messages(self) -> None:
        """Test LLM program."""
        messages = [Message(role=MessageRole.USER, content="Test")]
        prompt = ChatPromptTemplate(message_templates=messages)
        output_parser = PydanticParser(output_cls=ModelTest)
        text_llm = TextCompletionLLM(
            output_parser=output_parser,
            prompt=prompt,
            llm=MockLLM(),
        )

        obj_output = text_llm()
        assert isinstance(obj_output, ModelTest)
        assert obj_output.hello == "world"

    def test_llm_program_with_messages_and_chat(self) -> None:
        """Test LLM program."""
        messages = [Message(role=MessageRole.USER, content="Test")]
        prompt = ChatPromptTemplate(message_templates=messages)
        output_parser = PydanticParser(output_cls=ModelTest)
        text_llm = TextCompletionLLM(
            output_parser=output_parser,
            prompt=prompt,
            llm=MockChatLLM(),
        )

        obj_output = text_llm()
        assert isinstance(obj_output, ModelTest)
        assert obj_output.hello == "chat"
