"""Test LLM program."""

import json
from unittest.mock import MagicMock

from serapeum.core.base.llms.models import (
    Message,
    ChatResponse,
    CompletionResponse,
    Metadata,
    MessageRole,
)
from pydantic import BaseModel
from serapeum.core.output_parsers.models import PydanticOutputParser
from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
from serapeum.core.prompts import ChatPromptTemplate


class MockLLM(MagicMock):
    def complete(self, prompt: str) -> CompletionResponse:
        test_object = {"hello": "world"}
        text = json.dumps(test_object)
        return CompletionResponse(text=text)

    @property
    def metadata(self) -> Metadata:
        return Metadata()


class MockChatLLM(MagicMock):
    def chat(self, prompt: str) -> ChatResponse:
        test_object = {"hello": "chat"}
        text = json.dumps(test_object)
        return ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content=text)
        )

    @property
    def metadata(self) -> Metadata:
        metadata = Metadata()
        metadata.is_chat_model = True
        return metadata


class ModelTest(BaseModel):
    __test__ = False
    hello: str

class TestTextCompletionLLM:
    def test_text_completion_llm(self) -> None:
        """Test LLM program."""
        output_parser = PydanticOutputParser(output_cls=ModelTest)
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
        output_parser = PydanticOutputParser(output_cls=ModelTest)
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
        output_parser = PydanticOutputParser(output_cls=ModelTest)
        text_llm = TextCompletionLLM(
            output_parser=output_parser,
            prompt=prompt,
            llm=MockChatLLM(),
        )

        obj_output = text_llm()
        assert isinstance(obj_output, ModelTest)
        assert obj_output.hello == "chat"
