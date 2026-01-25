"""Tests for Ollama text completion integration with serapeum-core."""

import pytest
from pydantic import BaseModel

from serapeum.core.base.llms.models import Message, MessageRole
from serapeum.core.configs.configs import Configs
from serapeum.core.output_parsers import PydanticParser
from serapeum.core.prompts import ChatPromptTemplate
from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM
from serapeum.llms.ollama import Ollama

LLM = Ollama(
    model="llama3.1",
    request_timeout=180,
)


class DummyModel(BaseModel):
    """Dummy model for testing output parsing."""

    value: str


class SecondaryModel(BaseModel):
    """Secondary model for type mismatch tests."""

    flag: bool


class RecordingPydanticParser(PydanticParser):
    """PydanticParser that records parse calls for testing."""

    def __init__(
        self,
        *,
        output_cls: type[BaseModel],
        override_result: BaseModel | None = None,
        custom_results: dict[str, BaseModel] | None = None,
    ) -> None:
        """Initialize RecordingPydanticParser."""
        super().__init__(output_cls=output_cls)
        self.override_result = override_result
        self.custom_results = custom_results or {}
        self.parse_calls: list[str] = []

    def parse(self, output: str):
        """Record and parse output string."""
        self.parse_calls.append(output)
        if output in self.custom_results:
            return self.custom_results[output]
        if self.override_result is not None:
            return self.override_result
        return super().parse(output)


@pytest.fixture(autouse=True)
def restore_configs_llm():
    """Ensure Configs.llm is restored after every test."""
    original_llm = Configs.llm
    yield
    Configs.llm = original_llm


class TestCallMethod:
    """Test synchronous call method for TextCompletionLLM."""

    @pytest.mark.e2e
    def test_call_non_chat_llm_success(self) -> None:
        """Test synchronous call with text LLM.

        Inputs: text LLM and prompt args with llm kwargs.
        Expected: parse returns DummyModel.
        Checks: complete path and kwargs forwarding.
        """
        parser = RecordingPydanticParser(output_cls=DummyModel)
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt="Value: {value}",
            llm=LLM,
        )

        result = text_llm(llm_kwargs={"temperature": 0.2}, value="input")

        assert isinstance(result, DummyModel)
        # the `value` attribute should have a value  of `input`
        assert result.value == "input"

    @pytest.mark.e2e
    def test_call_chat_llm_success(self) -> None:
        """Test synchronous call with chat LLM.

        Inputs: chat LLM with chat response.
        Expected: parse returns DummyModel.
        Checks: chat branch and message extension.
        """
        parser = RecordingPydanticParser(output_cls=DummyModel)
        messages = [
            Message(role=MessageRole.USER, content="Value"),
        ]
        prompt = ChatPromptTemplate(message_templates=messages)
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt=prompt,
            llm=LLM,
        )
        result = text_llm()

        assert isinstance(result, DummyModel)
        assert result.value is not None

    @pytest.mark.e2e
    def test_call_raises_when_parser_returns_wrong_type(self) -> None:
        """Test error when parser returns wrong type.

        Inputs: parser returning SecondaryModel.
        Expected: ValueError complaining about mismatch.
        Checks: runtime type guard.
        """
        parser = RecordingPydanticParser(
            output_cls=DummyModel,
            override_result=SecondaryModel(flag=True),
        )
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt="Value: {value}",
            llm=LLM,
        )

        with pytest.raises(ValueError):
            text_llm(value="anything")


class TestAcallMethod:
    """Test asynchronous call method for TextCompletionLLM."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_acall_non_chat_llm_success(self) -> None:
        """Test async call with text LLM.

        Inputs: async call on text LLM.
        Expected: DummyModel returned.
        Checks: asynchronous complete branch.
        """
        parser = RecordingPydanticParser(output_cls=DummyModel)
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt="Value: {value}",
            llm=LLM,
        )

        result = await text_llm.acall(value="input")

        assert isinstance(result, DummyModel)
        assert result.value == "input"

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_acall_chat_llm_success(self) -> None:
        """Test async call with chat LLM.

        Inputs: async call on chat LLM.
        Expected: DummyModel returned.
        Checks: asynchronous chat branch.
        """
        parser = RecordingPydanticParser(output_cls=DummyModel)
        messages = [
            Message(role=MessageRole.USER, content="Value"),
        ]
        prompt = ChatPromptTemplate(message_templates=messages)
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt=prompt,
            llm=LLM,
        )

        result = await text_llm.acall()

        assert isinstance(result, DummyModel)
        assert result.value is not None

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_acall_raises_when_parser_returns_wrong_type(self) -> None:
        """Test error when parser returns wrong type in async call.

        Inputs: parser returning SecondaryModel.
        Expected: ValueError for wrong type.
        Checks: async guard mirrored from sync path.
        """
        parser = RecordingPydanticParser(
            output_cls=DummyModel,
            override_result=SecondaryModel(flag=True),
        )
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt="Value: {value}",
            llm=LLM,
        )

        with pytest.raises(ValueError):
            await text_llm.acall(value="anything")
