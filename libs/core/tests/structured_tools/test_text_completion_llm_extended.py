"""Tests for TextCompletionLLM."""

import pytest
from pydantic import BaseModel

from serapeum.core.base.llms.types import (
    ChatResponse,
    CompletionResponse,
    Message,
    MessageRole,
    Metadata,
)
from serapeum.core.configs.configs import Configs
from serapeum.core.output_parsers import BaseParser, PydanticParser
from serapeum.core.prompts import ChatPromptTemplate
from serapeum.core.prompts.base import PromptTemplate
from serapeum.core.structured_tools.text_completion_llm import TextCompletionLLM


class DummyModel(BaseModel):
    """Dummy model for testing."""

    value: str


class SecondaryModel(BaseModel):
    """Secondary model for testing."""

    flag: bool


class RecordingPydanticParser(PydanticParser):
    """Recording Pydantic parser for testing."""

    def __init__(
        self,
        *,
        output_cls: type[BaseModel],
        override_result: BaseModel | None = None,
        custom_results: dict[str, BaseModel] | None = None,
    ) -> None:
        """Initialize RecordingPydanticParser with customizable results."""
        super().__init__(output_cls=output_cls)
        self.override_result = override_result
        self.custom_results = custom_results or {}
        self.parse_calls: list[str] = []

    def parse(self, output: str):
        self.parse_calls.append(output)
        if output in self.custom_results:
            return self.custom_results[output]
        if self.override_result is not None:
            return self.override_result
        return super().parse(output)


class DummyNonPydanticParser(BaseParser):
    """Dummy non-Pydantic parser for testing."""

    def parse(self, output: str):
        """Parse output by returning it unchanged."""
        return output


class DummyLLM:
    """Dummy LLM for testing."""

    def __init__(
        self,
        *,
        is_chat: bool = False,
        completion_text: str = '{"value": "from-complete"}',
        chat_content: str | None = '{"value": "from-chat"}',
    ) -> None:
        """Initialize DummyLLM with predefined responses."""
        self.metadata = Metadata(is_chat_model=is_chat)
        self.completion_response = CompletionResponse(text=completion_text)
        self.chat_response = ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content=chat_content)
        )
        self.complete_calls: list[tuple[str, dict]] = []
        self.chat_calls: list[tuple[list[Message], dict]] = []
        self.extend_calls: list[tuple[list[Message], list[Message]]] = []
        self.acomplete_calls: list[tuple[str, dict]] = []
        self.achat_calls: list[tuple[list[Message], dict]] = []

    def complete(self, prompt: str, **kwargs):
        self.complete_calls.append((prompt, kwargs))
        return self.completion_response

    async def acomplete(self, prompt: str, **kwargs):
        self.acomplete_calls.append((prompt, kwargs))
        return self.completion_response

    def chat(self, messages, **kwargs):
        self.chat_calls.append((messages, kwargs))
        return self.chat_response

    async def achat(self, messages, **kwargs):
        self.achat_calls.append((messages, kwargs))
        return self.chat_response

    def _extend_messages(self, messages):
        extended = list(messages) + [
            Message(role=MessageRole.SYSTEM, content="extended")
        ]
        self.extend_calls.append((messages, extended))
        return extended


@pytest.fixture(autouse=True)
def restore_configs_llm():
    """Fixture ensuring Configs.llm is restored after every test."""
    original_llm = Configs.llm
    yield
    Configs.llm = original_llm


class TestTextCompletionLLMInit:
    """Test suite for TextCompletionLLM initialization."""

    def test_init_with_provided_parser_and_prompt_template(self) -> None:
        """Test initialization with provided parser and prompt template.

        Inputs: PromptTemplate and ready PydanticParser
        Expected: instance keeps parser/prompt/llm
        Checks: assignments done without mutation.
        """
        prompt_template = PromptTemplate("Value: {value}")
        output_parser = RecordingPydanticParser(output_cls=DummyModel)
        llm = DummyLLM()

        text_llm = TextCompletionLLM(
            output_parser=output_parser,
            prompt=prompt_template,
            llm=llm,
        )

        assert text_llm.output_cls is DummyModel
        assert text_llm.prompt is prompt_template
        assert text_llm._output_parser is output_parser  # type: ignore[attr-defined]
        assert text_llm._llm is llm  # type: ignore[attr-defined]

    def test_init_with_output_cls_only_creates_parser(self) -> None:
        """Test initialization with output_cls only.

        Inputs: output_cls without parser
        Expected: factory builds Pydantic parser and stores model
        Checks: automatic parser and prompt coercion.
        """
        llm = DummyLLM()
        text_llm = TextCompletionLLM(
            output_parser=None,
            prompt="Number: {value}",
            output_cls=DummyModel,
            llm=llm,
        )

        assert isinstance(text_llm._output_parser, PydanticParser)  # type: ignore[attr-defined]
        assert text_llm.output_cls is DummyModel
        assert isinstance(text_llm.prompt, PromptTemplate)

    def test_init_with_string_prompt_coerces_to_template(self) -> None:
        """Test initialization with string prompt.

        Inputs: bare string prompt
        Expected: init wraps string into PromptTemplate
        Checks: prompt getter returns PromptTemplate.
        """
        llm = DummyLLM()
        output_parser = RecordingPydanticParser(output_cls=DummyModel)

        text_llm = TextCompletionLLM(
            output_parser=output_parser,
            prompt="Hello {value}",
            llm=llm,
        )

        assert isinstance(text_llm.prompt, PromptTemplate)

    def test_init_rejects_invalid_prompt_type(self) -> None:
        """Test initialization rejects invalid prompt type.

        Inputs: unsupported prompt type
        Expected: initializer raises ValueError
        Checks: validate_prompt enforces type safety.
        """
        llm = DummyLLM()
        output_parser = RecordingPydanticParser(output_cls=DummyModel)

        with pytest.raises(ValueError):
            TextCompletionLLM(
                output_parser=output_parser,
                prompt=123,  # type: ignore[arg-type]
                llm=llm,
            )


class TestValidatePrompt:
    """Test suite for _validate_prompt method."""

    def test_validate_prompt_with_template_instance(self) -> None:
        """Test validation with PromptTemplate instance.

        Inputs: PromptTemplate already built
        Expected: method returns same template
        Checks: passthrough behavior.
        """
        prompt_template = PromptTemplate("Value: {value}")

        validated = TextCompletionLLM._validate_prompt(prompt_template)

        assert validated is prompt_template

    def test_validate_prompt_with_string(self) -> None:
        """Test validation with string prompt.

        Inputs: simple string
        Expected: method wraps string into PromptTemplate
        Checks: coercion branch.
        """
        validated = TextCompletionLLM._validate_prompt("Hi {value}")

        assert isinstance(validated, PromptTemplate)
        assert validated.template == "Hi {value}"

    def test_validate_prompt_with_invalid_type(self) -> None:
        """Test validation rejects invalid prompt type.

        Inputs: non-string, non-template
        Expected: ValueError raised
        Checks: guard rails for prompt types.
        """
        with pytest.raises(ValueError):
            TextCompletionLLM._validate_prompt(object())  # type: ignore[arg-type]


class TestValidateLlm:
    """Test suite for _validate_llm method."""

    def test_validate_llm_with_explicit_instance(self) -> None:
        """Test validation with explicit LLM instance.

        Inputs: direct DummyLLM
        Expected: method returns same instance
        Checks: short-circuit branch.
        """
        llm = DummyLLM()

        validated = TextCompletionLLM._validate_llm(llm)

        assert validated is llm

    def test_validate_llm_with_configs_default(self) -> None:
        """Test validation with Configs.llm default.

        Inputs: None while Configs.llm preset
        Expected: method falls back to Configs.llm
        Checks: default resolution.
        """
        llm = DummyLLM()
        Configs.llm = llm

        validated = TextCompletionLLM._validate_llm(None)  # type: ignore[arg-type]

        assert validated is llm

    def test_validate_llm_without_any_source(self) -> None:
        """Test validation without any LLM source.

        Inputs: None and Configs.llm unset
        Expected: AssertionError
        Checks: strict requirement for available LLM.
        """
        Configs.llm = None  # type: ignore[assignment]

        with pytest.raises(AssertionError):
            TextCompletionLLM._validate_llm(None)  # type: ignore[arg-type]


class TestValidateOutputParserCls:
    """Test suite for _validate_output_parser_cls method."""

    def test_validate_output_parser_without_output_cls_uses_parser_model(self) -> None:
        """Test validation without output_cls uses parser model.

        Inputs: Pydantic parser and no output class
        Expected: returns provided parser and its model
        Checks: default model resolution.
        """
        parser = PydanticParser(output_cls=DummyModel)

        validated_parser, validated_cls = TextCompletionLLM._validate_output_parser_cls(
            parser,
            None,  # type: ignore[arg-type]
        )

        assert validated_parser is parser
        assert validated_cls is DummyModel

    def test_validate_output_parser_requires_pydantic_when_no_output_cls(self) -> None:
        """Test validation requires Pydantic parser without output_cls.

        Inputs: non-Pydantic parser without output class
        Expected: ValueError
        Checks: guard for parser type.
        """
        parser = DummyNonPydanticParser()

        with pytest.raises(ValueError):
            TextCompletionLLM._validate_output_parser_cls(
                parser,
                None,  # type: ignore[arg-type]
            )

    def test_validate_output_parser_creates_parser_when_missing(self) -> None:
        """Test validation creates parser when missing.

        Inputs: output class without parser
        Expected: new PydanticParser returned
        Checks: factory instantiation.
        """
        parser, output_cls = TextCompletionLLM._validate_output_parser_cls(
            None,  # type: ignore[arg-type]
            DummyModel,
        )

        assert isinstance(parser, PydanticParser)
        assert output_cls is DummyModel

    def test_validate_output_parser_with_both_arguments_preserves_instances(
        self,
    ) -> None:
        """Test validation preserves instances with both arguments.

        Inputs: explicit parser and output class
        Expected: returns unchanged inputs
        Checks: bypass branch when values supplied.
        """
        parser = PydanticParser(output_cls=DummyModel)

        validated_parser, validated_cls = TextCompletionLLM._validate_output_parser_cls(
            parser,
            DummyModel,
        )

        assert validated_parser is parser
        assert validated_cls is DummyModel


class TestOutputClsProperty:
    """Test suite for output_cls property."""

    def test_output_cls_returns_configured_model(self) -> None:
        """Test output_cls property returns configured model.

        Inputs: text_llm built with DummyModel
        Expected: property exposes DummyModel
        Checks: property passthrough.
        """
        text_llm = TextCompletionLLM(
            output_parser=PydanticParser(output_cls=DummyModel),
            prompt="Value: {value}",
            llm=DummyLLM(),
        )

        assert text_llm.output_cls is DummyModel


class TestPromptProperty:
    """Test suite for prompt property."""

    def test_prompt_getter_returns_current_prompt(self) -> None:
        """Test prompt getter returns current prompt.

        Inputs: initial PromptTemplate
        Expected: getter returns same instance
        Checks: property pass-through.
        """
        prompt = PromptTemplate("Value: {value}")
        text_llm = TextCompletionLLM(
            output_parser=PydanticParser(output_cls=DummyModel),
            prompt=prompt,
            llm=DummyLLM(),
        )

        assert text_llm.prompt is prompt

    def test_prompt_setter_updates_prompt(self) -> None:
        """Test prompt setter updates prompt.

        Inputs: replacement PromptTemplate
        Expected: setter swaps prompt reference
        Checks: mutability of prompt property.
        """
        text_llm = TextCompletionLLM(
            output_parser=PydanticParser(output_cls=DummyModel),
            prompt="Start {value}",
            llm=DummyLLM(),
        )
        new_prompt = PromptTemplate("New {value}")

        text_llm.prompt = new_prompt

        assert text_llm.prompt is new_prompt


class TestCallMethod:
    """Test suite for __call__ method."""

    def test_call_non_chat_llm_success(self) -> None:
        """Test call with non-chat LLM success.

        Inputs: text LLM and prompt args with llm kwargs
        Expected: parse returns DummyModel
        Checks: complete path and kwargs forwarding.
        """
        llm = DummyLLM(is_chat=False, completion_text='{"value": "complete"}')
        parser = RecordingPydanticParser(output_cls=DummyModel)
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt="Value: {value}",
            llm=llm,
        )

        result = text_llm(llm_kwargs={"temperature": 0.2}, value="input")

        assert result == DummyModel(value="complete")
        assert llm.complete_calls[0][1] == {"temperature": 0.2}
        assert parser.parse_calls == ['{"value": "complete"}']

    def test_call_chat_llm_success(self) -> None:
        """Test call with chat LLM success.

        Inputs: chat LLM with chat response
        Expected: parse returns DummyModel
        Checks: chat branch and message extension.
        """
        llm = DummyLLM(is_chat=True, chat_content='{"value": "chat"}')
        parser = RecordingPydanticParser(output_cls=DummyModel)
        messages = [
            Message(role=MessageRole.USER, content="Value"),
        ]
        prompt = ChatPromptTemplate(message_templates=messages)
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt=prompt,
            llm=llm,
        )

        result = text_llm()

        assert result == DummyModel(value="chat")
        assert llm.extend_calls  # ensure extension occurred
        assert parser.parse_calls == ['{"value": "chat"}']

    def test_call_chat_llm_uses_empty_string_when_no_content(self) -> None:
        """Test call with chat LLM using empty string when no content.

        Inputs: chat LLM returning None content
        Expected: parser sees empty string
        Checks: fallback to empty string.
        """
        llm = DummyLLM(is_chat=True, chat_content=None)
        parser = RecordingPydanticParser(
            output_cls=DummyModel,
            custom_results={"": DummyModel(value="fallback")},
        )
        messages = [
            Message(role=MessageRole.USER, content="Value"),
        ]
        prompt = ChatPromptTemplate(message_templates=messages)
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt=prompt,
            llm=llm,
        )

        result = text_llm()

        assert result == DummyModel(value="fallback")
        assert parser.parse_calls == [""]

    def test_call_raises_when_parser_returns_wrong_type(self) -> None:
        """Test call raises when parser returns wrong type.

        Inputs: parser returning SecondaryModel
        Expected: ValueError complaining about mismatch
        Checks: runtime type guard.
        """
        llm = DummyLLM(is_chat=False, completion_text='{"value": "complete"}')
        parser = RecordingPydanticParser(
            output_cls=DummyModel,
            override_result=SecondaryModel(flag=True),
        )
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt="Value: {value}",
            llm=llm,
        )

        with pytest.raises(ValueError):
            text_llm(value="anything")


class TestAcallMethod:
    """Test suite for acall method."""

    @pytest.mark.asyncio
    async def test_acall_non_chat_llm_success(self) -> None:
        """Test async call with non-chat LLM success.

        Inputs: async call on text LLM
        Expected: DummyModel returned
        Checks: asynchronous complete branch.
        """
        llm = DummyLLM(is_chat=False, completion_text='{"value": "complete"}')
        parser = RecordingPydanticParser(output_cls=DummyModel)
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt="Value: {value}",
            llm=llm,
        )

        result = await text_llm.acall(value="input")

        assert result == DummyModel(value="complete")
        assert llm.acomplete_calls  # ensure async path invoked
        assert parser.parse_calls == ['{"value": "complete"}']

    @pytest.mark.asyncio
    async def test_acall_chat_llm_success(self) -> None:
        """Test async call with chat LLM success.

        Inputs: async call on chat LLM
        Expected: DummyModel returned
        Checks: asynchronous chat branch.
        """
        llm = DummyLLM(is_chat=True, chat_content='{"value": "chat"}')
        parser = RecordingPydanticParser(output_cls=DummyModel)
        messages = [
            Message(role=MessageRole.USER, content="Value"),
        ]
        prompt = ChatPromptTemplate(message_templates=messages)
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt=prompt,
            llm=llm,
        )

        result = await text_llm.acall()

        assert result == DummyModel(value="chat")
        assert llm.extend_calls  # ensure message extension reused
        assert parser.parse_calls == ['{"value": "chat"}']

    @pytest.mark.asyncio
    async def test_acall_raises_when_parser_returns_wrong_type(self) -> None:
        """Test async call raises when parser returns wrong type.

        Inputs: parser returning SecondaryModel
        Expected: ValueError for wrong type
        Checks: async guard mirrored from sync path.
        """
        llm = DummyLLM(is_chat=False, completion_text='{"value": "complete"}')
        parser = RecordingPydanticParser(
            output_cls=DummyModel,
            override_result=SecondaryModel(flag=True),
        )
        text_llm = TextCompletionLLM(
            output_parser=parser,
            prompt="Value: {value}",
            llm=llm,
        )

        with pytest.raises(ValueError):
            await text_llm.acall(value="anything")
