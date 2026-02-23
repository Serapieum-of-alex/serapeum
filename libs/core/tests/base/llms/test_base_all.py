from typing import Any, AsyncGenerator, ClassVar, Generator, Sequence

import pytest
from pydantic import BaseModel

from serapeum.core.base.llms.types import (
    ChatResponse,
    CompletionResponse,
    Message,
    MessageList,
    MessageRole,
    Metadata,
)
from serapeum.core.llms import TextCompletionLLM, ToolOrchestratingLLM
from serapeum.core.llms.base import (
    LLM,
    CompletionToPromptType,
    MessagesToPromptType,
    astream_response_to_tokens,
    default_completion_to_prompt,
    stream_response_to_tokens,
)
from serapeum.core.output_parsers import BaseParser
from serapeum.core.prompts import ChatPromptTemplate, PromptTemplate
from serapeum.core.types import StructuredOutputMode


class _FormatParser(BaseParser):
    def parse(self, output: str) -> str:
        return output

    def format(self, query: str) -> str:
        return f"[[{query}]]"


class _UpperParser(BaseParser):
    def parse(self, output: str) -> str:
        return output.upper()

    def format(self, query: str) -> str:
        return query.upper()


class _StripParser(BaseParser):
    def parse(self, output: str) -> str:
        return output.strip()


class _RecordingLLM(LLM):
    _metadata: ClassVar[Metadata] = Metadata.model_construct(
        is_chat_model=False,
        is_function_calling_model=False,
        model_name="recording",
    )

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    def _log_template_data(self, prompt: Any, **kwargs: Any) -> None:
        self.logged = (prompt, kwargs)

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError()

    def stream_chat(self, messages: Sequence[Message], **kwargs: Any) -> Generator:
        raise NotImplementedError()

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self.last_complete = (prompt, formatted, kwargs)
        return CompletionResponse(text=prompt, delta=prompt)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Generator[CompletionResponse, None, None]:
        self.last_stream_complete = (prompt, formatted, kwargs)

        def gen() -> Generator[CompletionResponse, None, None]:
            yield CompletionResponse(text=prompt, delta="a")
            yield CompletionResponse(text=prompt, delta="b")

        return gen()

    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError()

    async def astream_chat(self, messages: Sequence[Message], **kwargs: Any) -> Any:
        raise NotImplementedError()

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self.last_acomplete = (prompt, formatted, kwargs)
        return CompletionResponse(text=prompt[::-1], delta=prompt[::-1])

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> AsyncGenerator[CompletionResponse, None]:
        self.last_astream_complete = (prompt, formatted, kwargs)

        async def gen() -> AsyncGenerator[CompletionResponse, None]:
            yield CompletionResponse(text=prompt, delta="x")
            yield CompletionResponse(text=prompt, delta="y")

        return gen()


class _ChatLLM(LLM):
    _metadata: ClassVar[Metadata] = Metadata.model_construct(
        is_chat_model=True,
        is_function_calling_model=False,
        model_name="chat",
    )

    @property
    def metadata(self) -> Metadata:
        return self._metadata

    def _log_template_data(self, prompt: Any, **kwargs: Any) -> None:
        self.logged = (prompt, kwargs)

    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        self.last_chat = (messages, kwargs)
        return ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content="pong"),
            delta=None,
        )

    def stream_chat(self, messages: Sequence[Message], **kwargs: Any) -> Generator:
        self.last_stream_chat = (messages, kwargs)

        def gen() -> Generator[ChatResponse, None, None]:
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="ok"),
                delta="o",
            )
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="ok"),
                delta="k",
            )

        return gen()

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError()

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Generator:
        raise NotImplementedError()

    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        self.last_achat = (messages, kwargs)
        return ChatResponse(
            message=Message(role=MessageRole.ASSISTANT, content="pong"),
            delta=None,
        )

    async def astream_chat(self, messages: Sequence[Message], **kwargs: Any) -> Any:
        self.last_astream_chat = (messages, kwargs)

        async def gen() -> AsyncGenerator[ChatResponse, None]:
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="ok"),
                delta="o",
            )
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="ok"),
                delta="k",
            )

        return gen()

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError()

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> Any:
        raise NotImplementedError()


class _FunctionCallingLLM(_RecordingLLM):
    _metadata: ClassVar[Metadata] = Metadata.model_construct(
        is_chat_model=False,
        is_function_calling_model=True,
        model_name="function",
    )


class _OutputModel(BaseModel):
    name: str


class TestMessagesToPromptTypeProtocol:
    def test_callable_is_instance(self) -> None:
        """
        Inputs: a lambda that accepts a MessageList and returns a string.
        Expected result: runtime Protocol check returns True.
        Checks: isinstance reports callable satisfies MessagesToPromptType.
        """
        # arrange
        fn = lambda message_list: "ok"

        # act
        result = isinstance(fn, MessagesToPromptType)

        # assert
        assert result is True


class TestCompletionToPromptTypeProtocol:
    def test_callable_is_instance(self) -> None:
        """
        Inputs: a lambda that accepts a prompt string and returns a string.
        Expected result: runtime Protocol check returns True.
        Checks: isinstance reports callable satisfies CompletionToPromptType.
        """
        # arrange
        fn = lambda prompt: prompt

        # act
        result = isinstance(fn, CompletionToPromptType)

        # assert
        assert result is True


class TestStreamResponseToTokens:
    def test_completion_response_deltas(self) -> None:
        """
        Inputs: completion responses with delta values.
        Expected result: tokens mirror the delta values in order.
        Checks: list(token_gen) equals the deltas.
        """

        # arrange
        def responses() -> Generator[CompletionResponse, None, None]:
            yield CompletionResponse(text="hello", delta="he")
            yield CompletionResponse(text="hello", delta="llo")

        # act
        tokens = list(stream_response_to_tokens(responses()))

        # assert
        assert tokens == ["he", "llo"]

    def test_chat_response_missing_delta(self) -> None:
        """
        Inputs: chat responses with None and empty delta values.
        Expected result: tokens yield empty strings for missing deltas.
        Checks: list(token_gen) returns empty strings.
        """

        # arrange
        def responses() -> Generator[ChatResponse, None, None]:
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="partial"),
                delta=None,
            )
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="done"),
                delta="",
            )

        # act
        tokens = list(stream_response_to_tokens(responses()))

        # assert
        assert tokens == ["", ""]


class TestAstreamResponseToTokens:
    @pytest.mark.asyncio
    async def test_async_completion_deltas(self) -> None:
        """
        Inputs: async completion responses with delta values.
        Expected result: async tokens mirror the delta values in order.
        Checks: collected tokens match the deltas.
        """

        # arrange
        async def responses() -> AsyncGenerator[CompletionResponse, None]:
            yield CompletionResponse(text="hi", delta="h")
            yield CompletionResponse(text="hi", delta="i")

        # act
        token_gen = await astream_response_to_tokens(responses())
        tokens = [token async for token in token_gen]

        # assert
        assert tokens == ["h", "i"]

    @pytest.mark.asyncio
    async def test_async_chat_missing_delta(self) -> None:
        """
        Inputs: async chat responses with missing deltas.
        Expected result: async tokens emit empty strings for missing deltas.
        Checks: collected tokens are empty strings.
        """

        # arrange
        async def responses() -> AsyncGenerator[ChatResponse, None]:
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="partial"),
                delta=None,
            )
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="done"),
                delta="",
            )

        # act
        token_gen = await astream_response_to_tokens(responses())
        tokens = [token async for token in token_gen]

        # assert
        assert tokens == ["", ""]


class TestDefaultCompletionToPrompt:
    def test_identity(self) -> None:
        """
        Inputs: a prompt string.
        Expected result: identity output equals input.
        Checks: returned string matches the input.
        """
        # arrange
        prompt = "Draft a report"

        # act
        result = default_completion_to_prompt(prompt)

        # assert
        assert result == prompt


class TestSetMessagesToPrompt:
    def test_defaults_to_message_list_renderer(self) -> None:
        """
        Inputs: None for messages_to_prompt.
        Expected result: callable that renders MessageList.to_prompt().
        Checks: output matches MessageList.to_prompt for sample data.
        """
        # arrange
        messages = MessageList.from_list([Message(role=MessageRole.USER, content="hi")])

        # act
        adapter = LLM.set_messages_to_prompt(None)
        rendered = adapter(messages)

        # assert
        assert rendered == messages.to_prompt()

    def test_preserves_custom_adapter(self) -> None:
        """
        Inputs: custom adapter callable.
        Expected result: same callable returned without wrapping.
        Checks: returned object is the same as the input.
        """

        # arrange
        def reverse_messages(message_list: MessageList) -> str:
            return "|".join(message.content or "" for message in reversed(message_list))

        # act
        adapter = LLM.set_messages_to_prompt(reverse_messages)

        # assert
        assert adapter is reverse_messages


class TestSetCompletionToPrompt:
    def test_defaults_to_identity_adapter(self) -> None:
        """
        Inputs: None for completion_to_prompt.
        Expected result: default completion adapter is returned.
        Checks: adapter is default_completion_to_prompt.
        """
        # arrange
        # act
        adapter = LLM.set_completion_to_prompt(None)

        # assert
        assert adapter is default_completion_to_prompt

    def test_preserves_custom_adapter(self) -> None:
        """
        Inputs: custom completion adapter.
        Expected result: same callable returned.
        Checks: returned adapter is the original callable.
        """

        # arrange
        def prefix(prompt: str) -> str:
            return f"PREFIX: {prompt}"

        # act
        adapter = LLM.set_completion_to_prompt(prefix)

        # assert
        assert adapter is prefix


class TestCheckPrompts:
    def test_populates_missing_adapters(self) -> None:
        """
        Inputs: LLM instance without adapter values.
        Expected result: messages_to_prompt and completion_to_prompt are populated.
        Checks: attributes are callable after initialization.
        """
        # arrange
        llm = _RecordingLLM()

        # act
        messages_adapter = llm.messages_to_prompt
        completion_adapter = llm.completion_to_prompt

        # assert
        assert callable(messages_adapter)
        assert callable(completion_adapter)

    def test_preserves_explicit_adapters(self) -> None:
        """
        Inputs: custom adapters passed to constructor.
        Expected result: adapters preserved on the instance.
        Checks: adapter outputs match the custom implementations.
        """

        # arrange
        def to_prompt(message_list: MessageList) -> str:
            return "CUSTOM"

        def to_completion(prompt: str) -> str:
            return prompt.upper()

        # act
        llm = _RecordingLLM(
            messages_to_prompt=to_prompt,
            completion_to_prompt=to_completion,
        )

        # assert
        assert llm.messages_to_prompt(MessageList.from_list([])) == "CUSTOM"
        assert llm.completion_to_prompt("hi") == "HI"


class TestGetPrompt:
    def test_applies_completion_adapter_and_output_parser(self) -> None:
        """
        Inputs: prompt template, completion adapter, and output parser.
        Expected result: completion adapter runs before output parser formatting.
        Checks: formatted prompt reflects both transformations.
        """
        # arrange
        llm = _RecordingLLM(
            completion_to_prompt=lambda text: f"<<{text}>>",
            output_parser=_FormatParser(),
        )
        prompt = PromptTemplate("Hello {name}")

        # act
        formatted = llm._get_prompt(prompt, name="Ada")

        # assert
        assert formatted == "[[<<Hello Ada>>]]"


class TestGetMessages:
    def test_applies_output_parser_and_system_prompt(self) -> None:
        """
        Inputs: chat prompt template and output parser with a system prompt.
        Expected result: user message is formatted, then system prompt prepended.
        Checks: content ordering and casing of user message.
        """
        # arrange
        llm = _ChatLLM(output_parser=_UpperParser(), system_prompt="SYS")
        prompt = ChatPromptTemplate.from_messages([("user", "Hello {name}")])

        # act
        messages = llm._get_messages(prompt, name="Ada")

        # assert
        assert messages[0].role == MessageRole.SYSTEM
        assert messages[0].content == "SYS"
        assert messages[1].role == MessageRole.USER
        assert messages[1].content == "HELLO ADA"


class TestParseOutput:
    def test_returns_raw_output_without_parser(self) -> None:
        """
        Inputs: output string with no parser configured.
        Expected result: raw output is returned unchanged.
        Checks: output equals input.
        """
        # arrange
        llm = _RecordingLLM()

        # act
        result = llm._parse_output("ready")

        # assert
        assert result == "ready"

    def test_parses_with_custom_parser(self) -> None:
        """
        Inputs: output string with custom parser configured.
        Expected result: parsed output is returned.
        Checks: output string is transformed by parser.
        """
        # arrange
        llm = _RecordingLLM(output_parser=_StripParser())

        # act
        result = llm._parse_output("  ok  ")

        # assert
        assert result == "ok"


class TestExtendPrompt:
    def test_adds_system_and_wrapper_prompts(self) -> None:
        """
        Inputs: system prompt and query wrapper prompt.
        Expected result: system prompt prepended then wrapped by query template.
        Checks: final string matches expected composition.
        """
        # arrange
        llm = _RecordingLLM(
            system_prompt="SYS",
            query_wrapper_prompt=PromptTemplate("Question: {query_str}"),
        )

        # act
        extended = llm._extend_prompt("Ask")

        # assert
        assert extended == "Question: SYS\n\nAsk"


class TestExtendMessages:
    def test_no_system_prompt_returns_original(self) -> None:
        """
        Inputs: message list without system prompt configured.
        Expected result: messages returned unchanged.
        Checks: list contents and ordering are preserved.
        """
        # arrange
        llm = _ChatLLM()
        messages = [Message(role=MessageRole.USER, content="Hi")]

        # act
        extended = llm._extend_messages(messages)

        # assert
        assert extended == messages

    def test_system_prompt_is_prepended(self) -> None:
        """
        Inputs: message list with system prompt configured.
        Expected result: system message is prepended to list.
        Checks: first message is system prompt and original message follows.
        """
        # arrange
        llm = _ChatLLM(system_prompt="SYS")
        messages = [Message(role=MessageRole.USER, content="Hi")]

        # act
        extended = llm._extend_messages(messages)

        # assert
        assert extended[0].role == MessageRole.SYSTEM
        assert extended[0].content == "SYS"
        assert extended[1].content == "Hi"


class TestGetProgram:
    def test_default_mode_function_calling(self) -> None:
        """
        Inputs: DEFAULT mode with function-calling model metadata.
        Expected result: ToolOrchestratingLLM is instantiated.
        Checks: returned instance type.
        """
        # arrange
        llm = _FunctionCallingLLM()
        prompt = PromptTemplate("{name}")

        # act
        program = llm._get_program(_OutputModel, prompt)

        # assert
        assert isinstance(program, ToolOrchestratingLLM)

    def test_default_mode_text_completion(self) -> None:
        """
        Inputs: DEFAULT mode with non-function-calling metadata.
        Expected result: TextCompletionLLM is instantiated.
        Checks: returned instance type.
        """
        # arrange
        llm = _RecordingLLM()
        prompt = PromptTemplate("{name}")

        # act
        program = llm._get_program(_OutputModel, prompt)

        # assert
        assert isinstance(program, TextCompletionLLM)

    def test_function_mode_forces_tool_orchestrating(self) -> None:
        """
        Inputs: FUNCTION mode regardless of metadata.
        Expected result: ToolOrchestratingLLM is instantiated.
        Checks: returned instance type.
        """
        # arrange
        llm = _FunctionCallingLLM(structured_output_mode=StructuredOutputMode.FUNCTION)
        prompt = PromptTemplate("{name}")

        # act
        program = llm._get_program(_OutputModel, prompt)

        # assert
        assert isinstance(program, ToolOrchestratingLLM)

    def test_llm_mode_forces_text_completion(self) -> None:
        """
        Inputs: LLM mode regardless of metadata.
        Expected result: TextCompletionLLM is instantiated.
        Checks: returned instance type.
        """
        # arrange
        llm = _RecordingLLM(structured_output_mode=StructuredOutputMode.LLM)
        prompt = PromptTemplate("{name}")

        # act
        program = llm._get_program(_OutputModel, prompt)

        # assert
        assert isinstance(program, TextCompletionLLM)

    def test_unsupported_mode_raises(self) -> None:
        """
        Inputs: unsupported pydantic program mode value.
        Expected result: ValueError is raised.
        Checks: exception message includes mode string.
        """
        # arrange
        llm = _RecordingLLM(structured_output_mode=StructuredOutputMode.GUIDANCE)
        prompt = PromptTemplate("{name}")

        # act
        with pytest.raises(ValueError, match="Unsupported pydantic program mode"):
            llm._get_program(_OutputModel, prompt)


class TestStructuredPredict:
    def test_program_invoked_with_llm_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Inputs: fake program callable and llm_kwargs passed to structured_predict.
        Expected result: program is invoked with llm_kwargs and prompt args.
        Checks: returned model uses forwarded llm_kwargs.
        """
        # arrange
        llm = _RecordingLLM()
        prompt = PromptTemplate("{name}")

        def fake_program(llm_kwargs: dict | None = None, **kwargs: Any) -> _OutputModel:
            return _OutputModel(name=f"{kwargs['name']}:{llm_kwargs['temp']}")

        monkeypatch.setattr(llm, "_get_program", lambda *args, **kwargs: fake_program)

        # act
        result = llm.structured_predict(
            _OutputModel, prompt, llm_kwargs={"temp": 0.3}, name="ada"
        )

        # assert
        assert result.name == "ada:0.3"


class TestAStructuredPredict:
    @pytest.mark.asyncio
    async def test_program_acall_invoked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Inputs: fake program with async acall and llm_kwargs.
        Expected result: acall receives llm_kwargs and prompt args.
        Checks: returned model includes forwarded data.
        """
        # arrange
        llm = _RecordingLLM()
        prompt = PromptTemplate("{name}")

        class FakeProgram:
            async def acall(
                self, llm_kwargs: dict | None = None, **kwargs: Any
            ) -> _OutputModel:
                return _OutputModel(name=f"{kwargs['name']}:{llm_kwargs['seed']}")

        monkeypatch.setattr(llm, "_get_program", lambda *args, **kwargs: FakeProgram())

        # act
        result = await llm.astructured_predict(
            _OutputModel, prompt, llm_kwargs={"seed": 7}, name="ada"
        )

        # assert
        assert result.name == "ada:7"


class TestStreamStructuredPredict:
    def test_streams_values_from_program(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Inputs: fake program stream_call generator.
        Expected result: values streamed in order.
        Checks: collected values match generator output.
        """
        # arrange
        llm = _RecordingLLM()
        prompt = PromptTemplate("{name}")

        class FakeProgram:
            def stream_call(self, llm_kwargs: dict | None = None, **kwargs: Any):
                yield _OutputModel(name=kwargs["name"])
                yield _OutputModel(name=kwargs["name"].upper())

        monkeypatch.setattr(llm, "_get_program", lambda *args, **kwargs: FakeProgram())

        # act
        results = [
            item.name
            for item in llm.stream_structured_predict(_OutputModel, prompt, name="flow")
        ]

        # assert
        assert results == ["flow", "FLOW"]


class TestStructuredAstreamCall:
    @pytest.mark.asyncio
    async def test_returns_async_generator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Inputs: fake program returning async generator.
        Expected result: _structured_astream_call returns that generator.
        Checks: async iteration yields expected values.
        """
        # arrange
        llm = _RecordingLLM()
        prompt = PromptTemplate("{name}")

        class FakeProgram:
            async def astream_call(self, llm_kwargs: dict | None = None, **kwargs: Any):
                async def gen() -> AsyncGenerator[_OutputModel, None]:
                    yield _OutputModel(name=kwargs["name"])
                    yield _OutputModel(name=kwargs["name"].upper())

                return gen()

        monkeypatch.setattr(llm, "_get_program", lambda *args, **kwargs: FakeProgram())

        # act
        stream = await llm._structured_astream_call(_OutputModel, prompt, name="ada")
        collected = [item.name async for item in stream]

        # assert
        assert collected == ["ada", "ADA"]


class TestAstreamStructuredPredict:
    @pytest.mark.asyncio
    async def test_wraps_program_astream_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        Inputs: fake program with async stream results.
        Expected result: astream_structured_predict yields values from program.
        Checks: async iteration returns expected names.
        """
        # arrange
        llm = _RecordingLLM()
        prompt = PromptTemplate("{name}")

        class FakeProgram:
            async def astream_call(self, llm_kwargs: dict | None = None, **kwargs: Any):
                async def gen() -> AsyncGenerator[_OutputModel, None]:
                    yield _OutputModel(name=kwargs["name"])
                    yield _OutputModel(name=kwargs["name"].upper())

                return gen()

        monkeypatch.setattr(llm, "_get_program", lambda *args, **kwargs: FakeProgram())

        # act
        stream = await llm.astream_structured_predict(_OutputModel, prompt, name="eta")
        collected = [item.name async for item in stream]

        # assert
        assert collected == ["eta", "ETA"]


class TestPredict:
    def test_completion_path(self) -> None:
        """
        Inputs: prompt template for non-chat LLM.
        Expected result: completion response text is returned.
        Checks: prompt formatting and return value.
        """
        # arrange
        llm = _RecordingLLM()
        prompt = PromptTemplate("Hello {name}")

        # act
        result = llm.predict(prompt, name="Ada")

        # assert
        assert result == "Hello Ada"
        assert llm.last_complete[1] is True

    def test_chat_path(self) -> None:
        """
        Inputs: chat prompt template for chat LLM.
        Expected result: assistant message content returned.
        Checks: return value equals chat response content.
        """
        # arrange
        llm = _ChatLLM()
        prompt = ChatPromptTemplate.from_messages([("user", "ping")])

        # act
        result = llm.predict(prompt)

        # assert
        assert result == "pong"


class TestStream:
    def test_completion_stream(self) -> None:
        """
        Inputs: prompt template for non-chat LLM streaming.
        Expected result: token stream yields deltas.
        Checks: collected tokens equal expected deltas.
        """
        # arrange
        llm = _RecordingLLM()
        prompt = PromptTemplate("{verb}")

        # act
        tokens = list(llm.stream(prompt, verb="run"))

        # assert
        assert tokens == ["a", "b"]
        assert llm.last_stream_complete[1] is True

    def test_chat_stream(self) -> None:
        """
        Inputs: chat prompt template for chat LLM streaming.
        Expected result: token stream yields chat deltas.
        Checks: collected tokens equal expected deltas.
        """
        # arrange
        llm = _ChatLLM()
        prompt = ChatPromptTemplate.from_messages([("user", "ping")])

        # act
        tokens = list(llm.stream(prompt))

        # assert
        assert tokens == ["o", "k"]

    def test_output_parser_not_supported(self) -> None:
        """
        Inputs: prompt with output parser configured.
        Expected result: NotImplementedError is raised.
        Checks: error message matches unsupported streaming.
        """
        # arrange
        llm = _RecordingLLM()
        prompt = PromptTemplate("{word}", output_parser=_StripParser())

        # act
        with pytest.raises(NotImplementedError, match="Output parser is not supported"):
            list(llm.stream(prompt, word="hi"))


class TestAPredict:
    @pytest.mark.asyncio
    async def test_completion_path(self) -> None:
        """
        Inputs: prompt template for non-chat LLM.
        Expected result: async completion response text is returned.
        Checks: output equals reversed prompt per test double.
        """
        # arrange
        llm = _RecordingLLM()
        prompt = PromptTemplate("{word}")

        # act
        result = await llm.apredict(prompt, word="abc")

        # assert
        assert result == "cba"
        assert llm.last_acomplete[1] is True

    @pytest.mark.asyncio
    async def test_chat_path(self) -> None:
        """
        Inputs: chat prompt template for chat LLM.
        Expected result: assistant message content returned.
        Checks: output equals chat response content.
        """
        # arrange
        llm = _ChatLLM()
        prompt = ChatPromptTemplate.from_messages([("user", "ping")])

        # act
        result = await llm.apredict(prompt)

        # assert
        assert result == "pong"


class TestAstream:
    @pytest.mark.asyncio
    async def test_completion_stream(self) -> None:
        """
        Inputs: prompt template for non-chat async streaming.
        Expected result: token stream yields completion deltas.
        Checks: collected tokens equal expected values.
        """
        # arrange
        llm = _RecordingLLM()
        prompt = PromptTemplate("{verb}")

        # act
        stream = await llm.astream(prompt, verb="run")
        tokens = [token async for token in stream]

        # assert
        assert tokens == ["x", "y"]

    @pytest.mark.asyncio
    async def test_chat_stream(self) -> None:
        """
        Inputs: chat prompt template for chat async streaming.
        Expected result: token stream yields chat deltas.
        Checks: collected tokens equal expected deltas.
        """
        # arrange
        llm = _ChatLLM()
        prompt = ChatPromptTemplate.from_messages([("user", "ping")])

        # act
        stream = await llm.astream(prompt)
        tokens = [token async for token in stream]

        # assert
        assert tokens == ["o", "k"]

    @pytest.mark.asyncio
    async def test_output_parser_not_supported(self) -> None:
        """
        Inputs: LLM with output parser configured.
        Expected result: NotImplementedError is raised for streaming.
        Checks: error message matches unsupported streaming.
        """
        # arrange
        llm = _RecordingLLM(output_parser=_StripParser())
        prompt = PromptTemplate("{word}")

        # act
        with pytest.raises(NotImplementedError, match="Output parser is not supported"):
            await llm.astream(prompt, word="hi")


class TestAsStructuredLLM:
    def test_wraps_llm(self) -> None:
        """
        Inputs: output model class and base LLM.
        Expected result: StructuredOutputLLM wraps the LLM.
        Checks: wrapper.llm is the original LLM.
        """
        # arrange
        llm = _RecordingLLM()

        # act
        wrapper = llm.as_structured_llm(_OutputModel)

        # assert
        assert wrapper.llm is llm
