from typing import Any, List, Sequence

import pytest

from serapeum.core.llm.base import (
    LLM,
    astream_response_to_tokens,
    default_completion_to_prompt,
    stream_response_to_tokens,
)
from serapeum.core.base.llms.models import (
    ChatResponse,
    ChatResponseGen,
    ChatResponseAsyncGen,
    CompletionResponse,
    CompletionResponseGen,
    CompletionResponseAsyncGen,
    Message,
    MessageRole,
    Metadata,
)
from serapeum.core.prompts import PromptTemplate, ChatPromptTemplate
from serapeum.core.output_parsers.models import BaseOutputParser


# --------------------
# Test doubles & helpers
# --------------------


class UpperParser(BaseOutputParser):
    """Simple parser that uppercases text and messages.

    - parse: returns uppercased string
    - format: uppercases prompt string
    - format_messages: uppercases message contents (in-place)
    """

    def parse(self, output: str) -> str:
        return (output or "").upper()

    def format(self, query: str) -> str:
        return (query or "").upper()

    def format_messages(self, messages: List[Message]) -> List[Message]:
        for m in messages:
            if m.content is not None:
                m.content = m.content.upper()  # type: ignore
        return messages


class FailingParser(BaseOutputParser):
    """Parser that always raises during parse to test error propagation."""

    def parse(self, output: str) -> str:
        raise ValueError("invalid output")


class CompletionStubLLM(LLM):
    """LLM stub that implements completion endpoints (non-chat model)."""

    @property
    def metadata(self) -> Metadata:
        return Metadata.model_construct(is_chat_model=False)

    # -- sync
    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError()

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        # return uppercase to make transformations visible
        return CompletionResponse(text=prompt.upper(), delta=prompt.upper())

    def stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        raise NotImplementedError()

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        def gen() -> CompletionResponseGen:
            # yield delta pieces deterministically
            yield CompletionResponse(text=prompt, delta=prompt[:1])
            yield CompletionResponse(text=prompt, delta=prompt[1:])

        return gen()

    # -- async
    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        raise NotImplementedError()

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return CompletionResponse(text=prompt[::-1], delta=prompt[::-1])

    async def astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        raise NotImplementedError()

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        async def agen() -> CompletionResponseAsyncGen:
            yield CompletionResponse(text=prompt, delta=prompt[:1])
            yield CompletionResponse(text=prompt, delta=prompt[1:])

        return agen()


class ChatStubLLM(LLM):
    """LLM stub that implements chat endpoints (chat model)."""

    @property
    def metadata(self) -> Metadata:
        return Metadata.model_construct(is_chat_model=True)

    # -- sync
    def chat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=Message(content="pong", role=MessageRole.ASSISTANT))

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError()

    def stream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseGen:
        def gen() -> ChatResponseGen:
            yield ChatResponse(
                message=Message(content="ok", role=MessageRole.ASSISTANT), delta="o"
            )
            yield ChatResponse(
                message=Message(content="ok", role=MessageRole.ASSISTANT), delta="k"
            )

        return gen()

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError()

    # -- async
    async def achat(self, messages: Sequence[Message], **kwargs: Any) -> ChatResponse:
        return ChatResponse(message=Message(content="pong", role=MessageRole.ASSISTANT))

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        raise NotImplementedError()

    async def astream_chat(
        self, messages: Sequence[Message], **kwargs: Any
    ) -> ChatResponseAsyncGen:
        async def agen() -> ChatResponseAsyncGen:
            yield ChatResponse(
                message=Message(content="ok", role=MessageRole.ASSISTANT), delta="o"
            )
            yield ChatResponse(
                message=Message(content="ok", role=MessageRole.ASSISTANT), delta="k"
            )

        return agen()

    async def astream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseAsyncGen:
        raise NotImplementedError()


class TestStreamResponseToTokens:

    def test_nominal_and_empty_deltas_completion_response(self):
        """Inputs: two responses with non-empty deltas and then empty/None.
        Expected: tokens reflect exact delta values or empty strings for falsy deltas.
        Checks: list(stream_response_to_tokens(gen())) matches expected sequence.
        """

        def responses() -> CompletionResponseGen:
            yield CompletionResponse(text="Hello", delta="He")
            yield CompletionResponse(text="Hello", delta="llo")
            yield CompletionResponse(text="Hello", delta="")
            yield CompletionResponse(text="Hello", delta=None)

        tokens = list(stream_response_to_tokens(responses()))
        assert tokens == ["He", "llo", "", ""]

    def test_nominal_and_empty_deltas_chat_response(self):
        """Inputs: chat responses with deltas including empty and None.
        Expected: yielded tokens equal the deltas or empty strings when falsy.
        Checks: list(stream_response_to_tokens(gen())) equals expected.
        """

        def responses() -> ChatResponseGen:
            yield ChatResponse(
                message=Message(content="Hi", role=MessageRole.ASSISTANT), delta="H"
            )
            yield ChatResponse(
                message=Message(content="Hi", role=MessageRole.ASSISTANT), delta="i"
            )
            yield ChatResponse(
                message=Message(content="Hi", role=MessageRole.ASSISTANT), delta=""
            )
            yield ChatResponse(
                message=Message(content="Hi", role=MessageRole.ASSISTANT), delta=None
            )

        tokens = list(stream_response_to_tokens(responses()))
        assert tokens == ["H", "i", "", ""]


class TestAStreamResponseToTokens:
    @pytest.mark.asyncio
    async def test_async_nominal_and_empty_deltas_completion_response(self):
        """Inputs: async completion responses with normal and falsy deltas.
        Expected: async generator yields the same sequence of tokens, empty for falsy.
        Checks: collected list equals expected sequence.
        """

        async def responses() -> CompletionResponseAsyncGen:
            yield CompletionResponse(text="Hello", delta="He")
            yield CompletionResponse(text="Hello", delta="llo")
            yield CompletionResponse(text="Hello", delta="")
            yield CompletionResponse(text="Hello", delta=None)

        agen = await astream_response_to_tokens(responses())
        assert [t async for t in agen] == ["He", "llo", "", ""]

    @pytest.mark.asyncio
    async def test_async_nominal_and_empty_deltas_chat_response(self):
        """Inputs: async chat responses with deltas including empty and None.
        Expected: tokens are passed through as-is, empty for falsy values.
        Checks: collected list equals expected.
        """

        async def responses() -> ChatResponseAsyncGen:
            yield ChatResponse(
                message=Message(content="Hi", role=MessageRole.ASSISTANT), delta="H"
            )
            yield ChatResponse(
                message=Message(content="Hi", role=MessageRole.ASSISTANT), delta="i"
            )
            yield ChatResponse(
                message=Message(content="Hi", role=MessageRole.ASSISTANT), delta=""
            )
            yield ChatResponse(
                message=Message(content="Hi", role=MessageRole.ASSISTANT), delta=None
            )

        agen = await astream_response_to_tokens(responses())
        assert [t async for t in agen] == ["H", "i", "", ""]


class TestDefaultCompletionToPrompt:

    def test_identity_behavior(self):
        """Inputs: various strings including empty.
        Expected: identity function, returns the same exact string.
        Checks: equality for typical and boundary values.
        """
        assert default_completion_to_prompt("abc") == "abc"
        assert default_completion_to_prompt("") == ""


class TestValidators:

    def test_set_messages_to_prompt_defaults_and_custom(self):
        """Inputs: None or a custom callable.
        Expected: when None, default generic adapter returned; when custom, same callable returned.
        Checks: callable type and identity for custom.
        """
        defaulted = LLM.set_messages_to_prompt(None)
        assert callable(defaulted)

        def custom(messages: Sequence[Message]) -> str:
            return "|".join(m.content or "" for m in messages)

        assert LLM.set_messages_to_prompt(custom) is custom

    def test_set_completion_to_prompt_defaults_and_custom(self):
        """Inputs: None or custom callable.
        Expected: default fallback when None; preserve identity when custom provided.
        Checks: callable type and identity.
        """
        defaulted = LLM.set_completion_to_prompt(None)
        assert callable(defaulted)

        def custom(prompt: str) -> str:
            return "X:" + prompt

        assert LLM.set_completion_to_prompt(custom) is custom

    def test_check_prompts_sets_defaults_and_respects_customs(self):
        """Inputs: LLM instances with and without explicit adapters.
        Expected: missing adapters are populated; explicit adapters preserved.
        Checks: callables exist and custom adapter is used.
        """

        class Demo(CompletionStubLLM):
            pass

        # Defaults
        demo = Demo()
        assert callable(demo.messages_to_prompt)
        assert callable(demo.completion_to_prompt)

        # Customs
        def identity_messages(messages: Sequence[Message]) -> str:
            return " ".join(m.content or "" for m in messages)

        def upper_prompt(prompt: str) -> str:
            return prompt.upper()

        demo2 = Demo(
            messages_to_prompt=identity_messages, completion_to_prompt=upper_prompt
        )
        assert demo2.messages_to_prompt is identity_messages
        assert demo2.completion_to_prompt is upper_prompt


class TestGetPrompt:

    def test_get_prompt_without_parser(self):
        """Inputs: simple PromptTemplate with a variable.
        Expected: formatted string with variable substituted and no further changes.
        Checks: equals expected formatted output.
        """
        llm = CompletionStubLLM()
        out = llm._get_prompt(PromptTemplate("{subject} summary"), subject="Release")
        assert out == "Release summary"


#
#     @staticmethod
#     def test_get_prompt_with_output_parser_formatting():
#         """Inputs: LLM configured with an UpperParser.
#         Expected: formatted prompt is uppercased by parser.format and returned after _extend_prompt.
#         Checks: output equals the uppercase version.
#         """
#         llm = CompletionStubLLM(output_parser=UpperParser())
#         out = llm._get_prompt(PromptTemplate("summarize {item}"), item="notes")
#         assert out == "SUMMARIZE NOTES"
#
#
# class TestGetMessages:
#     @staticmethod
#     def test_get_messages_without_parser():
#         """Inputs: ChatPromptTemplate with one user message.
#         Expected: message content formatted with variables; no system message prepended by default.
#         Checks: first message content equals expected; list length is 1.
#         """
#         llm = ChatStubLLM()
#         msgs = llm._get_messages(ChatPromptTemplate.from_messages([("user", "Hello {name}!")]), name="Ada")
#         assert len(msgs) == 1
#         assert msgs[0].role == MessageRole.USER
#         assert msgs[0].content == "Hello Ada!"
#
#     @staticmethod
#     def test_get_messages_with_output_parser_formatting():
#         """Inputs: LLM with UpperParser.
#         Expected: formatted messages then passed to parser.format_messages, causing uppercase content.
#         Checks: message content uppercased.
#         """
#         llm = ChatStubLLM(output_parser=UpperParser())
#         msgs = llm._get_messages(ChatPromptTemplate.from_messages([("user", "Hello {name}!")]), name="Ada")
#         assert msgs[0].content == "HELLO ADA!"
#
#
# class TestExtendPromptAndMessages:
#     @staticmethod
#     def test_extend_prompt_with_system_and_wrapper():
#         """Inputs: system_prompt string and query_wrapper PromptTemplate.
#         Expected: system prompt prepended, wrapper applied around query string.
#         Checks: final string shows both decorations in order.
#         """
#         llm = CompletionStubLLM(system_prompt="You are an assistant.", query_wrapper_prompt=PromptTemplate("Question: {query_str}"))
#         result = llm._extend_prompt("List priorities")
#         # Wrapper is applied after the system prompt, wrapping the full text
#         assert result == "Question: You are an assistant.\n\nList priorities"
#
#     @staticmethod
#     def test_extend_messages_with_system_prompt():
#         """Inputs: message list and an LLM with system_prompt set.
#         Expected: a system message is prepended to the list.
#         Checks: first message has system role and expected content.
#         """
#         llm = ChatStubLLM(system_prompt="You are helpful.")
#         messages = [Message(content="Hi", role=MessageRole.USER)]
#         extended = llm._extend_messages(messages)
#         assert extended[0].role == MessageRole.SYSTEM
#         assert extended[0].content == "You are helpful."
#         assert extended[1].content == "Hi"
#
#
# # --------------------
# # Parsing output
# # --------------------
#
# class TestParseOutput:
#     @staticmethod
#     def test_parse_output_without_parser():
#         """Inputs: plain output string with no parser configured.
#         Expected: identical string returned.
#         Checks: equality.
#         """
#         llm = CompletionStubLLM()
#         assert llm._parse_output("ready") == "ready"
#
#     @staticmethod
#     def test_parse_output_with_parser_success():
#         """Inputs: LLM configured with UpperParser and mixed-case output.
#         Expected: parser.parse applied and stringified (already string), returning uppercase.
#         Checks: equality to uppercased output.
#         """
#         llm = CompletionStubLLM(output_parser=UpperParser())
#         assert llm._parse_output("Ok") == "OK"
#
#     @staticmethod
#     def test_parse_output_with_parser_failure_propagates():
#         """Inputs: LLM configured with parser that raises ValueError during parse.
#         Expected: the exception propagates unchanged.
#         Checks: pytest.raises(ValueError) context.
#         """
#         llm = CompletionStubLLM(output_parser=FailingParser())
#         with pytest.raises(ValueError, match="invalid output"):
#             llm._parse_output("oops")
#
#
# # --------------------
# # Predict and Stream (sync + async)
# # --------------------
#
# class TestPredictAndStreamCompletionMode:
#     @staticmethod
#     def test_predict_completion_path_parses_and_returns():
#         """Inputs: non-chat LLM and a simple PromptTemplate.
#         Expected: _get_prompt invoked and .complete called; output parsed and returned.
#         Checks: uppercase text from stubbed .complete.
#         """
#         llm = CompletionStubLLM()
#         out = llm.predict(PromptTemplate("{greet}"), greet="hi")
#         assert out == "HI"
#
#     @staticmethod
#     def test_stream_completion_tokens_and_parser_not_allowed():
#         """Inputs: non-chat LLM streaming with/without output parsers.
#         Expected: when no parser configured, yields tokens from stream_complete; when parser present on prompt or llm, NotImplementedError.
#         Checks: token list equals expected; error raised when parser configured.
#         """
#         llm = CompletionStubLLM()
#         tokens = list(llm.stream(PromptTemplate("{verb}"), verb="go"))
#         assert tokens == ["g", "o"]
#
#         # output parser on wrapper llm is not allowed for streaming
#         llm_with_parser = CompletionStubLLM(output_parser=UpperParser())
#         with pytest.raises(NotImplementedError, match="Output parser is not supported for streaming"):
#             list(llm_with_parser.stream(PromptTemplate("{x}"), x="z"))
#
#         # output parser on prompt is not allowed either
#         prompt_with_parser = PromptTemplate("{x}", output_parser=UpperParser())
#         with pytest.raises(NotImplementedError):
#             list(llm.stream(prompt_with_parser, x="z"))
#
#     @pytest.mark.asyncio
#     async def test_apredict_and_astream_completion(self):
#         """Inputs: non-chat LLM with async endpoints implemented.
#         Expected: apredict returns reversed string (per stub); astream yields deltas split in 2.
#         Checks: exact string and token list.
#         """
#         llm = CompletionStubLLM()
#         out = await llm.apredict(PromptTemplate("{w}"), w="abc")
#         assert out == "cba"
#
#         agen = await llm.astream(PromptTemplate("{v}"), v="go")
#         assert [t async for t in agen] == ["g", "o"]
#
#
# class TestPredictAndStreamChatMode:
#     @staticmethod
#     def test_predict_chat_path_returns_message_content():
#         """Inputs: chat LLM and ChatPromptTemplate.
#         Expected: predict formats messages and returns assistant content from .chat.
#         Checks: equals 'pong' per stub.
#         """
#         llm = ChatStubLLM()
#         prompt = ChatPromptTemplate.from_messages([("user", "ping")])
#         out = llm.predict(prompt)
#         assert out == "pong"
#
#     @staticmethod
#     def test_stream_chat_tokens_and_parser_not_allowed():
#         """Inputs: chat LLM streaming and parser configurations.
#         Expected: stream yields deltas ["o","k"]; configuring parsers on LLM or prompt raises NotImplementedError.
#         Checks: token list and error raising.
#         """
#         llm = ChatStubLLM()
#         prompt = ChatPromptTemplate.from_messages([("user", "ping")])
#         assert list(llm.stream(prompt)) == ["o", "k"]
#
#         with pytest.raises(NotImplementedError):
#             list(ChatStubLLM(output_parser=UpperParser()).stream(prompt))
#
#         with pytest.raises(NotImplementedError):
#             list(llm.stream(ChatPromptTemplate.from_messages([("user", "x")], output_parser=UpperParser())))
#
#     @pytest.mark.asyncio
#     async def test_apredict_and_astream_chat(self):
#         """Inputs: chat LLM and ChatPromptTemplate using async endpoints.
#         Expected: apredict returns 'pong'; astream yields ['o','k'].
#         Checks: equality and token sequence.
#         """
#         llm = ChatStubLLM()
#         prompt = ChatPromptTemplate.from_messages([("user", "ping")])
#         out = await llm.apredict(prompt)
#         assert out == "pong"
#
#         agen = await llm.astream(prompt)
#         assert [t async for t in agen] == ["o", "k"]
#
#
# # --------------------
# # Structured predictions
# # --------------------
#
# class TestStructuredPredict:
#     class Item(BaseModel):
#         value: str
#
#     class FakeProgram:
#         def __init__(self, suffix: str = "") -> None:
#             self.suffix = suffix
#
#         # sync call
#         def call(self, llm_kwargs: Optional[dict] = None, **kwargs: Any) -> BaseModel:
#             return TestStructuredPredict.Item(value=str(kwargs.get("name")) + self.suffix)
#
#         # async call
#         async def acall(self, llm_kwargs: Optional[dict] = None, **kwargs: Any) -> BaseModel:
#             return TestStructuredPredict.Item(value=str(kwargs.get("name")) + self.suffix)
#
#         # sync stream
#         def stream_call(self, llm_kwargs: Optional[dict] = None, **kwargs: Any):
#             yield TestStructuredPredict.Item(value=str(kwargs.get("name")))
#             yield TestStructuredPredict.Item(value=str(kwargs.get("name")).upper())
#
#         # async stream
#         async def astream_call(self, llm_kwargs: Optional[dict] = None, **kwargs: Any):
#             yield TestStructuredPredict.Item(value=str(kwargs.get("name")))
#             yield TestStructuredPredict.Item(value=str(kwargs.get("name")) + "!")
#
#     def _patch_program(self, monkeypatch, program):
#         monkeypatch.setattr(
#             "serapeum.core.structured_tools.utils.get_program_for_llm",
#             lambda output_cls, prompt, llm, pydantic_program_mode=None: program,
#         )
#
#     def test_structured_predict_sync(self, monkeypatch):
#         """Inputs: fake program with .call returning Item(name+suffix).
#         Expected: LLM.structured_predict returns a BaseModel of requested type from program.
#         Checks: returned model has expected value.
#         """
#         self._patch_program(monkeypatch, self.FakeProgram("!"))
#         llm = CompletionStubLLM()
#         item = llm.structured_predict(self.Item, PromptTemplate("{name}"), name="bob")
#         assert isinstance(item, BaseModel)
#         assert item.value == "bob!"
#
#     @pytest.mark.asyncio
#     async def test_astructured_predict_async(self, monkeypatch):
#         """Inputs: fake program with .acall.
#         Expected: astructured_predict awaits and returns BaseModel.
#         Checks: instance type and value.
#         """
#         self._patch_program(monkeypatch, self.FakeProgram("?"))
#         llm = CompletionStubLLM()
#         item = await llm.astructured_predict(self.Item, PromptTemplate("{name}"), name="zoe")
#         assert isinstance(item, BaseModel)
#         assert item.value == "zoe?"
#
#     def test_stream_structured_predict_sync(self, monkeypatch):
#         """Inputs: fake program with .stream_call yielding Items and lists.
#         Expected: stream_structured_predict yields BaseModel instances progressively.
#         Checks: sequence of yielded .value fields.
#         """
#         self._patch_program(monkeypatch, self.FakeProgram())
#         llm = CompletionStubLLM()
#         values = [part.value for part in llm.stream_structured_predict(self.Item, PromptTemplate("{name}"), name="sig")]
#         assert values == ["sig", "SIG"]
#
#     @pytest.mark.asyncio
#     async def test_astructured_stream_async(self, monkeypatch):
#         """Inputs: fake program with .astream_call yielding two items.
#         Expected: astream_structured_predict returns async gen yielding two models.
#         Checks: collected values match expected.
#         """
#         self._patch_program(monkeypatch, self.FakeProgram())
#         llm = CompletionStubLLM()
#         agen = await llm.astream_structured_predict(self.Item, PromptTemplate("{name}"), name="yo")
#         values = [m.value async for m in agen]
#         assert values == ["yo", "yo!"]
#
#     @pytest.mark.asyncio
#     async def test__structured_astream_call_private_helper(self, monkeypatch):
#         """Inputs: same fake program, exercising private helper.
#         Expected: helper returns async generator relaying program.astream_call.
#         Checks: collected values from helper stream.
#         """
#         self._patch_program(monkeypatch, self.FakeProgram())
#         llm = CompletionStubLLM()
#         agen = await llm._structured_astream_call(self.Item, PromptTemplate("{name}"), name="ok")
#         values = [m.value async for m in agen]
#         assert values == ["ok", "ok!"]
#
#
# # --------------------
# # Wrapper creation
# # --------------------
#
# class TestAsStructuredLLM:
#     @staticmethod
#     def test_wrapper_returns_configured_structured_llm():
#         """Inputs: a BaseModel class and optional kwargs.
#         Expected: wrapper.llm is the original LLM; output_cls set; kwargs passed through.
#         Checks: instance type and attribute wiring.
#         """
#         class Person(BaseModel):
#             name: str
#
#         llm = CompletionStubLLM()
#         wrapper = llm.as_structured_llm(Person, retries=2)
#         # We don't import the type directly (to avoid circular import in tests),
#         # but we can check core attributes that must be present.
#         assert wrapper.llm is llm
#         assert wrapper.output_cls is Person
#         assert getattr(wrapper, "retries", 2) == 2
