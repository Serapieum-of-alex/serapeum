import pytest

from serapeum.core.base.llms.models import ChatResponse
from serapeum.core.base.llms.utils import (
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    get_from_param_or_env,
)

from serapeum.core.base.llms.models import Message, MessageRole, TextChunk, Image


class TestDecorators:
    def test_chat_to_completion_decorator(self):
        """
        Inputs: Decorate a chat-style function that receives messages and returns a ChatResponse("OK"). Call wrapper with prompt string.
        Expected: Wrapper converts prompt to single user Message; returns CompletionResponse with text "OK".
        Checks: The inner function sees exactly one message with role=user and content matching prompt; output text mapped.
        """
        seen = {}

        def chat_impl(messages, **kwargs):
            # validate normalization
            assert isinstance(messages, list)
            assert len(messages) == 1
            assert isinstance(messages[0], Message)
            assert messages[0].role == MessageRole.USER
            assert messages[0].content == "Ping"
            seen["called"] = True
            return ChatResponse(message=Message(role=MessageRole.ASSISTANT, content="OK"))

        wrapped = chat_to_completion_decorator(chat_impl)
        out = wrapped("Ping")
        assert seen.get("called") is True
        assert out.text == "OK"

    def test_stream_chat_to_completion_decorator(self):
        """
        Inputs: Decorate a generator-based chat function yielding two ChatResponse chunks.
        Expected: Wrapper returns a generator yielding two CompletionResponse with the same deltas and text.
        Checks: Full mapping and order.
        """
        def chat_stream(messages, **kwargs):
            assert len(messages) == 1 and messages[0].content == "Start"
            yield ChatResponse(message=Message(role=MessageRole.ASSISTANT, content="S1"), delta="s1")
            yield ChatResponse(message=Message(role=MessageRole.ASSISTANT, content="S2"), delta="s2")

        wrapped = stream_chat_to_completion_decorator(chat_stream)
        out = list(wrapped("Start"))
        assert [c.text for c in out] == ["S1", "S2"]
        assert [c.delta for c in out] == ["s1", "s2"]

    @pytest.mark.asyncio
    async def test_achat_to_completion_decorator(self):
        """
        Inputs: Decorate an async chat function returning a single ChatResponse("OK"). Call with prompt string.
        Expected: Awaited wrapper returns CompletionResponse with text "OK".
        Checks: Input normalization and output mapping.
        """
        async def chat_async(messages, **kwargs):
            assert len(messages) == 1 and messages[0].content == "Hello"
            return ChatResponse(message=Message(role=MessageRole.ASSISTANT, content="OK"))

        wrapped = achat_to_completion_decorator(chat_async)
        out = await wrapped("Hello")
        assert out.text == "OK"

    @pytest.mark.asyncio
    async def test_astream_chat_to_completion_decorator(self):
        """
        Inputs: Decorate an async function that returns an async generator of ChatResponse.
        Expected: Awaited wrapper returns an async generator of CompletionResponse with correctly mapped fields.
        Checks: Order and values preserved through the wrapper.
        """
        async def chat_async_gen(messages, **kwargs):
            assert len(messages) == 1 and messages[0].content == "Go"

            async def inner():
                yield ChatResponse(message=Message(role=MessageRole.ASSISTANT, content="C1"), delta="d1")
                yield ChatResponse(message=Message(role=MessageRole.ASSISTANT, content="C2"), delta="d2")

            return inner()

        wrapped = astream_chat_to_completion_decorator(chat_async_gen)
        agen = await wrapped("Go")
        texts, deltas = [], []
        async for item in agen:
            texts.append(item.text)
            deltas.append(item.delta)
        assert texts == ["C1", "C2"]
        assert deltas == ["d1", "d2"]


class TestGetFromParamOrEnv:
    def test_param_takes_precedence_over_env_and_default(self, monkeypatch):
        """
        Inputs: param="VAL", env_key set in environment to another value, default="DEF".
        Expected: Function returns param value regardless of env/default.
        Checks: Exact equality "VAL".
        """
        monkeypatch.setenv("MY_KEY", "ENV_VAL")
        assert get_from_param_or_env("k", param="VAL", env_key="MY_KEY", default="DEF") == "VAL"

    def test_env_used_when_param_none_and_env_present(self, monkeypatch):
        """
        Inputs: param=None, env_key present with non-empty value, default set.
        Expected: Returns environment value.
        Checks: Exact string from environ.
        """
        monkeypatch.setenv("API_TOKEN", "token123")
        assert get_from_param_or_env("token", param=None, env_key="API_TOKEN", default="zzz") == "token123"

    def test_default_used_when_param_none_env_missing_or_empty(self, monkeypatch):
        """
        Inputs: param=None, env_key missing or empty, default provided.
        Expected: Returns default string.
        Checks: Exact match to provided default.
        """
        # ensure env var not present or empty
        monkeypatch.delenv("EMPTY_KEY", raising=False)
        assert get_from_param_or_env("x", param=None, env_key="EMPTY_KEY", default="D") == "D"
        monkeypatch.setenv("EMPTY_KEY", "")
        assert get_from_param_or_env("x", param=None, env_key="EMPTY_KEY", default="D2") == "D2"

    def test_raises_when_all_missing_with_message(self, monkeypatch):
        """
        Inputs: No param, no env value, no default.
        Expected: Raises ValueError with a message guiding to set env or pass param containing the key name and env key placeholder.
        Checks: Use regex to match both the key and env variable name in the error message.
        """
        monkeypatch.delenv("NOT_SET", raising=False)
        with pytest.raises(ValueError, match=r"Did not find secret,.*`NOT_SET`.*`secret`"):
            get_from_param_or_env("secret", param=None, env_key="NOT_SET", default=None)
