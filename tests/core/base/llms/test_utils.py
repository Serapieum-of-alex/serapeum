import pytest

from serapeum.core.base.llms.models import ChatResponse
from serapeum.core.base.llms.utils import (
    stream_chat_response_to_completion_response,
    astream_chat_response_to_completion_response,
    chat_to_completion_decorator,
    stream_chat_to_completion_decorator,
    achat_to_completion_decorator,
    astream_chat_to_completion_decorator,
    get_from_param_or_env,
)

from serapeum.core.base.llms.models import Message, MessageRole, TextChunk, Image
from serapeum.core.base.llms.utils import MessageList


class TestMessageList:
    def test_happy_path_system_and_user(self):
        """
        Inputs: Two messages — system("You are a bot."), user("Hello").
        Expected: Lines formatted as "system: You are a bot." and "user: Hello", followed by a trailing "assistant: " line.
        Checks: Exact string equality; ordering preserved; single trailing assistant line; no trailing newline at end.
        """

        messages = [
            Message(role=MessageRole.SYSTEM, content="You are a bot."),
            Message(role=MessageRole.USER, content="Hello"),
        ]
        message_list = MessageList(messages)
        prompt = message_list.to_prompt()

        expected = "\n".join([
            "system: You are a bot.",
            "user: Hello",
            "assistant: ",
        ])
        assert prompt == expected
        assert not prompt.endswith("\n")

    def test_empty_messages_yields_assistant_only(self):
        """
        Inputs: Empty message sequence.
        Expected: String equals exactly "assistant: " (no preceding or trailing newlines).
        Checks: Deterministic minimal output; no errors.
        """
        messages: list[Message] = []
        message_list = MessageList(messages)
        prompt = message_list.to_prompt()

        assert prompt == "assistant: "

    def test_additional_kwargs_are_appended_on_new_line(self):
        """
        Inputs: One user message with content "Hi" and additional_kwargs {"tool": {"name": "calc"}}.
        Expected: Two lines for the message — first "user: Hi", then the dict repr on the next line; final line "assistant: ".
        Checks: Dict structure and ordering preserved in the string; overall line ordering correct.
        """
        msg = Message(role=MessageRole.USER, content="Hi", additional_kwargs={"tool": {"name": "calc"}})
        message_list = MessageList([msg])
        prompt = message_list.to_prompt()

        expected = "\n".join([
            "user: Hi",
            "{'tool': {'name': 'calc'}}",
            "assistant: ",
        ])
        assert prompt == expected

    def test_multiple_text_chunks_joined_with_newline(self):
        """
        Inputs: One user message with two TextChunks: "Line1" and "Line2".
        Expected: Content property becomes "Line1\nLine2" so the rendered line is "user: Line1\nLine2"; final line "assistant: ".
        Checks: Correct newline joining within a single message; no extra blank lines.
        """
        msg = Message(role=MessageRole.USER, content=[TextChunk(content="Line1"), TextChunk(content="Line2")])
        message_list = MessageList([msg])
        prompt = message_list.to_prompt()

        expected = "\n".join([
            "user: Line1\nLine2",
            "assistant: ",
        ])
        assert prompt == expected

    def test_non_text_chunk_results_in_none_content(self):
        """
        Inputs: One user message with a single Image chunk and no text chunks.
        Expected: Message.content is None; formatted line is "user: None"; final line "assistant: ".
        Checks: Graceful handling of non-text content without exceptions.
        """
        img = Image(content=b"\x89PNG", image_mimetype="image/png")
        msg = Message(role=MessageRole.USER, content=[img])
        message_list = MessageList([msg])
        prompt = message_list.to_prompt()

        expected = "\n".join([
            "user: None",
            "assistant: ",
        ])
        assert prompt == expected

    def test_ordering_is_preserved_and_trailing_assistant_always_added(self):
        """
        Inputs: Three messages in order: user("A"), assistant("B"), tool("C").
        Expected: Rendered in the same order with their roles and contents, followed by an extra trailing "assistant: " line.
        Checks: Ordering stability and presence of the final assistant prompt starter even when an assistant message exists in input.
        """

        messages = [
            Message(role=MessageRole.USER, content="A"),
            Message(role=MessageRole.ASSISTANT, content="B"),
            Message(role=MessageRole.TOOL, content="C"),
        ]
        message_list = MessageList(messages)
        prompt = message_list.to_prompt()

        expected = "\n".join([
            "user: A",
            "assistant: B",
            "tool: C",
            "assistant: ",
        ])
        assert prompt == expected


class TestMessageListBasics:
    def test_from_list_and_len_getitem_slice_and_append(self):
        """
        Inputs:
            - Start with two messages (system and user).
            - Use MessageList.from_list to construct, then test __len__, __getitem__ (int), slicing, and append.
        Expected:
            - Length reflects number of messages.
            - Integer indexing returns Message; slicing returns MessageList with correct subset.
            - Append adds to the end; iteration order preserved.
        Checks:
            - Types of returned objects; content and roles remain intact.
        """
        m1 = Message(role=MessageRole.SYSTEM, content="You are a bot.")
        m2 = Message(role=MessageRole.USER, content="Hello")
        ml = MessageList.from_list([m1, m2])

        # __len__ and __getitem__
        assert len(ml) == 2
        assert ml[0] is m1
        assert ml[1] is m2

        # slice returns MessageList
        sub = ml[0:1]
        assert isinstance(sub, MessageList)
        assert len(sub) == 1
        assert sub[0] is m1

        # append maintains order
        m3 = Message(role=MessageRole.ASSISTANT, content="Hi!")
        ml.append(m3)
        assert list(ml)[-1] is m3
        assert [m.role for m in ml] == [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT]

    def test_from_str_constructs_user_message(self):
        """
        Inputs: Use MessageList.from_str with prompt "Ping".
        Expected: Single Message with role=user and content="Ping".
        Checks: Role and content correct; to_prompt adds trailing assistant line.
        """
        ml = MessageList.from_str("Ping")
        msgs = list(ml)
        assert len(msgs) == 1
        assert msgs[0].role == MessageRole.USER
        assert msgs[0].content == "Ping"
        assert ml.to_prompt().splitlines() == ["user: Ping", "assistant: "]

    def test_filter_by_role(self):
        """
        Inputs: Mixed roles (system, user, assistant, tool).
        Expected: filter_by_role returns only messages of that role, as a MessageList.
        Checks: Type and ordering preserved; other roles excluded.
        """
        messages = [
            Message(role=MessageRole.SYSTEM, content="S"),
            Message(role=MessageRole.USER, content="U1"),
            Message(role=MessageRole.ASSISTANT, content="A"),
            Message(role=MessageRole.USER, content="U2"),
            Message(role=MessageRole.TOOL, content="T"),
        ]
        ml = MessageList(messages)
        only_users = ml.filter_by_role(MessageRole.USER)
        assert isinstance(only_users, MessageList)
        assert [m.content for m in only_users] == ["U1", "U2"]


class TestConversionFunctions:

    def test_stream_chat_response_to_completion_response(self):
        """
        Inputs: Generator of two ChatResponse items with different content and delta values.
        Expected: Generator yields corresponding CompletionResponse items mapping fields 1:1 (text, additional_kwargs, delta, raw).
        Checks: Order preserved; values mapped correctly.
        """
        def chat_gen():
            yield ChatResponse(message=Message(role=MessageRole.ASSISTANT, content="A"), delta="A", raw={"i": 0})
            yield ChatResponse(message=Message(role=MessageRole.ASSISTANT, content="B", additional_kwargs={"x": 2}), delta="B", raw={"i": 1})

        comp_gen = stream_chat_response_to_completion_response(chat_gen())
        out = list(comp_gen)
        assert [c.text for c in out] == ["A", "B"]
        assert [c.delta for c in out] == ["A", "B"]
        assert [c.raw for c in out] == [{"i": 0}, {"i": 1}]
        assert out[0].additional_kwargs == {}
        assert out[1].additional_kwargs == {"x": 2}

    @pytest.mark.asyncio
    async def test_astream_chat_response_to_completion_response(self):
        """
        Inputs: Async generator yielding two ChatResponse objects.
        Expected: Async generator of CompletionResponse with field mapping identical to sync version.
        Checks: Sequence and field values preserved.
        """
        async def agen():
            yield ChatResponse(message=Message(role=MessageRole.ASSISTANT, content="X"), delta="x", raw={"k": 0})
            yield ChatResponse(message=Message(role=MessageRole.ASSISTANT, content="Y", additional_kwargs={"a": 1}), delta="y", raw={"k": 1})

        comp_agen = astream_chat_response_to_completion_response(agen())
        results = []
        async for item in comp_agen:
            results.append(item)
        assert [c.text for c in results] == ["X", "Y"]
        assert [c.delta for c in results] == ["x", "y"]
        assert [c.raw for c in results] == [{"k": 0}, {"k": 1}]
        assert results[0].additional_kwargs == {}
        assert results[1].additional_kwargs == {"a": 1}


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
