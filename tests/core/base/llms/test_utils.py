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


