from serapeum.core.base.llms.models import Message, MessageRole, TextChunk, Image
from serapeum.core.base.llms.utils import MessageList


class TestMessagesToPrompt:
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
