"""Unit tests for LLM models in serapeum-core."""

import base64
from io import BytesIO
from pathlib import Path
from unittest import mock

import httpx
import pytest
from pydantic import AnyUrl, BaseModel

from serapeum.core.base.llms.models import (
    ChatResponse,
    CompletionResponse,
    Image,
    Message,
    MessageList,
    MessageRole,
    TextChunk,
)


@pytest.fixture()
def empty_bytes() -> bytes:
    """Empty bytes."""
    return b""


@pytest.fixture()
def png_1px_b64() -> bytes:
    """Base64-encoded 1px PNG."""
    return b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture()
def png_1px(png_1px_b64) -> bytes:
    """Decoded 1px PNG bytes."""
    return base64.b64decode(png_1px_b64)


@pytest.fixture()
def pdf_url() -> str:
    """PDF test URL."""
    return "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


@pytest.fixture()
def mock_pdf_bytes(pdf_url) -> bytes:
    """Returns a byte string representing a very simple, minimal PDF file."""
    return httpx.get(pdf_url).content


@pytest.fixture()
def pdf_base64(mock_pdf_bytes) -> bytes:
    """Base64-encoded PDF bytes."""
    return base64.b64encode(mock_pdf_bytes)


@pytest.fixture()
def mp4_bytes() -> bytes:
    """Minimal MP4 header bytes."""
    # Minimal fake MP4 header bytes (ftyp box)
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


@pytest.fixture()
def mp4_base64(mp4_bytes: bytes) -> bytes:
    """Base64-encoded MP4 bytes."""
    return base64.b64encode(mp4_bytes)


class TestMessage:
    """Test Message."""

    def test_chat_message_from_str(self):
        m = Message.from_str(content="test content")
        assert m.content == "test content"
        assert len(m.chunks) == 1
        assert type(m.chunks[0]) is TextChunk
        assert m.chunks[0].content == "test content"

    def test_chat_message_content_legacy_get(self):
        m = Message(content="test content")
        assert m.content == "test content"
        assert len(m.chunks) == 1
        assert type(m.chunks[0]) is TextChunk
        assert m.chunks[0].content == "test content"

        m = Message(role="user", content="test content")
        assert m.role == "user"
        assert m.content == "test content"
        assert len(m.chunks) == 1
        assert isinstance(m.chunks[0], TextChunk)
        assert m.chunks[0].content == "test content"

        m = Message(
            chunks=[
                TextChunk(content="test content 1"),
                TextChunk(content="test content 2"),
            ]
        )
        assert m.content == "test content 1\ntest content 2"
        assert len(m.chunks) == 2
        assert all(isinstance(block, TextChunk) for block in m.chunks)

    def test_chat_message_content_legacy_set(self):
        m = Message()
        m.content = "test content"
        assert len(m.chunks) == 1
        assert type(m.chunks[0]) is TextChunk
        assert m.chunks[0].content == "test content"

        m = Message(content="some original content")
        m.content = "test content"
        assert len(m.chunks) == 1
        assert type(m.chunks[0]) is TextChunk
        assert m.chunks[0].content == "test content"

        m = Message(content=[TextChunk(content="test content"), Image()])
        with pytest.raises(ValueError):
            m.content = "test content"

    def test_chat_message_content_returns_empty_string(self):
        m = Message(content=[TextChunk(content="test content"), Image()])
        assert m.content == "test content"
        m = Message()
        assert m.content is None

    def test_chat_message__str__(self):
        assert str(Message(content="test content")) == "user: test content"

    def test_chat_message_serializer(self):
        class SimpleModel(BaseModel):
            some_field: str = ""

        m = Message(
            content="test content",
            additional_kwargs={
                "some_list": ["a", "b", "c"],
                "some_object": SimpleModel(),
            },
        )
        assert m.model_dump(exclude_none=True) == {
            "role": MessageRole.USER,
            "additional_kwargs": {
                "some_list": ["a", "b", "c"],
                "some_object": {"some_field": ""},
            },
            "chunks": [{"type": "text", "content": "test content"}],
        }

    def test_chat_message_legacy_roundtrip(self):
        legacy_message = {
            "role": MessageRole.USER,
            "content": "foo",
            "additional_kwargs": {},
        }
        m = Message(**legacy_message)
        assert m.model_dump(exclude_none=True) == {
            "additional_kwargs": {},
            "chunks": [{"type": "text", "content": "foo"}],
            "role": MessageRole.USER,
        }


class TestImageBlock:
    """Test ImageBlock."""

    def test_image_block_resolve_image(self, png_1px: bytes, png_1px_b64: bytes):
        b = Image(content=png_1px)

        img = b.resolve_image()
        assert isinstance(img, BytesIO)
        assert img.read() == png_1px

        img = b.resolve_image(as_base64=True)
        assert isinstance(img, BytesIO)
        assert img.read() == png_1px_b64

    def test_image_block_resolve_image_path(
        self, tmp_path: Path, png_1px_b64: bytes, png_1px: bytes
    ):
        png_path = tmp_path / "test.png"
        png_path.write_bytes(png_1px)

        b = Image(path=png_path)
        img = b.resolve_image()
        assert isinstance(img, BytesIO)
        assert img.read() == png_1px

        img = b.resolve_image(as_base64=True)
        assert isinstance(img, BytesIO)
        assert img.read() == png_1px_b64

    def test_image_block_resolve_image_url(self, png_1px_b64: bytes, png_1px: bytes):
        with mock.patch("serapeum.core.utils.base.requests") as mocked_req:
            url_str = "http://example.com"
            mocked_req.get.return_value = mock.MagicMock(content=png_1px)
            b = Image(url=AnyUrl(url=url_str))
            img = b.resolve_image()
            assert isinstance(img, BytesIO)
            assert img.read() == png_1px

            img = b.resolve_image(as_base64=True)
            assert isinstance(img, BytesIO)
            assert img.read() == png_1px_b64

    def test_image_block_resolve_image_data_url_base64(
        self, png_1px_b64: bytes, png_1px: bytes
    ):
        # Test data URL with base64 encoding
        data_url = f"data:image/png;base64,{png_1px_b64.decode('utf-8')}"
        b = Image(url=AnyUrl(url=data_url))

        img = b.resolve_image()
        assert isinstance(img, BytesIO)
        assert img.read() == png_1px

        img = b.resolve_image(as_base64=True)
        assert isinstance(img, BytesIO)
        assert img.read() == png_1px_b64

    def test_image_block_resolve_image_data_url_plain_text(self):
        # Test data URL with plain text (no base64)
        test_text = "Hello, World!"
        data_url = f"data:text/plain,{test_text}"
        b = Image(url=AnyUrl(url=data_url))

        img = b.resolve_image()
        assert isinstance(img, BytesIO)
        assert img.read() == test_text.encode("utf-8")

        img = b.resolve_image(as_base64=True)
        assert isinstance(img, BytesIO)
        assert img.read() == base64.b64encode(test_text.encode("utf-8"))

    def test_image_block_resolve_image_data_url_invalid(self):
        # Test invalid data URL format (missing comma)
        invalid_data_url = "data:image/png;base64"
        b = Image(url=AnyUrl(url=invalid_data_url))

        with pytest.raises(
            ValueError, match="Invalid data URL format: missing comma separator"
        ):
            b.resolve_image()

    def test_image_block_resolve_error(self):
        with pytest.raises(
            ValueError, match="No valid source provided to resolve binary data!"
        ):
            b = Image()
            b.resolve_image()

    def test_image_block_store_as_anyurl(self):
        url_str = "http://example.com"
        b = Image(url=url_str)
        assert b.url == AnyUrl(url=url_str)

    def test_image_block_store_as_base64(self, png_1px_b64: bytes, png_1px: bytes):
        # Store regular bytes
        assert Image(content=png_1px).content == png_1px_b64
        # Store already encoded data
        assert Image(content=png_1px_b64).content == png_1px_b64

    def test_legacy_image_additional_kwargs(self, png_1px_b64: bytes):
        msg = Message(chunks=[Image(content=png_1px_b64)])
        assert len(msg.chunks) == 1
        assert msg.chunks[0].content == png_1px_b64


def test_chat_response():
    """Test ChatResponse string representation."""
    message = Message("some content")
    cr = ChatResponse(message=message)
    assert str(cr) == str(message)


class TestChatResponseToCompletionResponse:
    """Test suite for ChatResponse to CompletionResponse conversion."""

    def test_to_completion_response(self):
        """Test conversion to CompletionResponse.

        Inputs: ChatResponse with message text "Hello" and raw payload. The Message carries additional_kwargs; the ChatResponse also has its own additional_kwargs.
        Expected: CompletionResponse.text == "Hello"; additional_kwargs are taken from the MESSAGE (not the response) per implementation; raw propagated.
        Checks: Exact equality for text; verify precedence/selection of additional_kwargs.
        """
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hello",
            additional_kwargs={"from": "message"},
        )
        cr = ChatResponse(
            message=msg, additional_kwargs={"from": "response"}, raw={"r": True}
        )
        out = cr.to_completion_response()
        assert out.text == "Hello"
        assert out.additional_kwargs == {"from": "message"}
        assert out.raw == {"r": True}

    def test_when_none_text(self):
        """Test conversion when message has no text.

        Inputs: ChatResponse whose message has no text chunks (only an Image), so message.content is None.
        Expected: CompletionResponse.text becomes empty string "".
        Checks: Exact empty string, not None.
        """
        img = Image(content=b"\x89PNG", image_mimetype="image/png")
        msg = Message(role=MessageRole.ASSISTANT, content=[img])
        assert msg.content is None  # guard
        cr = ChatResponse(message=msg)
        out = cr.to_completion_response()
        assert out.text == ""

    def test_stream_to_completion_response(self):
        """Test streaming conversion to CompletionResponse.

        Inputs: Generator of two ChatResponse items with different content and delta values.
        Expected: Generator yields corresponding CompletionResponse items mapping fields 1:1 (text, additional_kwargs, delta, raw).
        Checks: Order preserved; values mapped correctly.
        """

        def chat_gen():
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="A"),
                delta="A",
                raw={"i": 0},
            )
            yield ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT, content="B", additional_kwargs={"x": 2}
                ),
                delta="B",
                raw={"i": 1},
            )

        comp_gen = ChatResponse.stream_to_completion_response(chat_gen())
        out = list(comp_gen)
        assert [c.text for c in out] == ["A", "B"]
        assert [c.delta for c in out] == ["A", "B"]
        assert [c.raw for c in out] == [{"i": 0}, {"i": 1}]
        assert out[0].additional_kwargs == {}
        assert out[1].additional_kwargs == {"x": 2}

    @pytest.mark.asyncio
    async def test_astream_completion_response(self):
        """Test async streaming conversion to CompletionResponse.

        Inputs: Async generator yielding two ChatResponse objects.
        Expected: Async generator of CompletionResponse with field mapping identical to sync version.
        Checks: Sequence and field values preserved.
        """

        async def agen():
            yield ChatResponse(
                message=Message(role=MessageRole.ASSISTANT, content="X"),
                delta="x",
                raw={"k": 0},
            )
            yield ChatResponse(
                message=Message(
                    role=MessageRole.ASSISTANT, content="Y", additional_kwargs={"a": 1}
                ),
                delta="y",
                raw={"k": 1},
            )

        comp_agen = ChatResponse.astream_to_completion_response(agen())
        results = []
        async for item in comp_agen:
            results.append(item)
        assert [c.text for c in results] == ["X", "Y"]
        assert [c.delta for c in results] == ["x", "y"]
        assert [c.raw for c in results] == [{"k": 0}, {"k": 1}]
        assert results[0].additional_kwargs == {}
        assert results[1].additional_kwargs == {"a": 1}


def test_completion_response():
    """Test CompletionResponse string representation."""
    cr = CompletionResponse(text="some text")
    assert str(cr) == "some text"


class TestMessageLists:
    """Test suite for MessageList."""

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
            message_list = MessageList(messages=messages)
            prompt = message_list.to_prompt()

            expected = "\n".join(
                [
                    "system: You are a bot.",
                    "user: Hello",
                    "assistant: ",
                ]
            )
            assert prompt == expected
            assert not prompt.endswith("\n")

        def test_empty_messages_yields_assistant_only(self):
            """
            Inputs: Empty message sequence.
            Expected: String equals exactly "assistant: " (no preceding or trailing newlines).
            Checks: Deterministic minimal output; no errors.
            """
            messages: list[Message] = []
            message_list = MessageList(messages=messages)
            prompt = message_list.to_prompt()

            assert prompt == "assistant: "

        def test_additional_kwargs_are_appended_on_new_line(self):
            """
            Inputs: One user message with content "Hi" and additional_kwargs {"tool": {"name": "calc"}}.
            Expected: Two lines for the message — first "user: Hi", then the dict repr on the next line; final line "assistant: ".
            Checks: Dict structure and ordering preserved in the string; overall line ordering correct.
            """
            msg = Message(
                role=MessageRole.USER,
                content="Hi",
                additional_kwargs={"tool": {"name": "calc"}},
            )
            message_list = MessageList(messages=[msg])
            prompt = message_list.to_prompt()

            expected = "\n".join(
                [
                    "user: Hi",
                    "{'tool': {'name': 'calc'}}",
                    "assistant: ",
                ]
            )
            assert prompt == expected

        def test_multiple_text_chunks_joined_with_newline(self):
            """
            Inputs: One user message with two TextChunks: "Line1" and "Line2".
            Expected: Content property becomes "Line1\nLine2" so the rendered line is "user: Line1\nLine2"; final line "assistant: ".
            Checks: Correct newline joining within a single message; no extra blank lines.
            """
            msg = Message(
                role=MessageRole.USER,
                content=[TextChunk(content="Line1"), TextChunk(content="Line2")],
            )
            message_list = MessageList(messages=[msg])
            prompt = message_list.to_prompt()

            expected = "\n".join(
                [
                    "user: Line1\nLine2",
                    "assistant: ",
                ]
            )
            assert prompt == expected

        def test_non_text_chunk_results_in_none_content(self):
            """
            Inputs: One user message with a single Image chunk and no text chunks.
            Expected: Message.content is None; formatted line is "user: None"; final line "assistant: ".
            Checks: Graceful handling of non-text content without exceptions.
            """
            img = Image(content=b"\x89PNG", image_mimetype="image/png")
            msg = Message(role=MessageRole.USER, content=[img])
            message_list = MessageList(messages=[msg])
            prompt = message_list.to_prompt()

            expected = "\n".join(
                [
                    "user: None",
                    "assistant: ",
                ]
            )
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
            message_list = MessageList(messages=messages)
            prompt = message_list.to_prompt()

            expected = "\n".join(
                [
                    "user: A",
                    "assistant: B",
                    "tool: C",
                    "assistant: ",
                ]
            )
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
            assert [m.role for m in ml] == [
                MessageRole.SYSTEM,
                MessageRole.USER,
                MessageRole.ASSISTANT,
            ]

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
            ml = MessageList(messages=messages)
            only_users = ml.filter_by_role(MessageRole.USER)
            assert isinstance(only_users, MessageList)
            assert [m.content for m in only_users] == ["U1", "U2"]
