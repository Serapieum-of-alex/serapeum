import base64
from io import BytesIO
from pathlib import Path
from unittest import mock
import pytest
import httpx

from serapeum.core.base.llms.models import (
    Message,
    ChatResponse,
    CompletionResponse,
    Image,
    MessageRole,
    TextChunk,
)
from pydantic import BaseModel
from pydantic import AnyUrl


@pytest.fixture()
def empty_bytes() -> bytes:
    return b""


@pytest.fixture()
def png_1px_b64() -> bytes:
    return b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


@pytest.fixture()
def png_1px(png_1px_b64) -> bytes:
    return base64.b64decode(png_1px_b64)


@pytest.fixture()
def pdf_url() -> str:
    return "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"


@pytest.fixture()
def mock_pdf_bytes(pdf_url) -> bytes:
    """
    Returns a byte string representing a very simple, minimal PDF file.
    """
    return httpx.get(pdf_url).content


@pytest.fixture()
def pdf_base64(mock_pdf_bytes) -> bytes:
    return base64.b64encode(mock_pdf_bytes)


@pytest.fixture()
def mp4_bytes() -> bytes:
    # Minimal fake MP4 header bytes (ftyp box)
    return b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"


@pytest.fixture()
def mp4_base64(mp4_bytes: bytes) -> bytes:
    return base64.b64encode(mp4_bytes)


class TestMessage:
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
                TextChunk(content="test content 2")
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
            additional_kwargs={"some_list": ["a", "b", "c"], "some_object": SimpleModel()},
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

    def test_image_block_resolve_image_data_url_base64(self, png_1px_b64: bytes, png_1px: bytes):
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
    message = Message("some content")
    cr = ChatResponse(message=message)
    assert str(cr) == str(message)


def test_completion_response():
    cr = CompletionResponse(text="some text")
    assert str(cr) == "some text"
