"""Integration tests for retry in the Ollama LLM class."""

from __future__ import annotations

from unittest.mock import MagicMock, AsyncMock

import ollama as ollama_sdk
import pytest

from serapeum.core.llms import Message, MessageRole, TextChunk
from serapeum.ollama import Ollama


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ollama(*, max_retries: int = 3, client: MagicMock | None = None) -> Ollama:
    """Create an Ollama instance with an injected mock client."""
    mock_client = client or MagicMock()
    return Ollama(
        model="test-model",
        client=mock_client,
        max_retries=max_retries,
    )


def _user_message() -> list[Message]:
    return [Message(role=MessageRole.USER, chunks=[TextChunk(content="hello")])]


def _chat_response_dict() -> dict:
    return {
        "message": {"role": "assistant", "content": "hi there", "tool_calls": []},
        "done": True,
        "model": "test-model",
        "prompt_eval_count": 5,
        "eval_count": 3,
    }


# ---------------------------------------------------------------------------
# Non-streaming chat
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChatRetry:
    def test_succeeds_first_try(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.return_value = _chat_response_dict()
        llm = _make_ollama(client=mock_client)

        response = llm.chat(_user_message())
        assert response.message.content == "hi there"
        assert mock_client.chat.call_count == 1

    def test_retries_on_connection_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.side_effect = [
            ConnectionError("refused"),
            ConnectionError("refused"),
            _chat_response_dict(),
        ]
        llm = _make_ollama(client=mock_client, max_retries=3)

        response = llm.chat(_user_message())
        assert response.message.content == "hi there"
        assert mock_client.chat.call_count == 3

    def test_no_retry_on_request_error(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.side_effect = ollama_sdk.RequestError("bad input")
        llm = _make_ollama(client=mock_client, max_retries=3)

        with pytest.raises(ollama_sdk.RequestError):
            llm.chat(_user_message())
        assert mock_client.chat.call_count == 1

    def test_max_retries_zero_disables_retry(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.side_effect = ConnectionError("refused")
        llm = _make_ollama(client=mock_client, max_retries=0)

        with pytest.raises(ConnectionError):
            llm.chat(_user_message())
        assert mock_client.chat.call_count == 1

    def test_max_retries_field_default(self) -> None:
        llm = _make_ollama()
        assert llm.max_retries == 3


# ---------------------------------------------------------------------------
# Async non-streaming chat
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAChatRetry:
    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self) -> None:
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            side_effect=[
                ConnectionError("refused"),
                _chat_response_dict(),
            ]
        )
        llm = Ollama(
            model="test-model",
            async_client=mock_client,
            max_retries=3,
        )

        response = await llm.achat(_user_message())
        assert response.message.content == "hi there"
        assert mock_client.chat.call_count == 2


# ---------------------------------------------------------------------------
# Streaming chat
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStreamChatRetry:
    def test_retries_on_mid_stream_failure(self) -> None:
        chunk1 = {
            "message": {"role": "assistant", "content": "partial", "tool_calls": []},
            "done": False,
            "model": "test-model",
        }
        chunk_success_1 = {
            "message": {"role": "assistant", "content": "hello ", "tool_calls": []},
            "done": False,
            "model": "test-model",
        }
        chunk_success_2 = {
            "message": {"role": "assistant", "content": "world", "tool_calls": []},
            "done": True,
            "model": "test-model",
            "prompt_eval_count": 5,
            "eval_count": 3,
        }

        # First call: yields one chunk then raises
        def fail_stream(**kwargs):
            yield chunk1
            raise ConnectionError("stream broken")

        # Second call: succeeds fully
        def success_stream(**kwargs):
            yield chunk_success_1
            yield chunk_success_2

        mock_client = MagicMock()
        mock_client.chat.side_effect = [fail_stream(), success_stream()]
        llm = _make_ollama(client=mock_client, max_retries=2)

        chunks = list(llm.chat(_user_message(), stream=True))
        # Partial chunk from the failed attempt is yielded before the retry
        # restarts, followed by the 2 chunks from the successful retry.
        assert len(chunks) == 3
        assert chunks[0].delta == "partial"
        assert chunks[1].delta == "hello "
        assert chunks[2].delta == "world"
        assert mock_client.chat.call_count == 2

    def test_stream_no_retry_on_request_error(self) -> None:
        def fail_stream(**kwargs):
            raise ollama_sdk.RequestError("bad")
            yield  # noqa: RET503 — make it a generator

        mock_client = MagicMock()
        mock_client.chat.return_value = fail_stream()
        llm = _make_ollama(client=mock_client, max_retries=3)

        with pytest.raises(ollama_sdk.RequestError):
            list(llm.chat(_user_message(), stream=True))
        assert mock_client.chat.call_count == 1


# ---------------------------------------------------------------------------
# Async streaming chat
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAStreamChatRetry:
    @pytest.mark.asyncio
    async def test_retries_on_mid_stream_failure(self) -> None:
        chunk_success = {
            "message": {"role": "assistant", "content": "ok", "tool_calls": []},
            "done": True,
            "model": "test-model",
            "prompt_eval_count": 5,
            "eval_count": 3,
        }

        class FailingAsyncIter:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise ConnectionError("stream broken")

        class SuccessAsyncIter:
            def __init__(self):
                self._yielded = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._yielded:
                    self._yielded = True
                    return chunk_success
                raise StopAsyncIteration

        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(
            side_effect=[FailingAsyncIter(), SuccessAsyncIter()]
        )
        llm = Ollama(
            model="test-model",
            async_client=mock_client,
            max_retries=2,
        )

        gen = await llm.achat(_user_message(), stream=True)
        chunks = [c async for c in gen]
        assert len(chunks) == 1
        assert chunks[0].message.content == "ok"
