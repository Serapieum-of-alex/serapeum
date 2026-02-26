"""Tests for Ollama LLM cloud backend — api_key field, base-URL resolution, and live cloud calls.

Unit tests validate Pydantic field behaviour and client header construction with
no network calls.  End-to-end tests send real requests to https://api.ollama.com
and are skipped automatically when ``cloud_client`` is ``None`` (unreachable or
invalid key).

Run unit tests only::

    pytest tests/llm/test_ollama_cloud.py -m unit

Run cloud e2e tests only::

    pytest tests/llm/test_ollama_cloud.py -m e2e
"""

from __future__ import annotations

import pytest

from serapeum.core.llms import Message
from serapeum.ollama import Ollama
from serapeum.ollama.client import DEFAULT_BASE_URL, OLLAMA_CLOUD_BASE_URL
from ..models import cloud_client


class TestApiKeyField:
    """Tests for the api_key Pydantic field."""

    @pytest.mark.unit
    def test_api_key_defaults_to_none(self) -> None:
        """
        Inputs: Ollama constructed with no api_key.
        Expected: api_key is None.
        Checks: Default value is correct.
        """
        llm = Ollama(model="m")
        assert llm.api_key is None

    @pytest.mark.unit
    def test_api_key_stored_as_provided(self) -> None:
        """
        Inputs: api_key="my-secret-key".
        Expected: llm.api_key == "my-secret-key".
        Checks: Field assignment round-trips correctly.
        """
        llm = Ollama(model="m", api_key="my-secret-key")
        assert llm.api_key == "my-secret-key"

    @pytest.mark.unit
    @pytest.mark.parametrize("key", ["", "short", "x" * 256])
    def test_api_key_accepts_any_string(self, key: str) -> None:
        """
        Inputs: Various string lengths (empty, short, very long).
        Expected: api_key stored as-is with no validation error.
        Checks: No length or format constraints are applied.
        """
        llm = Ollama(model="m", api_key=key)
        assert llm.api_key == key


class TestResolveBaseUrl:
    """Tests for the _resolve_base_url model_validator."""

    @pytest.mark.unit
    def test_local_default_unchanged_without_api_key(self) -> None:
        """
        Inputs: No api_key provided.
        Expected: base_url remains http://localhost:11434.
        Checks: Validator does not modify base_url when api_key is absent.
        """
        llm = Ollama(model="m")
        assert llm.base_url == DEFAULT_BASE_URL

    @pytest.mark.unit
    def test_cloud_url_auto_resolved_when_api_key_provided(self) -> None:
        """
        Inputs: api_key="secret", no explicit base_url.
        Expected: base_url switches to OLLAMA_CLOUD_BASE_URL.
        Checks: Validator redirects localhost → cloud when api_key is set.
        """
        llm = Ollama(model="m", api_key="secret")
        assert llm.base_url == OLLAMA_CLOUD_BASE_URL

    @pytest.mark.unit
    def test_explicit_base_url_not_overridden(self) -> None:
        """
        Inputs: api_key="secret", base_url="http://my-server:11434".
        Expected: base_url stays "http://my-server:11434".
        Checks: Explicit non-default base_url is always respected.
        """
        custom = "http://my-server:11434"
        llm = Ollama(model="m", api_key="secret", base_url=custom)
        assert llm.base_url == custom

    @pytest.mark.unit
    def test_cloud_url_not_re_resolved_when_already_cloud(self) -> None:
        """
        Inputs: api_key="secret", base_url explicitly set to OLLAMA_CLOUD_BASE_URL.
        Expected: base_url unchanged (no double-resolution).
        Checks: Validator is idempotent when base_url is already the cloud URL.
        """
        llm = Ollama(model="m", api_key="secret", base_url=OLLAMA_CLOUD_BASE_URL)
        assert llm.base_url == OLLAMA_CLOUD_BASE_URL

    @pytest.mark.unit
    def test_none_api_key_leaves_base_url_as_local(self) -> None:
        """
        Inputs: api_key=None explicitly.
        Expected: base_url stays at DEFAULT_BASE_URL.
        Checks: Explicit None behaves the same as omitting api_key.
        """
        llm = Ollama(model="m", api_key=None)
        assert llm.base_url == DEFAULT_BASE_URL


class TestSyncClientAuthHeader:
    """Tests that the sync client property injects the correct Authorization header."""

    @pytest.mark.unit
    def test_auth_header_injected_when_api_key_set(self) -> None:
        """
        Inputs: api_key="test-key".
        Expected: The underlying httpx client carries 'Authorization: Bearer test-key'.
        Checks: Header is present and has exact expected value.
        """
        llm = Ollama(model="m", api_key="test-key")
        headers = dict(llm.client._client.headers)
        assert headers.get("authorization") == "Bearer test-key"

    @pytest.mark.unit
    def test_client_reused_on_second_access(self) -> None:
        """
        Inputs: llm.client accessed twice.
        Expected: Same object returned both times (lazy singleton).
        Checks: _client is not re-created on repeated access.
        """
        llm = Ollama(model="m", api_key="key")
        c1 = llm.client
        c2 = llm.client
        assert c1 is c2


class TestAsyncClientAuthHeader:
    """Tests that the async client injects the correct Authorization header."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_auth_header_injected_when_api_key_set(self) -> None:
        """
        Inputs: api_key="async-key".
        Expected: The underlying httpx async client carries 'Authorization: Bearer async-key'.
        Checks: Header is present and has exact expected value.
        """
        llm = Ollama(model="m", api_key="async-key")
        headers = dict(llm.async_client._client.headers)
        assert headers.get("authorization") == "Bearer async-key"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_async_no_auth_header_when_api_key_absent(self) -> None:
        """
        Inputs: No api_key.
        Expected: _build_client_kwargs does not include a headers entry.
        Checks: Our code does not inject Authorization when api_key is absent.
        Note: We test _build_client_kwargs directly because the underlying ollama SDK
        reads OLLAMA_API_KEY from the environment independently of our code.
        """
        llm = Ollama(model="m")
        assert "headers" not in llm._build_client_kwargs()


class TestCloudChat:
    """End-to-end tests for chat methods against the Ollama Cloud backend."""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    def test_chat_returns_non_empty_response(self, cloud_llm: Ollama) -> None:
        """
        Inputs: Single user message "Say hello.".
        Expected: Non-empty assistant content string.
        Checks: Response object is valid; message content is non-empty.
        """
        response = cloud_llm.chat([Message(role="user", content="Say hello.")])

        assert response is not None
        assert response.message.content is not None
        assert response.message.content.strip() != ""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    def test_complete_returns_non_empty_response(self, cloud_llm: Ollama) -> None:
        """
        Inputs: Plain text prompt "The capital of France is".
        Expected: Non-empty completion string.
        Checks: Text completion works through the cloud backend.
        """
        response = cloud_llm.complete("The capital of France is")

        assert response is not None
        assert str(response).strip() != ""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    def test_stream_chat_yields_chunks_with_deltas(self, cloud_llm: Ollama) -> None:
        """
        Inputs: Single user message "Count to 3.".
        Expected: At least one chunk; every chunk has a non-None delta; final
                  accumulated content is non-empty.
        Checks: Streaming protocol works correctly against the cloud backend.
        """
        chunks = list(
            cloud_llm.chat([Message(role="user", content="Count to 3.")], stream=True)
        )

        assert len(chunks) > 0
        assert all(c.delta is not None for c in chunks)
        assert chunks[-1].message.content.strip() != ""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    @pytest.mark.asyncio
    async def test_achat_returns_non_empty_response(self, cloud_llm: Ollama) -> None:
        """
        Inputs: Single user message "Say hello." via async interface.
        Expected: Non-empty assistant content string.
        Checks: Async chat works through the cloud backend.
        """
        response = await cloud_llm.achat([Message(role="user", content="Say hello.")])

        assert response is not None
        assert response.message.content is not None
        assert response.message.content.strip() != ""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    @pytest.mark.asyncio
    async def test_astream_chat_yields_chunks_with_deltas(
        self, cloud_llm: Ollama
    ) -> None:
        """
        Inputs: Single user message "Count to 3." via async streaming interface.
        Expected: At least one chunk; every chunk has a non-None delta; final
                  accumulated content is non-empty.
        Checks: Async streaming protocol works correctly against the cloud backend.
        """
        chunks = []
        async for chunk in await cloud_llm.achat(
            [Message(role="user", content="Count to 3.")], stream=True
        ):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert all(c.delta is not None for c in chunks)
        assert chunks[-1].message.content.strip() != ""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    @pytest.mark.asyncio
    async def test_acomplete_returns_non_empty_response(
        self, cloud_llm: Ollama
    ) -> None:
        """
        Inputs: Plain text prompt "The capital of France is" via async interface.
        Expected: Non-empty completion string.
        Checks: Async text completion works through the cloud backend.
        """
        response = await cloud_llm.acomplete("The capital of France is")

        assert response is not None
        assert str(response).strip() != ""
