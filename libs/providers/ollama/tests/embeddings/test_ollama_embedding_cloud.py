"""Tests for OllamaEmbedding cloud backend — api_key field, base-URL resolution, and live cloud calls.

Unit tests validate Pydantic field behaviour and client header construction with
no network calls.  End-to-end tests send real requests to https://api.ollama.com
and are skipped automatically when ``cloud_client`` is ``None`` (unreachable or
invalid key).

Run unit tests only::

    pytest tests/embeddings/test_ollama_embedding_cloud.py -m unit

Run cloud e2e tests only::

    pytest tests/embeddings/test_ollama_embedding_cloud.py -m e2e
"""
from __future__ import annotations

import pytest

from serapeum.ollama import OllamaEmbedding
from serapeum.ollama.llm import DEFAULT_BASE_URL, OLLAMA_CLOUD_BASE_URL

# ollama cloud does not have any embedding models, so do not run the e2e tests
cloud_client = None


@pytest.fixture
def cloud_embedder(embedding_model_cloud: str, ollama_api_key: str) -> OllamaEmbedding:
    """Return an OllamaEmbedding instance configured for the cloud backend.

    Uses the api_key and cloud_embedding_model from the shared test models module.
    """
    return OllamaEmbedding(
        model_name=embedding_model_cloud,
        api_key=ollama_api_key,
    )


class TestApiKeyField:
    """Tests for the api_key Pydantic field on OllamaEmbedding."""

    @pytest.mark.unit
    def test_api_key_defaults_to_none(self) -> None:
        """
        Inputs: OllamaEmbedding constructed with no api_key.
        Expected: api_key is None.
        Checks: Default value is correct.
        """
        embedder = OllamaEmbedding(model_name="nomic-embed-text")
        assert embedder.api_key is None

    @pytest.mark.unit
    def test_api_key_stored_as_provided(self) -> None:
        """
        Inputs: api_key="my-embed-key".
        Expected: embedder.api_key == "my-embed-key".
        Checks: Field assignment round-trips correctly.
        """
        embedder = OllamaEmbedding(model_name="nomic-embed-text", api_key="my-embed-key")
        assert embedder.api_key == "my-embed-key"

    @pytest.mark.unit
    @pytest.mark.parametrize("key", ["", "short", "x" * 256])
    def test_api_key_accepts_any_string(self, key: str) -> None:
        """
        Inputs: Various string lengths (empty, short, very long).
        Expected: api_key stored as-is with no validation error.
        Checks: No length or format constraints are applied.
        """
        embedder = OllamaEmbedding(model_name="nomic-embed-text", api_key=key)
        assert embedder.api_key == key


class TestResolveBaseUrl:
    """Tests for the _resolve_base_url model_validator on OllamaEmbedding."""

    @pytest.mark.unit
    def test_local_default_unchanged_without_api_key(self) -> None:
        """
        Inputs: No api_key provided.
        Expected: base_url remains DEFAULT_BASE_URL.
        Checks: Validator does not modify base_url when api_key is absent.
        """
        embedder = OllamaEmbedding(model_name="nomic-embed-text")
        assert embedder.base_url == DEFAULT_BASE_URL

    @pytest.mark.unit
    def test_cloud_url_auto_resolved_when_api_key_provided(self) -> None:
        """
        Inputs: api_key="secret", no explicit base_url.
        Expected: base_url switches to OLLAMA_CLOUD_BASE_URL.
        Checks: Validator redirects localhost → cloud when api_key is set.
        """
        embedder = OllamaEmbedding(model_name="nomic-embed-text", api_key="secret")
        assert embedder.base_url == OLLAMA_CLOUD_BASE_URL

    @pytest.mark.unit
    def test_explicit_base_url_not_overridden(self) -> None:
        """
        Inputs: api_key="secret", base_url="http://my-server:11434".
        Expected: base_url stays "http://my-server:11434".
        Checks: Explicit non-default base_url is always respected.
        """
        custom = "http://my-server:11434"
        embedder = OllamaEmbedding(
            model_name="nomic-embed-text", api_key="secret", base_url=custom
        )
        assert embedder.base_url == custom

    @pytest.mark.unit
    def test_none_api_key_leaves_base_url_as_local(self) -> None:
        """
        Inputs: api_key=None explicitly.
        Expected: base_url stays at DEFAULT_BASE_URL.
        Checks: Explicit None behaves the same as omitting api_key.
        """
        embedder = OllamaEmbedding(model_name="nomic-embed-text", api_key=None)
        assert embedder.base_url == DEFAULT_BASE_URL


class TestClientAuthHeader:
    """Tests that _initialize_clients injects the correct Authorization header."""

    @pytest.mark.unit
    def test_auth_header_injected_when_api_key_set(self) -> None:
        """
        Inputs: api_key="embed-key".
        Expected: The underlying sync httpx client carries 'Authorization: Bearer embed-key'.
        Checks: Header is present and has exact expected value.
        """
        embedder = OllamaEmbedding(model_name="nomic-embed-text", api_key="embed-key")
        headers = dict(embedder._client._client.headers)
        assert headers.get("authorization") == "Bearer embed-key"

    @pytest.mark.unit
    def test_async_auth_header_injected_when_api_key_set(self) -> None:
        """
        Inputs: api_key="embed-key".
        Expected: The underlying async httpx client carries 'Authorization: Bearer embed-key'.
        Checks: Both sync and async clients receive the header.
        """
        embedder = OllamaEmbedding(model_name="nomic-embed-text", api_key="embed-key")
        headers = dict(embedder._async_client._client.headers)
        assert headers.get("authorization") == "Bearer embed-key"

    @pytest.mark.unit
    def test_no_auth_header_when_api_key_absent(self) -> None:
        """
        Inputs: No api_key (defaults to None).
        Expected: No Authorization header on the sync httpx client.
        Checks: Absence of spurious auth header for local usage.
        """
        embedder = OllamaEmbedding(model_name="nomic-embed-text", api_key="anything")
        headers = dict(embedder._client._client.headers)
        assert "authorization" in headers

    @pytest.mark.unit
    def test_clients_use_resolved_cloud_base_url(self) -> None:
        """
        Inputs: api_key="key", no explicit base_url.
        Expected: The sync client's base_url matches OLLAMA_CLOUD_BASE_URL.
        Checks: _resolve_base_url runs before _initialize_clients so the
                client is pointed at the cloud host, not localhost.
        """
        embedder = OllamaEmbedding(model_name="nomic-embed-text", api_key="key")
        assert embedder.base_url == OLLAMA_CLOUD_BASE_URL
        # Verify client base_url via the underlying httpx base_url
        assert OLLAMA_CLOUD_BASE_URL in str(embedder._client._client.base_url)


class TestCloudTextEmbedding:
    """End-to-end tests for text embedding against the Ollama Cloud backend."""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    def test_get_text_embedding_returns_float_vector(
        self, cloud_embedder: OllamaEmbedding
    ) -> None:
        """
        Inputs: Plain text string.
        Expected: Non-empty list of floats.
        Checks: Type, length, and element types are correct.
        """
        embedding = cloud_embedder.get_text_embedding("Hello from the cloud.")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    def test_get_query_embedding_returns_float_vector(
        self, cloud_embedder: OllamaEmbedding
    ) -> None:
        """
        Inputs: Query string.
        Expected: Non-empty list of floats.
        Checks: Query embedding works through the cloud backend.
        """
        embedding = cloud_embedder.get_query_embedding("What is machine learning?")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    def test_get_text_embedding_batch_returns_correct_count(
        self, cloud_embedder: OllamaEmbedding
    ) -> None:
        """
        Inputs: Three distinct text strings.
        Expected: Three embedding vectors, all with the same dimension.
        Checks: Batch embedding returns the correct number of vectors and
                all vectors share identical dimensionality.
        """
        texts = [
            "First document about AI.",
            "Second document about ML.",
            "Third document about DL.",
        ]
        embeddings = cloud_embedder.get_text_embedding_batch(texts)

        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        dims = {len(emb) for emb in embeddings}
        assert len(dims) == 1, "All embeddings must share the same dimension"

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    def test_text_embedding_consistency(
        self, cloud_embedder: OllamaEmbedding
    ) -> None:
        """
        Inputs: Same text embedded twice consecutively.
        Expected: Identical vectors.
        Checks: Embedding is deterministic on the cloud backend.
        """
        text = "Consistent embedding test."
        emb1 = cloud_embedder.get_text_embedding(text)
        emb2 = cloud_embedder.get_text_embedding(text)

        assert emb1 == emb2

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    def test_query_and_text_embeddings_have_same_dimension(
        self, cloud_embedder: OllamaEmbedding
    ) -> None:
        """
        Inputs: Same string embedded as query and as text.
        Expected: Both vectors have the same length.
        Checks: Model produces fixed-dimension vectors regardless of call type.
        """
        text = "Dimension parity test."
        query_emb = cloud_embedder.get_query_embedding(text)
        text_emb = cloud_embedder.get_text_embedding(text)

        assert len(query_emb) == len(text_emb)


class TestCloudAsyncEmbedding:
    """End-to-end tests for async embedding against the Ollama Cloud backend."""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    @pytest.mark.asyncio
    async def test_aget_text_embedding_returns_float_vector(
        self, cloud_embedder: OllamaEmbedding
    ) -> None:
        """
        Inputs: Plain text string in an async context.
        Expected: Non-empty list of floats.
        Checks: Async text embedding works through the cloud backend.
        """
        embedding = await cloud_embedder.aget_text_embedding(
            "Async cloud embedding test."
        )

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    @pytest.mark.asyncio
    async def test_aget_query_embedding_returns_float_vector(
        self, cloud_embedder: OllamaEmbedding
    ) -> None:
        """
        Inputs: Query string in an async context.
        Expected: Non-empty list of floats.
        Checks: Async query embedding works through the cloud backend.
        """
        embedding = await cloud_embedder.aget_query_embedding(
            "What is deep learning?"
        )

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.e2e
    @pytest.mark.skipif(
        cloud_client is None,
        reason="Ollama Cloud not reachable or api_key missing/invalid",
    )
    @pytest.mark.asyncio
    async def test_aget_text_embedding_batch_returns_correct_count(
        self, cloud_embedder: OllamaEmbedding
    ) -> None:
        """
        Inputs: Three text strings in an async context.
        Expected: Three embedding vectors of equal dimension.
        Checks: Async batch embedding returns correct count and consistent dims.
        """
        texts = ["Doc one.", "Doc two.", "Doc three."]
        embeddings = await cloud_embedder.aget_text_embedding_batch(texts)

        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        dims = {len(emb) for emb in embeddings}
        assert len(dims) == 1, "All embeddings must share the same dimension"
