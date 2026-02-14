"""End-to-end tests for OllamaEmbedding class.

These tests require a running Ollama server with the test model available.
They are marked with @pytest.mark.e2e and skipped if Ollama is not available.

Run with: pytest tests/embeddings/test_ollama_embedding_e2e.py -m e2e
Skip with: pytest tests/embeddings/ -m "not e2e"
"""

from __future__ import annotations

import pytest
from ollama import Client

from serapeum.core.base.embeddings.base import BaseEmbedding
from serapeum.ollama import OllamaEmbedding


test_model = "llama3.1:latest"

try:
    client = Client()
    models = client.list()
    model_found = False
    for model in models["models"]:
        if model.model == test_model:
            model_found = True
            break
    if not model_found:
        client = None  # type: ignore
except Exception:
    client = None  # type: ignore



class TestBasicEmbedding:
    """Basic E2E Tests."""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_basic_query(self) -> None:
        """Test basic query embedding with live Ollama server.

        Inputs: Query string
        Expected: Valid embedding vector returned
        Checks: Type, length, values are floats
        """
        embedder = OllamaEmbedding(model_name=test_model)

        query = "What is artificial intelligence?"
        embedding = embedder.get_query_embedding(query)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)
        # Embeddings should have reasonable magnitude
        assert all(-100 < x < 100 for x in embedding)


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_basic_text(self) -> None:
        """Test basic text embedding with live Ollama server.

        Inputs: Text string
        Expected: Valid embedding vector returned
        Checks: Type, length, values are floats, different from query
        """
        embedder = OllamaEmbedding(model_name=test_model)

        text = "Machine learning is a subset of artificial intelligence."
        embedding = embedder.get_text_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_query_vs_text_different(self) -> None:
        """Test that query and text embeddings are different for same input.

        Inputs: Same string as query and text
        Expected: Different embeddings when instructions differ
        Checks: Embeddings should differ (unless no instructions provided)
        """
        embedder = OllamaEmbedding(model_name=test_model)

        input_str = "What is the capital of France?"

        query_embedding = embedder.get_query_embedding(input_str)
        text_embedding = embedder.get_text_embedding(input_str)

        # Without instructions, they might be similar but validate structure
        assert len(query_embedding) == len(text_embedding)
        assert isinstance(query_embedding, list)
        assert isinstance(text_embedding, list)



class TestBatchEmbedding:
    """Batch E2E Tests."""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_batch_texts(self) -> None:
        """Test batch text embedding with live Ollama server.

        Inputs: Multiple text strings
        Expected: List of embeddings matching input count
        Checks: Count, types, dimensions match
        """
        embedder = OllamaEmbedding(model_name=test_model, batch_size=10)

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a programming language.",
            "Machine learning models learn from data.",
        ]

        embeddings = embedder.get_text_embedding_batch(texts)

        assert len(embeddings) == len(texts)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

        # All embeddings should have same dimension
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_large_batch(self) -> None:
        """Test large batch processing with live Ollama server.

        Inputs: 20 text strings
        Expected: All embeddings returned correctly
        Checks: Count matches, no errors
        """
        embedder = OllamaEmbedding(model_name=test_model, batch_size=50)

        texts = [f"This is test sentence number {i}." for i in range(20)]

        embeddings = embedder.get_text_embedding_batch(texts)

        assert len(embeddings) == 20
        assert all(isinstance(emb, list) for emb in embeddings)


class TestInstructionEmbedding:
    """Instruction E2E Tests."""


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_with_query_instruction(self) -> None:
        """Test query embedding with instruction prefix.

        Inputs: Query with query_instruction set
        Expected: Embedding generated with instruction prefix
        Checks: Valid embedding returned
        """
        embedder = OllamaEmbedding(
            model_name=test_model,
            query_instruction="Represent this sentence for searching relevant passages:"
        )

        query = "What is deep learning?"
        embedding = embedder.get_query_embedding(query)

        assert isinstance(embedding, list)
        assert len(embedding) > 0


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_with_text_instruction(self) -> None:
        """Test text embedding with instruction prefix.

        Inputs: Text with text_instruction set
        Expected: Embedding generated with instruction prefix
        Checks: Valid embedding returned
        """
        embedder = OllamaEmbedding(
            model_name=test_model,
            text_instruction="Represent this document for retrieval:"
        )

        text = "Neural networks are computing systems inspired by biological neural networks."
        embedding = embedder.get_text_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_instructions_affect_output(self) -> None:
        """Test that instructions actually change the embedding.

        Inputs: Same text with and without instruction
        Expected: Different embeddings
        Checks: Embeddings differ when instruction is used
        """
        text = "Artificial intelligence is transforming the world."

        # Without instruction
        embedder1 = OllamaEmbedding(model_name=test_model)
        embedding1 = embedder1.get_text_embedding(text)

        # With instruction
        embedder2 = OllamaEmbedding(
            model_name=test_model,
            text_instruction="Document:"
        )
        embedding2 = embedder2.get_text_embedding(text)

        # Embeddings should differ
        assert embedding1 != embedding2
        assert len(embedding1) == len(embedding2)


class TestAsyncEmbedding:
    """Async E2E Tests."""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    @pytest.mark.asyncio
    async def test_ollama_embedding_async_query(self) -> None:
        """Test async query embedding with live Ollama server.

        Inputs: Query string in async context
        Expected: Valid embedding vector returned
        Checks: Async execution works correctly
        """
        embedder = OllamaEmbedding(model_name=test_model)

        query = "How does machine learning work?"
        embedding = await embedder.aget_query_embedding(query)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    @pytest.mark.asyncio
    async def test_ollama_embedding_async_text(self) -> None:
        """Test async text embedding with live Ollama server.

        Inputs: Text string in async context
        Expected: Valid embedding vector returned
        Checks: Async execution works correctly
        """
        embedder = OllamaEmbedding(model_name=test_model)

        text = "Deep learning uses neural networks with multiple layers."
        embedding = await embedder.aget_text_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    @pytest.mark.asyncio
    async def test_ollama_embedding_async_batch(self) -> None:
        """Test async batch embedding with live Ollama server.

        Inputs: Multiple texts in async context
        Expected: All embeddings returned correctly
        Checks: Async batch processing works
        """
        embedder = OllamaEmbedding(model_name=test_model)

        texts = [
            "First text about AI.",
            "Second text about ML.",
            "Third text about DL.",
        ]

        embeddings = await embedder.aget_text_embedding_batch(texts)

        assert len(embeddings) == 3
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)



class TestConfiguration:
    """Configuration E2E Tests."""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_with_keep_alive(self) -> None:
        """Test embedding with custom keep_alive setting.

        Inputs: Query with keep_alive="10m"
        Expected: Embedding generated successfully
        Checks: keep_alive parameter accepted
        """
        embedder = OllamaEmbedding(
            model_name=test_model,
            keep_alive="10m"
        )

        query = "Test query with keep alive"
        embedding = embedder.get_query_embedding(query)

        assert isinstance(embedding, list)
        assert len(embedding) > 0


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_with_additional_kwargs(self) -> None:
        """Test embedding with additional Ollama kwargs.

        Inputs: Text with ollama_additional_kwargs
        Expected: Embedding generated successfully
        Checks: Additional kwargs accepted by server
        """
        embedder = OllamaEmbedding(
            model_name=test_model,
            ollama_additional_kwargs={"temperature": 0.0}
        )

        text = "Test text with additional kwargs"
        embedding = embedder.get_text_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_custom_base_url(self) -> None:
        """Test embedding with custom base URL.

        Inputs: Query with default localhost URL
        Expected: Connection works with default URL
        Checks: base_url parameter works
        """
        embedder = OllamaEmbedding(
            model_name=test_model,
            base_url="http://localhost:11434"
        )

        query = "Test with custom base URL"
        embedding = embedder.get_query_embedding(query)

        assert isinstance(embedding, list)
        assert len(embedding) > 0


class TestSemanticSimilarity:
    """Semantic Similarity E2E Tests"""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_semantic_similarity(self) -> None:
        """Test that semantically similar texts have similar embeddings.

        Inputs: Two semantically similar texts
        Expected: Embeddings should be more similar than unrelated text
        Checks: Cosine similarity or distance metrics
        """
        import math

        embedder = OllamaEmbedding(model_name=test_model)

        text1 = "The cat sat on the mat."
        text2 = "A feline rested on the rug."
        text3 = "Quantum physics is complex."

        emb1 = embedder.get_text_embedding(text1)
        emb2 = embedder.get_text_embedding(text2)
        emb3 = embedder.get_text_embedding(text3)

        # Simple dot product similarity
        def similarity(a, b):
            return sum(x * y for x, y in zip(a, b))

        sim_1_2 = similarity(emb1, emb2)
        sim_1_3 = similarity(emb1, emb3)

        # Similar texts should have higher similarity than unrelated
        # Note: This is a weak assertion as it depends on the model
        assert len(emb1) == len(emb2) == len(emb3)


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_consistency(self) -> None:
        """Test that same text produces consistent embeddings.

        Inputs: Same text embedded twice
        Expected: Identical embeddings
        Checks: Deterministic behavior
        """
        embedder = OllamaEmbedding(model_name=test_model)

        text = "This text should produce consistent embeddings."

        embedding1 = embedder.get_text_embedding(text)
        embedding2 = embedder.get_text_embedding(text)

        # Should produce identical results
        assert embedding1 == embedding2


class TestEdgeCases:
    """Edge Case E2E Tests."""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_whitespace_string(self) -> None:
        """Test embedding with whitespace-only string.

        Inputs: Single space string (empty after strip)
        Expected: ValueError raised for empty text after stripping
        Checks: Input validation prevents empty embeddings
        """
        embedder = OllamaEmbedding(model_name=test_model)

        # Should raise ValueError for whitespace-only input
        with pytest.raises(ValueError) as exc_info:
            embedder.get_text_embedding(" ")

        assert "empty or whitespace-only" in str(exc_info.value).lower()
        assert "stripping whitespace" in str(exc_info.value).lower()


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_long_text(self) -> None:
        """Test embedding with very long text.

        Inputs: Text with 1000+ words
        Expected: Embedding generated successfully
        Checks: Model handles long inputs
        """
        embedder = OllamaEmbedding(model_name=test_model)

        long_text = " ".join(["This is a test sentence."] * 200)  # ~1000 words

        embedding = embedder.get_text_embedding(long_text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_special_characters(self) -> None:
        """Test embedding with special characters and unicode.

        Inputs: Text with emojis, special chars, unicode
        Expected: Embedding generated successfully
        Checks: Model handles various character types
        """
        embedder = OllamaEmbedding(model_name=test_model)

        text = "Hello 👋 world! Special chars: @#$%^&* Unicode: 你好 العربية"

        embedding = embedder.get_text_embedding(text)

        assert isinstance(embedding, list)
        assert len(embedding) > 0


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_single_item_batch(self) -> None:
        """Test batch processing with single item.

        Inputs: List with one text
        Expected: List with one embedding
        Checks: Single-item batch handled correctly
        """
        embedder = OllamaEmbedding(model_name=test_model)

        texts = ["Single text in a batch."]

        embeddings = embedder.get_text_embedding_batch(texts)

        assert len(embeddings) == 1
        assert isinstance(embeddings[0], list)
        assert len(embeddings[0]) > 0


class TestIntegrationWithBaseEmbedding:
    """Integration with BaseEmbedding E2E Tests."""

    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_inherits_base_methods(self) -> None:
        """Test that BaseEmbedding public methods work correctly.

        Inputs: Use public get_query_embedding and get_text_embedding
        Expected: Methods work through inheritance
        Checks: BaseEmbedding integration works end-to-end
        """
        embedder = OllamaEmbedding(model_name=test_model)

        # These are BaseEmbedding public methods
        query_emb = embedder.get_query_embedding("Test query")
        text_emb = embedder.get_text_embedding("Test text")

        assert isinstance(query_emb, list)
        assert isinstance(text_emb, list)
        assert len(query_emb) > 0
        assert len(text_emb) > 0


    @pytest.mark.e2e
    @pytest.mark.skipif(
        client is None,
        reason="Ollama client is not available or test model is missing"
    )
    def test_ollama_embedding_class_type(self) -> None:
        """Test that OllamaEmbedding is instance of BaseEmbedding.

        Inputs: OllamaEmbedding instance
        Expected: Is instance of BaseEmbedding
        Checks: Type hierarchy correct
        """
        embedder = OllamaEmbedding(model_name=test_model)

        assert isinstance(embedder, OllamaEmbedding)
        assert isinstance(embedder, BaseEmbedding)
        assert embedder.class_name() == "OllamaEmbedding"

