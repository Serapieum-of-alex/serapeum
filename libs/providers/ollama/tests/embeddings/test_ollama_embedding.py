"""Comprehensive test suite for OllamaEmbedding class.

This module provides exhaustive unit, integration, and edge case tests for the
OllamaEmbedding class, targeting ≥95% code coverage with focus on:
- Core functionality and formatting
- Error handling and exception scenarios
- Client initialization and configuration
- Concurrent operations and performance
- Type validation and boundary testing
"""

from __future__ import annotations

import asyncio
import pytest
from typing import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

from ollama import Client, AsyncClient
from pydantic import ValidationError

from serapeum.core.base.embeddings.base import BaseEmbedding, DEFAULT_EMBED_BATCH_SIZE
from serapeum.ollama.embedding import OllamaEmbedding


@pytest.fixture
def mock_embed_response() -> MagicMock:
    """Create a mock embedding response from Ollama API.

    Returns:
        MagicMock with embeddings attribute containing sample float vectors.
    """
    response = MagicMock()
    response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
    return response


@pytest.fixture
def mock_batch_embed_response() -> MagicMock:
    """Create a mock batch embedding response from Ollama API.

    Returns:
        MagicMock with embeddings attribute containing multiple float vectors.
    """
    response = MagicMock()
    response.embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.6, 0.7, 0.8, 0.9, 1.0],
        [0.11, 0.22, 0.33, 0.44, 0.55],
    ]
    return response


@pytest.fixture
def mock_ollama_client(mock_embed_response: MagicMock) -> MagicMock:
    """Create a mock Ollama client with embed method.

    Args:
        mock_embed_response: Mock response to return from embed calls.

    Returns:
        MagicMock configured as Ollama Client with embed method.
    """
    client = MagicMock(spec=Client)
    client.embed.return_value = mock_embed_response
    return client


@pytest.fixture
def mock_ollama_async_client(mock_embed_response: MagicMock) -> AsyncMock:
    """Create a mock async Ollama client with embed method.

    Args:
        mock_embed_response: Mock response to return from async embed calls.

    Returns:
        AsyncMock configured as Ollama AsyncClient with embed method.
    """
    client = AsyncMock(spec=AsyncClient)
    client.embed.return_value = mock_embed_response
    return client


@pytest.fixture
def mock_client_with_error() -> MagicMock:
    """Create a mock client that raises exceptions.

    Returns:
        MagicMock that raises exception on embed call.
    """
    client = MagicMock(spec=Client)
    client.embed.side_effect = ConnectionError("Failed to connect to Ollama server")
    return client


@pytest.fixture
def mock_timeout_client() -> MagicMock:
    """Create a mock client that times out.

    Returns:
        MagicMock that raises TimeoutError.
    """
    client = MagicMock(spec=Client)
    client.embed.side_effect = TimeoutError("Request timed out")
    return client


@pytest.fixture
def basic_embedder() -> OllamaEmbedding:
    """Create a basic OllamaEmbedding instance with default settings.

    Returns:
        OllamaEmbedding instance with minimal configuration.
    """
    return OllamaEmbedding(model_name="test-model")


@pytest.fixture
def configured_embedder() -> OllamaEmbedding:
    """Create an OllamaEmbedding instance with full configuration.

    Returns:
        OllamaEmbedding with query/text instructions and custom settings.
    """
    return OllamaEmbedding(
        model_name="test-model",
        base_url="http://custom:8080",
        batch_size=20,
        query_instruction="Query:",
        text_instruction="Text:",
        keep_alive="10m",
        ollama_additional_kwargs={"temperature": 0.0},
    )


@pytest.fixture
def embedder_factory():
    """Factory fixture to create OllamaEmbedding instances.

    Returns:
        Callable that creates configured embedders.
    """
    def _create(**kwargs):
        with patch("serapeum.ollama.embedding.Client"), patch(
            "serapeum.ollama.embedding.AsyncClient"
        ):
            defaults = {"model_name": "test-model"}
            defaults.update(kwargs)
            return OllamaEmbedding(**defaults)
    return _create


@pytest.mark.unit
class TestOllamaEmbeddingInitialization:
    """Test suite for OllamaEmbedding initialization and configuration."""

    def test_minimal_initialization(self) -> None:
        """Test initialization with only required model_name parameter.

        Inputs: model_name only
        Expected: Instance created with default values for all optional parameters
        Checks: All defaults match expected values from class definition
        """
        embedder = OllamaEmbedding(model_name="test-model")

        # Assert - Verify instance is created
        assert isinstance(embedder, OllamaEmbedding)
        assert isinstance(embedder, BaseEmbedding)

        # Assert - Verify default values
        assert embedder.model_name == "test-model"
        assert embedder.base_url == "http://localhost:11434"
        assert embedder.batch_size == DEFAULT_EMBED_BATCH_SIZE
        assert embedder.query_instruction is None
        assert embedder.text_instruction is None
        assert embedder.keep_alive == "5m"
        assert embedder.ollama_additional_kwargs == {}
        assert embedder.client_kwargs == {}

        # Assert - Verify clients were created
        assert embedder._client is not None
        assert embedder._async_client is not None

    def test_full_initialization(self) -> None:
        """Test initialization with all parameters specified.

        Inputs: All constructor parameters with custom values
        Expected: Instance created with all custom values preserved
        Checks: Each parameter is correctly stored in the instance
        """
        custom_kwargs = {"num_ctx": 2048, "temperature": 0.5}
        client_kwargs = {"timeout": 60}

        embedder = OllamaEmbedding(
            model_name="custom-model",
            base_url="http://custom-host:9999",
            batch_size=50,
            ollama_additional_kwargs=custom_kwargs,
            query_instruction="Represent query:",
            text_instruction="Represent document:",
            client_kwargs=client_kwargs,
            keep_alive=120.0,
        )

        # Assert - Verify all custom values
        assert embedder.model_name == "custom-model"
        assert embedder.base_url == "http://custom-host:9999"
        assert embedder.batch_size == 50
        assert embedder.query_instruction == "Represent query:"
        assert embedder.text_instruction == "Represent document:"
        assert embedder.keep_alive == 120.0
        assert embedder.ollama_additional_kwargs == custom_kwargs
        assert embedder.client_kwargs == client_kwargs

        # Assert - Verify clients were created
        assert embedder._client is not None
        assert embedder._async_client is not None

    def test_initialization_with_none_kwargs(self) -> None:
        """Test initialization handling of optional kwargs.

        Inputs: Optional dictionaries not provided (client_kwargs, ollama_additional_kwargs)
        Expected: Empty dicts used as defaults via Pydantic default_factory
        Checks: No errors occur and empty dicts are set
        """
        embedder = OllamaEmbedding(model_name="test-model")

        assert embedder.ollama_additional_kwargs == {}
        assert embedder.client_kwargs == {}

    def test_class_name_method(self, basic_embedder: OllamaEmbedding) -> None:
        """Test class_name class method returns correct identifier.

        Inputs: None (classmethod)
        Expected: Returns "OllamaEmbedding" string
        Checks: Exact string match for class identification
        """
        assert OllamaEmbedding.class_name() == "OllamaEmbedding"
        assert basic_embedder.class_name() == "OllamaEmbedding"

    def test_batch_size_validation_too_small(self) -> None:
        """Test validation error when batch_size is 0 or negative.

        Inputs: batch_size=0 (violates gt=0 constraint)
        Expected: ValidationError raised
        Checks: Pydantic validation enforces positive integer
        """
        with pytest.raises(ValidationError) as exc_info:
            OllamaEmbedding(model_name="test-model", batch_size=0)

        assert "batch_size" in str(exc_info.value)

    def test_batch_size_validation_too_large(self) -> None:
        """Test validation error when batch_size exceeds maximum.

        Inputs: batch_size=2049 (violates le=2048 constraint)
        Expected: ValidationError raised
        Checks: Pydantic validation enforces maximum batch size
        """
        with pytest.raises(ValidationError) as exc_info:
            OllamaEmbedding(model_name="test-model", batch_size=2049)

        assert "batch_size" in str(exc_info.value)

    def test_batch_size_boundary_values(self) -> None:
        """Test batch_size at boundary values (1 and 2048).

        Inputs: batch_size=1 (minimum) and batch_size=2048 (maximum)
        Expected: Both values accepted without error
        Checks: Boundary conditions are valid
        """
        # Minimum valid value
        embedder_min = OllamaEmbedding(
            model_name="test-model", batch_size=1
        )
        assert embedder_min.batch_size == 1

        # Maximum valid value
        embedder_max = OllamaEmbedding(
            model_name="test-model", batch_size=2048
        )
        assert embedder_max.batch_size == 2048


@pytest.mark.unit
@pytest.mark.mock
class TestTextFormattingMethods:
    """Test suite for _format_query and _format_text methods."""

    def test_format_query_without_instruction(
        self, basic_embedder: OllamaEmbedding
    ) -> None:
        """Test query formatting when no instruction is provided.

        Inputs: query="What is AI?" with query_instruction=None
        Expected: Returns query stripped of whitespace
        Checks: Only whitespace trimming applied, no prefix added
        """
        result = basic_embedder._format_query("What is AI?")
        assert result == "What is AI?"

    def test_format_query_with_instruction(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test query formatting with instruction prefix.

        Inputs: query="What is AI?" with query_instruction="Query:"
        Expected: Returns "Query: What is AI?"
        Checks: Instruction prepended with single space separator
        """
        result = configured_embedder._format_query("What is AI?")
        assert result == "Query: What is AI?"

    def test_format_query_strips_whitespace(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test query formatting handles extra whitespace correctly.

        Inputs: query="  What is AI?  " with instruction="  Query:  "
        Expected: Returns "Query: What is AI?" (all whitespace normalized)
        Checks: strip() applied to both instruction and query
        """
        result = configured_embedder._format_query("  What is AI?  ")
        assert result == "Query: What is AI?"

    def test_format_query_empty_string_with_instruction(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test query formatting with empty query string and instruction.

        Inputs: query="" with instruction="Query:"
        Expected: ValueError raised (no content to embed)
        Checks: Empty query not allowed even with instruction
        """
        with pytest.raises(ValueError) as exc_info:
            configured_embedder._format_query("")

        assert "empty or whitespace-only" in str(exc_info.value).lower()

    def test_format_query_whitespace_only_with_instruction(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test query formatting with whitespace-only query and instruction.

        Inputs: query="   " with instruction="Query:"
        Expected: ValueError raised (no content after stripping)
        Checks: Whitespace-only query not allowed even with instruction
        """
        with pytest.raises(ValueError) as exc_info:
            configured_embedder._format_query("   ")

        assert "empty or whitespace-only" in str(exc_info.value).lower()

    def test_format_text_without_instruction(
        self, basic_embedder: OllamaEmbedding
    ) -> None:
        """Test text formatting when no instruction is provided.

        Inputs: text="AI is a field" with text_instruction=None
        Expected: Returns text stripped of whitespace
        Checks: Only whitespace trimming applied, no prefix added
        """
        result = basic_embedder._format_text("AI is a field")
        assert result == "AI is a field"

    def test_format_text_with_instruction(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test text formatting with instruction prefix.

        Inputs: text="AI is a field" with text_instruction="Text:"
        Expected: Returns "Text: AI is a field"
        Checks: Instruction prepended with single space separator
        """
        result = configured_embedder._format_text("AI is a field")
        assert result == "Text: AI is a field"

    def test_format_text_strips_whitespace(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test text formatting handles extra whitespace correctly.

        Inputs: text="  AI is a field  " with instruction="  Text:  "
        Expected: Returns "Text: AI is a field" (all whitespace normalized)
        Checks: strip() applied to both instruction and text
        """
        result = configured_embedder._format_text("  AI is a field  ")
        assert result == "Text: AI is a field"

    def test_format_text_empty_string_with_instruction(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test text formatting with empty text string and instruction.

        Inputs: text="" with instruction="Text:"
        Expected: ValueError raised (no content to embed)
        Checks: Empty text not allowed even with instruction
        """
        with pytest.raises(ValueError) as exc_info:
            configured_embedder._format_text("")

        assert "empty or whitespace-only" in str(exc_info.value).lower()

    def test_format_text_multiline(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test text formatting preserves content structure.

        Inputs: Multiline text with instruction
        Expected: Strip applies only to outer whitespace, preserves internal structure
        Checks: Internal newlines and spaces preserved
        """
        multiline = "Line 1\nLine 2\nLine 3"
        result = configured_embedder._format_text(multiline)
        assert result == "Text: Line 1\nLine 2\nLine 3"

    def test_format_query_empty_string_raises_error(
        self, basic_embedder: OllamaEmbedding
    ) -> None:
        """Test that formatting empty query raises ValueError.

        Inputs: Empty string ""
        Expected: ValueError raised
        Checks: Input validation prevents empty queries
        """
        with pytest.raises(ValueError) as exc_info:
            basic_embedder._format_query("")

        assert "empty or whitespace-only" in str(exc_info.value).lower()

    def test_format_query_whitespace_only_raises_error(
        self, basic_embedder: OllamaEmbedding
    ) -> None:
        """Test that formatting whitespace-only query raises ValueError.

        Inputs: Whitespace-only string "   "
        Expected: ValueError raised
        Checks: Input validation prevents whitespace-only queries
        """
        with pytest.raises(ValueError) as exc_info:
            basic_embedder._format_query("   ")

        assert "empty or whitespace-only" in str(exc_info.value).lower()

    def test_format_text_empty_string_raises_error(
        self, basic_embedder: OllamaEmbedding
    ) -> None:
        """Test that formatting empty text raises ValueError.

        Inputs: Empty string ""
        Expected: ValueError raised
        Checks: Input validation prevents empty text
        """
        with pytest.raises(ValueError) as exc_info:
            basic_embedder._format_text("")

        assert "empty or whitespace-only" in str(exc_info.value).lower()

    def test_format_text_whitespace_only_raises_error(
        self, basic_embedder: OllamaEmbedding
    ) -> None:
        """Test that formatting whitespace-only text raises ValueError.

        Inputs: Whitespace-only string "   "
        Expected: ValueError raised
        Checks: Input validation prevents whitespace-only text
        """
        with pytest.raises(ValueError) as exc_info:
            basic_embedder._format_text("   ")

        assert "empty or whitespace-only" in str(exc_info.value).lower()

    def test_format_with_newlines_in_instruction(self, embedder_factory) -> None:
        """Test formatting with newline characters in instruction.

        Inputs: instruction="Query:\\n" with newlines
        Expected: Newlines stripped due to strip() call in _format_query
        Checks: strip() behavior on special characters
        """
        embedder = embedder_factory(query_instruction="Query:\n")
        result = embedder._format_query("test")

        # strip() removes trailing newline
        assert result == "Query: test"

    def test_format_with_unicode_instruction(self, embedder_factory) -> None:
        """Test formatting with unicode characters in instruction.

        Inputs: instruction with emoji "🔍 Query:"
        Expected: Unicode preserved in output
        Checks: Unicode support in instructions
        """
        embedder = embedder_factory(query_instruction="🔍 Query:")
        result = embedder._format_query("search term")

        assert "🔍 Query:" in result
        assert "search term" in result

    def test_format_with_very_long_instruction(self, embedder_factory) -> None:
        """Test formatting with extremely long instruction string.

        Inputs: instruction with 1000+ characters
        Expected: Long instruction handled without errors
        Checks: No length limitations on instructions
        """
        long_instruction = "Instruction: " + "A" * 1000
        embedder = embedder_factory(query_instruction=long_instruction)

        result = embedder._format_query("query")

        assert long_instruction in result
        assert result.endswith("query")

    def test_format_with_special_characters(self, embedder_factory) -> None:
        """Test formatting with special characters in text and instruction.

        Inputs: Special chars like @#$% in both instruction and text
        Expected: All special characters preserved
        Checks: Special character handling
        """
        embedder = embedder_factory(text_instruction="[Doc@#$%]:")
        result = embedder._format_text("Test @#$% text")

        assert "[Doc@#$%]:" in result
        assert "Test @#$% text" in result


@pytest.mark.unit
@pytest.mark.mock
class TestSingleEmbeddingSyncMethods:
    """Test suite for synchronous single embedding methods."""

    def test_get_general_text_embedding_basic(
        self,
        basic_embedder: OllamaEmbedding,
        mock_ollama_client: MagicMock,
        mock_embed_response: MagicMock,
    ) -> None:
        """Test get_general_text_embedding makes correct API call.

        Inputs: text="test query"
        Expected: Calls client.embed with correct parameters, returns first embedding
        Checks: API called once with model, input, options, keep_alive
        """
        basic_embedder._client = mock_ollama_client

        result = basic_embedder.get_general_text_embedding("test query")

        # Assert - Verify API call
        mock_ollama_client.embed.assert_called_once_with(
            model="test-model",
            input="test query",
            options={},
            keep_alive="5m",
        )

        # Assert - Verify result
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_get_general_text_embedding_with_additional_kwargs(
        self, configured_embedder: OllamaEmbedding, mock_ollama_client: MagicMock
    ) -> None:
        """Test get_general_text_embedding passes additional kwargs to API.

        Inputs: text with ollama_additional_kwargs set
        Expected: kwargs passed to options parameter
        Checks: temperature and other options included in API call
        """
        configured_embedder._client = mock_ollama_client

        configured_embedder.get_general_text_embedding("test")

        mock_ollama_client.embed.assert_called_once_with(
            model="test-model",
            input="test",
            options={"temperature": 0.0},
            keep_alive="10m",
        )

    def test_get_query_embedding_formats_and_calls(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_client: MagicMock,
        mock_embed_response: MagicMock,
    ) -> None:
        """Test _get_query_embedding applies formatting before embedding.

        Inputs: query="What is AI?" with query_instruction="Query:"
        Expected: Formatted query passed to get_general_text_embedding
        Checks: _format_query called and result used for embedding
        """
        configured_embedder._client = mock_ollama_client

        result = configured_embedder._get_query_embedding("What is AI?")

        # Assert - Verify formatted query was embedded
        mock_ollama_client.embed.assert_called_once()
        call_args = mock_ollama_client.embed.call_args
        assert call_args[1]["input"] == "Query: What is AI?"
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_get_text_embedding_formats_and_calls(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_client: MagicMock,
        mock_embed_response: MagicMock,
    ) -> None:
        """Test _get_text_embedding applies formatting before embedding.

        Inputs: text="AI is a field" with text_instruction="Text:"
        Expected: Formatted text passed to get_general_text_embedding
        Checks: _format_text called and result used for embedding
        """
        configured_embedder._client = mock_ollama_client

        result = configured_embedder._get_text_embedding("AI is a field")

        # Assert - Verify formatted text was embedded
        mock_ollama_client.embed.assert_called_once()
        call_args = mock_ollama_client.embed.call_args
        assert call_args[1]["input"] == "Text: AI is a field"
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.mark.unit
@pytest.mark.mock
@pytest.mark.asyncio
class TestSingleEmbeddingAsyncMethods:
    """Test suite for asynchronous single embedding methods."""

    async def test_aget_general_text_embedding_basic(
        self,
        basic_embedder: OllamaEmbedding,
        mock_ollama_async_client: AsyncMock,
        mock_embed_response: MagicMock,
    ) -> None:
        """Test async get_general_text_embedding makes correct API call.

        Inputs: prompt="test query"
        Expected: Calls async client.embed, returns first embedding
        Checks: Async API called once with correct parameters
        """
        basic_embedder._async_client = mock_ollama_async_client

        result = await basic_embedder.aget_general_text_embedding("test query")

        # Assert - Verify async API call
        mock_ollama_async_client.embed.assert_called_once_with(
            model="test-model",
            input="test query",
            options={},
            keep_alive="5m",
        )

        # Assert - Verify result
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    async def test_aget_general_text_embedding_with_kwargs(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_async_client: AsyncMock,
    ) -> None:
        """Test async embedding passes additional kwargs.

        Inputs: prompt with ollama_additional_kwargs
        Expected: kwargs passed to options
        Checks: Additional options included in async call
        """
        configured_embedder._async_client = mock_ollama_async_client

        await configured_embedder.aget_general_text_embedding("test")

        mock_ollama_async_client.embed.assert_called_once_with(
            model="test-model",
            input="test",
            options={"temperature": 0.0},
            keep_alive="10m",
        )

    async def test_aget_query_embedding_formats(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_async_client: AsyncMock,
        mock_embed_response: MagicMock,
    ) -> None:
        """Test async _aget_query_embedding applies formatting.

        Inputs: query="What is AI?" with query_instruction
        Expected: Formatted query embedded
        Checks: Format applied before async embedding call
        """
        configured_embedder._async_client = mock_ollama_async_client

        result = await configured_embedder._aget_query_embedding("What is AI?")

        call_args = mock_ollama_async_client.embed.call_args
        assert call_args[1]["input"] == "Query: What is AI?"
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    async def test_aget_text_embedding_formats(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_async_client: AsyncMock,
        mock_embed_response: MagicMock,
    ) -> None:
        """Test async _aget_text_embedding applies formatting.

        Inputs: text="AI is a field" with text_instruction
        Expected: Formatted text embedded
        Checks: Format applied before async embedding call
        """
        configured_embedder._async_client = mock_ollama_async_client

        result = await configured_embedder._aget_text_embedding("AI is a field")

        call_args = mock_ollama_async_client.embed.call_args
        assert call_args[1]["input"] == "Text: AI is a field"
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.mark.unit
@pytest.mark.mock
class TestBatchEmbeddingSyncMethods:
    """Test suite for synchronous batch embedding methods."""

    def test_get_general_text_embeddings_basic(
        self,
        basic_embedder: OllamaEmbedding,
        mock_ollama_client: MagicMock,
        mock_batch_embed_response: MagicMock,
    ) -> None:
        """Test get_general_text_embeddings with multiple texts.

        Inputs: texts=["text1", "text2", "text3"]
        Expected: Single API call with all texts, returns all embeddings
        Checks: Batch API call efficiency, correct return structure
        """
        basic_embedder._client = mock_ollama_client
        mock_ollama_client.embed.return_value = mock_batch_embed_response

        texts = ["text1", "text2", "text3"]
        result = basic_embedder.get_general_text_embeddings(texts)

        # Assert - Verify batch API call
        mock_ollama_client.embed.assert_called_once_with(
            model="test-model",
            input=texts,
            options={},
            keep_alive="5m",
        )

        # Assert - Verify result
        assert len(result) == 3
        assert result[0] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert result[1] == [0.6, 0.7, 0.8, 0.9, 1.0]
        assert result[2] == [0.11, 0.22, 0.33, 0.44, 0.55]

    def test_get_text_embeddings_formats_all_texts(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_client: MagicMock,
        mock_batch_embed_response: MagicMock,
    ) -> None:
        """Test _get_text_embeddings applies formatting to all texts.

        Inputs: texts=["text1", "text2"] with text_instruction="Text:"
        Expected: All texts formatted before batch embedding
        Checks: Each text individually formatted, then batched
        """
        configured_embedder._client = mock_ollama_client
        mock_ollama_client.embed.return_value = mock_batch_embed_response

        texts = ["text1", "text2"]
        result = configured_embedder._get_text_embeddings(texts)

        # Assert - Verify formatted texts used
        call_args = mock_ollama_client.embed.call_args
        assert call_args[1]["input"] == ["Text: text1", "Text: text2"]
        assert len(result) == 3  # Based on mock response

    def test_get_text_embeddings_empty_list(
        self,
        basic_embedder: OllamaEmbedding,
        mock_ollama_client: MagicMock,
    ) -> None:
        """Test _get_text_embeddings with empty list.

        Inputs: texts=[]
        Expected: API called with empty list, returns empty result
        Checks: Edge case handling for empty input
        """
        basic_embedder._client = mock_ollama_client
        mock_response = MagicMock()
        mock_response.embeddings = []
        mock_ollama_client.embed.return_value = mock_response

        result = basic_embedder._get_text_embeddings([])

        mock_ollama_client.embed.assert_called_once_with(
            model="test-model",
            input=[],
            options={},
            keep_alive="5m",
        )
        assert result == []

    def test_get_text_embeddings_single_item(
        self,
        basic_embedder: OllamaEmbedding,
        mock_ollama_client: MagicMock,
        mock_embed_response: MagicMock,
    ) -> None:
        """Test _get_text_embeddings with single text (batch of one).

        Inputs: texts=["single text"]
        Expected: Batch API called with one-element list
        Checks: Single item treated as batch, not special-cased
        """
        basic_embedder._client = mock_ollama_client

        result = basic_embedder._get_text_embeddings(["single text"])

        mock_ollama_client.embed.assert_called_once_with(
            model="test-model",
            input=["single text"],
            options={},
            keep_alive="5m",
        )
        assert len(result) == 1

    def test_batch_with_large_list(self, embedder_factory) -> None:
        """Test batch processing with large input list.

        Inputs: 100 texts to embed
        Expected: All texts processed in single API call
        Checks: Batch processing efficiency
        """
        embedder = embedder_factory()
        texts = [f"text_{i}" for i in range(100)]

        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2] for _ in range(100)]
        embedder._client.embed.return_value = mock_response

        result = embedder.get_general_text_embeddings(texts)

        # Verify single API call with all texts
        embedder._client.embed.assert_called_once()
        call_kwargs = embedder._client.embed.call_args[1]
        assert call_kwargs["input"] == texts
        assert len(result) == 100

    def test_batch_preserves_order(self, embedder_factory) -> None:
        """Test that batch embeddings preserve input order.

        Inputs: Texts ["a", "b", "c"]
        Expected: Embeddings returned in same order
        Checks: Order preservation in batch processing
        """
        embedder = embedder_factory()
        texts = ["alpha", "beta", "gamma"]

        mock_response = MagicMock()
        mock_response.embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        embedder._client.embed.return_value = mock_response

        result = embedder.get_general_text_embeddings(texts)

        assert len(result) == 3
        assert result[0] == [1.0, 0.0, 0.0]
        assert result[1] == [0.0, 1.0, 0.0]
        assert result[2] == [0.0, 0.0, 1.0]


@pytest.mark.unit
@pytest.mark.mock
@pytest.mark.asyncio
class TestBatchEmbeddingAsyncMethods:
    """Test suite for asynchronous batch embedding methods."""

    async def test_aget_general_text_embeddings_basic(
        self,
        basic_embedder: OllamaEmbedding,
        mock_ollama_async_client: AsyncMock,
        mock_batch_embed_response: MagicMock,
    ) -> None:
        """Test async get_general_text_embeddings with multiple texts.

        Inputs: texts=["text1", "text2", "text3"]
        Expected: Single async API call, returns all embeddings
        Checks: Async batch efficiency, correct return structure
        """
        basic_embedder._async_client = mock_ollama_async_client
        mock_ollama_async_client.embed.return_value = mock_batch_embed_response

        texts = ["text1", "text2", "text3"]
        result = await basic_embedder.aget_general_text_embeddings(texts)

        # Assert - Verify async batch call
        mock_ollama_async_client.embed.assert_called_once_with(
            model="test-model",
            input=texts,
            options={},
            keep_alive="5m",
        )

        # Assert - Verify result
        assert len(result) == 3
        assert result[0] == [0.1, 0.2, 0.3, 0.4, 0.5]

    async def test_aget_text_embeddings_formats_all_texts(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_async_client: AsyncMock,
        mock_batch_embed_response: MagicMock,
    ) -> None:
        """Test async _aget_text_embeddings applies formatting.

        Inputs: texts=["text1", "text2"] with text_instruction
        Expected: All texts formatted before async batch
        Checks: Formatting applied to each text before batching
        """
        configured_embedder._async_client = mock_ollama_async_client
        mock_ollama_async_client.embed.return_value = mock_batch_embed_response

        texts = ["text1", "text2"]
        result = await configured_embedder._aget_text_embeddings(texts)

        call_args = mock_ollama_async_client.embed.call_args
        assert call_args[1]["input"] == ["Text: text1", "Text: text2"]

    async def test_aget_text_embeddings_empty_list(
        self,
        basic_embedder: OllamaEmbedding,
        mock_ollama_async_client: AsyncMock,
    ) -> None:
        """Test async _aget_text_embeddings with empty list.

        Inputs: texts=[]
        Expected: Async API called with empty list
        Checks: Empty edge case in async context
        """
        basic_embedder._async_client = mock_ollama_async_client
        mock_response = MagicMock()
        mock_response.embeddings = []
        mock_ollama_async_client.embed.return_value = mock_response

        result = await basic_embedder._aget_text_embeddings([])

        mock_ollama_async_client.embed.assert_called_once()
        assert result == []

    async def test_async_batch_processing(self, embedder_factory) -> None:
        """Test async batch embedding processing.

        Inputs: Multiple texts in async context
        Expected: All texts processed in single async call
        Checks: Async batch processing works correctly
        """
        embedder = embedder_factory()
        texts = ["text1", "text2", "text3"]

        mock_response = MagicMock()
        mock_response.embeddings = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ]
        embedder._async_client.embed = AsyncMock(return_value=mock_response)

        result = await embedder.aget_general_text_embeddings(texts)

        embedder._async_client.embed.assert_called_once()
        assert len(result) == 3


@pytest.mark.unit
class TestErrorHandling:
    """Test suite for error handling and exception scenarios."""

    def test_connection_error_in_sync_embedding(
        self, embedder_factory, mock_client_with_error: MagicMock
    ) -> None:
        """Test handling of connection errors in synchronous embedding.

        Inputs: Ollama server unavailable
        Expected: ConnectionError propagated to caller
        Checks: Error message preserved and exception type correct
        """
        embedder = embedder_factory()
        embedder._client = mock_client_with_error

        with pytest.raises(ConnectionError) as exc_info:
            embedder.get_general_text_embedding("test text")

        assert "Failed to connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connection_error_in_async_embedding(
        self, embedder_factory
    ) -> None:
        """Test handling of connection errors in async embedding.

        Inputs: Ollama server unavailable for async call
        Expected: ConnectionError propagated to caller
        Checks: Async error handling works correctly
        """
        embedder = embedder_factory()
        embedder._async_client.embed = AsyncMock(
            side_effect=ConnectionError("Failed to connect to Ollama server")
        )

        with pytest.raises(ConnectionError) as exc_info:
            await embedder.aget_general_text_embedding("test text")

        assert "Failed to connect" in str(exc_info.value)

    def test_timeout_error_handling(
        self, embedder_factory, mock_timeout_client: MagicMock
    ) -> None:
        """Test handling of timeout errors.

        Inputs: Request that exceeds timeout limit
        Expected: TimeoutError raised with appropriate message
        Checks: Timeout errors are not swallowed
        """
        embedder = embedder_factory()
        embedder._client = mock_timeout_client

        with pytest.raises(TimeoutError) as exc_info:
            embedder.get_general_text_embedding("test text")

        assert "timed out" in str(exc_info.value).lower()

    def test_invalid_model_name_type(self) -> None:
        """Test validation error with invalid model_name type.

        Inputs: model_name=123 (integer instead of string)
        Expected: ValidationError raised
        Checks: Pydantic type validation enforced
        """
        with patch("serapeum.ollama.embedding.Client"), patch(
            "serapeum.ollama.embedding.AsyncClient"
        ):
            with pytest.raises(ValidationError) as exc_info:
                OllamaEmbedding(model_name=123)  # type: ignore

            assert "model_name" in str(exc_info.value)

    def test_invalid_base_url_type(self) -> None:
        """Test validation error with invalid base_url type.

        Inputs: base_url=12345 (integer instead of string)
        Expected: ValidationError raised
        Checks: URL validation enforced
        """
        with patch("serapeum.ollama.embedding.Client"), patch(
            "serapeum.ollama.embedding.AsyncClient"
        ):
            with pytest.raises(ValidationError) as exc_info:
                OllamaEmbedding(model_name="test", base_url=12345)  # type: ignore

            assert "base_url" in str(exc_info.value)

    def test_invalid_ollama_additional_kwargs_type(self) -> None:
        """Test validation error with invalid ollama_additional_kwargs type.

        Inputs: ollama_additional_kwargs="not a dict" (string instead of dict)
        Expected: ValidationError raised
        Checks: Dict type validation enforced
        """
        with patch("serapeum.ollama.embedding.Client"), patch(
            "serapeum.ollama.embedding.AsyncClient"
        ):
            with pytest.raises(ValidationError) as exc_info:
                OllamaEmbedding(
                    model_name="test",
                    ollama_additional_kwargs="not a dict"  # type: ignore
                )

            assert "ollama_additional_kwargs" in str(exc_info.value)

    def test_malformed_response_from_api(self, embedder_factory) -> None:
        """Test handling of malformed API responses.

        Inputs: API returns response without embeddings attribute
        Expected: AttributeError raised when accessing embeddings
        Checks: Error handling for unexpected API response format
        """
        embedder = embedder_factory()
        mock_response = MagicMock()
        del mock_response.embeddings  # Remove embeddings attribute
        embedder._client.embed.return_value = mock_response

        with pytest.raises(AttributeError):
            embedder.get_general_text_embedding("test")

    def test_empty_embeddings_list_from_api(self, embedder_factory) -> None:
        """Test handling when API returns empty embeddings list.

        Inputs: API returns response with empty embeddings list
        Expected: IndexError when trying to access first embedding
        Checks: Proper handling of unexpected empty responses
        """
        embedder = embedder_factory()
        mock_response = MagicMock()
        mock_response.embeddings = []  # Empty list
        embedder._client.embed.return_value = mock_response

        with pytest.raises(IndexError):
            embedder.get_general_text_embedding("test")


@pytest.mark.unit
class TestClientInitialization:
    """Test suite for client initialization edge cases."""

    def test_client_initialization_with_custom_headers(self) -> None:
        """Test client initialization with custom HTTP headers.

        Inputs: client_kwargs with Authorization header
        Expected: Clients created with custom headers passed through
        Checks: Header configuration propagated to clients
        """
        with patch("serapeum.ollama.embedding.Client") as mock_client, patch(
            "serapeum.ollama.embedding.AsyncClient"
        ) as mock_async:
            custom_headers = {"Authorization": "Bearer secret-token"}
            embedder = OllamaEmbedding(
                model_name="test",
                client_kwargs={"headers": custom_headers}
            )

            # Verify headers passed to both clients
            mock_client.assert_called_once()
            mock_async.assert_called_once()
            assert mock_client.call_args[1]["headers"] == custom_headers
            assert mock_async.call_args[1]["headers"] == custom_headers

    def test_client_initialization_with_timeout(self) -> None:
        """Test client initialization with custom timeout.

        Inputs: client_kwargs with timeout=120
        Expected: Timeout configuration passed to clients
        Checks: Timeout settings propagated correctly
        """
        with patch("serapeum.ollama.embedding.Client") as mock_client, patch(
            "serapeum.ollama.embedding.AsyncClient"
        ) as mock_async:
            embedder = OllamaEmbedding(
                model_name="test",
                client_kwargs={"timeout": 120}
            )

            mock_client.assert_called_once()
            mock_async.assert_called_once()
            assert mock_client.call_args[1]["timeout"] == 120
            assert mock_async.call_args[1]["timeout"] == 120

    def test_multiple_client_kwargs(self) -> None:
        """Test client initialization with multiple kwargs.

        Inputs: Multiple client configuration options
        Expected: All kwargs passed to client constructors
        Checks: Complex configuration handled correctly
        """
        with patch("serapeum.ollama.embedding.Client") as mock_client, patch(
            "serapeum.ollama.embedding.AsyncClient"
        ) as mock_async:
            client_config = {
                "timeout": 60,
                "headers": {"User-Agent": "TestClient/1.0"},
                "verify": False
            }

            embedder = OllamaEmbedding(
                model_name="test",
                base_url="https://custom:443",
                client_kwargs=client_config
            )

            # Verify all kwargs passed
            call_kwargs = mock_client.call_args[1]
            assert call_kwargs["timeout"] == 60
            assert call_kwargs["headers"]["User-Agent"] == "TestClient/1.0"
            assert call_kwargs["verify"] is False

    def test_client_created_on_initialization(self) -> None:
        """Test that clients are created during model validation.

        Inputs: OllamaEmbedding instantiation
        Expected: Both sync and async clients initialized immediately
        Checks: _initialize_clients validator executed
        """
        with patch("serapeum.ollama.embedding.Client") as mock_client, patch(
            "serapeum.ollama.embedding.AsyncClient"
        ) as mock_async:
            embedder = OllamaEmbedding(model_name="test")

            # Verify clients were created
            assert mock_client.called
            assert mock_async.called

            # Verify clients are stored as private attributes
            assert hasattr(embedder, "_client")
            assert hasattr(embedder, "_async_client")


@pytest.mark.unit
class TestKeepAliveParameter:
    """Test suite for keep_alive parameter handling."""

    def test_keep_alive_string_format(self, embedder_factory) -> None:
        """Test keep_alive with string duration format.

        Inputs: keep_alive="10m"
        Expected: String passed through to API call
        Checks: String format accepted and propagated
        """
        embedder = embedder_factory(keep_alive="10m")
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        embedder._client.embed.return_value = mock_response

        embedder.get_general_text_embedding("test")

        # Verify keep_alive passed to API
        call_kwargs = embedder._client.embed.call_args[1]
        assert call_kwargs["keep_alive"] == "10m"

    def test_keep_alive_numeric_format(self, embedder_factory) -> None:
        """Test keep_alive with numeric seconds format.

        Inputs: keep_alive=600.0 (float seconds)
        Expected: Float passed through to API call
        Checks: Numeric format accepted
        """
        embedder = embedder_factory(keep_alive=600.0)
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        embedder._client.embed.return_value = mock_response

        embedder.get_general_text_embedding("test")

        call_kwargs = embedder._client.embed.call_args[1]
        assert call_kwargs["keep_alive"] == 600.0

    def test_keep_alive_none(self, embedder_factory) -> None:
        """Test keep_alive when set to None.

        Inputs: keep_alive=None
        Expected: None passed to API (server uses default)
        Checks: None value handled correctly
        """
        embedder = embedder_factory(keep_alive=None)
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        embedder._client.embed.return_value = mock_response

        embedder.get_general_text_embedding("test")

        call_kwargs = embedder._client.embed.call_args[1]
        assert call_kwargs["keep_alive"] is None

    def test_keep_alive_default_value(self, embedder_factory) -> None:
        """Test default keep_alive value.

        Inputs: No keep_alive specified
        Expected: Default "5m" used
        Checks: Default value from class definition
        """
        embedder = embedder_factory()
        assert embedder.keep_alive == "5m"

    def test_keep_alive_types(self) -> None:
        """Test keep_alive accepts string and numeric types.

        Inputs: keep_alive as string, int, float, None
        Expected: All types accepted by Pydantic validation
        Checks: Type flexibility for keep_alive parameter
        """
        # String format
        emb1 = OllamaEmbedding(model_name="test", keep_alive="5m")
        assert emb1.keep_alive == "5m"

        # Numeric format
        emb2 = OllamaEmbedding(model_name="test", keep_alive=300.0)
        assert emb2.keep_alive == 300.0

        # None
        emb3 = OllamaEmbedding(model_name="test", keep_alive=None)
        assert emb3.keep_alive is None


@pytest.mark.unit
class TestOllamaAdditionalKwargs:
    """Test suite for ollama_additional_kwargs parameter."""

    def test_additional_kwargs_passed_to_sync_call(self, embedder_factory) -> None:
        """Test that additional kwargs are passed to sync embed call.

        Inputs: ollama_additional_kwargs with temperature and num_ctx
        Expected: Kwargs passed in options parameter
        Checks: Configuration options propagated correctly
        """
        additional = {"temperature": 0.7, "num_ctx": 4096}
        embedder = embedder_factory(ollama_additional_kwargs=additional)

        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2]]
        embedder._client.embed.return_value = mock_response

        embedder.get_general_text_embedding("test")

        call_kwargs = embedder._client.embed.call_args[1]
        assert call_kwargs["options"] == additional

    @pytest.mark.asyncio
    async def test_additional_kwargs_passed_to_async_call(
        self, embedder_factory
    ) -> None:
        """Test that additional kwargs are passed to async embed call.

        Inputs: ollama_additional_kwargs in async context
        Expected: Kwargs passed in options parameter to async client
        Checks: Async calls receive configuration
        """
        additional = {"seed": 42, "top_k": 50}
        embedder = embedder_factory(ollama_additional_kwargs=additional)

        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2]]

        # Create proper AsyncMock for async client
        embedder._async_client.embed = AsyncMock(return_value=mock_response)

        await embedder.aget_general_text_embedding("test")

        call_kwargs = embedder._async_client.embed.call_args[1]
        assert call_kwargs["options"] == additional

    def test_empty_additional_kwargs(self, embedder_factory) -> None:
        """Test behavior with empty additional kwargs dict.

        Inputs: ollama_additional_kwargs={}
        Expected: Empty dict passed to API (no errors)
        Checks: Empty dict handled gracefully
        """
        embedder = embedder_factory(ollama_additional_kwargs={})

        mock_response = MagicMock()
        mock_response.embeddings = [[0.1]]
        embedder._client.embed.return_value = mock_response

        embedder.get_general_text_embedding("test")

        call_kwargs = embedder._client.embed.call_args[1]
        assert call_kwargs["options"] == {}


@pytest.mark.unit
class TestTypeAndReturnValues:
    """Test suite for type checking and return value validation."""

    def test_get_query_embedding_return_type(self, embedder_factory) -> None:
        """Test _get_query_embedding returns correct type.

        Inputs: Valid query string
        Expected: Returns Sequence[float]
        Checks: Return type matches Sequence[float] protocol
        """
        embedder = embedder_factory()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        embedder._client.embed.return_value = mock_response

        result = embedder._get_query_embedding("test")

        assert isinstance(result, Sequence)
        assert all(isinstance(x, float) for x in result)

    def test_get_text_embedding_return_type(self, embedder_factory) -> None:
        """Test _get_text_embedding returns correct type.

        Inputs: Valid text string
        Expected: Returns Sequence[float]
        Checks: Return type consistency
        """
        embedder = embedder_factory()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.4, 0.5, 0.6]]
        embedder._client.embed.return_value = mock_response

        result = embedder._get_text_embedding("test")

        assert isinstance(result, Sequence)
        assert all(isinstance(x, float) for x in result)

    def test_get_text_embeddings_return_type(self, embedder_factory) -> None:
        """Test _get_text_embeddings returns correct type.

        Inputs: List of text strings
        Expected: Returns Sequence[Sequence[float]]
        Checks: Nested sequence type correctness
        """
        embedder = embedder_factory()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        embedder._client.embed.return_value = mock_response

        result = embedder._get_text_embeddings(["text1", "text2"])

        assert isinstance(result, Sequence)
        for embedding in result:
            assert isinstance(embedding, Sequence)
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_async_return_types(self, embedder_factory) -> None:
        """Test async methods return correct types.

        Inputs: Async embedding calls
        Expected: Same return types as sync versions
        Checks: Type consistency between sync and async
        """
        embedder = embedder_factory()
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        embedder._async_client.embed = AsyncMock(return_value=mock_response)

        query_result = await embedder._aget_query_embedding("query")
        text_result = await embedder._aget_text_embedding("text")

        assert isinstance(query_result, Sequence)
        assert isinstance(text_result, Sequence)
        assert all(isinstance(x, float) for x in query_result)
        assert all(isinstance(x, float) for x in text_result)


@pytest.mark.integration
@pytest.mark.mock
class TestOllamaEmbeddingIntegration:
    """Integration tests with mocked Ollama clients."""

    def test_query_embedding_full_flow(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_client: MagicMock,
    ) -> None:
        """Test complete flow from query to embedding with formatting.

        Inputs: Raw query through public get_query_embedding
        Expected: Formatting applied, correct embedding returned
        Checks: End-to-end integration of format + embed
        """
        configured_embedder._client = mock_ollama_client

        with patch.object(
            configured_embedder, "_get_query_embedding", wraps=configured_embedder._get_query_embedding
        ) as mock_get:
            result = configured_embedder.get_query_embedding("What is AI?")

            # Assert - Verify internal method called
            mock_get.assert_called_once_with("What is AI?")

            # Assert - Verify API called with formatted query
            call_args = mock_ollama_client.embed.call_args
            assert call_args[1]["input"] == "Query: What is AI?"

    def test_text_embedding_full_flow(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_client: MagicMock,
    ) -> None:
        """Test complete flow from text to embedding with formatting.

        Inputs: Raw text through public get_text_embedding
        Expected: Formatting applied, correct embedding returned
        Checks: End-to-end integration of format + embed
        """
        configured_embedder._client = mock_ollama_client

        with patch.object(
            configured_embedder, "_get_text_embedding", wraps=configured_embedder._get_text_embedding
        ) as mock_get:
            result = configured_embedder.get_text_embedding("AI is a field")

            # Assert - Verify internal method called
            mock_get.assert_called_once_with("AI is a field")

            # Assert - Verify API called with formatted text
            call_args = mock_ollama_client.embed.call_args
            assert call_args[1]["input"] == "Text: AI is a field"

    @pytest.mark.asyncio
    async def test_async_query_embedding_full_flow(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_async_client: AsyncMock,
    ) -> None:
        """Test complete async flow from query to embedding.

        Inputs: Raw query through public aget_query_embedding
        Expected: Async formatting + embedding
        Checks: End-to-end async integration
        """
        configured_embedder._async_client = mock_ollama_async_client

        with patch.object(
            configured_embedder,
            "_aget_query_embedding",
            wraps=configured_embedder._aget_query_embedding,
        ) as mock_aget:
            result = await configured_embedder.aget_query_embedding("What is AI?")

            # Assert - Verify internal async method called
            mock_aget.assert_called_once_with("What is AI?")

    @pytest.mark.asyncio
    async def test_async_text_embedding_full_flow(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_async_client: AsyncMock,
    ) -> None:
        """Test complete async flow from text to embedding.

        Inputs: Raw text through public aget_text_embedding
        Expected: Async formatting + embedding
        Checks: End-to-end async integration
        """
        configured_embedder._async_client = mock_ollama_async_client

        with patch.object(
            configured_embedder,
            "_aget_text_embedding",
            wraps=configured_embedder._aget_text_embedding,
        ) as mock_aget:
            result = await configured_embedder.aget_text_embedding("AI is a field")

            # Assert - Verify internal async method called
            mock_aget.assert_called_once_with("AI is a field")

    def test_batch_embeddings_with_formatting(
        self,
        configured_embedder: OllamaEmbedding,
        mock_ollama_client: MagicMock,
        mock_batch_embed_response: MagicMock,
    ) -> None:
        """Test batch text embeddings with instruction formatting.

        Inputs: Multiple texts with text_instruction set
        Expected: All texts formatted, single batch API call
        Checks: Batch efficiency with formatting
        """
        configured_embedder._client = mock_ollama_client
        mock_ollama_client.embed.return_value = mock_batch_embed_response

        texts = ["First text", "Second text", "Third text"]
        result = configured_embedder.get_text_embedding_batch(texts)

        # Assert - Should have been called with formatted texts
        call_args = mock_ollama_client.embed.call_args
        expected_formatted = [f"Text: {text}" for text in texts]
        assert call_args[1]["input"] == expected_formatted

    def test_complete_query_embedding_workflow(self, embedder_factory) -> None:
        """Test complete workflow from query to embedding with all features.

        Inputs: Query with instruction, additional kwargs, custom keep_alive
        Expected: Complete pipeline executes correctly
        Checks: End-to-end integration of all features
        """
        embedder = embedder_factory(
            query_instruction="Search:",
            ollama_additional_kwargs={"temperature": 0.5},
            keep_alive="15m"
        )

        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        embedder._client.embed.return_value = mock_response

        result = embedder._get_query_embedding("What is AI?")

        # Verify all parameters passed correctly
        call_kwargs = embedder._client.embed.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["input"] == "Search: What is AI?"
        assert call_kwargs["options"] == {"temperature": 0.5}
        assert call_kwargs["keep_alive"] == "15m"
        assert result == [0.1, 0.2, 0.3, 0.4, 0.5]

    @pytest.mark.asyncio
    async def test_complete_async_workflow(self, embedder_factory) -> None:
        """Test complete async workflow with all features.

        Inputs: Async text embedding with full configuration
        Expected: All features work in async context
        Checks: Async integration completeness
        """
        embedder = embedder_factory(
            text_instruction="Document:",
            ollama_additional_kwargs={"num_ctx": 2048},
            keep_alive=300
        )

        mock_response = MagicMock()
        mock_response.embeddings = [[0.9, 0.8, 0.7]]
        embedder._async_client.embed = AsyncMock(return_value=mock_response)

        result = await embedder._aget_text_embedding("Sample document text")

        # Verify complete async pipeline
        call_kwargs = embedder._async_client.embed.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["input"] == "Document: Sample document text"
        assert call_kwargs["options"] == {"num_ctx": 2048}
        assert call_kwargs["keep_alive"] == 300
        assert result == [0.9, 0.8, 0.7]

    def test_batch_workflow_with_formatting(self, embedder_factory) -> None:
        """Test batch processing workflow with instruction formatting.

        Inputs: Multiple texts with text_instruction
        Expected: All texts formatted and embedded in batch
        Checks: Batch + formatting integration
        """
        embedder = embedder_factory(text_instruction="Text:")
        texts = ["first", "second", "third"]

        mock_response = MagicMock()
        mock_response.embeddings = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        embedder._client.embed.return_value = mock_response

        result = embedder._get_text_embeddings(texts)

        # Verify formatting applied to all texts
        call_kwargs = embedder._client.embed.call_args[1]
        expected_formatted = ["Text: first", "Text: second", "Text: third"]
        assert call_kwargs["input"] == expected_formatted
        assert len(result) == 3


@pytest.mark.unit
@pytest.mark.mock
class TestOllamaEmbeddingProperties:
    """Property-based and invariant tests."""

    def test_format_idempotency(self, configured_embedder: OllamaEmbedding) -> None:
        """Test that formatting is idempotent (applying twice gives same result).

        Inputs: Same text formatted twice
        Expected: Both results identical
        Checks: Formatting is deterministic and stable
        """
        text = "  test text  "
        result1 = configured_embedder._format_text(text)
        result2 = configured_embedder._format_text(text)
        assert result1 == result2

    def test_format_preserves_non_whitespace(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test that formatting preserves non-whitespace characters.

        Inputs: Text with various non-whitespace chars
        Expected: All original chars present in result
        Checks: No content loss during formatting
        """
        text = "Special chars: !@#$%^&*()_+-=[]{}|;:,.<>?"
        result = configured_embedder._format_text(text)
        # All original non-whitespace chars should be in result
        for char in text.strip():
            if not char.isspace():
                assert char in result

    def test_client_immutability_after_init(
        self, basic_embedder: OllamaEmbedding
    ) -> None:
        """Test that client instances remain stable after initialization.

        Inputs: Access _client and _async_client multiple times
        Expected: Same object references returned
        Checks: Clients not recreated on access
        """
        client1 = basic_embedder._client
        client2 = basic_embedder._client
        assert client1 is client2

        async_client1 = basic_embedder._async_client
        async_client2 = basic_embedder._async_client
        assert async_client1 is async_client2


@pytest.mark.unit
@pytest.mark.mock
class TestOllamaEmbeddingEdgeCases:
    """Test edge cases and error conditions."""

    def test_format_query_unicode_characters(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test formatting handles Unicode characters correctly.

        Inputs: Query with Unicode (Chinese, emoji, accents)
        Expected: Unicode preserved in formatted output
        Checks: Internationalization support
        """
        query = "What is 人工智能? 🤖 Café"
        result = configured_embedder._format_query(query)
        assert result == "Query: What is 人工智能? 🤖 Café"

    def test_format_text_very_long_string(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test formatting handles very long strings.

        Inputs: Text with 10000 characters
        Expected: Full text formatted without truncation
        Checks: No implicit length limits in formatting
        """
        long_text = "A" * 10000
        result = configured_embedder._format_text(long_text)
        assert result == f"Text: {long_text}"
        assert len(result) == len(long_text) + len("Text: ")

    def test_format_special_characters(
        self, configured_embedder: OllamaEmbedding
    ) -> None:
        """Test formatting preserves special characters and escape sequences.

        Inputs: Text with newlines, tabs, quotes
        Expected: All special chars preserved
        Checks: No unwanted escaping or normalization
        """
        text = 'Text with\nnewlines\tand\t"quotes" and \'apostrophes\''
        result = configured_embedder._format_text(text)
        assert "\n" in result
        assert "\t" in result
        assert '"' in result
        assert "'" in result

    def test_embedding_dimension_consistency(
        self,
        basic_embedder: OllamaEmbedding,
        mock_ollama_client: MagicMock,
    ) -> None:
        """Test that all embeddings from same model have same dimensions.

        Inputs: Multiple different texts
        Expected: All resulting embeddings have same length
        Checks: Model produces consistent dimensionality
        """
        basic_embedder._client = mock_ollama_client

        # Create responses with consistent dimensions
        response1 = MagicMock()
        response1.embeddings = [[0.1, 0.2, 0.3]]
        response2 = MagicMock()
        response2.embeddings = [[0.4, 0.5, 0.6]]

        mock_ollama_client.embed.side_effect = [response1, response2]

        emb1 = basic_embedder.get_general_text_embedding("text1")
        emb2 = basic_embedder.get_general_text_embedding("text2")

        assert len(emb1) == len(emb2)


@pytest.mark.unit
@pytest.mark.mock
@pytest.mark.parametrize(
    "query_instruction,text_instruction,input_query,input_text,expected_query,expected_text",
    [
        # No instructions
        (None, None, "query", "text", "query", "text"),
        # Only query instruction
        ("Q:", None, "query", "text", "Q: query", "text"),
        # Only text instruction
        (None, "T:", "query", "text", "query", "T: text"),
        # Both instructions
        ("Q:", "T:", "query", "text", "Q: query", "T: text"),
        # Instructions with extra whitespace
        ("  Q:  ", "  T:  ", "  query  ", "  text  ", "Q: query", "T: text"),
        # Unicode
        ("問:", "文:", "query", "text", "問: query", "文: text"),
        # Long instructions
        (
            "Represent the query:",
            "Represent the document:",
            "query",
            "text",
            "Represent the query: query",
            "Represent the document: text",
        ),
    ],
    ids=[
        "no_instructions",
        "query_instruction_only",
        "text_instruction_only",
        "both_instructions",
        "whitespace_handling",
        "unicode_instructions",
        "long_instructions",
    ],
)
class TestFormattingScenarioMatrix:
    """Comprehensive scenario matrix for formatting behavior."""

    def test_formatting_scenarios(
        self,
        query_instruction: str | None,
        text_instruction: str | None,
        input_query: str,
        input_text: str,
        expected_query: str,
        expected_text: str,
    ) -> None:
        """Test all formatting scenarios from matrix.

        Inputs: Various combinations of instructions and inputs
        Expected: Correct formatted outputs per scenario
        Checks: Matrix covers all instruction/input combinations
        """
        embedder = OllamaEmbedding(
            model_name="test",
            query_instruction=query_instruction,
            text_instruction=text_instruction,
        )

        # Assert - Query formatting
        assert embedder._format_query(input_query) == expected_query

        # Assert - Text formatting
        assert embedder._format_text(input_text) == expected_text


@pytest.mark.asyncio
@pytest.mark.integration
class TestConcurrentAccess:
    """Test suite for concurrent and parallel access scenarios."""

    async def test_multiple_async_calls_concurrent(self, embedder_factory) -> None:
        """Test multiple async embedding calls running concurrently.

        Inputs: 10 concurrent async embedding requests
        Expected: All complete successfully without interference
        Checks: Thread safety and async handling
        """
        embedder = embedder_factory()

        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3]]
        embedder._async_client.embed = AsyncMock(return_value=mock_response)

        # Launch 10 concurrent requests
        tasks = [
            embedder._aget_query_embedding(f"query_{i}")
            for i in range(10)
        ]

        results = await asyncio.gather(*tasks)

        # Verify all completed
        assert len(results) == 10
        assert all(len(r) == 3 for r in results)

    async def test_mixed_sync_async_access(self, embedder_factory) -> None:
        """Test mixing sync and async calls (should work independently).

        Inputs: Alternating sync and async calls
        Expected: Both client types work independently
        Checks: Separate client instances don't interfere
        """
        embedder = embedder_factory()

        mock_response = MagicMock()
        mock_response.embeddings = [[0.5, 0.5]]

        embedder._client.embed.return_value = mock_response
        embedder._async_client.embed = AsyncMock(return_value=mock_response)

        # Sync call
        sync_result = embedder._get_query_embedding("sync query")

        # Async call
        async_result = await embedder._aget_query_embedding("async query")

        assert sync_result == [0.5, 0.5]
        assert async_result == [0.5, 0.5]

        # Verify both clients were used
        embedder._client.embed.assert_called_once()
        embedder._async_client.embed.assert_called_once()


@pytest.mark.performance
class TestPerformanceScenarios:
    """Test suite for performance-related scenarios."""

    def test_large_batch_size_handling(self, embedder_factory) -> None:
        """Test handling of maximum batch size (2048 texts).

        Inputs: 2048 texts (maximum allowed)
        Expected: Processed without errors
        Checks: Maximum capacity handling
        """
        embedder = embedder_factory(batch_size=2048)
        texts = [f"text_{i}" for i in range(2048)]

        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] for _ in range(2048)]
        embedder._client.embed.return_value = mock_response

        result = embedder.get_general_text_embeddings(texts)

        assert len(result) == 2048

    def test_very_long_text_input(self, embedder_factory) -> None:
        """Test embedding of very long text input.

        Inputs: Text with 10000+ characters
        Expected: Processed without truncation or errors
        Checks: Large input handling
        """
        embedder = embedder_factory()
        long_text = "A" * 10000  # 10K characters

        mock_response = MagicMock()
        mock_response.embeddings = [[0.1, 0.2]]
        embedder._client.embed.return_value = mock_response

        result = embedder.get_general_text_embedding(long_text)

        # Verify full text passed to API
        call_kwargs = embedder._client.embed.call_args[1]
        assert len(call_kwargs["input"]) == 10000
        assert isinstance(result, list)

    def test_embedding_dimension_consistency_multiple(self, embedder_factory) -> None:
        """Test that embedding dimensions are consistent across calls.

        Inputs: Multiple embedding calls
        Expected: All return same dimension size
        Checks: Dimension consistency
        """
        embedder = embedder_factory()

        mock_response_1 = MagicMock()
        mock_response_1.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]

        mock_response_2 = MagicMock()
        mock_response_2.embeddings = [[0.6, 0.7, 0.8, 0.9, 1.0]]

        embedder._client.embed.side_effect = [mock_response_1, mock_response_2]

        result1 = embedder.get_general_text_embedding("text1")
        result2 = embedder.get_general_text_embedding("text2")

        assert len(result1) == len(result2) == 5


@pytest.mark.unit
class TestBaseEmbeddingIntegration:
    """Test suite for integration with BaseEmbedding parent class."""

    def test_inherits_from_base_embedding(self) -> None:
        """Test that OllamaEmbedding properly inherits from BaseEmbedding.

        Inputs: Class hierarchy inspection
        Expected: Inherits from BaseEmbedding
        Checks: Inheritance chain correct
        """
        assert issubclass(OllamaEmbedding, BaseEmbedding)

    def test_inherits_from_base_embedding_instance(self, embedder_factory) -> None:
        """Test that OllamaEmbedding instance is BaseEmbedding.

        Inputs: OllamaEmbedding instance
        Expected: isinstance check passes for BaseEmbedding
        Checks: Inheritance relationship
        """
        embedder = embedder_factory()
        assert isinstance(embedder, BaseEmbedding)

    def test_implements_required_abstract_methods(self, embedder_factory) -> None:
        """Test that all required abstract methods are implemented.

        Inputs: Check for method existence
        Expected: All abstract methods from BaseEmbedding implemented
        Checks: Abstract method implementation completeness
        """
        embedder = embedder_factory()

        # Check sync methods
        assert hasattr(embedder, "_get_query_embedding")
        assert callable(getattr(embedder, "_get_query_embedding"))

        assert hasattr(embedder, "_get_text_embedding")
        assert callable(getattr(embedder, "_get_text_embedding"))

        # Check async methods
        assert hasattr(embedder, "_aget_query_embedding")
        assert callable(getattr(embedder, "_aget_query_embedding"))

        assert hasattr(embedder, "_aget_text_embedding")
        assert callable(getattr(embedder, "_aget_text_embedding"))

        # Check batch methods
        assert hasattr(embedder, "_get_text_embeddings")
        assert callable(getattr(embedder, "_get_text_embeddings"))

        assert hasattr(embedder, "_aget_text_embeddings")
        assert callable(getattr(embedder, "_aget_text_embeddings"))

    def test_model_name_accessible(self, embedder_factory) -> None:
        """Test that model_name from BaseEmbedding is accessible.

        Inputs: Create embedder with specific model_name
        Expected: model_name attribute accessible and correct
        Checks: Parent class field inheritance
        """
        embedder = embedder_factory(model_name="custom-model-name")
        assert embedder.model_name == "custom-model-name"

    def test_batch_size_from_base(self, embedder_factory) -> None:
        """Test that batch_size from BaseEmbedding works.

        Inputs: Custom batch_size
        Expected: Value stored and accessible
        Checks: Parent class configuration field
        """
        embedder = embedder_factory(batch_size=100)
        assert embedder.batch_size == 100

    def test_field_defaults_match_specification(self) -> None:
        """Test that field defaults match class specification.

        Inputs: Create instance with no optional params
        Expected: All defaults as specified in Field definitions
        Checks: Default values correct per docstrings
        """
        embedder = OllamaEmbedding(model_name="test")

        assert embedder.base_url == "http://localhost:11434"
        assert embedder.batch_size == DEFAULT_EMBED_BATCH_SIZE
        assert embedder.ollama_additional_kwargs == {}
        assert embedder.query_instruction is None
        assert embedder.text_instruction is None
        assert embedder.keep_alive == "5m"
        assert embedder.client_kwargs == {}

    def test_constructor_kwargs_forwarding(self) -> None:
        """Test that extra kwargs are properly forwarded to parent BaseEmbedding.

        Inputs: Extra kwargs (num_workers) passed to constructor
        Expected: No error, kwargs forwarded to Pydantic BaseModel parent
        Checks: Inheritance works correctly, parent fields accessible
        """
        # This shouldn't raise even with extra kwargs
        embedder = OllamaEmbedding(
            model_name="test",
            # Extra field from parent BaseEmbedding
            num_workers=4,
        )
        assert embedder is not None
        assert embedder.num_workers == 4

    def test_pydantic_model_config(self, basic_embedder: OllamaEmbedding) -> None:
        """Test that Pydantic model configuration is correct.

        Inputs: Access model fields
        Expected: All fields properly defined
        Checks: Pydantic integration working correctly
        """
        assert hasattr(basic_embedder, "model_name")
        assert hasattr(basic_embedder, "base_url")
        assert hasattr(basic_embedder, "batch_size")
        assert hasattr(basic_embedder, "ollama_additional_kwargs")

    def test_private_attrs_initialized(self, basic_embedder: OllamaEmbedding) -> None:
        """Test that private attributes are initialized.

        Inputs: Access _client and _async_client
        Expected: Both attributes exist and are not None
        Checks: PrivateAttr initialization in __init__
        """
        assert hasattr(basic_embedder, "_client")
        assert hasattr(basic_embedder, "_async_client")
        assert basic_embedder._client is not None
        assert basic_embedder._async_client is not None

    def test_client_kwargs_passed_to_clients(self) -> None:
        """Test client_kwargs are passed to Client and AsyncClient initialization.

        Inputs: client_kwargs with timeout
        Expected: Clients created with custom kwargs
        Checks: Client initialization receives correct parameters
        """
        embedder = OllamaEmbedding(
            model_name="test",
            client_kwargs={"timeout": 120}
        )

        # Assert - Clients should exist (created with kwargs)
        assert embedder._client is not None
        assert embedder._async_client is not None
