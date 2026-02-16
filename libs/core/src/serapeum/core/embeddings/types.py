"""Mock embedding implementations for testing and development.

This module provides lightweight mock embedding models that can be used for
testing without requiring access to real embedding services or models. Mock
embeddings return constant vectors, making tests deterministic and fast.

The primary use case is unit testing code that depends on embedding functionality
without the overhead of loading actual embedding models or making API calls.
"""

from __future__ import annotations
from pydantic import Field, field_validator

from serapeum.core.base.embeddings.base import BaseEmbedding, Embedding


class MockEmbedding(BaseEmbedding):
    """Mock embedding model for testing purposes.

    Returns constant embedding vectors (all 0.5 values) for any input,
    allowing tests to run without requiring a real embedding model. This is
    useful for unit testing, integration testing, and development without
    the overhead of loading actual models or making API calls.

    All embeddings returned are deterministic vectors of the specified dimension,
    filled with 0.5 values. This makes tests reproducible and fast.

    Attributes:
        embed_dim: Embedding dimension (must be positive).
        model_name: Model name identifier (defaults to "mock-embedding").

    Examples:
        - Creating a mock embedding model
            ```python
            >>> from serapeum.core.embeddings import MockEmbedding
            >>> emb = MockEmbedding(embed_dim=3)
            >>> emb.model_name
            'mock-embedding'
            >>> emb.embed_dim
            3

            ```

        - Getting embeddings returns constant vectors
            ```python
            >>> emb = MockEmbedding(embed_dim=4)
            >>> result = emb.get_text_embedding("any text")
            >>> result
            [0.5, 0.5, 0.5, 0.5]

            ```

        - All inputs produce identical embeddings
            ```python
            >>> emb = MockEmbedding(embed_dim=2)
            >>> emb.get_text_embedding("hello") == emb.get_text_embedding("world")
            True

            ```

        - Validation of embed_dim
            ```python
            >>> MockEmbedding(embed_dim=0)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            pydantic_core._pydantic_core.ValidationError: 1 validation error...

            ```

    See Also:
        BaseEmbedding: Abstract base class that MockEmbedding implements.
    """

    embed_dim: int = Field(
        ..., gt=0, description="Embedding dimension (must be positive)"
    )
    model_name: str = Field(
        default="mock-embedding", description="Model name identifier"
    )

    @field_validator("embed_dim")
    @classmethod
    def validate_embed_dim(cls, v: int) -> int:
        """Validate that embed_dim is positive.

        Args:
            v: The embed_dim value to validate.

        Returns:
            The validated embed_dim.

        Raises:
            ValueError: If embed_dim is not positive.
        """
        if v <= 0:
            raise ValueError(f"embed_dim must be positive, got {v}")
        return v

    @classmethod
    def class_name(cls) -> str:
        """Return the class name identifier.

        Returns:
            String "MockEmbedding" identifying this class.

        Examples:
            - Getting the class name
                ```python
                >>> from serapeum.core.embeddings import MockEmbedding
                >>> MockEmbedding.class_name()
                'MockEmbedding'

                ```
        """
        return "MockEmbedding"

    def _get_mocked_vector(self) -> Embedding:
        """Generate a mock embedding vector of constant values.

        Creates a vector of length embed_dim where all values are 0.5. This is
        the core method that all other embedding methods delegate to.

        Returns:
            List of floats with length equal to embed_dim, all values 0.5.

        Examples:
            - Generating a mock vector
                ```python
                >>> from serapeum.core.embeddings import MockEmbedding
                >>> emb = MockEmbedding(embed_dim=5)
                >>> emb._get_mocked_vector()
                [0.5, 0.5, 0.5, 0.5, 0.5]

                ```

            - Vector length matches embed_dim
                ```python
                >>> from serapeum.core.embeddings import MockEmbedding
                >>> emb = MockEmbedding(embed_dim=3)
                >>> len(emb._get_mocked_vector())
                3

                ```
        """
        return [0.5] * self.embed_dim

    def _get_query_embedding(self, query: str) -> Embedding:
        """Get query embedding (returns constant mock vector).

        This method ignores the input query and always returns the same mock
        vector. Implements the abstract method from BaseEmbedding.

        Args:
            query: Query text (unused in mock implementation).

        Returns:
            Mock embedding vector with all values set to 0.5.

        Examples:
            - Query embedding returns mock vector
                ```python
                >>> from serapeum.core.embeddings import MockEmbedding
                >>> emb = MockEmbedding(embed_dim=3)
                >>> emb._get_query_embedding("test query")
                [0.5, 0.5, 0.5]

                ```

            - Different queries return identical vectors
                ```python
                >>> emb = MockEmbedding(embed_dim=2)
                >>> emb._get_query_embedding("query1") == emb._get_query_embedding("query2")
                True

                ```
        """
        return self._get_mocked_vector()

    def _get_text_embedding(self, text: str) -> Embedding:
        """Get text embedding (returns constant mock vector).

        This method ignores the input text and always returns the same mock
        vector. Implements the abstract method from BaseEmbedding.

        Args:
            text: Input text (unused in mock implementation).

        Returns:
            Mock embedding vector with all values set to 0.5.

        Examples:
            - Text embedding returns mock vector
                ```python
                >>> from serapeum.core.embeddings import MockEmbedding
                >>> emb = MockEmbedding(embed_dim=4)
                >>> emb._get_text_embedding("sample text")
                [0.5, 0.5, 0.5, 0.5]

                ```

            - Different texts return identical vectors
                ```python
                >>> emb = MockEmbedding(embed_dim=2)
                >>> emb._get_text_embedding("text1") == emb._get_text_embedding("text2")
                True

                ```
        """
        return self._get_mocked_vector()

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Asynchronously get query embedding (returns constant mock vector).

        Async version of _get_query_embedding. This method ignores the input
        query and always returns the same mock vector. Implements the abstract
        method from BaseEmbedding.

        Args:
            query: Query text (unused in mock implementation).

        Returns:
            Mock embedding vector with all values set to 0.5.

        Examples:
            - Async query embedding
                ```python
                >>> import asyncio
                >>> from serapeum.core.embeddings import MockEmbedding
                >>> emb = MockEmbedding(embed_dim=3)
                >>> asyncio.run(emb._aget_query_embedding("async query"))
                [0.5, 0.5, 0.5]

                ```
        """
        return self._get_mocked_vector()

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Asynchronously get text embedding (returns constant mock vector).

        Async version of _get_text_embedding. This method ignores the input
        text and always returns the same mock vector. Implements the abstract
        method from BaseEmbedding.

        Args:
            text: Input text (unused in mock implementation).

        Returns:
            Mock embedding vector with all values set to 0.5.

        Examples:
            - Async text embedding
                ```python
                >>> import asyncio
                >>> from serapeum.core.embeddings import MockEmbedding
                >>> emb = MockEmbedding(embed_dim=3)
                >>> asyncio.run(emb._aget_text_embedding("async text"))
                [0.5, 0.5, 0.5]

                ```
        """
        return self._get_mocked_vector()
