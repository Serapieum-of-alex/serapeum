"""Mock embedding model for testing."""

from pydantic import Field, field_validator

from serapeum.core.base.embeddings.base import BaseEmbedding, Embedding


class MockEmbedding(BaseEmbedding):
    """Mock embedding model for testing purposes.

    Returns constant embedding vectors (all 0.5 values) for any input,
    allowing tests to run without requiring a real embedding model.

    Attributes:
        embed_dim: Embedding dimension (must be positive).
        model_name: Model name identifier (defaults to "mock-embedding").
    """

    embed_dim: int = Field(..., gt=0, description="Embedding dimension (must be positive)")
    model_name: str = Field(default="mock-embedding", description="Model name identifier")

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
        """Return class name."""
        return "MockEmbedding"

    def _get_mocked_vector(self) -> Embedding:
        """Generate mock embedding vector."""
        return [0.5] * self.embed_dim

    def _get_query_embedding(self, query: str) -> Embedding:
        """Get query embedding (returns constant mock vector).

        Args:
            query: Query text (unused in mock).

        Returns:
            Mock embedding vector.
        """
        return self._get_mocked_vector()

    def _get_text_embedding(self, text: str) -> Embedding:
        """Get text embedding (returns constant mock vector).

        Args:
            text: Input text (unused in mock).

        Returns:
            Mock embedding vector.
        """
        return self._get_mocked_vector()

    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Async get query embedding (returns constant mock vector).

        Args:
            query: Query text (unused in mock).

        Returns:
            Mock embedding vector.
        """
        return self._get_mocked_vector()

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Async get text embedding (returns constant mock vector).

        Args:
            text: Input text (unused in mock).

        Returns:
            Mock embedding vector.
        """
        return self._get_mocked_vector()
