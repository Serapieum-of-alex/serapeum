"""Mock embedding model."""

from typing import Any

from serapeum.core.base.embeddings.base import BaseEmbedding


class MockEmbedding(BaseEmbedding):
    """Mock embedding.

    attributes:
        embed_dim (int): embedding dimension
    """

    embed_dim: int

    def __init__(self, embed_dim: int, **kwargs: Any) -> None:
        """Init params."""
        super().__init__(embed_dim=embed_dim, **kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "MockEmbedding"

    def _get_vector(self) -> list[float]:
        return [0.5] * self.embed_dim

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_vector()

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_vector()

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get query embedding."""
        return self._get_vector()

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get text embedding."""
        return self._get_vector()
