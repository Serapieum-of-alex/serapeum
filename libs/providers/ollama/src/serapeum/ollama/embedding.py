from typing import Any, Sequence

from serapeum.core.base.embeddings.base import BaseEmbedding
from pydantic import Field, PrivateAttr, model_validator
from serapeum.core.configs.defaults import DEFAULT_EMBED_BATCH_SIZE

from ollama import Client, AsyncClient


class OllamaEmbedding(BaseEmbedding):
    """Class for Ollama embeddings."""

    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted by Ollama"
    )
    model_name: str = Field(description="The Ollama model to use.")
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )
    ollama_additional_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Ollama API."
    )
    query_instruction: str | None = Field(
        default=None, description="Instruction to prepend to query text."
    )
    text_instruction: str | None = Field(
        default=None, description="Instruction to prepend to text."
    )
    keep_alive: float | str | None = Field(
        default="5m",
        description="controls how long the model will stay loaded into memory following the request(default: 5m)",
    )
    client_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the Ollama client initialization."
    )

    _client: Client = PrivateAttr()
    _async_client: AsyncClient = PrivateAttr()

    @model_validator(mode='after')
    def _initialize_clients(self) -> 'OllamaEmbedding':
        """Initialize Ollama clients after model validation."""
        self._client = Client(host=self.base_url, **self.client_kwargs)
        self._async_client = AsyncClient(host=self.base_url, **self.client_kwargs)
        return self

    @classmethod
    def class_name(cls) -> str:
        return "OllamaEmbedding"

    def _get_query_embedding(self, query: str) -> Sequence[float]:
        """Get query embedding."""
        formatted_query = self._format_query(query)
        return self.get_general_text_embedding(formatted_query)

    async def _aget_query_embedding(self, query: str) -> Sequence[float]:
        """The asynchronous version of _get_query_embedding."""
        formatted_query = self._format_query(query)
        return await self.aget_general_text_embedding(formatted_query)

    def _get_text_embedding(self, text: str) -> Sequence[float]:
        """Get text embedding."""
        formatted_text = self._format_text(text)
        return self.get_general_text_embedding(formatted_text)

    async def _aget_text_embedding(self, text: str) -> Sequence[float]:
        """Asynchronously get text embedding."""
        formatted_text = self._format_text(text)
        return await self.aget_general_text_embedding(formatted_text)

    def _get_text_embeddings(self, texts: list[str]) -> Sequence[Sequence[float]]:
        """Get text embeddings."""
        formatted_texts = [self._format_text(text) for text in texts]
        return self.get_general_text_embeddings(formatted_texts)

    async def _aget_text_embeddings(
        self, texts: list[str]
    ) -> Sequence[Sequence[float]]:
        """Asynchronously get text embeddings."""
        formatted_texts = [self._format_text(text) for text in texts]
        return await self.aget_general_text_embeddings(formatted_texts)

    def get_general_text_embeddings(
        self, texts: list[str]
    ) -> Sequence[Sequence[float]]:
        """Get Ollama embeddings."""
        result = self._client.embed(
            model=self.model_name,
            input=texts,
            options=self.ollama_additional_kwargs,
            keep_alive=self.keep_alive,
        )
        return result.embeddings

    async def aget_general_text_embeddings(
        self, texts: list[str]
    ) -> Sequence[Sequence[float]]:
        """Asynchronously get Ollama embeddings."""
        result = await self._async_client.embed(
            model=self.model_name,
            input=texts,
            options=self.ollama_additional_kwargs,
            keep_alive=self.keep_alive,
        )
        return result.embeddings

    def get_general_text_embedding(self, texts: str) -> Sequence[float]:
        """Get Ollama embedding."""
        result = self._client.embed(
            model=self.model_name,
            input=texts,
            options=self.ollama_additional_kwargs,
            keep_alive=self.keep_alive,
        )
        return result.embeddings[0]

    async def aget_general_text_embedding(self, prompt: str) -> Sequence[float]:
        """Asynchronously get Ollama embedding."""
        result = await self._async_client.embed(
            model=self.model_name,
            input=prompt,
            options=self.ollama_additional_kwargs,
            keep_alive=self.keep_alive,
        )
        return result.embeddings[0]

    def _format_query(self, query: str) -> str:
        """Format query with instruction if provided."""
        if self.query_instruction:
            val = f"{self.query_instruction.strip()} {query.strip()}".strip()
        else:
            val = query.strip()
        return val

    def _format_text(self, text: str) -> str:
        """Format text with instruction if provided."""
        if self.text_instruction:
            val = f"{self.text_instruction.strip()} {text.strip()}".strip()
        else:
            val = text.strip()
        return val
