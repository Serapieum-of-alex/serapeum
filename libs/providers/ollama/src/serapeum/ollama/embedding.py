"""Ollama embeddings implementation for text and query vectorization.

This module provides the OllamaEmbedding class for generating embeddings using
Ollama models. It supports both symmetric and asymmetric embedding patterns,
allowing different instructions for queries vs documents to optimize retrieval
performance. All operations support both synchronous and asynchronous execution.
"""

from __future__ import annotations
from typing import Any, Sequence

from pydantic import Field, PrivateAttr, model_validator

import ollama as ollama_sdk
from serapeum.core.embeddings import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):
    """Ollama-based embedding model for generating text and query vector representations.

    This class provides a complete embedding interface using Ollama models, supporting
    both symmetric and asymmetric embedding patterns. Asymmetric embeddings allow
    different instructions for queries vs documents, which can significantly improve
    retrieval accuracy by optimizing each embedding type for its specific role.

    The class manages both synchronous and asynchronous Ollama clients, automatically
    handling connection pooling and model lifecycle. All embedding operations support
    batching for efficient processing of multiple texts.

    Attributes:
        base_url: Base URL where Ollama is hosted. Defaults to "http://localhost:11434".
        model_name: Name of the Ollama model to use for embeddings (e.g., "nomic-embed-text").
        ollama_additional_kwargs: Additional options passed to Ollama's embed API.
        query_instruction: Optional instruction prepended to search queries for asymmetric
            embedding. Example: "search_query:" or "Represent this query for retrieval:".
        text_instruction: Optional instruction prepended to documents for asymmetric
            embedding. Example: "search_document:" or "Represent this document:".
        keep_alive: Duration to keep model loaded in memory after requests. Can be a
            duration string (e.g., "5m", "1h") or float (seconds). Defaults to "5m".
        client_kwargs: Additional kwargs for Ollama client initialization.

    Examples:
        - Basic embedding with default settings
            ```python
            >>> from serapeum.ollama import OllamaEmbedding
            >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
            >>> embedding = embedder.get_text_embedding("Hello world")  # doctest: +SKIP
            >>> len(embedding) > 0  # doctest: +SKIP
            True

            ```
        - Asymmetric embeddings for retrieval
            ```python
            >>> from serapeum.ollama import OllamaEmbedding
            >>> embedder = OllamaEmbedding(  # doctest: +SKIP
            ...     model_name="nomic-embed-text",
            ...     query_instruction="search_query:",
            ...     text_instruction="search_document:"
            ... )
            >>> # Query embedding optimized for search
            >>> query_emb = embedder.get_query_embedding("What is Python?")  # doctest: +SKIP
            >>> # Document embedding optimized for retrieval
            >>> doc_emb = embedder.get_text_embedding("Python is a programming language")  # doctest: +SKIP

            ```
        - Batch processing with async
            ```python
            >>> import asyncio
            >>> from serapeum.ollama import OllamaEmbedding
            >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
            >>> async def embed_batch():  # doctest: +SKIP
            ...     texts = ["First doc", "Second doc", "Third doc"]
            ...     embeddings = await embedder.aget_text_embeddings(texts)
            ...     return len(embeddings)
            >>> # asyncio.run(embed_batch())  # Returns 3

            ```
        - Custom Ollama server configuration
            ```python
            >>> from serapeum.ollama import OllamaEmbedding
            >>> embedder = OllamaEmbedding(  # doctest: +SKIP
            ...     model_name="llama2",
            ...     base_url="http://custom-server:11434",
            ...     keep_alive="10m",
            ...     ollama_additional_kwargs={"temperature": 0.0}
            ... )

            ```

    See Also:
        BaseEmbedding: Abstract base class defining the embedding interface.
        serapeum.core.embeddings: Core embedding abstractions and utilities.
    """

    base_url: str = Field(
        default="http://localhost:11434",
        description="Base url the model is hosted by Ollama",
    )
    model_name: str = Field(description="The Ollama model to use.")
    ollama_additional_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional kwargs for the Ollama API."
    )
    query_instruction: str | None = Field(
        default=None,
        description=(
            "Instruction to prepend to search queries for asymmetric embedding. "
            "Used by get_query_embedding() when embedding user questions/searches. "
            "Example: 'search_query:' or 'Represent this query for retrieving relevant documents:'. "
            "This helps the model optimize query embeddings to match against document embeddings."
        ),
    )
    text_instruction: str | None = Field(
        default=None,
        description=(
            "Instruction to prepend to documents/text for asymmetric embedding. "
            "Used by get_text_embedding() when embedding documents to be searched. "
            "Example: 'search_document:' or 'Represent this document for retrieval:'. "
            "This helps the model create document embeddings optimized for retrieval."
        ),
    )
    keep_alive: float | str | None = Field(
        default="5m",
        description="controls how long the model will stay loaded into memory following the request(default: 5m)",
    )
    client_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for the Ollama client initialization.",
    )

    _client: ollama_sdk.Client = PrivateAttr()
    _async_client: ollama_sdk.AsyncClient = PrivateAttr()

    @model_validator(mode="after")
    def _initialize_clients(self) -> OllamaEmbedding:
        """Initialize Ollama synchronous and asynchronous clients after model validation.

        This validator runs automatically after all fields are validated during
        instance creation. It creates both sync and async Ollama clients configured
        with the specified base_url and any additional client kwargs.

        Returns:
            The OllamaEmbedding instance with initialized clients.

        Examples:
            - Clients are initialized automatically on instantiation
                ```python
                >>> from serapeum.ollama import OllamaEmbedding
                >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
                >>> # Both _client and _async_client are now initialized

                ```
        """
        self._client = ollama_sdk.Client(host=self.base_url, **self.client_kwargs)
        self._async_client = ollama_sdk.AsyncClient(host=self.base_url, **self.client_kwargs)
        return self

    @classmethod
    def class_name(cls) -> str:
        """Return the canonical class name for this embedding implementation.

        Returns:
            The string "OllamaEmbedding".

        Examples:
            - Get the class name identifier
                ```python
                >>> from serapeum.ollama import OllamaEmbedding
                >>> OllamaEmbedding.class_name()
                'OllamaEmbedding'

                ```
        """
        return "OllamaEmbedding"

    def _get_query_embedding(self, query: str) -> Sequence[float]:
        """Generate an embedding vector for a search query.

        Formats the query with the optional query_instruction prefix (if configured)
        to optimize the embedding for search/retrieval tasks, then generates the
        embedding vector using the Ollama model.

        Args:
            query: The search query text to embed.

        Returns:
            A sequence of floats representing the query's embedding vector.

        Raises:
            ValueError: If the query is empty or whitespace-only.

        Examples:
            - Embed a search query
                ```python
                >>> from serapeum.ollama import OllamaEmbedding
                >>> embedder = OllamaEmbedding(  # doctest: +SKIP
                ...     model_name="nomic-embed-text",
                ...     query_instruction="search_query:"
                ... )
                >>> query_vec = embedder.get_query_embedding("What is machine learning?")  # doctest: +SKIP
                >>> len(query_vec) > 0  # doctest: +SKIP
                True

                ```

        See Also:
            _aget_query_embedding: Async version of this method.
            _format_query: Formats query with instruction prefix.
        """
        formatted_query = self._format_query(query)
        return self.get_general_text_embedding(formatted_query)

    async def _aget_query_embedding(self, query: str) -> Sequence[float]:
        """Asynchronously generate an embedding vector for a search query.

        Async version of _get_query_embedding. Formats the query with the optional
        query_instruction prefix, then generates the embedding using async Ollama client.

        Args:
            query: The search query text to embed.

        Returns:
            A sequence of floats representing the query's embedding vector.

        Raises:
            ValueError: If the query is empty or whitespace-only.

        Examples:
            - Async query embedding
                ```python
                >>> import asyncio
                >>> from serapeum.ollama import OllamaEmbedding
                >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
                >>> async def embed_query():  # doctest: +SKIP
                ...     vec = await embedder.aget_query_embedding("neural networks")
                ...     return len(vec) > 0
                >>> # asyncio.run(embed_query())  # Returns True

                ```

        See Also:
            _get_query_embedding: Synchronous version of this method.
            _format_query: Formats query with instruction prefix.
        """
        formatted_query = self._format_query(query)
        return await self.aget_general_text_embedding(formatted_query)

    def _get_text_embedding(self, text: str) -> Sequence[float]:
        """Generate an embedding vector for a document or text passage.

        Formats the text with the optional text_instruction prefix (if configured)
        to optimize the embedding for document retrieval, then generates the
        embedding vector using the Ollama model.

        Args:
            text: The document or text passage to embed.

        Returns:
            A sequence of floats representing the text's embedding vector.

        Raises:
            ValueError: If the text is empty or whitespace-only.

        Examples:
            - Embed a document
                ```python
                >>> from serapeum.ollama import OllamaEmbedding
                >>> embedder = OllamaEmbedding(  # doctest: +SKIP
                ...     model_name="nomic-embed-text",
                ...     text_instruction="search_document:"
                ... )
                >>> doc_vec = embedder.get_text_embedding("Python is a programming language")  # doctest: +SKIP
                >>> len(doc_vec) > 0  # doctest: +SKIP
                True

                ```

        See Also:
            _aget_text_embedding: Async version of this method.
            _format_text: Formats text with instruction prefix.
        """
        formatted_text = self._format_text(text)
        return self.get_general_text_embedding(formatted_text)

    async def _aget_text_embedding(self, text: str) -> Sequence[float]:
        """Asynchronously generate an embedding vector for a document or text passage.

        Async version of _get_text_embedding. Formats the text with the optional
        text_instruction prefix, then generates the embedding using async Ollama client.

        Args:
            text: The document or text passage to embed.

        Returns:
            A sequence of floats representing the text's embedding vector.

        Raises:
            ValueError: If the text is empty or whitespace-only.

        Examples:
            - Async document embedding
                ```python
                >>> import asyncio
                >>> from serapeum.ollama import OllamaEmbedding
                >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
                >>> async def embed_doc():  # doctest: +SKIP
                ...     vec = await embedder.aget_text_embedding("Machine learning basics")
                ...     return len(vec) > 0
                >>> # asyncio.run(embed_doc())  # Returns True

                ```

        See Also:
            _get_text_embedding: Synchronous version of this method.
            _format_text: Formats text with instruction prefix.
        """
        formatted_text = self._format_text(text)
        return await self.aget_general_text_embedding(formatted_text)

    def _get_text_embeddings(self, texts: list[str]) -> Sequence[Sequence[float]]:
        """Generate embedding vectors for multiple documents or text passages.

        Batch version of _get_text_embedding. Formats all texts with the optional
        text_instruction prefix, then generates embeddings for all texts in a single
        API call for efficiency.

        Args:
            texts: List of documents or text passages to embed.

        Returns:
            A sequence of embedding vectors, one for each input text, in the same order.

        Raises:
            ValueError: If any text is empty or whitespace-only.

        Examples:
            - Batch embed multiple documents
                ```python
                >>> from serapeum.ollama import OllamaEmbedding
                >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
                >>> docs = ["First document", "Second document", "Third document"]  # doctest: +SKIP
                >>> embeddings = embedder.get_text_embeddings(docs)  # doctest: +SKIP
                >>> len(embeddings) == 3  # doctest: +SKIP
                True

                ```

        See Also:
            _aget_text_embeddings: Async version of this method.
            _get_text_embedding: Single text version.
        """
        formatted_texts = [self._format_text(text) for text in texts]
        return self.get_general_text_embeddings(formatted_texts)

    async def _aget_text_embeddings(
        self, texts: list[str]
    ) -> Sequence[Sequence[float]]:
        """Asynchronously generate embedding vectors for multiple documents or text passages.

        Async batch version of _get_text_embedding. Formats all texts with the optional
        text_instruction prefix, then generates all embeddings in a single async API call.

        Args:
            texts: List of documents or text passages to embed.

        Returns:
            A sequence of embedding vectors, one for each input text, in the same order.

        Raises:
            ValueError: If any text is empty or whitespace-only.

        Examples:
            - Async batch embedding
                ```python
                >>> import asyncio
                >>> from serapeum.ollama import OllamaEmbedding
                >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
                >>> async def batch_embed():  # doctest: +SKIP
                ...     docs = ["Doc 1", "Doc 2", "Doc 3"]
                ...     vecs = await embedder.aget_text_embeddings(docs)
                ...     return len(vecs)
                >>> # asyncio.run(batch_embed())  # Returns 3

                ```

        See Also:
            _get_text_embeddings: Synchronous version of this method.
            _aget_text_embedding: Single text async version.
        """
        formatted_texts = [self._format_text(text) for text in texts]
        return await self.aget_general_text_embeddings(formatted_texts)

    def get_general_text_embeddings(
        self, texts: list[str]
    ) -> Sequence[Sequence[float]]:
        """Generate raw embeddings for multiple texts using the Ollama API.

        Low-level method that directly calls the Ollama embed API without any
        text formatting or instruction prefixes. Used internally by higher-level
        methods after text formatting is applied.

        Args:
            texts: List of text strings to embed (should already be formatted).

        Returns:
            A sequence of embedding vectors from the Ollama model.

        Examples:
            - Direct API call for batch embedding
                ```python
                >>> from serapeum.ollama import OllamaEmbedding
                >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
                >>> texts = ["text 1", "text 2"]  # doctest: +SKIP
                >>> embeddings = embedder.get_general_text_embeddings(texts)  # doctest: +SKIP
                >>> len(embeddings) == 2  # doctest: +SKIP
                True

                ```

        See Also:
            aget_general_text_embeddings: Async version of this method.
            get_general_text_embedding: Single text version.
        """
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
        """Asynchronously generate raw embeddings for multiple texts using the Ollama API.

        Async low-level method that directly calls the Ollama embed API without any
        text formatting or instruction prefixes. Used internally by higher-level
        async methods after text formatting is applied.

        Args:
            texts: List of text strings to embed (should already be formatted).

        Returns:
            A sequence of embedding vectors from the Ollama model.

        Examples:
            - Async direct API call for batch embedding
                ```python
                >>> import asyncio
                >>> from serapeum.ollama import OllamaEmbedding
                >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
                >>> async def get_embeddings():  # doctest: +SKIP
                ...     texts = ["text 1", "text 2", "text 3"]
                ...     vecs = await embedder.aget_general_text_embeddings(texts)
                ...     return len(vecs)
                >>> # asyncio.run(get_embeddings())  # Returns 3

                ```

        See Also:
            get_general_text_embeddings: Synchronous version of this method.
            aget_general_text_embedding: Single text async version.
        """
        result = await self._async_client.embed(
            model=self.model_name,
            input=texts,
            options=self.ollama_additional_kwargs,
            keep_alive=self.keep_alive,
        )
        return result.embeddings

    def get_general_text_embedding(self, texts: str) -> Sequence[float]:
        """Generate a raw embedding for a single text using the Ollama API.

        Low-level method that directly calls the Ollama embed API without any
        text formatting or instruction prefixes. Used internally by higher-level
        methods after text formatting is applied. Returns the first embedding
        from the API response.

        Args:
            texts: The text string to embed (should already be formatted).

        Returns:
            An embedding vector from the Ollama model.

        Examples:
            - Direct API call for single text embedding
                ```python
                >>> from serapeum.ollama import OllamaEmbedding
                >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
                >>> embedding = embedder.get_general_text_embedding("sample text")  # doctest: +SKIP
                >>> len(embedding) > 0  # doctest: +SKIP
                True

                ```

        See Also:
            aget_general_text_embedding: Async version of this method.
            get_general_text_embeddings: Batch version.
        """
        result = self._client.embed(
            model=self.model_name,
            input=texts,
            options=self.ollama_additional_kwargs,
            keep_alive=self.keep_alive,
        )
        return result.embeddings[0]

    async def aget_general_text_embedding(self, prompt: str) -> Sequence[float]:
        """Asynchronously generate a raw embedding for a single text using the Ollama API.

        Async low-level method that directly calls the Ollama embed API without any
        text formatting or instruction prefixes. Used internally by higher-level
        async methods after text formatting is applied. Returns the first embedding
        from the API response.

        Args:
            prompt: The text string to embed (should already be formatted).

        Returns:
            An embedding vector from the Ollama model.

        Examples:
            - Async direct API call for single text embedding
                ```python
                >>> import asyncio
                >>> from serapeum.ollama import OllamaEmbedding
                >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
                >>> async def get_embedding():  # doctest: +SKIP
                ...     vec = await embedder.aget_general_text_embedding("sample text")
                ...     return len(vec) > 0
                >>> # asyncio.run(get_embedding())  # Returns True

                ```

        See Also:
            get_general_text_embedding: Synchronous version of this method.
            aget_general_text_embeddings: Batch async version.
        """
        result = await self._async_client.embed(
            model=self.model_name,
            input=prompt,
            options=self.ollama_additional_kwargs,
            keep_alive=self.keep_alive,
        )
        return result.embeddings[0]

    def _format_query(self, query: str) -> str:
        """Format query with instruction if provided.

        Args:
            query: The query string to format.

        Returns:
            Formatted query string.

        Raises:
            ValueError: If query is empty or whitespace-only after stripping.
        """
        stripped_query = query.strip()

        if not stripped_query:
            raise ValueError(
                "Cannot embed empty or whitespace-only query. "
                "Query becomes empty after stripping whitespace."
            )

        if self.query_instruction:
            return f"{self.query_instruction.strip()} {stripped_query}"
        return stripped_query

    def _format_text(self, text: str) -> str:
        """Format text with instruction if provided.

        Args:
            text: The text string to format.

        Returns:
            Formatted text string.

        Raises:
            ValueError: If text is empty or whitespace-only after stripping.
        """
        stripped_text = text.strip()

        if not stripped_text:
            raise ValueError(
                "Cannot embed empty or whitespace-only text. "
                "Text becomes empty after stripping whitespace."
            )

        if self.text_instruction:
            return f"{self.text_instruction.strip()} {stripped_text}"
        return stripped_text
