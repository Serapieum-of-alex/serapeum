"""Ollama embeddings implementation for text and query vectorization.

This module provides the OllamaEmbedding class for generating embeddings using
Ollama models. It supports both symmetric and asymmetric embedding patterns,
allowing different instructions for queries vs. documents to optimize retrieval
performance. All operations support both synchronous and asynchronous execution.
"""

from __future__ import annotations
from typing import Any, Sequence

from pydantic import Field

from serapeum.core.embeddings import BaseEmbedding
from serapeum.ollama.client import OllamaClientMixin


class OllamaEmbedding(OllamaClientMixin, BaseEmbedding):  # type: ignore[misc]
    """Ollama-based embedding model for generating text and query vector representations.

    Wraps the Ollama SDK embed API to produce dense float vectors from text. Inherits
    connection management from ``OllamaClientMixin``, which supplies ``base_url``,
    ``api_key``, and lazily-created sync/async SDK clients.

    **Local vs Ollama Cloud**

    Without ``api_key`` the class talks to a local Ollama server at
    ``http://localhost:11434``. **To switch to Ollama Cloud, set** ``api_key``
    **— that is the only change required.** When ``api_key`` is provided and
    ``base_url`` is still the local default, ``base_url`` is automatically
    switched to ``https://api.ollama.com``; no manual URL update is needed.
    An explicit non-default ``base_url`` is always preserved so custom remote
    deployments are unaffected. ``api_key`` is excluded from ``model_dump()``
    and ``model_dump_json()`` to prevent accidental credential leakage.

    **Lazy client initialisation**

    The underlying ``ollama.Client`` and ``ollama.AsyncClient`` instances are created
    on first access of the ``client`` / ``async_client`` properties, not at
    construction time. Pass pre-built SDK clients via the ``client=`` and
    ``async_client=`` constructor kwargs to inject mock objects in tests — they are
    intercepted before Pydantic validation and stored in private attributes.

    **Asymmetric embeddings**

    Set ``query_instruction`` and ``text_instruction`` to apply different prefixes when
    embedding queries versus documents, which can significantly improve retrieval
    accuracy for models that support asymmetric representations (e.g., nomic-embed-text,
    mxbai-embed-large).

    Args:
        model_name: The Ollama model to use for embeddings (e.g., ``"nomic-embed-text"``).
        base_url: Base URL where the Ollama server is hosted. Defaults to
            ``"http://localhost:11434"``. Automatically switched to
            ``https://api.ollama.com`` when ``api_key`` is provided.
        api_key: The single switch between local and cloud. When ``None``
            (default), requests go to the local Ollama server. When set,
            requests are routed to Ollama Cloud and ``base_url`` is
            automatically updated. **Excluded from** ``model_dump()`` /
            ``model_dump_json()`` — use environment variables or a secrets
            manager rather than persisting the serialised model.
        ollama_additional_kwargs: Extra options forwarded to the Ollama ``embed`` API
            (e.g., ``{"mirostat": 1}``). Defaults to ``{}``.
        query_instruction: Instruction prefix prepended to search queries before
            embedding (e.g., ``"search_query:"``). Helps models distinguish query
            embeddings from document embeddings for asymmetric retrieval.
        text_instruction: Instruction prefix prepended to documents before embedding
            (e.g., ``"search_document:"``). Paired with ``query_instruction`` for
            asymmetric embedding patterns.
        keep_alive: How long to keep the model loaded in memory after a request.
            Accepts a duration string (e.g., ``"5m"``, ``"1h"``) or a float in
            seconds. Defaults to ``"5m"``.
        client_kwargs: Additional keyword arguments forwarded to the Ollama client
            constructor (merged with the base kwargs built from ``base_url`` and
            ``api_key``). Custom headers take precedence over the ``Authorization``
            header generated from ``api_key``.
        client: Pre-built ``ollama.Client`` injected for testing. Intercepted before
            Pydantic validation; the ``client`` property returns this object on first
            access without creating a new one.
        async_client: Pre-built ``ollama.AsyncClient`` injected for testing. Works
            the same way as the ``client`` parameter.

    Examples:
        - Basic text embedding against a local Ollama server
            ```python
            >>> from serapeum.ollama import OllamaEmbedding  # type: ignore
            >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")
            >>> embedding = embedder.get_text_embedding("Hello world")  # doctest: +SKIP
            >>> len(embedding) > 0  # doctest: +SKIP
            True

            ```
        - Connect to Ollama Cloud with an API key (base_url auto-switches)
            ```python
            >>> from serapeum.ollama import OllamaEmbedding  # type: ignore
            >>> from serapeum.ollama.client import OLLAMA_CLOUD_BASE_URL
            >>> embedder = OllamaEmbedding(
            ...     model_name="nomic-embed-text",
            ...     api_key="sk-my-ollama-key",
            ... )
            >>> embedder.base_url == OLLAMA_CLOUD_BASE_URL
            True

            ```
        - Verify api_key is excluded from serialisation
            ```python
            >>> from serapeum.ollama import OllamaEmbedding  # type: ignore
            >>> embedder = OllamaEmbedding(
            ...     model_name="nomic-embed-text",
            ...     api_key="sk-secret",
            ... )
            >>> "api_key" in embedder.model_dump()
            False

            ```
        - Inject a mock client to test embedding logic without a running server
            ```python
            >>> from unittest.mock import MagicMock
            >>> from serapeum.ollama import OllamaEmbedding  # type: ignore
            >>> mock_client = MagicMock()
            >>> mock_client.embed.return_value.embeddings = [[0.1, 0.2, 0.3]]
            >>> embedder = OllamaEmbedding(
            ...     model_name="nomic-embed-text",
            ...     client=mock_client,
            ... )
            >>> embedder.get_text_embedding("test")
            [0.1, 0.2, 0.3]

            ```
        - Asymmetric embeddings for retrieval (query vs. document)
            ```python
            >>> from serapeum.ollama import OllamaEmbedding  # type: ignore
            >>> embedder = OllamaEmbedding(  # doctest: +SKIP
            ...     model_name="nomic-embed-text",
            ...     query_instruction="search_query:",
            ...     text_instruction="search_document:",
            ... )
            >>> query_vec = embedder.get_query_embedding("What is Python?")    # doctest: +SKIP
            >>> doc_vec = embedder.get_text_embedding("Python is a language")  # doctest: +SKIP
            >>> len(query_vec) == len(doc_vec)  # doctest: +SKIP
            True

            ```
        - Batch embed multiple documents in one call
            ```python
            >>> from serapeum.ollama import OllamaEmbedding  # type: ignore
            >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
            >>> docs = ["First document", "Second document", "Third document"]
            >>> embeddings = embedder.get_text_embedding_batch(docs)  # doctest: +SKIP
            >>> len(embeddings) == 3  # doctest: +SKIP
            True

            ```
        - Async batch embedding
            ```python
            >>> import asyncio
            >>> from serapeum.ollama import OllamaEmbedding  # type: ignore
            >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")  # doctest: +SKIP
            >>> async def embed_batch():  # doctest: +SKIP
            ...     docs = ["Doc 1", "Doc 2", "Doc 3"]
            ...     vecs = await embedder.aget_text_embedding_batch(docs)
            ...     return len(vecs)
            >>> asyncio.run(embed_batch())  # doctest: +SKIP
            3

            ```
        - List available models on the connected server
            ```python
            >>> from serapeum.ollama import OllamaEmbedding  # type: ignore
            >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")
            >>> models = embedder.list_models()   # doctest: +SKIP
            >>> isinstance(models, list)          # doctest: +SKIP
            True

            ```
        - Async model listing
            ```python
            >>> import asyncio
            >>> from serapeum.ollama import OllamaEmbedding  # type: ignore
            >>> embedder = OllamaEmbedding(model_name="nomic-embed-text")
            >>> async def show_models():                # doctest: +SKIP
            ...     return await embedder.alist_models()
            >>> asyncio.run(show_models())              # doctest: +SKIP
            ['nomic-embed-text:latest', 'mxbai-embed-large:latest']

            ```

    See Also:
        Ollama: Chat / completion LLM from the same Ollama provider.
        OllamaClientMixin: Shared connection mixin (base_url, api_key, lazy clients).
        get_text_embedding: Embed a single document string.
        get_query_embedding: Embed a query string with optional instruction prefix.
        list_models: List all models available on the connected Ollama server.
        alist_models: Async variant of ``list_models``.
    """

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

    def _build_client_kwargs(self) -> dict[str, Any]:
        """Extend base client kwargs with any extra client_kwargs for the embedding client.

        Headers are merged rather than replaced so that an Authorization header
        from ``api_key`` is preserved alongside any custom headers in ``client_kwargs``.
        Custom headers in ``client_kwargs`` take precedence in case of key conflicts.
        """
        base = super()._build_client_kwargs()
        extra = dict(self.client_kwargs)
        if "headers" in extra and "headers" in base:
            extra["headers"] = {**base.pop("headers"), **extra["headers"]}
        return {**base, **extra}

    @classmethod
    def class_name(cls) -> str:
        """Return the canonical class name for this embedding implementation.

        Returns:
            The string "OllamaEmbedding".

        Examples:
            - Get the class name identifier
                ```python
                >>> from serapeum.ollama import OllamaEmbedding     # type: ignore
                >>> OllamaEmbedding.class_name()
                'OllamaEmbedding'

                ```
        """
        return "OllamaEmbedding"

    def _get_query_embedding(self, query: str) -> list[float]:
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
                >>> from serapeum.ollama import OllamaEmbedding     # type: ignore
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
        return self._embed_raw(formatted_query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
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
                >>> from serapeum.ollama import OllamaEmbedding     # type: ignore
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
        return await self._a_embed_raw(formatted_query)

    def _get_text_embedding(self, text: str) -> list[float]:
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
                >>> from serapeum.ollama import OllamaEmbedding     # type: ignore
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
        return self._embed_raw(formatted_text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
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
                >>> from serapeum.ollama import OllamaEmbedding     # type: ignore
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
        return await self._a_embed_raw(formatted_text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
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
                >>> from serapeum.ollama import OllamaEmbedding     # type: ignore
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
        return self._embed_batch_raw(formatted_texts)

    async def _aget_text_embeddings(
        self, texts: list[str]
    ) -> list[list[float]]:
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
                >>> from serapeum.ollama import OllamaEmbedding     # type: ignore
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
        return await self._a_embed_batch_raw(formatted_texts)

    def _embed_batch_raw(self, texts: list[str]) -> list[list[float]]:
        """Generate raw embeddings for multiple texts using the Ollama API.

        Low-level private method that directly calls the Ollama embed API without any
        text formatting or instruction prefixes. Used internally by higher-level
        methods after text formatting is applied.

        Args:
            texts: List of text strings to embed (should already be formatted).

        Returns:
            A sequence of embedding vectors from the Ollama model.

        See Also:
            _a_embed_batch_raw: Async version of this method.
            _embed_raw: Single text version.
        """
        result = self.client.embed(
            model=self.model_name,
            input=texts,
            options=self.ollama_additional_kwargs,
            keep_alive=self.keep_alive,
        )
        return result.embeddings  # type: ignore[no-any-return]

    async def _a_embed_batch_raw(self, texts: list[str]) -> list[list[float]]:
        """Asynchronously generate raw embeddings for multiple texts using the Ollama API.

        Async low-level private method that directly calls the Ollama embed API without any
        text formatting or instruction prefixes. Used internally by higher-level
        async methods after text formatting is applied.

        Args:
            texts: List of text strings to embed (should already be formatted).

        Returns:
            A sequence of embedding vectors from the Ollama model.

        See Also:
            _embed_batch_raw: Synchronous version of this method.
            _a_embed_raw: Single text async version.
        """
        result = await self.async_client.embed(
            model=self.model_name,
            input=texts,
            options=self.ollama_additional_kwargs,
            keep_alive=self.keep_alive,
        )
        return result.embeddings  # type: ignore[no-any-return]

    def _embed_raw(self, text: str) -> list[float]:
        """Generate a raw embedding for a single text using the Ollama API.

        Low-level private method that directly calls the Ollama embed API without any
        text formatting or instruction prefixes. Used internally by higher-level
        methods after text formatting is applied. Returns the first embedding
        from the API response.

        Args:
            text: The text string to embed (should already be formatted).

        Returns:
            An embedding vector from the Ollama model.

        See Also:
            _a_embed_raw: Async version of this method.
            _embed_batch_raw: Batch version.
        """
        result = self.client.embed(
            model=self.model_name,
            input=text,
            options=self.ollama_additional_kwargs,
            keep_alive=self.keep_alive,
        )
        return result.embeddings[0]

    async def _a_embed_raw(self, text: str) -> Sequence[float]:
        """Asynchronously generate a raw embedding for a single text using the Ollama API.

        Async low-level private method that directly calls the Ollama embed API without any
        text formatting or instruction prefixes. Used internally by higher-level
        async methods after text formatting is applied. Returns the first embedding
        from the API response.

        Args:
            text: The text string to embed (should already be formatted).

        Returns:
            An embedding vector from the Ollama model.

        See Also:
            _embed_raw: Synchronous version of this method.
            _a_embed_batch_raw: Batch async version.
        """
        result = await self.async_client.embed(
            model=self.model_name,
            input=text,
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
