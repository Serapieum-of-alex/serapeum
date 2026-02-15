"""Base embedding abstractions for text and query vectorization.

This module provides the foundational interfaces and utilities for embedding text
into high-dimensional vector spaces. It includes:

- BaseEmbedding: Abstract base class for all embedding implementations
- SimilarityMode: Enum for different similarity/distance metrics
- Utility functions for embedding aggregation and similarity computation
- Caching support for efficient repeated embedding operations

The module supports both synchronous and asynchronous operations, with optional
batching and progress tracking for large-scale embedding tasks.
"""

from __future__ import annotations
import asyncio
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Coroutine, Sequence

import numpy as np
from pydantic import (
    Field,
    ConfigDict,
)

from serapeum.core.types import SerializableModel
from serapeum.core.base.embeddings.types import (
    BaseNode,
    MetadataMode,
    CallMixin,
)
from serapeum.core.utils.base import get_tqdm_iterable, run_jobs

Embedding = list[float]

DEFAULT_EMBED_BATCH_SIZE = 10


class SimilarityMode(str, Enum):
    """Similarity and distance metrics for comparing embeddings.

    This enum defines the supported modes for calculating similarity or distance
    between embedding vectors. Different modes are suited for different use cases
    and model types.

    Attributes:
        DEFAULT: Cosine similarity (normalized dot product). Range: [-1, 1].
            Most commonly used for semantic similarity in NLP tasks.
        DOT_PRODUCT: Raw dot product of vectors. Range: unbounded.
            Faster than cosine but sensitive to vector magnitude.
        EUCLIDEAN: Negative Euclidean distance. Range: (-∞, 0].
            Uses negative distance so higher values indicate greater similarity,
            maintaining consistent ranking order with other modes.

    Examples:
        - Using the default cosine similarity
            ```python
            >>> from serapeum.core.base.embeddings.base import SimilarityMode
            >>> mode = SimilarityMode.DEFAULT
            >>> mode.value
            'cosine'

            ```

        - Comparing different similarity modes
            ```python
            >>> from serapeum.core.base.embeddings.base import SimilarityMode
            >>> SimilarityMode.DOT_PRODUCT.value
            'dot_product'
            >>> SimilarityMode.EUCLIDEAN.value
            'euclidean'

            ```

    See Also:
        similarity: Function that uses these modes to compute embedding similarity.
        BaseEmbedding.similarity: Static method for computing embedding similarity.
    """

    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def mean_agg(embeddings: list[Embedding]) -> Embedding:
    """Compute the mean (average) of multiple embedding vectors.

    This function aggregates a list of embedding vectors by computing their
    element-wise mean. This is useful for combining multiple embeddings into
    a single representative vector, such as when creating a document embedding
    from multiple sentence embeddings.

    Args:
        embeddings: List of embedding vectors to aggregate. Each embedding
            should be a list of floats with the same dimensionality.

    Returns:
        A single embedding vector representing the mean of all input embeddings.
        The output has the same dimensionality as the input embeddings.

    Raises:
        ValueError: If the embeddings list is empty.

    Examples:
        - Averaging two simple 2D embeddings
            ```python
            >>> from serapeum.core.base.embeddings.base import mean_agg
            >>> emb1 = [1.0, 2.0]
            >>> emb2 = [3.0, 4.0]
            >>> result = mean_agg([emb1, emb2])
            >>> result
            [2.0, 3.0]

            ```

        - Averaging three 3D embeddings
            ```python
            >>> emb1 = [1.0, 0.0, 0.0]
            >>> emb2 = [0.0, 1.0, 0.0]
            >>> emb3 = [0.0, 0.0, 1.0]
            >>> result = mean_agg([emb1, emb2, emb3])
            >>> [round(x, 6) for x in result]
            [0.333333, 0.333333, 0.333333]

            ```

        - Error handling for empty list
            ```python
            >>> mean_agg([])  # doctest: +ELLIPSIS
            Traceback (most recent call last):
                ...
            ValueError: No embeddings to aggregate

            ```

    See Also:
        BaseEmbedding.get_agg_embedding_from_queries: Uses this function by default
            for aggregating query embeddings.
    """
    if not embeddings:
        raise ValueError("No embeddings to aggregate")

    return np.array(embeddings).mean(axis=0).tolist()


def similarity(
    embedding1: Embedding,
    embedding2: Embedding,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    """Calculate similarity between two embedding vectors.

    This function computes the similarity or distance between two embedding vectors
    using the specified mode. The calculation method depends on the mode parameter.

    Args:
        embedding1: First embedding vector (list of floats).
        embedding2: Second embedding vector (list of floats). Must have the same
            dimensionality as embedding1.
        mode: Similarity computation mode. Defaults to SimilarityMode.DEFAULT
            (cosine similarity).

    Returns:
        Similarity score as a float. The interpretation depends on the mode:
            - Cosine: Range [-1, 1], where 1 is identical, 0 is orthogonal,
              -1 is opposite.
            - Dot product: Unbounded, higher values indicate greater similarity.
            - Euclidean: Range (-∞, 0], where 0 is identical, more negative
              values indicate greater distance.

    Examples:
        - Computing cosine similarity (default mode)
            ```python
            >>> from serapeum.core.base.embeddings.base import similarity
            >>> emb1 = [1.0, 0.0, 0.0]
            >>> emb2 = [1.0, 0.0, 0.0]
            >>> float(similarity(emb1, emb2))
            1.0

            ```

        - Comparing similar vs orthogonal vectors
            ```python
            >>> emb1 = [1.0, 0.0]
            >>> emb2 = [0.0, 1.0]
            >>> round(float(similarity(emb1, emb2)), 6)
            0.0

            ```

        - Using dot product mode
            ```python
            >>> emb1 = [2.0, 3.0]
            >>> emb2 = [4.0, 5.0]
            >>> float(similarity(emb1, emb2, mode=SimilarityMode.DOT_PRODUCT))
            23.0

            ```

        - Using Euclidean distance mode
            ```python
            >>> emb1 = [1.0, 1.0]
            >>> emb2 = [1.0, 1.0]
            >>> abs(float(similarity(emb1, emb2, mode=SimilarityMode.EUCLIDEAN)))
            0.0

            ```

    See Also:
        SimilarityMode: Enum defining available similarity modes.
        BaseEmbedding.similarity: Static method wrapper for this function.
    """
    if mode == SimilarityMode.EUCLIDEAN:
        # Using -euclidean distance as similarity to achieve same ranking order
        val = -float(np.linalg.norm(np.array(embedding1) - np.array(embedding2)))
    elif mode == SimilarityMode.DOT_PRODUCT:
        val = np.dot(embedding1, embedding2)
    else:
        product = np.dot(embedding1, embedding2)
        norm = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        val = product / norm

    return val


class BaseEmbedding(SerializableModel, CallMixin, ABC):
    """Abstract base class for all embedding model implementations.

    This class provides the core interface and shared functionality for converting
    text into dense vector embeddings. It supports both query and document embedding,
    with optional caching, batching, and async operations.

    Subclasses must implement the abstract methods for generating embeddings from
    text and queries. The class handles caching, batching, and progress tracking
    automatically.

    Attributes:
        model_name: Name of the embedding model. Defaults to "unknown".
        batch_size: Number of texts to process in each batch. Must be between
            1 and 2048. Defaults to 10.
        num_workers: Number of worker threads for async operations. If None,
            uses default async behavior without worker pooling.
        cache_store: Optional key-value store for caching embeddings. Must implement
            get(), aget(), put(), and aput() methods. When provided, embeddings are
            cached using a key combining text and model configuration.

    Notes:
        This is an abstract base class and cannot be instantiated directly.
        Subclasses must implement _get_query_embedding, _aget_query_embedding,
        and _get_text_embedding methods.

    See Also:
        serapeum.providers.ollama.embeddings.OllamaEmbedding: Concrete implementation
            for Ollama embedding models.
        CallMixin: Mixin providing __call__ and acall methods.
        SerializableModel: Base Pydantic model with serialization support.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_name: str = Field(
        default="unknown", description="The name of the embedding model."
    )
    batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )
    num_workers: int | None = Field(
        default=None,
        description="The number of workers to use for async embedding calls.",
    )

    cache_store: Any | None = Field(
        default=None,
        description=(
            "Key-value store for caching embeddings. Must implement get(), aget(), "
            "put(), and aput() methods with signature: get(key: str, collection: str) -> dict | None. "
            "When provided, embeddings are cached using a key that combines the text and model configuration. "
            "If None, embeddings are not cached and will be recomputed on each call."
        ),
    )

    def _get_cache_key(self, text: str) -> str:
        """Generate a unique cache key combining text and model configuration.

        The cache key includes both the input text and a JSON representation of
        the model configuration, ensuring that different models or configurations
        don't share cached embeddings. Sensitive fields like api_key are excluded.

        Args:
            text: Input text to generate a cache key for.

        Returns:
            Cache key string in format "{text}::{model_config_json}".

        Notes:
            The following fields are excluded from the cache key:
                - api_key: Sensitive credential information
                - cache_store: Avoid circular reference in serialization
        """
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        model_dict.pop("cache_store", None)  # Avoid circular reference

        # Create a deterministic string representation
        import json

        model_str = json.dumps(model_dict, sort_keys=True)
        return f"{text}::{model_str}"

    @abstractmethod
    def _get_query_embedding(self, query: str) -> Embedding:
        """Embed the input query synchronously (internal implementation).

        This is an internal method that subclasses must implement to provide
        the core query embedding functionality. The public method get_query_embedding()
        handles caching and calls this method when needed.

        Query embeddings may use special instructions or prefixes depending on the
        model. For example, some models prepend "Represent the question for
        retrieving supporting documents: " to optimize for retrieval tasks.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector as a list of floats.

        See Also:
            get_query_embedding: Public method that handles caching and delegates
                to this method.
            _aget_query_embedding: Async version of this method.
        """

    @abstractmethod
    async def _aget_query_embedding(self, query: str) -> Embedding:
        """Embed the input query asynchronously (internal implementation).

        This is an internal async method that subclasses must implement to provide
        the core query embedding functionality. The public method aget_query_embedding()
        handles caching and calls this method when needed.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector as a list of floats.

        See Also:
            aget_query_embedding: Public async method that handles caching and
                delegates to this method.
            _get_query_embedding: Sync version of this method.
        """

    def get_query_embedding(self, query: str) -> Embedding:
        """Generate an embedding vector for a query string.

        Embeds the input query into a dense vector representation optimized for
        retrieval tasks. When caching is enabled, checks the cache first and stores
        new embeddings automatically.

        Depending on the model, a special instruction may be prepended to the raw
        query string to optimize for specific tasks. For example, some models use
        "Represent the question for retrieving supporting documents: ".

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector as a list of floats.

        See Also:
            aget_query_embedding: Async version of this method.
            get_text_embedding: For embedding document text (not queries).
            _get_query_embedding: Internal implementation method.
        """
        query_embedding = None
        if self.cache_store:
            cache_key = self._get_cache_key(query)
            cached = self.cache_store.get(key=cache_key, collection="embeddings")
            if cached:
                cached_key = next(iter(cached.keys()))
                query_embedding = cached[cached_key]

        if query_embedding is None:
            query_embedding = self._get_query_embedding(query)
            if self.cache_store:
                cache_key = self._get_cache_key(query)
                self.cache_store.put(
                    key=cache_key,
                    val={str(uuid.uuid4()): query_embedding},
                    collection="embeddings",
                )

        return query_embedding

    async def aget_query_embedding(self, query: str) -> Embedding:
        """Asynchronously generate an embedding vector for a query string.

        Async version of get_query_embedding(). Embeds the input query into a dense
        vector representation with cache support.

        Args:
            query: Query text to embed.

        Returns:
            Embedding vector as a list of floats.

        See Also:
            get_query_embedding: Sync version of this method.
            aget_text_embedding: For embedding document text asynchronously.
            _aget_query_embedding: Internal async implementation method.
        """
        query_embedding = None
        if self.cache_store:
            cache_key = self._get_cache_key(query)
            cached = await self.cache_store.aget(key=cache_key, collection="embeddings")
            if cached:
                cached_key = next(iter(cached.keys()))
                query_embedding = cached[cached_key]

        if query_embedding is None:
            query_embedding = await self._aget_query_embedding(query)
            if self.cache_store:
                cache_key = self._get_cache_key(query)
                await self.cache_store.aput(
                    key=cache_key,
                    val={str(uuid.uuid4()): query_embedding},
                    collection="embeddings",
                )

        return query_embedding

    def get_agg_embedding_from_queries(
        self,
        queries: list[str],
        agg_fn: Callable[..., Embedding] | None = None,
    ) -> Embedding:
        """Generate a single aggregated embedding from multiple query strings.

        Embeds each query individually and then combines them using an aggregation
        function. This is useful for creating a unified representation from multiple
        related queries or questions.

        Args:
            queries: List of query strings to embed and aggregate.
            agg_fn: Optional aggregation function that takes a list of embeddings
                and returns a single embedding. Defaults to mean_agg (arithmetic mean).

        Returns:
            Single aggregated embedding vector as a list of floats.

        See Also:
            aget_agg_embedding_from_queries: Async version of this method.
            mean_agg: Default aggregation function.
            get_query_embedding: Used internally to embed each query.
        """
        query_embeddings = [self.get_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    async def aget_agg_embedding_from_queries(
        self,
        queries: list[str],
        agg_fn: Callable[..., Embedding] | None = None,
    ) -> Embedding:
        """Asynchronously generate an aggregated embedding from multiple queries.

        Async version of get_agg_embedding_from_queries(). Embeds each query
        asynchronously and then combines them using an aggregation function.

        Args:
            queries: List of query strings to embed and aggregate.
            agg_fn: Optional aggregation function that takes a list of embeddings
                and returns a single embedding. Defaults to mean_agg.

        Returns:
            Single aggregated embedding vector as a list of floats.

        See Also:
            get_agg_embedding_from_queries: Sync version of this method.
            aget_query_embedding: Used internally to embed each query.
            mean_agg: Default aggregation function.
        """
        query_embeddings = [await self.aget_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    @abstractmethod
    def _get_text_embedding(self, text: str) -> Embedding:
        """Embed document text synchronously (internal implementation).

        This is an internal method that subclasses must implement to provide
        the core text embedding functionality. The public method get_text_embedding()
        handles caching and calls this method when needed.

        Text embeddings may use different instructions or prefixes than query
        embeddings. For example, some models prepend "Represent the document for
        retrieval: " to optimize for document representation.

        Args:
            text: Document text to embed.

        Returns:
            Embedding vector as a list of floats.

        See Also:
            get_text_embedding: Public method that handles caching and delegates
                to this method.
            _aget_text_embedding: Async version of this method.
            _get_query_embedding: For embedding queries (not documents).
        """

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Embed document text asynchronously (internal implementation).

        This is an internal async method that subclasses can override to provide
        true async text embedding. The default implementation falls back to the
        sync method _get_text_embedding().

        Args:
            text: Document text to embed.

        Returns:
            Embedding vector as a list of floats.

        Notes:
            Subclasses should override this method if they have a native async
            implementation. Otherwise, the default fallback to the sync method
            is used.

        See Also:
            aget_text_embedding: Public async method that handles caching.
            _get_text_embedding: Sync version used as fallback.
        """
        # Default implementation just falls back on _get_text_embedding
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """Embed multiple texts synchronously in batch (internal implementation).

        This internal method provides batch embedding functionality. Subclasses
        can override this method to provide more efficient batch processing if
        supported by the underlying model API.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one for each input text, in the same order.

        Notes:
            The default implementation simply loops over _get_text_embedding().
            Subclasses should override this if they can process batches more
            efficiently.

        See Also:
            get_text_embedding_batch: Public method for batch embedding with
                batching and progress tracking.
            _aget_text_embeddings: Async version of this method.
        """
        # Default implementation just loops over _get_text_embedding
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """Embed multiple texts asynchronously in batch (internal implementation).

        This internal async method provides batch embedding functionality using
        asyncio.gather for concurrent processing. Subclasses can override this
        for more efficient batch processing.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one for each input text, in the same order.

        Notes:
            The default implementation uses asyncio.gather to process all texts
            concurrently via _aget_text_embedding(). Subclasses should override
            this if they can process batches more efficiently.

        See Also:
            aget_text_embedding_batch: Public async method for batch embedding.
            _get_text_embeddings: Sync version of this method.
        """
        return await asyncio.gather(
            *[self._aget_text_embedding(text) for text in texts]
        )

    def _get_text_embeddings_cached(self, texts: list[str]) -> list[Embedding]:
        """Retrieve text embeddings from cache or generate if not cached.

        This internal method checks the cache for each text and only generates
        embeddings for texts not found in the cache. Newly generated embeddings
        are automatically stored in the cache.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one for each input text, preserving order.

        Raises:
            ValueError: If cache_store is None when this method is called.

        See Also:
            _aget_text_embeddings_cached: Async version of this method.
            get_text_embedding_batch: Public method that uses this for caching.
        """
        if self.cache_store is None:
            raise ValueError("embeddings_cache must be defined")

        embeddings: list[Embedding | None] = [None for i in range(len(texts))]
        # Tuples of (index, text) to be able to keep same order of embeddings
        non_cached_texts: list[tuple[int, str]] = []
        for i, txt in enumerate(texts):
            cache_key = self._get_cache_key(txt)
            cached_emb = self.cache_store.get(key=cache_key, collection="embeddings")
            if cached_emb is not None:
                cached_key = next(iter(cached_emb.keys()))
                embeddings[i] = cached_emb[cached_key]
            else:
                non_cached_texts.append((i, txt))
        if len(non_cached_texts) > 0:
            text_embeddings = self._get_text_embeddings(
                [x[1] for x in non_cached_texts]
            )
            for j, text_embedding in enumerate(text_embeddings):
                orig_i = non_cached_texts[j][0]
                embeddings[orig_i] = text_embedding

                cache_key = self._get_cache_key(texts[orig_i])
                self.cache_store.put(
                    key=cache_key,
                    val={str(uuid.uuid4()): text_embedding},
                    collection="embeddings",
                )
        return embeddings

    async def _aget_text_embeddings_cached(self, texts: list[str]) -> list[Embedding]:
        """Asynchronously retrieve text embeddings from cache or generate them.

        Async version of _get_text_embeddings_cached(). Checks the cache for each
        text and generates embeddings only for texts not found. Newly generated
        embeddings are automatically stored in the cache.

        Args:
            texts: List of document texts to embed.

        Returns:
            List of embedding vectors, one for each input text, preserving order.

        Raises:
            ValueError: If cache_store is None when this method is called.

        See Also:
            _get_text_embeddings_cached: Sync version of this method.
            aget_text_embedding_batch: Public async method that uses this.
        """
        if self.cache_store is None:
            raise ValueError("embeddings_cache must be defined")

        embeddings: list[Embedding | None] = [None for i in range(len(texts))]
        # Tuples of (index, text) to be able to keep same order of embeddings
        non_cached_texts: list[tuple[int, str]] = []
        for i, txt in enumerate(texts):
            cache_key = self._get_cache_key(txt)
            cached_emb = await self.cache_store.aget(
                key=cache_key, collection="embeddings"
            )
            if cached_emb is not None:
                cached_key = next(iter(cached_emb.keys()))
                embeddings[i] = cached_emb[cached_key]
            else:
                non_cached_texts.append((i, txt))

        if len(non_cached_texts) > 0:
            text_embeddings = await self._aget_text_embeddings(
                [x[1] for x in non_cached_texts]
            )
            for j, text_embedding in enumerate(text_embeddings):
                orig_i = non_cached_texts[j][0]
                embeddings[orig_i] = text_embedding
                cache_key = self._get_cache_key(texts[orig_i])
                await self.cache_store.aput(
                    key=cache_key,
                    val={str(uuid.uuid4()): text_embedding},
                    collection="embeddings",
                )
        return embeddings

    def get_text_embedding(self, text: str) -> Embedding:
        """Generate an embedding vector for document text.

        Embeds the input text into a dense vector representation optimized for
        document representation tasks. When caching is enabled, checks the cache
        first and stores new embeddings automatically.

        Depending on the model, a special instruction may be prepended to the raw
        text string to optimize for document retrieval. For example, some models
        use "Represent the document for retrieval: ".

        Args:
            text: Document text to embed.

        Returns:
            Embedding vector as a list of floats.

        See Also:
            aget_text_embedding: Async version of this method.
            get_query_embedding: For embedding queries (not documents).
            get_text_embedding_batch: For embedding multiple texts efficiently.
            _get_text_embedding: Internal implementation method.
        """
        if not self.cache_store:
            text_embedding = self._get_text_embedding(text)
        elif self.cache_store is not None:
            cache_key = self._get_cache_key(text)
            cached_emb = self.cache_store.get(key=cache_key, collection="embeddings")
            if cached_emb is not None:
                cached_key = next(iter(cached_emb.keys()))
                text_embedding = cached_emb[cached_key]
            else:
                text_embedding = self._get_text_embedding(text)
                cache_key = self._get_cache_key(text)
                self.cache_store.put(
                    key=cache_key,
                    val={str(uuid.uuid4()): text_embedding},
                    collection="embeddings",
                )

        return text_embedding

    async def aget_text_embedding(self, text: str) -> Embedding:
        """Asynchronously generate an embedding vector for document text.

        Async version of get_text_embedding(). Embeds the input text into a dense
        vector representation with cache support.

        Args:
            text: Document text to embed.

        Returns:
            Embedding vector as a list of floats.

        See Also:
            get_text_embedding: Sync version of this method.
            aget_query_embedding: For embedding queries asynchronously.
            aget_text_embedding_batch: For embedding multiple texts efficiently.
            _aget_text_embedding: Internal async implementation method.
        """
        if not self.cache_store:
            text_embedding = await self._aget_text_embedding(text)
        elif self.cache_store is not None:
            cache_key = self._get_cache_key(text)
            cached_emb = await self.cache_store.aget(
                key=cache_key, collection="embeddings"
            )
            if cached_emb is not None:
                cached_key = next(iter(cached_emb.keys()))
                text_embedding = cached_emb[cached_key]
            else:
                text_embedding = await self._aget_text_embedding(text)
                cache_key = self._get_cache_key(text)
                await self.cache_store.aput(
                    key=cache_key,
                    val={str(uuid.uuid4()): text_embedding},
                    collection="embeddings",
                )

        return text_embedding

    def get_text_embedding_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[Embedding]:
        """Generate embeddings for multiple texts with automatic batching.

        Processes a list of texts in batches according to self.batch_size. Supports
        optional progress tracking and automatic caching if cache_store is configured.

        Args:
            texts: List of document texts to embed.
            show_progress: Whether to display a progress bar. Defaults to False.
            **kwargs: Additional keyword arguments (reserved for future use).

        Returns:
            List of embedding vectors, one for each input text, in the same order.

        See Also:
            aget_text_embedding_batch: Async version with parallel processing.
            get_text_embedding: For embedding a single text.
            _get_text_embeddings: Internal batch processing method.
            _get_text_embeddings_cached: Internal cached batch processing.
        """
        cur_batch: list[str] = []
        result_embeddings: list[Embedding] = []

        queue_with_progress = enumerate(
            get_tqdm_iterable(texts, show_progress, "Generating embeddings")
        )

        for idx, text in queue_with_progress:
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.batch_size:
                # flush
                if not self.cache_store:
                    embeddings = self._get_text_embeddings(cur_batch)
                elif self.cache_store is not None:
                    embeddings = self._get_text_embeddings_cached(cur_batch)
                result_embeddings.extend(embeddings)

                cur_batch = []

        return result_embeddings

    async def aget_text_embedding_batch(
        self,
        texts: list[str],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> list[Embedding]:
        """Asynchronously generate embeddings for multiple texts with batching.

        Async version of get_text_embedding_batch(). Processes texts in batches
        with concurrent execution for improved performance. Supports worker pooling
        if num_workers is set.

        Args:
            texts: List of document texts to embed.
            show_progress: Whether to display a progress bar. Defaults to False.
                Requires tqdm package for progress tracking.
            **kwargs: Additional keyword arguments (reserved for future use).

        Returns:
            List of embedding vectors, one for each input text, in the same order.

        Notes:
            When num_workers > 1, uses worker pooling for concurrent batch processing.
            When show_progress=True, attempts to use tqdm.asyncio for progress tracking.

        See Also:
            get_text_embedding_batch: Sync version of this method.
            aget_text_embedding: For embedding a single text asynchronously.
            _aget_text_embeddings: Internal async batch processing method.
        """
        num_workers = self.num_workers

        cur_batch: list[str] = []
        embeddings_coroutines: list[Coroutine] = []

        # for idx, text in queue_with_progress:
        for idx, text in enumerate(texts):
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.batch_size:
                # flush

                if not self.cache_store:
                    embeddings_coroutines.append(self._aget_text_embeddings(cur_batch))
                elif self.cache_store is not None:
                    embeddings_coroutines.append(
                        self._aget_text_embeddings_cached(cur_batch)
                    )

                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        if len(embeddings_coroutines) > 0:
            if num_workers and num_workers > 1:
                nested_embeddings = await run_jobs(
                    embeddings_coroutines,
                    show_progress=show_progress,
                    workers=self.num_workers,
                    desc="Generating embeddings",
                )
            elif show_progress:
                try:
                    from tqdm.asyncio import tqdm_asyncio

                    nested_embeddings = await tqdm_asyncio.gather(
                        *embeddings_coroutines,
                        total=len(embeddings_coroutines),
                        desc="Generating embeddings",
                    )
                except ImportError:
                    nested_embeddings = await asyncio.gather(*embeddings_coroutines)
            else:
                nested_embeddings = await asyncio.gather(*embeddings_coroutines)
        else:
            nested_embeddings = []

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]
        return result_embeddings

    @staticmethod
    def similarity(
        embedding1: Embedding,
        embedding2: Embedding,
        mode: SimilarityMode = SimilarityMode.DEFAULT,
    ) -> float:
        """Calculate similarity between two embedding vectors.

        Static method wrapper for the module-level similarity() function. Provides
        a convenient way to compute similarity directly from the class.

        Args:
            embedding1: First embedding vector (list of floats).
            embedding2: Second embedding vector (list of floats).
            mode: Similarity computation mode. Defaults to cosine similarity.

        Returns:
            Similarity score as a float. Interpretation depends on the mode.

        Examples:
            - Computing cosine similarity
                ```python
                >>> from serapeum.core.embeddings import BaseEmbedding
                >>> emb1 = [1.0, 0.0]
                >>> emb2 = [1.0, 0.0]
                >>> float(BaseEmbedding.similarity(emb1, emb2))
                1.0

                ```

            - Using different similarity modes
                ```python
                >>> emb1 = [3.0, 4.0]
                >>> emb2 = [3.0, 4.0]
                >>> float(BaseEmbedding.similarity(emb1, emb2, mode=SimilarityMode.DOT_PRODUCT))
                25.0

                ```

        See Also:
            similarity: Module-level function that performs the actual calculation.
            SimilarityMode: Enum defining available similarity modes.
        """
        return similarity(embedding1=embedding1, embedding2=embedding2, mode=mode)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        """Embed a sequence of nodes by calling the embedding model.

        This makes the embedding model callable, allowing it to be used as a function.
        Extracts text content from each node, generates embeddings, and assigns them
        back to the nodes.

        Args:
            nodes: Sequence of BaseNode objects to embed.
            **kwargs: Additional keyword arguments passed to get_text_embedding_batch.

        Returns:
            The input sequence of nodes with embeddings assigned to each node's
            embedding attribute.

        See Also:
            acall: Async version of this method.
            get_text_embedding_batch: Method used internally for batch embedding.
            MetadataMode.EMBED: Mode used to extract content from nodes.
        """
        embeddings = self.get_text_embedding_batch(
            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
            **kwargs,
        )

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        """Asynchronously embed a sequence of nodes.

        Async version of __call__(). Extracts text content from each node,
        generates embeddings asynchronously, and assigns them back to the nodes.

        Args:
            nodes: Sequence of BaseNode objects to embed.
            **kwargs: Additional keyword arguments passed to aget_text_embedding_batch.

        Returns:
            The input sequence of nodes with embeddings assigned to each node's
            embedding attribute.

        See Also:
            __call__: Sync version of this method.
            aget_text_embedding_batch: Method used internally for async batch embedding.
            MetadataMode.EMBED: Mode used to extract content from nodes.
        """
        embeddings = await self.aget_text_embedding_batch(
            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
            **kwargs,
        )

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes
