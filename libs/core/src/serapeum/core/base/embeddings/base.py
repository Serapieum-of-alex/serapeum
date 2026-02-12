"""Base embeddings file."""

import asyncio
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Coroutine, Sequence, cast

import numpy as np
from pydantic import (
    Field,
    ConfigDict,
)
from serapeum.core.configs import (
    DEFAULT_EMBED_BATCH_SIZE,
)
from serapeum.core.base.embeddings.types import (
    BaseNode,
    MetadataMode,
    TransformComponent,
)
from serapeum.core.utils.base import get_tqdm_iterable, run_jobs

Embedding = list[float]


class SimilarityMode(str, Enum):
    """Modes for similarity/distance."""

    DEFAULT = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def mean_agg(embeddings: list[Embedding]) -> Embedding:
    """Mean aggregation for embeddings."""
    if not embeddings:
        raise ValueError("No embeddings to aggregate")

    return np.array(embeddings).mean(axis=0).tolist()


def similarity(
    embedding1: Embedding,
    embedding2: Embedding,
    mode: SimilarityMode = SimilarityMode.DEFAULT,
) -> float:
    """Get embedding similarity."""
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


class BaseEmbedding(TransformComponent, ABC):
    """Base class for embeddings."""

    model_config = ConfigDict(
        protected_namespaces=("pydantic_model_",), arbitrary_types_allowed=True
    )
    model_name: str = Field(
        default="unknown", description="The name of the embedding model."
    )
    embed_batch_size: int = Field(
        default=DEFAULT_EMBED_BATCH_SIZE,
        description="The batch size for embedding calls.",
        gt=0,
        le=2048,
    )
    num_workers: int | None = Field(
        default=None,
        description="The number of workers to use for async embedding calls.",
    )

    embeddings_cache: Any | None = Field(
        default=None,
        description="Cache for the embeddings: if None, the embeddings are not cached",
    )

    @abstractmethod
    def _get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query synchronously.

        Subclasses should implement this method. Reference get_query_embedding's
        docstring for more information.
        """

    @abstractmethod
    async def _aget_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query asynchronously.

        Subclasses should implement this method. Reference get_query_embedding's
        docstring for more information.
        """

    def get_query_embedding(self, query: str) -> Embedding:
        """
        Embed the input query.

        When embedding a query, depending on the model, a special instruction
        can be prepended to the raw query string. For example, "Represent the
        question for retrieving supporting documents: ". If you're curious,
        other examples of predefined instructions can be found in
        embeddings/huggingface_utils.py.
        """
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)

        query_embedding = None
        if self.embeddings_cache:
            cached = self.embeddings_cache.get(key=query, collection="embeddings")
            if cached:
                cached_key = next(iter(cached.keys()))
                query_embedding = cached[cached_key]

        if query_embedding is None:
            query_embedding = self._get_query_embedding(query)
            if self.embeddings_cache:
                self.embeddings_cache.put(
                    key=query,
                    val={str(uuid.uuid4()): query_embedding},
                    collection="embeddings",
                )

        return query_embedding

    async def aget_query_embedding(self, query: str) -> Embedding:
        """Get query embedding."""
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)

        query_embedding = None
        if self.embeddings_cache:
            cached = await self.embeddings_cache.aget(key=query, collection="embeddings")
            if cached:
                cached_key = next(iter(cached.keys()))
                query_embedding = cached[cached_key]

        if query_embedding is None:
            query_embedding = await self._aget_query_embedding(query)
            if self.embeddings_cache:
                await self.embeddings_cache.aput(
                    key=query,
                    val={str(uuid.uuid4()): query_embedding},
                    collection="embeddings",
                )

        return query_embedding

    def get_agg_embedding_from_queries(
        self,
        queries: list[str],
        agg_fn: Callable[..., Embedding] | None = None,
    ) -> Embedding:
        """Get aggregated embedding from multiple queries."""
        query_embeddings = [self.get_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    async def aget_agg_embedding_from_queries(
        self,
        queries: list[str],
        agg_fn: Callable[..., Embedding] | None = None,
    ) -> Embedding:
        """Async get aggregated embedding from multiple queries."""
        query_embeddings = [await self.aget_query_embedding(query) for query in queries]
        agg_fn = agg_fn or mean_agg
        return agg_fn(query_embeddings)

    @abstractmethod
    def _get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text synchronously.

        Subclasses should implement this method. Reference get_text_embedding's
        docstring for more information.
        """

    async def _aget_text_embedding(self, text: str) -> Embedding:
        """Embed the input text asynchronously.

        Subclasses can implement this method if there is a true async
        implementation. Reference get_text_embedding's docstring for more
        information.
        """
        # Default implementation just falls back on _get_text_embedding
        return self._get_text_embedding(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """Embed the input sequence of text synchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        # Default implementation just loops over _get_text_embedding
        return [self._get_text_embedding(text) for text in texts]

    async def _aget_text_embeddings(self, texts: list[str]) -> list[Embedding]:
        """Embed the input sequence of text asynchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        return await asyncio.gather(
            *[self._aget_text_embedding(text) for text in texts]
        )

    def _get_text_embeddings_cached(self, texts: list[str]) -> list[Embedding]:
        """Get text embeddings from cache.

        If not in cache, generate them.
        """
        if self.embeddings_cache is None:
            raise ValueError("embeddings_cache must be defined")

        embeddings: list[Embedding | None] = [None for i in range(len(texts))]
        # Tuples of (index, text) to be able to keep same order of embeddings
        non_cached_texts: list[tuple[int, str]] = []
        for i, txt in enumerate(texts):
            cached_emb = self.embeddings_cache.get(key=txt, collection="embeddings")
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

                self.embeddings_cache.put(
                    key=texts[orig_i],
                    val={str(uuid.uuid4()): text_embedding},
                    collection="embeddings",
                )
        return embeddings

    async def _aget_text_embeddings_cached(self, texts: list[str]) -> list[Embedding]:
        """Asynchronously get text embeddings from cache.

        If not in cache, generate them.
        """
        if self.embeddings_cache is None:
            raise ValueError("embeddings_cache must be defined")

        embeddings: list[Embedding | None] = [None for i in range(len(texts))]
        # Tuples of (index, text) to be able to keep same order of embeddings
        non_cached_texts: list[tuple[int, str]] = []
        for i, txt in enumerate(texts):
            cached_emb = await self.embeddings_cache.aget(
                key=txt, collection="embeddings"
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
                await self.embeddings_cache.aput(
                    key=texts[orig_i],
                    val={str(uuid.uuid4()): text_embedding},
                    collection="embeddings",
                )
        return embeddings

    def get_text_embedding(self, text: str) -> Embedding:
        """
        Embed the input text.

        When embedding text, depending on the model, a special instruction
        can be prepended to the raw text string. For example, "Represent the
        document for retrieval: ". If you're curious, other examples of
        predefined instructions can be found in embeddings/huggingface_utils.py.
        """
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)

        if not self.embeddings_cache:
            text_embedding = self._get_text_embedding(text)
        elif self.embeddings_cache is not None:
            cached_emb = self.embeddings_cache.get(key=text, collection="embeddings")
            if cached_emb is not None:
                cached_key = next(iter(cached_emb.keys()))
                text_embedding = cached_emb[cached_key]
            else:
                text_embedding = self._get_text_embedding(text)
                self.embeddings_cache.put(
                    key=text,
                    val={str(uuid.uuid4()): text_embedding},
                    collection="embeddings",
                )

        return text_embedding

    async def aget_text_embedding(self, text: str) -> Embedding:
        """Async get text embedding."""
        model_dict = self.to_dict()
        model_dict.pop("api_key", None)

        if not self.embeddings_cache:
            text_embedding = await self._aget_text_embedding(text)
        elif self.embeddings_cache is not None:
            cached_emb = await self.embeddings_cache.aget(
                key=text, collection="embeddings"
            )
            if cached_emb is not None:
                cached_key = next(iter(cached_emb.keys()))
                text_embedding = cached_emb[cached_key]
            else:
                text_embedding = await self._aget_text_embedding(text)
                await self.embeddings_cache.aput(
                    key=text,
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
        """Get a list of text embeddings, with batching."""
        cur_batch: list[str] = []
        result_embeddings: list[Embedding] = []

        queue_with_progress = enumerate(
            get_tqdm_iterable(texts, show_progress, "Generating embeddings")
        )

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)
        for idx, text in queue_with_progress:
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush
                if not self.embeddings_cache:
                    embeddings = self._get_text_embeddings(cur_batch)
                elif self.embeddings_cache is not None:
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
        """Asynchronously get a list of text embeddings, with batching."""
        num_workers = self.num_workers

        model_dict = self.to_dict()
        model_dict.pop("api_key", None)

        cur_batch: list[str] = []
        embeddings_coroutines: list[Coroutine] = []

        # for idx, text in queue_with_progress:
        for idx, text in enumerate(texts):
            cur_batch.append(text)
            if idx == len(texts) - 1 or len(cur_batch) == self.embed_batch_size:
                # flush

                if not self.embeddings_cache:
                    embeddings_coroutines.append(self._aget_text_embeddings(cur_batch))
                elif self.embeddings_cache is not None:
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
        """Get embedding similarity."""
        return similarity(embedding1=embedding1, embedding2=embedding2, mode=mode)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
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
        embeddings = await self.aget_text_embedding_batch(
            [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes],
            **kwargs,
        )

        for node, embedding in zip(nodes, embeddings):
            node.embedding = embedding

        return nodes
