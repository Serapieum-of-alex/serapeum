"""Base embeddings file."""

import asyncio
from abc import abstractmethod
from typing import Coroutine, List, Tuple

from serapeum.core.base.embeddings.base import (
    BaseEmbedding,
    Embedding,
)
# from serapeum.core.callbacks.schema import CBEventType, EventPayload
from serapeum.core.base.llms.models import ImageType
from serapeum.core.utils.base import get_tqdm_iterable


class MultiModalEmbedding(BaseEmbedding):
    """Base class for Multi Modal embeddings."""

    @abstractmethod
    def _get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """
        Embed the input image synchronously.

        Subclasses should implement this method. Reference get_image_embedding's
        docstring for more information.
        """

    @abstractmethod
    async def _aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """
        Embed the input image asynchronously.

        Subclasses should implement this method. Reference get_image_embedding's
        docstring for more information.
        """

    def get_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """
        Embed the input image.
        """
        image_embedding = self._get_image_embedding(img_file_path)
        return image_embedding

    async def aget_image_embedding(self, img_file_path: ImageType) -> Embedding:
        """Get image embedding."""
        image_embedding = await self._aget_image_embedding(img_file_path)
        return image_embedding

    def _get_image_embeddings(self, img_file_paths: List[ImageType]) -> List[Embedding]:
        """
        Embed the input sequence of image synchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        # Default implementation just loops over _get_image_embedding
        return [
            self._get_image_embedding(img_file_path) for img_file_path in img_file_paths
        ]

    async def _aget_image_embeddings(
        self, img_file_paths: List[ImageType]
    ) -> List[Embedding]:
        """
        Embed the input sequence of image asynchronously.

        Subclasses can implement this method if batch queries are supported.
        """
        return await asyncio.gather(
            *[
                self._aget_image_embedding(img_file_path)
                for img_file_path in img_file_paths
            ]
        )

    def get_image_embedding_batch(
        self, img_file_paths: List[ImageType], show_progress: bool = False
    ) -> List[Embedding]:
        """Get a list of image embeddings, with batching."""
        cur_batch: List[ImageType] = []
        result_embeddings: List[Embedding] = []

        queue_with_progress = enumerate(
            get_tqdm_iterable(
                img_file_paths, show_progress, "Generating image embeddings"
            )
        )

        for idx, img_file_path in queue_with_progress:
            cur_batch.append(img_file_path)
            if (
                idx == len(img_file_paths) - 1
                or len(cur_batch) == self.embed_batch_size
            ):
                embeddings = self._get_image_embeddings(cur_batch)
                result_embeddings.extend(embeddings)
                cur_batch = []

        return result_embeddings

    async def aget_image_embedding_batch(
        self, img_file_paths: List[ImageType], show_progress: bool = False
    ) -> List[Embedding]:
        """Asynchronously get a list of image embeddings, with batching."""
        cur_batch: List[ImageType] = []
        callback_payloads: List[Tuple[str, List[ImageType]]] = []
        result_embeddings: List[Embedding] = []
        embeddings_coroutines: List[Coroutine] = []
        for idx, img_file_path in enumerate(img_file_paths):
            cur_batch.append(img_file_path)
            if (
                idx == len(img_file_paths) - 1
                or len(cur_batch) == self.embed_batch_size
            ):
                # flush
                embeddings_coroutines.append(self._aget_image_embeddings(cur_batch))
                cur_batch = []

        # flatten the results of asyncio.gather, which is a list of embeddings lists
        nested_embeddings = []
        if show_progress:
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

        result_embeddings = [
            embedding for embeddings in nested_embeddings for embedding in embeddings
        ]

        return result_embeddings
