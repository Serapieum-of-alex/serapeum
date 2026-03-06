from __future__ import annotations

from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from tiktoken import Encoding as Tokenizer


class OpenAIModelMixin:
    """Shared model metadata helpers for OpenAI providers.

    Provides ``_get_model_name`` (strips fine-tuning prefixes) and a
    ``_tokenizer`` property backed by *tiktoken*.  Both ``OpenAI`` and
    ``OpenAIResponses`` inherit this mixin so the logic lives in one place.

    The concrete class must declare a ``model: str`` field.
    """

    model: str  # provided by the concrete Pydantic class

    def _get_model_name(self) -> str:
        model_name = self.model
        if "ft-" in model_name:  # legacy fine-tuning
            model_name = model_name.split(":")[0]
        elif model_name.startswith("ft:"):
            model_name = model_name.split(":")[1]
        return model_name

    @property
    def _tokenizer(self) -> Tokenizer | None:
        """Get a tiktoken tokenizer for this model, or *None* if unknown."""
        return tiktoken.encoding_for_model(self._get_model_name())
