"""Model metadata helpers: tokenizer resolution and fine-tune prefix stripping.

Provides the :class:`Tokenizer` protocol and the :class:`ModelMetadata` mixin
used by all OpenAI-flavoured provider classes to resolve *tiktoken* encodings
and normalise fine-tuned model identifiers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import tiktoken


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for objects that can tokenize text into integer token IDs.

    Any object whose ``encode`` method accepts a ``str`` and returns a
    ``list[int]`` satisfies this protocol.  The *tiktoken* ``Encoding``
    class is the canonical implementation used throughout the OpenAI
    provider package.

    Examples:
        - Verify that a tiktoken encoding satisfies the protocol:
            ```python
            >>> import tiktoken
            >>> from serapeum.openai.llm.base.model import Tokenizer
            >>> enc = tiktoken.get_encoding("cl100k_base")
            >>> tokens = enc.encode("hello world")
            >>> len(tokens) > 0
            True

            ```

    See Also:
        ModelMetadata._tokenizer: Property that returns a ``Tokenizer``
            for the current model.
    """

    def encode(self, text: str) -> list[int]:  # fmt: skip
        """Encode *text* into a list of integer token IDs.

        Args:
            text: The input string to tokenize.

        Returns:
            A list of non-negative integers representing token IDs.
        """
        ...


class ModelMetadata:
    """Shared model metadata helpers for OpenAI provider classes.

    Provides :meth:`_get_model_name` (strips fine-tuning prefixes) and a
    :pyattr:`_tokenizer` property backed by *tiktoken*.  Both
    ``Completions`` and ``Responses`` inherit this mixin so the logic
    lives in one place.

    The concrete class must declare a ``model: str`` field (typically a
    Pydantic ``Field``).

    Examples:
        - Use via a concrete subclass to obtain a tokenizer:
            ```python
            >>> import tiktoken
            >>> from serapeum.openai.llm.base.model import ModelMetadata
            >>> class MyModel(ModelMetadata):
            ...     model = "gpt-4o-mini"
            >>> m = MyModel()
            >>> enc = m._tokenizer
            >>> tokens = enc.encode("hello")
            >>> len(tokens) > 0
            True

            ```
        - Fine-tuned model names are normalised to their base model:
            ```python
            >>> from serapeum.openai.llm.base.model import ModelMetadata
            >>> class FTModel(ModelMetadata):
            ...     model = "ft:gpt-4o-mini:my-org:custom:id"
            >>> FTModel()._get_model_name()
            'gpt-4o-mini'

            ```

    See Also:
        Tokenizer: Protocol satisfied by tiktoken encodings.
    """

    model: str  # provided by the concrete Pydantic class

    def _get_model_name(self) -> str:
        """Return the base model name with fine-tuning prefixes stripped.

        Handles both legacy (``ft-`` infix) and current (``ft:`` prefix)
        fine-tuning naming conventions.  For non-fine-tuned models the
        ``model`` string is returned unchanged.
        """
        model_name = self.model
        if "ft-" in model_name:  # legacy fine-tuning
            model_name = model_name.split(":")[0]
        elif model_name.startswith("ft:"):
            model_name = model_name.split(":")[1]
        return model_name

    @property
    def _tokenizer(self) -> Tokenizer | None:
        """Tiktoken tokenizer for this model, or ``None`` if unknown.

        Resolves the encoding via ``tiktoken.encoding_for_model`` using
        the base model name returned by :meth:`_get_model_name`.
        """
        try:
            value = tiktoken.encoding_for_model(self._get_model_name())
        except KeyError:
            value = None

        return value
