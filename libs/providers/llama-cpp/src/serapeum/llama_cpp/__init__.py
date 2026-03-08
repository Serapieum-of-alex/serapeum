"""serapeum-llama-cpp — local LLM inference via llama-cpp-python.

This package provides a ``LlamaCPP`` class that runs GGUF quantised models
locally using the `llama-cpp-python <https://github.com/abetlen/llama-cpp-python>`_
backend.  Models can be loaded from a local path, downloaded from a direct URL,
or fetched from HuggingFace Hub.

Typical usage:

- Load a local GGUF model with Llama 3 formatters and explore the instance
    ```python
    >>> import os
    >>> from serapeum.llama_cpp import LlamaCPP
    >>> from serapeum.llama_cpp.formatters.llama3 import (
    ...     messages_to_prompt_v3_instruct,
    ...     completion_to_prompt_v3_instruct,
    ... )
    >>> llm = LlamaCPP(
    ...     model_path=os.environ["LLAMA_MODEL_PATH"],
    ...     temperature=0.1,
    ...     max_new_tokens=256,
    ...     context_window=512,
    ...     messages_to_prompt=messages_to_prompt_v3_instruct,
    ...     completion_to_prompt=completion_to_prompt_v3_instruct,
    ... )
    >>> llm.temperature
    0.1
    >>> llm.max_new_tokens
    256
    >>> LlamaCPP.class_name()
    'LlamaCPP'

    ```

See Also:
    serapeum.llama_cpp.formatters.llama2: Prompt formatters for Llama 2 / Mistral models.
    serapeum.llama_cpp.formatters.llama3: Prompt formatters for Llama 3 Instruct models.
"""


from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serapeum.llama_cpp.llm import LlamaCPP


def __getattr__(name: str) -> object:
    """Lazily import heavy symbols to avoid triggering llama_cpp at collection time."""
    if name == "LlamaCPP":
        from serapeum.llama_cpp.llm import LlamaCPP  # noqa: N811

        result: object = LlamaCPP
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return result


__all__ = [
    "LlamaCPP",
]
