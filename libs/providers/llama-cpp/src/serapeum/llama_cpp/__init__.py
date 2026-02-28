"""serapeum-llama-cpp — local LLM inference via llama-cpp-python.

This package provides a ``LlamaCPP`` class that runs GGUF quantised models
locally using the `llama-cpp-python <https://github.com/abetlen/llama-cpp-python>`_
backend.  Models can be loaded from a local path, downloaded from a direct URL,
or fetched from HuggingFace Hub.

Typical usage:

    ```python
    from serapeum.llama_cpp import LlamaCPP
    from serapeum.llama_cpp.formatters.llama3 import (
        messages_to_prompt_v3_instruct,
        completion_to_prompt_v3_instruct,
    )

    llm = LlamaCPP(  # doctest: +SKIP
        model_path="/models/llama-3-8b-instruct.Q4_0.gguf",
        messages_to_prompt=messages_to_prompt_v3_instruct,
        completion_to_prompt=completion_to_prompt_v3_instruct,
    )
    response = llm.complete("Hello!")  # doctest: +SKIP
    ```

See Also:
    serapeum.llama_cpp.formatters.llama2: Prompt formatters for Llama 2 / Mistral models.
    serapeum.llama_cpp.formatters.llama3: Prompt formatters for Llama 3 Instruct models.
"""

from serapeum.llama_cpp.llm import LlamaCPP

__all__ = [
    "LlamaCPP",
]
