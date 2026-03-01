"""Prompt formatters for llama-cpp-python chat templates.

This subpackage provides ready-made prompt formatters for popular GGUF model
families.  Select the formatter that matches the model you are loading and pass
it to :class:`~serapeum.llama_cpp.LlamaCPP` at construction time.

Available formatters:

- :mod:`serapeum.llama_cpp.formatters.llama2` — Llama 2 ``[INST]``/``<<SYS>>``
  format, compatible with Llama 2 Chat and Mistral Instruct.
- :mod:`serapeum.llama_cpp.formatters.llama3` — Llama 3 Instruct
  ``<|start_header_id|>``/``<|eot_id|>`` format.

Typical usage::

    from serapeum.llama_cpp.formatters.llama3 import (
        messages_to_prompt_v3_instruct,
        completion_to_prompt_v3_instruct,
    )
    llm = LlamaCPP(
        model_path="...",
        messages_to_prompt=messages_to_prompt_v3_instruct,
        completion_to_prompt=completion_to_prompt_v3_instruct,
    )

See Also:
    serapeum.llama_cpp.formatters.llama2: Formatter for Llama 2 / Mistral models.
    serapeum.llama_cpp.formatters.llama3: Formatter for Llama 3 Instruct models.
"""

from serapeum.llama_cpp.formatters import llama2, llama3

__all__ = [
    "llama2",
    "llama3",
]
