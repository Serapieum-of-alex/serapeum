"""Core LLM abstractions and base classes.

This module contains the foundational LLM classes that providers should inherit from
or use to build specialized LLM implementations.
"""

from serapeum.core.llms.abstractions.function_calling import FunctionCallingLLM
from serapeum.core.llms.abstractions.structured_output_llm import StructuredOutputLLM

__all__ = [
    "FunctionCallingLLM",
    "StructuredOutputLLM",
]
