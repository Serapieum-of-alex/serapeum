"""Ollama provider package for Serapeum.

Note: This package is named 'ollama' which conflicts with the external 'ollama' PyPI package.
The external package is imported as 'ollama_sdk' in the implementation files to avoid circular imports.
"""

from __future__ import annotations
from serapeum.ollama.embedding import OllamaEmbedding
from serapeum.ollama.llm import Ollama

__all__ = ["Ollama", "OllamaEmbedding"]
