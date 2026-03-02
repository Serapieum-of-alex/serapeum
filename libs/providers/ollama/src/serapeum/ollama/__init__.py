"""Ollama provider package for Serapeum.

Note: This package is named 'ollama' which conflicts with the external 'ollama' PyPI package.
The external package is imported as 'ollama_sdk' in the implementation files to avoid circular imports.
Symbols are lazily imported via ``__getattr__`` so that loading this package
does not eagerly pull in the third-party ``ollama`` SDK.
"""

from __future__ import annotations

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Ollama": ("serapeum.ollama.llm", "Ollama"),
    "OllamaEmbedding": ("serapeum.ollama.embedding", "OllamaEmbedding"),
}


def __getattr__(name: str) -> object:
    """Lazily import heavy symbols to avoid triggering ollama SDK at collection time."""
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        result: object = getattr(module, attr)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return result


__all__ = ["Ollama", "OllamaEmbedding"]
