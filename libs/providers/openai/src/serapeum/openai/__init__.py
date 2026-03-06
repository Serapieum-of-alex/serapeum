"""OpenAI provider package for Serapeum.

Note: This package is named 'openai' which conflicts with the external 'openai' PyPI package.
Symbols are lazily imported via ``__getattr__`` so that loading this package
does not eagerly pull in the third-party ``openai`` SDK.
"""

from __future__ import annotations

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "OpenAI": ("serapeum.openai.llm", "OpenAI"),
    "SyncOpenAI": ("serapeum.openai.llm", "SyncOpenAI"),
    "AsyncOpenAI": ("serapeum.openai.llm", "AsyncOpenAI"),
    "Tokenizer": ("serapeum.openai.llm", "Tokenizer"),
    "OpenAIResponses": ("serapeum.openai.responses", "OpenAIResponses"),
    "OpenAIClientMixin": ("serapeum.openai.client", "OpenAIClientMixin"),
    "OpenAIModelMixin": ("serapeum.openai.model", "OpenAIModelMixin"),
    "OpenAIStructuredOutputMixin": ("serapeum.openai.structured", "OpenAIStructuredOutputMixin"),
}


def __getattr__(name: str) -> object:
    """Lazily import heavy symbols to avoid triggering openai SDK at collection time."""
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        result: object = getattr(module, attr)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return result


__all__ = [
    "OpenAI",
    "OpenAIClientMixin",
    "OpenAIModelMixin",
    "OpenAIResponses",
    "OpenAIStructuredOutputMixin",
    "Tokenizer",
    "SyncOpenAI",
    "AsyncOpenAI",
]