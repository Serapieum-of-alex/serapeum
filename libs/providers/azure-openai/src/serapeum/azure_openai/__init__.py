"""Azure OpenAI provider package for Serapeum.

Symbols are lazily imported via ``__getattr__`` so that loading this package
does not eagerly pull in the third-party ``openai`` SDK.
"""

from __future__ import annotations

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AzureClient": ("serapeum.azure_openai.llm", "AzureClient"),
    "Completions": ("serapeum.azure_openai.llm", "Completions"),
    "Responses": ("serapeum.azure_openai.llm", "Responses"),
    "SyncAzureOpenAI": ("serapeum.azure_openai.llm", "SyncAzureOpenAI"),
    "AsyncAzureOpenAI": ("serapeum.azure_openai.llm", "AsyncAzureOpenAI"),
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
    "AzureClient",
    "Completions",
    "Responses",
    "SyncAzureOpenAI",
    "AsyncAzureOpenAI",
]
