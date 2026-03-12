"""OpenAI provider package for Serapeum.

Note: This package is named 'openai' which conflicts with the external 'openai' PyPI package.
Symbols are lazily imported via ``__getattr__`` so that loading this package
does not eagerly pull in the third-party ``openai`` SDK.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serapeum.openai.llm import Completions, Responses

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Completions": ("serapeum.openai.llm", "Completions"),
    "Responses": ("serapeum.openai.llm", "Responses"),
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
    "Completions",
    "Responses",
]
