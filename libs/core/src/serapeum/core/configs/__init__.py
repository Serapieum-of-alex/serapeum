"""Configuration package for default settings and environment management.

This module exposes a small public API (Configs plus default constants) while
avoiding import-time side effects. The Configs class is provided lazily via
PEP 562 module __getattr__, so importing this package does not immediately
import serapeum.core.configs.configs. That helps prevent circular imports and
keeps startup costs low. The __dir__ implementation mirrors __all__ so
introspection and IDE autocompletion show the intended public surface even
though Configs is loaded on first access.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serapeum.core.configs.configs import Configs as Configs

from serapeum.core.configs.defaults import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)

__all__ = [
    "Configs",
    "DEFAULT_CONTEXT_WINDOW",
    "DEFAULT_NUM_OUTPUTS",
    "DEFAULT_TEMPERATURE",
]


def __getattr__(name: str):
    """Lazily expose Configs to avoid import-time side effects/cycles."""
    if name == "Configs":
        from serapeum.core.configs.configs import Configs as _Configs

        return _Configs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Ensure introspection/autocomplete reflects the public API in __all__."""
    return sorted(__all__)
