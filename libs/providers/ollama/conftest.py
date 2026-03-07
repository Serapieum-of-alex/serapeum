"""Conftest for ollama — pre-imports the third-party ollama SDK.

Pre-importing the external ``ollama`` package BEFORE pytest's --doctest-modules
collection adds ``src/serapeum/`` to sys.path ensures that ``import ollama``
in the implementation files resolves to the correct third-party package, not the
local ``serapeum/ollama/`` namespace package.
"""

from __future__ import annotations

# Pre-import the third-party ollama BEFORE pytest's --doctest-modules
# collection adds src/serapeum/ to sys.path.  Once cached in sys.modules,
# llm.py's `import ollama as ollama_sdk` will find the correct package.
import ollama as _ollama_ext  # noqa: F401
