"""Conftest for openai — loads .env and pre-imports the third-party openai SDK.

Pre-importing the external ``openai`` package BEFORE pytest's --doctest-modules
collection adds ``src/serapeum/`` to sys.path ensures that ``from openai import ...``
in the implementation files resolves to the correct third-party package, not the
local ``serapeum/openai/`` namespace package.
"""

from __future__ import annotations

# Pre-import the third-party openai BEFORE pytest's --doctest-modules
# collection adds src/serapeum/ to sys.path.  Once cached in sys.modules,
# llm.py's `from openai import OpenAI` will find the correct package.
import openai as _openai_ext  # noqa: F401