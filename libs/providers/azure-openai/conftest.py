"""Conftest for azure-openai — pre-imports the third-party openai SDK.

Pre-importing the external ``openai`` package BEFORE pytest's --doctest-modules
collection adds ``src/serapeum/`` to sys.path ensures that ``from openai import ...``
in the implementation files resolves to the correct third-party package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Pre-import the third-party openai BEFORE pytest's --doctest-modules
# collection adds src/serapeum/ to sys.path.  Once cached in sys.modules,
# llm.py's `from openai import AzureOpenAI` will find the correct package.
import openai as _openai_ext  # noqa: F401

# Re-export shared pytest hooks from the core conftest (--no-skip-doctest flag).
_core_conftest = Path(__file__).resolve().parents[2] / "core" / "conftest.py"
_spec = importlib.util.spec_from_file_location("_core_conftest", _core_conftest)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pytest_addoption = _mod.pytest_addoption  # noqa: F401
pytest_collection_modifyitems = _mod.pytest_collection_modifyitems  # noqa: F401
