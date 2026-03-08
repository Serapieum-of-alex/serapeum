"""Root conftest — loads .env and re-exports shared hooks from core."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Re-export shared pytest hooks from the core conftest
# (--no-skip-doctest, --md-marker flags).
_core_conftest = Path(__file__).resolve().parent / "libs" / "core" / "conftest.py"
_spec = importlib.util.spec_from_file_location("_core_conftest", _core_conftest)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pytest_addoption = _mod.pytest_addoption  # noqa: F401
pytest_collection_modifyitems = _mod.pytest_collection_modifyitems  # noqa: F401
