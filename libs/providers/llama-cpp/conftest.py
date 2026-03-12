"""Conftest for llama-cpp — loads .env and injects a shared doctest namespace."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

# Pre-import the third-party llama_cpp BEFORE pytest's --doctest-modules
# collection adds src/serapeum/ to sys.path.  Once cached in sys.modules,
# llm.py's `from llama_cpp import Llama` will find the correct package.
import llama_cpp as _llama_cpp_ext  # noqa: F401
import pytest

# Re-export shared pytest hooks from the core conftest (--no-skip-doctest flag).
_core_conftest = Path(__file__).resolve().parents[2] / "core" / "conftest.py"
_spec = importlib.util.spec_from_file_location("_core_conftest", _core_conftest)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
pytest_addoption = _mod.pytest_addoption  # noqa: F401
pytest_collection_modifyitems = _mod.pytest_collection_modifyitems  # noqa: F401
from dotenv import load_dotenv  # noqa: E402

load_dotenv()


@pytest.fixture(autouse=True)
def doctest_namespace(doctest_namespace: dict[str, Any]) -> dict[str, Any]:
    """Inject pre-loaded LlamaCPP and common imports into every doctest."""
    import asyncio

    from serapeum.llama_cpp import LlamaCPP
    from serapeum.llama_cpp.formatters.llama2 import (
        completion_to_prompt,
        messages_to_prompt,
    )
    from serapeum.llama_cpp.formatters.llama3 import (
        completion_to_prompt_v3_instruct,
        messages_to_prompt_v3_instruct,
    )

    doctest_namespace["os"] = os
    doctest_namespace["asyncio"] = asyncio
    doctest_namespace["LlamaCPP"] = LlamaCPP
    doctest_namespace["messages_to_prompt_v3_instruct"] = messages_to_prompt_v3_instruct
    doctest_namespace["completion_to_prompt_v3_instruct"] = (
        completion_to_prompt_v3_instruct
    )
    doctest_namespace["messages_to_prompt"] = messages_to_prompt
    doctest_namespace["completion_to_prompt"] = completion_to_prompt

    model_path = os.environ.get("LLAMA_MODEL_PATH")
    if model_path:
        llm = LlamaCPP(
            model_path=model_path,
            temperature=0.1,
            max_new_tokens=256,
            context_window=512,
            messages_to_prompt=messages_to_prompt_v3_instruct,
            completion_to_prompt=completion_to_prompt_v3_instruct,
            verbose=False,
        )
        doctest_namespace["llm"] = llm

    return doctest_namespace
