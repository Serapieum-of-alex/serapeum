"""Tests for conftest."""

import pytest

from serapeum.llms.ollama import Ollama


@pytest.fixture
def model_name() -> str:
    """Model name."""
    return "llama3.1"


@pytest.fixture
def llm_model(model_name: str) -> Ollama:
    return Ollama(
        model=model_name,
        request_timeout=180,
        temperature=0.0,  # Use temperature=0 for deterministic test results
    )
