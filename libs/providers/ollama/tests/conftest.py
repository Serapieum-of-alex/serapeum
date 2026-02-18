"""Tests for conftest."""

import pytest


@pytest.fixture
def local_model() -> str:
    """Model name."""
    return "llama3.1"


@pytest.fixture
def embedding_model_cloud() -> str:
    return "nomic-embed-text"


@pytest.fixture
def cloud_model() -> str:
    return "qwen3-next:80b"


@pytest.fixture
def llm_model(local_model: str):
    from serapeum.ollama import Ollama as serapeum_ollama
    return serapeum_ollama(
        model=local_model,
        request_timeout=180,
        temperature=0.0,  # Use temperature=0 for deterministic test results
    )