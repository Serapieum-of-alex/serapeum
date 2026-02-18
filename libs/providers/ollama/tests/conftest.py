"""Tests for conftest."""
from __future__ import annotations
import os
import pytest


@pytest.fixture
def ollama_api_key() -> str:
    """Model name."""
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key is None:
        raise ValueError("OLLAMA_API_KEY environment variable is not set")
    return api_key


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

@pytest.fixture
def cloud_llm(cloud_model: str, ollama_api_key: str):
    """Return an Ollama instance configured for the cloud backend.

    Uses the api_key and test_model from the shared test models module.
    temperature=0.0 for deterministic responses; request_timeout=120 for
    cloud latency.
    """
    from serapeum.ollama import Ollama
    return Ollama(
        model=cloud_model,
        api_key=ollama_api_key,
        request_timeout=120,
        temperature=0.0,
    )