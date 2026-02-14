"""Tests for conftest."""
import os
import pytest

from ollama import Client
from serapeum.ollama import Ollama


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


test_model = os.environ.get("OLLAMA_TEST_MODEL", "llama3.1:latest")
try:
    client = Client()
    models = client.list()

    model_found = False
    for model in models["models"]:
        if model["model"] == test_model:
            model_found = True
            break

    if not model_found:
        client = None
except Exception:
    client = None


@pytest.fixture
def model_name():
    return test_model