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
