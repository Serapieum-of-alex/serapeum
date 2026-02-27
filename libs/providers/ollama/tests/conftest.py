"""Tests for conftest."""

from __future__ import annotations
import os
import pytest
from pydantic import BaseModel, Field
from .models import is_ci


class Song(BaseModel):
    """A song data model used in tests."""

    title: str = Field(description="The title of the song")


class Album(BaseModel):
    """Album model used as the program output."""

    name: str = Field(description="The name/title of the album")
    artist: str = Field(description="The name of the artist or band")
    songs: list[Song] = Field(description="List of songs in the album")


@pytest.fixture
def ollama_api_key() -> str | None:
    """Return the Ollama API key, or None when running locally without one set."""
    return os.environ.get("OLLAMA_API_KEY")


@pytest.fixture
def ollama_api_key_required() -> str:
    """Return the Ollama API key, raising if not set. Use this for cloud/e2e fixtures only."""
    api_key = os.environ.get("OLLAMA_API_KEY")
    if api_key is None:
        raise ValueError("OLLAMA_API_KEY environment variable is not set")
    return api_key


@pytest.fixture
def local_model() -> str:
    """Model name."""
    return "llama3.1"


@pytest.fixture
def cloud_model() -> str:
    return "qwen3-next:80b"


@pytest.fixture
def model_name(local_model: str, cloud_model: str) -> str:
    if is_ci:
        name = cloud_model
    else:
        name = local_model
    return name


@pytest.fixture
def embedding_model_cloud() -> str:
    return "nomic-embed-text"


@pytest.fixture
def llm_model(model_name: str, ollama_api_key: str):
    from serapeum.ollama import Ollama as serapeum_ollama

    if is_ci:
        api_key = ollama_api_key
    else:
        api_key = None

    return serapeum_ollama(
        model=model_name,
        request_timeout=180,
        temperature=0.0,  # Use temperature=0 for deterministic test results
        api_key=api_key,
    )


@pytest.fixture
def cloud_llm(cloud_model: str, ollama_api_key_required: str):
    """Return an Ollama instance configured for the cloud backend.

    Uses the api_key and test_model from the shared test models module.
    temperature=0.0 for deterministic responses; request_timeout=120 for
    cloud latency.
    """
    from serapeum.ollama import Ollama

    return Ollama(
        model=cloud_model,
        api_key=ollama_api_key_required,
        request_timeout=120,
        temperature=0.0,
    )


@pytest.fixture
def album() -> type[Album]:
    return Album
