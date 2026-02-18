"""Test data models for Ollama integration tests."""
from __future__ import annotations
import os
from typing import List

import ollama as ollama_sdk

from pydantic import BaseModel, Field


# created clients in this test file are used in the pytest.mark.skipif decorator and can not be made as fixtures
# so keep them as they are.
# currently there are no embedding models in ollama cloud

api_key: str = os.environ.get("OLLAMA_API_KEY")


# Local client (used by existing tests)
try:
    client = ollama_sdk.Client()        # type: ignore
    client.list()
except Exception:
    client = None


# Cloud client (used by cloud tests)
try:
    cloud_client = ollama_sdk.Client(       # type: ignore
        host="https://api.ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    cloud_client.list()   # connectivity probe — raises if unreachable / unauthorized
except Exception:
    cloud_client = None


class Song(BaseModel):
    """A song data model used in tests."""

    title: str = Field(description="The title of the song")


class Album(BaseModel):
    """Album model used as the program output."""

    name: str = Field(description="The name/title of the album")
    artist: str = Field(description="The name of the artist or band")
    songs: List[Song] = Field(description="List of songs in the album")
