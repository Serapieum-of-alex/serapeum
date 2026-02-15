"""Test data models for Ollama integration tests."""

import os
from typing import List

from ollama import Client
from pydantic import BaseModel, Field

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



class Song(BaseModel):
    """A song data model used in tests."""

    title: str = Field(description="The title of the song")


class Album(BaseModel):
    """Album model used as the program output."""

    name: str = Field(description="The name/title of the album")
    artist: str = Field(description="The name of the artist or band")
    songs: List[Song] = Field(description="List of songs in the album")
