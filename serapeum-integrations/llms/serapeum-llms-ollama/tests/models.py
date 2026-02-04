"""Test data models for Ollama integration tests."""

from typing import List

from pydantic import BaseModel, Field


class Song(BaseModel):
    """A song data model used in tests."""

    title: str = Field(description="The title of the song")


class Album(BaseModel):
    """Album model used as the program output."""

    name: str = Field(description="The name/title of the album")
    artist: str = Field(description="The name of the artist or band")
    songs: List[Song] = Field(description="List of songs in the album")
