from typing import List
from pydantic import BaseModel


class Song(BaseModel):
    """A song data model used in tests."""

    title: str


class Album(BaseModel):
    """Album model used as the program output."""

    name: str
    artist: str
    songs: List[Song]
