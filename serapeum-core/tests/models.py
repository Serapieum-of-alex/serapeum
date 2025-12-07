from typing import List

from pydantic import BaseModel


class MockSong(BaseModel):
    """Mock Song class."""

    title: str


class MockAlbum(BaseModel):
    title: str
    artist: str
    songs: List[MockSong]


MOCK_ALBUM = MockAlbum(
    title="hello",
    artist="world",
    songs=[MockSong(title="song1"), MockSong(title="song2")],
)

MOCK_ALBUM_2 = MockAlbum(
    title="hello2",
    artist="world2",
    songs=[MockSong(title="song3"), MockSong(title="song4")],
)
