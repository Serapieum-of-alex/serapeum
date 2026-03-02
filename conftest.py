"""Root conftest — loads .env variables before any test session."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()
