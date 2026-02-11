"""Utilities for resolving binary data from various sources into BytesIO objects."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Union



def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
