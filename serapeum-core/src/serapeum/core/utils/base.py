"""Utilities for resolving binary data from various sources into BytesIO objects."""
import os
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import requests


def resolve_binary(
    raw_bytes: Optional[bytes] = None,
    path: Optional[Union[str, Path]] = None,
    url: Optional[str] = None,
    as_base64: bool = False,
) -> BytesIO:
    """Resolve binary data from various sources into a BytesIO object.

    Args:
        raw_bytes:
            Raw bytes data
        path:
            File path to read bytes from
        url:
            URL to fetch bytes from
        as_base64:
            Whether to base64 encode the output bytes

    Returns:
        BytesIO object containing the binary data

    Raises:
        ValueError: If no valid source is provided
    """
    # Handle raw bytes input
    if raw_bytes is not None:
        try:
            # Try to decode the bytes as base64
            decoded_bytes = base64.b64decode(raw_bytes)
        except Exception:
            # the bytes are already raw binary data
            decoded_bytes = raw_bytes

        if as_base64:
            # Re-encode the decoded bytes to base64
            return BytesIO(base64.b64encode(decoded_bytes))

        # Return the decoded binary data
        buffer = BytesIO(decoded_bytes)

    # Handle file path input
    elif path is not None:
        path = Path(path) if isinstance(path, str) else path

        # Read file content as bytes
        data = path.read_bytes()

        # Check if the caller wants base64-encoded output
        if as_base64:
            # Encode the file bytes to base64 and wrap in BytesIO
            return BytesIO(base64.b64encode(data))

        # Create a BytesIO buffer containing the raw file bytes
        buffer = BytesIO(data)

    # Handle URL input
    elif url is not None:
        # Parse the URL to extract its components (scheme, path, etc.)
        parsed_url = urlparse(url)

        # Special handling for data: URLs (embedded data in the URL itself)
        if parsed_url.scheme == "data":
            # Data URL format: data:[<mediatype>][;base64],<data>
            # The path attribute contains everything after "data:"
            data_part = parsed_url.path

            # Validate that the data URL has the required comma separator
            if "," not in data_part:
                raise ValueError("Invalid data URL format: missing comma separator")

            # Split the data URL into metadata (mediatype and encoding) and actual data
            # Only split on the first comma to preserve commas in the data portion
            metadata, url_data = data_part.split(",", 1)

            # Check if the metadata indicates base64 encoding
            is_base64_encoded = metadata.endswith(";base64")

            # Handle base64-encoded data URLs
            if is_base64_encoded:
                # Decode the base64 string from the URL to get raw binary data
                decoded_data = base64.b64decode(url_data)

                # Check if the caller wants base64-encoded output
                if as_base64:
                    # Re-encode to base64 and wrap in BytesIO
                    return BytesIO(base64.b64encode(decoded_data))
                else:
                    # Return the decoded binary data wrapped in BytesIO
                    return BytesIO(decoded_data)

            # Handle plain text data URLs (not base64-encoded)
            else:
                # Check if the caller wants base64-encoded output
                if as_base64:
                    # Convert the text to UTF-8 bytes, then encode to base64
                    return BytesIO(base64.b64encode(url_data.encode("utf-8")))
                else:
                    # Convert the text to UTF-8 bytes and wrap in BytesIO
                    return BytesIO(url_data.encode("utf-8"))

        # Handle HTTP(S) URLs - fetch data from the network
        # Create empty headers dict (placeholder for future authentication/headers)
        headers = {}

        # Make HTTP GET request to fetch the data from the URL
        response = requests.get(url, headers=headers)

        # Raise an exception if the request failed (4xx or 5xx status codes)
        response.raise_for_status()

        # Check if the caller wants base64-encoded output
        if as_base64:
            # Encode the response content to base64 and wrap in BytesIO
            return BytesIO(base64.b64encode(response.content))

        # Create a BytesIO buffer containing the raw response content
        buffer = BytesIO(response.content)

    # Branch 4: No valid source provided
    else:
        # Raise an error if none of the input parameters were provided
        raise ValueError("No valid source provided to resolve binary data!")

    # Return the buffer created in one of the branches above
    return buffer


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."
