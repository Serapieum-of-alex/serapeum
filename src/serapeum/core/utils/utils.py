import base64
import requests
from io import BytesIO
from urllib.parse import urlparse
from pathlib import Path
from typing import Optional, Union


def resolve_binary(
    raw_bytes: Optional[bytes] = None,
    path: Optional[Union[str, Path]] = None,
    url: Optional[str] = None,
    as_base64: bool = False,
) -> BytesIO:
    """Resolve binary data from various sources into a BytesIO object.

    Args:
        raw_bytes: Raw bytes data
        path: File path to read bytes from
        url: URL to fetch bytes from
        as_base64: Whether to base64 encode the output bytes

    Returns:
        BytesIO object containing the binary data

    Raises:
        ValueError: If no valid source is provided
    """
    if raw_bytes is not None:
        # check if raw_bytes is base64 encoded
        try:
            decoded_bytes = base64.b64decode(raw_bytes)
        except Exception:
            decoded_bytes = raw_bytes

        if as_base64:
            return BytesIO(base64.b64encode(decoded_bytes))
        return BytesIO(decoded_bytes)

    elif path is not None:
        path = Path(path) if isinstance(path, str) else path
        data = path.read_bytes()
        if as_base64:
            return BytesIO(base64.b64encode(data))
        return BytesIO(data)

    elif url is not None:
        parsed_url = urlparse(url)
        if parsed_url.scheme == "data":
            # Parse data URL: data:[<mediatype>][;base64],<data>
            # The path contains everything after "data:"
            data_part = parsed_url.path

            # Split on the first comma to separate metadata from data
            if "," not in data_part:
                raise ValueError("Invalid data URL format: missing comma separator")

            metadata, url_data = data_part.split(",", 1)
            is_base64_encoded = metadata.endswith(";base64")

            if is_base64_encoded:
                # Data is base64 encoded in the URL
                decoded_data = base64.b64decode(url_data)
                if as_base64:
                    # Return as base64 bytes
                    return BytesIO(base64.b64encode(decoded_data))
                else:
                    # Return decoded binary data
                    return BytesIO(decoded_data)
            else:
                # Data is not base64 encoded in the URL (URL-encoded text)
                if as_base64:
                    # Encode the text data as base64
                    return BytesIO(base64.b64encode(url_data.encode("utf-8")))
                else:
                    # Return as text bytes
                    return BytesIO(url_data.encode("utf-8"))
        headers = {}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        if as_base64:
            return BytesIO(base64.b64encode(response.content))
        return BytesIO(response.content)

    raise ValueError("No valid source provided to resolve binary data!")


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."