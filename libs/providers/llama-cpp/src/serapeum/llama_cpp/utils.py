from __future__ import annotations
from pathlib import Path
import logging
import math
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _fetch_model_file(model_url: str, model_path: Path) -> None:
    """Download a model file from a URL to a local path with progress reporting.

    Streams the response in 1 MiB chunks to avoid loading the entire file into
    memory. Validates that the server reports a ``Content-Length`` of at least
    1 MB before writing any bytes. On any failure the partially written file is
    removed and the original exception is re-raised unchanged.

    Args:
        model_url: Fully-qualified URL of the GGUF model file to download.
        model_path: Destination path where the model file will be written.
            The parent directory must already exist.

    Raises:
        ValueError: If the server's ``Content-Length`` header is absent or
            reports fewer than 1 000 000 bytes, indicating the response is
            not a valid model file.
        requests.exceptions.HTTPError: If the server returns a 4xx or 5xx
            HTTP status code.
        requests.exceptions.ConnectionError: If a network-level error occurs
            before or during the download.
        OSError: If ``model_path`` cannot be opened for writing.

    Examples:
        - Download a GGUF model file to a local directory
            ```python
            >>> from pathlib import Path
            >>> _fetch_model_file(  # doctest: +SKIP
            ...     "https://example.com/model.gguf",
            ...     Path("/models/model.gguf"),
            ... )

            ```
        - Partial files are cleaned up automatically on failure
            ```python
            >>> from pathlib import Path
            >>> from unittest.mock import patch, MagicMock
            >>> mock_resp = MagicMock()
            >>> mock_resp.raise_for_status.return_value = None
            >>> mock_resp.headers.get.return_value = "500"  # < 1 MB → ValueError
            >>> mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            >>> mock_resp.__exit__ = MagicMock(return_value=False)
            >>> dest = Path("/tmp/model.gguf")
            >>> with patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp):
            ...     try:
            ...         _fetch_model_file("https://example.com/model.gguf", dest)
            ...     except ValueError as exc:
            ...         print(exc)
            Content-Length is 500 bytes; expected at least 1 MB

            ```

    See Also:
        tqdm: Progress bar library used to display download progress.
        requests.Response.iter_content: Underlying streaming iterator.
    """
    logger.info("Downloading %s to %s", model_url, model_path)
    try:
        with requests.get(model_url, stream=True, timeout=(10, None)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("Content-Length") or 0)
            if total_size < 1_000_000:
                raise ValueError(
                    f"Content-Length is {total_size} bytes; expected at least 1 MB"
                )
            logger.info("Total size: %.2f MB", total_size / 1_000_000)
            chunk_size = 1024 * 1024  # 1 MB
            with model_path.open("wb") as file:
                for chunk in tqdm(
                    r.iter_content(chunk_size=chunk_size),
                    total=math.ceil(total_size / chunk_size),
                    unit="MB",
                ):
                    file.write(chunk)
    except Exception:
        logger.exception("Download failed, removing partial file at %s", model_path)
        model_path.unlink(missing_ok=True)
        raise
