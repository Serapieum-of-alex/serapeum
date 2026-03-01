"""Internal utilities for downloading GGUF model files.

This module provides two private helpers used by
:class:`~serapeum.llama_cpp.LlamaCPP` to resolve a model path before loading:

- :func:`_fetch_model_file` — streams a GGUF file from an arbitrary URL with
  progress reporting and automatic cleanup on failure.
- :func:`_fetch_model_file_hf` — downloads from HuggingFace Hub using the
  ``huggingface_hub`` library (optional dependency).

These functions are **internal** — they are not part of the public API.
External callers should use :class:`~serapeum.llama_cpp.LlamaCPP` directly.

See Also:
    serapeum.llama_cpp.llm: The LlamaCPP class that consumes these helpers.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
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
            >>> _fetch_model_file(
            ...     "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_0.gguf",
            ...     Path("/models/llama-2-7b.Q4_0.gguf"),
            ... ) # doctest: +SKIP
            >>> Path("/models/llama-2-7b.Q4_0.gguf").exists() # doctest: +SKIP
            True
            >>> Path("/models/llama-2-7b.Q4_0.gguf").stat().st_size # doctest: +SKIP
            3791725568

            ```

    See Also:
        tqdm: Progress bar library used to display download progress.
        requests.Response.iter_content: Underlying streaming iterator.
    """
    logger.info("Downloading %s to %s", model_url, model_path)
    try:
        with requests.get(model_url, stream=True, timeout=(10, 120)) as r:
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


def _fetch_model_file_hf(repo_id: str, filename: str, cache_dir: Path) -> Path:
    """Download a GGUF model from HuggingFace Hub to a local cache directory.

    Uses the ``huggingface_hub`` library (installed with ``serapeum-llama-cpp``)
    for authenticated downloads, automatic caching, and SHA-256 verification.

    Args:
        repo_id: HuggingFace Hub repository ID
            (e.g. ``'TheBloke/Llama-2-13B-chat-GGUF'``).
        filename: File name within the repository
            (e.g. ``'llama-2-13b-chat.Q4_0.gguf'``).
        cache_dir: Local directory passed to ``hf_hub_download`` as its cache
            root.  The library manages the exact subdirectory layout.

    Returns:
        Path to the downloaded (or already-cached) model file.

    Raises:
        Exception: Any exception raised by ``hf_hub_download`` — network
            errors, authentication failures, missing files, etc.

    Examples:
        - Download a model file from HuggingFace Hub and explore the result
            ```python
            >>> from pathlib import Path
            >>> model_path = _fetch_model_file_hf(
            ...     "TheBloke/Llama-2-13B-chat-GGUF",
            ...     "llama-2-13b-chat.Q4_0.gguf",
            ...     Path("/tmp/hf_cache"),
            ... ) # doctest: +SKIP
            >>> model_path.name # doctest: +SKIP
            'llama-2-13b-chat.Q4_0.gguf'
            >>> model_path.suffix # doctest: +SKIP
            '.gguf'
            >>> model_path.exists() # doctest: +SKIP
            True

            ```

    See Also:
        _fetch_model_file: Direct-URL download alternative.
        huggingface_hub.hf_hub_download: Underlying download function.
    """
    logger.info("Downloading %s/%s from HuggingFace Hub", repo_id, filename)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download '{filename}' from HuggingFace repo '{repo_id}'. "
            f"Check that the repo ID and filename are correct and that you have "
            f"access. Original error: {exc}"
        ) from exc
    logger.info("Model cached at %s", local_path)
    return Path(local_path)
