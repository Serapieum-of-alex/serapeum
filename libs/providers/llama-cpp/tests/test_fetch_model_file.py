"""Unit tests for _fetch_model_file in serapeum.llama_cpp.utils."""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from serapeum.llama_cpp.utils import _fetch_model_file

ONE_MB = 1024 * 1024
TWO_MB = 2 * ONE_MB


def _make_mock_response(
    content_length: int | None,
    chunks: list[bytes],
    http_error: Exception | None = None,
) -> MagicMock:
    """Build a mock requests.Response for use as a context manager.

    Args:
        content_length: Value returned by headers.get("Content-Length"). Pass
            None to simulate a missing header.
        chunks: Bytes chunks yielded by iter_content().
        http_error: If provided, raise_for_status() will raise this exception.

    Returns:
        MagicMock: Configured mock response usable with ``with requests.get()``.
    """
    mock_resp = MagicMock()
    if http_error is not None:
        mock_resp.raise_for_status.side_effect = http_error
    else:
        mock_resp.raise_for_status.return_value = None
    mock_resp.headers.get.return_value = (
        str(content_length) if content_length is not None else None
    )
    mock_resp.iter_content.return_value = iter(chunks)
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


@pytest.mark.unit
class TestFetchModelFile:
    """Tests for _fetch_model_file."""

    def test_success_writes_file_content(self, tmp_path: Path, mocker) -> None:
        """Test that all downloaded chunks are written to model_path.

        Test scenario:
            A 2 MB response (two 1 MB chunks) should produce a file at
            model_path containing all bytes in order.
        """
        model_path = tmp_path / "model.gguf"
        chunk1, chunk2 = b"a" * ONE_MB, b"b" * ONE_MB
        mock_resp = _make_mock_response(TWO_MB, [chunk1, chunk2])
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)
        mocker.patch("serapeum.llama_cpp.utils.tqdm", side_effect=lambda it, **kw: it)

        _fetch_model_file("http://example.com/model.gguf", model_path)

        assert model_path.exists(), "model file should exist after successful download"
        assert model_path.read_bytes() == chunk1 + chunk2, (
            "file content should match all downloaded chunks in order"
        )

    def test_success_does_not_delete_file(self, tmp_path: Path, mocker) -> None:
        """Test that model_path is not removed after a successful download.

        Test scenario:
            unlink() must NOT be called when the download completes without error.
        """
        model_path = tmp_path / "model.gguf"
        mock_resp = _make_mock_response(ONE_MB, [b"x" * ONE_MB])
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)
        mocker.patch("serapeum.llama_cpp.utils.tqdm", side_effect=lambda it, **kw: it)

        _fetch_model_file("http://example.com/model.gguf", model_path)

        assert model_path.exists(), "model file must still exist after a clean download"

    def test_requests_get_called_with_stream_true(self, tmp_path: Path, mocker) -> None:
        """Test that requests.get is invoked with stream=True.

        Test scenario:
            Streaming must be enabled so large GGUF files are not fully
            buffered in memory before writing.
        """
        model_path = tmp_path / "model.gguf"
        mock_resp = _make_mock_response(ONE_MB, [b"x" * ONE_MB])
        mock_get = mocker.patch(
            "serapeum.llama_cpp.utils.requests.get", return_value=mock_resp
        )
        mocker.patch("serapeum.llama_cpp.utils.tqdm", side_effect=lambda it, **kw: it)

        _fetch_model_file("http://example.com/model.gguf", model_path)

        mock_get.assert_called_once_with("http://example.com/model.gguf", stream=True)

    def test_iter_content_chunk_size_is_1mb(self, tmp_path: Path, mocker) -> None:
        """Test that iter_content is called with chunk_size equal to 1 MiB.

        Test scenario:
            1 MiB chunks (1024 * 1024) balance I/O efficiency and memory
            usage; the value must not silently change.
        """
        model_path = tmp_path / "model.gguf"
        mock_resp = _make_mock_response(ONE_MB, [b"x" * ONE_MB])
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)
        mocker.patch("serapeum.llama_cpp.utils.tqdm", side_effect=lambda it, **kw: it)

        _fetch_model_file("http://example.com/model.gguf", model_path)

        mock_resp.iter_content.assert_called_once_with(chunk_size=ONE_MB)

    def test_tqdm_total_uses_ceil(self, tmp_path: Path, mocker) -> None:
        """Test that tqdm receives math.ceil(total_size / chunk_size) as total.

        Test scenario:
            For a 2.5 MiB file, total should be ceil(2.5) = 3, not floor = 2,
            so the progress bar reaches 100%.
        """
        model_path = tmp_path / "model.gguf"
        total_bytes = int(2.5 * ONE_MB)
        chunks = [b"x" * ONE_MB, b"x" * ONE_MB, b"x" * (ONE_MB // 2)]
        mock_resp = _make_mock_response(total_bytes, chunks)
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)
        mock_tqdm = mocker.patch(
            "serapeum.llama_cpp.utils.tqdm", side_effect=lambda it, **kw: it
        )

        _fetch_model_file("http://example.com/model.gguf", model_path)

        mock_tqdm.assert_called_once()
        _, kwargs = mock_tqdm.call_args
        expected = math.ceil(total_bytes / ONE_MB)
        assert kwargs["total"] == expected, (
            f"Expected tqdm total={expected} (ceil), got {kwargs['total']}"
        )

    def test_logs_download_start(self, tmp_path: Path, mocker) -> None:
        """Test that logger.info is called with the URL and destination path.

        Test scenario:
            At the start of every download, an info log should record the
            source URL and local model_path for observability.
        """
        model_path = tmp_path / "model.gguf"
        mock_resp = _make_mock_response(ONE_MB, [b"x" * ONE_MB])
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)
        mocker.patch("serapeum.llama_cpp.utils.tqdm", side_effect=lambda it, **kw: it)
        mock_logger = mocker.patch("serapeum.llama_cpp.utils.logger")

        _fetch_model_file("http://example.com/model.gguf", model_path)

        mock_logger.info.assert_any_call(
            "Downloading %s to %s",
            "http://example.com/model.gguf",
            model_path,
        )

    @pytest.mark.parametrize(
        "content_length, label",
        [
            (0, "zero"),
            (999_999, "one byte below 1 MB"),
            (500_000, "500 KB"),
        ],
    )
    def test_raises_for_small_content_length(
        self, tmp_path: Path, mocker, content_length: int, label: str
    ) -> None:
        """Test ValueError is raised when Content-Length is below 1 MB.

        Args:
            content_length: Response Content-Length value to test.
            label: Human-readable description of the boundary case.

        Test scenario:
            Any Content-Length smaller than 1_000_000 bytes must raise
            ValueError mentioning the actual size.
        """
        model_path = tmp_path / "model.gguf"
        mock_resp = _make_mock_response(content_length, [])
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)

        with pytest.raises(ValueError, match="expected at least 1 MB") as exc_info:
            _fetch_model_file("http://example.com/model.gguf", model_path)

        assert str(content_length) in str(exc_info.value), (
            f"[{label}] Error message should include actual size {content_length}, "
            f"got: {exc_info.value}"
        )

    def test_raises_when_content_length_header_missing(
        self, tmp_path: Path, mocker
    ) -> None:
        """Test ValueError when Content-Length header is absent (returns None).

        Test scenario:
            A missing Content-Length header evaluates to 0, which must raise
            ValueError just like an explicit 0.
        """
        model_path = tmp_path / "model.gguf"
        mock_resp = _make_mock_response(None, [])
        mock_resp.headers.get.return_value = None
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)

        with pytest.raises(ValueError, match="expected at least 1 MB"):
            _fetch_model_file("http://example.com/model.gguf", model_path)

    def test_boundary_exactly_1_000_000_bytes_passes(
        self, tmp_path: Path, mocker
    ) -> None:
        """Test that Content-Length == 1_000_000 is accepted without error.

        Test scenario:
            The threshold is a strict less-than (<), so exactly 1_000_000
            bytes must not raise.
        """
        model_path = tmp_path / "model.gguf"
        mock_resp = _make_mock_response(1_000_000, [b"x" * 1_000_000])
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)
        mocker.patch("serapeum.llama_cpp.utils.tqdm", side_effect=lambda it, **kw: it)

        _fetch_model_file("http://example.com/model.gguf", model_path)  # must not raise

    @pytest.mark.parametrize(
        "status_code, message",
        [
            (404, "404 Not Found"),
            (500, "500 Server Error"),
            (403, "403 Forbidden"),
        ],
    )
    def test_http_error_is_reraised(
        self, tmp_path: Path, mocker, status_code: int, message: str
    ) -> None:
        """Test that HTTP errors from raise_for_status() propagate to the caller.

        Args:
            status_code: HTTP status code being simulated.
            message: Error message embedded in the exception.

        Test scenario:
            raise_for_status() raising HTTPError must not be swallowed —
            the original exception should reach the caller.
        """
        model_path = tmp_path / "model.gguf"
        http_error = requests.exceptions.HTTPError(message)
        mock_resp = _make_mock_response(TWO_MB, [], http_error=http_error)
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)

        with pytest.raises(requests.exceptions.HTTPError, match=str(status_code)):
            _fetch_model_file("http://example.com/model.gguf", model_path)

    def test_http_error_cleans_up_partial_file(self, tmp_path: Path, mocker) -> None:
        """Test that model_path is removed when an HTTP error occurs.

        Test scenario:
            A 404 HTTPError should trigger cleanup so no partial file is left
            on disk.
        """
        model_path = tmp_path / "model.gguf"
        http_error = requests.exceptions.HTTPError("404 Not Found")
        mock_resp = _make_mock_response(TWO_MB, [], http_error=http_error)
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)

        with pytest.raises(requests.exceptions.HTTPError):
            _fetch_model_file("http://example.com/model.gguf", model_path)

        assert not model_path.exists(), "partial file should be removed after HTTP error"

    def test_partial_file_removed_on_value_error(self, tmp_path: Path, mocker) -> None:
        """Test that a pre-existing partial file is deleted when ValueError is raised.

        Test scenario:
            If a partially downloaded file exists when ValueError occurs
            (e.g. bad Content-Length), unlink(missing_ok=True) should remove it.
        """
        model_path = tmp_path / "model.gguf"
        model_path.write_bytes(b"partial data")
        mock_resp = _make_mock_response(100, [])
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)

        with pytest.raises(ValueError):
            _fetch_model_file("http://example.com/model.gguf", model_path)

        assert not model_path.exists(), "partial file should be removed after ValueError"

    def test_connection_error_cleans_up_and_reraises(
        self, tmp_path: Path, mocker
    ) -> None:
        """Test that a network-level error triggers cleanup and is re-raised.

        Test scenario:
            A ConnectionError from requests.get (before any bytes are written)
            must be re-raised and must not leave a partial file.
        """
        model_path = tmp_path / "model.gguf"
        mocker.patch(
            "serapeum.llama_cpp.utils.requests.get",
            side_effect=requests.exceptions.ConnectionError("no route to host"),
        )

        with pytest.raises(requests.exceptions.ConnectionError, match="no route to host"):
            _fetch_model_file("http://example.com/model.gguf", model_path)

        assert not model_path.exists(), "model_path should not exist after ConnectionError"

    def test_failure_logs_exception(self, tmp_path: Path, mocker) -> None:
        """Test that logger.exception is called on any download failure.

        Test scenario:
            When an exception occurs, logger.exception should be invoked with
            the model_path so the failure appears in application logs.
        """
        model_path = tmp_path / "model.gguf"
        mock_resp = _make_mock_response(500_000, [])
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)
        mock_logger = mocker.patch("serapeum.llama_cpp.utils.logger")

        with pytest.raises(ValueError):
            _fetch_model_file("http://example.com/model.gguf", model_path)

        mock_logger.exception.assert_called_once_with(
            "Download failed, removing partial file at %s",
            model_path,
        )

    def test_original_exception_type_is_preserved(self, tmp_path: Path, mocker) -> None:
        """Test that the bare raise preserves the original exception type.

        Test scenario:
            The except block must not wrap or replace exceptions — the caller
            should receive the exact type that was raised internally.
        """
        model_path = tmp_path / "model.gguf"
        mock_resp = _make_mock_response(TWO_MB, [], http_error=TimeoutError("timed out"))
        mocker.patch("serapeum.llama_cpp.utils.requests.get", return_value=mock_resp)

        with pytest.raises(TimeoutError, match="timed out"):
            _fetch_model_file("http://example.com/model.gguf", model_path)
