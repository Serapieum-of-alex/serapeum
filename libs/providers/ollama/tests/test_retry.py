"""Unit tests for serapeum.ollama.retry.is_retryable."""

from __future__ import annotations

import httpx
import ollama as ollama_sdk
import pytest

from serapeum.ollama.retry import is_retryable


@pytest.mark.unit
class TestIsRetryable:
    def test_connection_error(self) -> None:
        assert is_retryable(ConnectionError("refused")) is True

    def test_httpx_timeout(self) -> None:
        assert is_retryable(httpx.TimeoutException("timed out")) is True

    def test_httpx_connect_error(self) -> None:
        assert is_retryable(httpx.ConnectError("failed")) is True

    def test_httpx_stream_error(self) -> None:
        assert is_retryable(httpx.StreamError("broken")) is True

    def test_response_error_500(self) -> None:
        exc = ollama_sdk.ResponseError("server error")
        exc.status_code = 500
        assert is_retryable(exc) is True

    def test_response_error_502(self) -> None:
        exc = ollama_sdk.ResponseError("bad gateway")
        exc.status_code = 502
        assert is_retryable(exc) is True

    def test_response_error_503(self) -> None:
        exc = ollama_sdk.ResponseError("unavailable")
        exc.status_code = 503
        assert is_retryable(exc) is True

    def test_response_error_429(self) -> None:
        exc = ollama_sdk.ResponseError("rate limited")
        exc.status_code = 429
        assert is_retryable(exc) is True

    def test_response_error_408(self) -> None:
        exc = ollama_sdk.ResponseError("request timeout")
        exc.status_code = 408
        assert is_retryable(exc) is True

    def test_response_error_400(self) -> None:
        exc = ollama_sdk.ResponseError("bad request")
        exc.status_code = 400
        assert is_retryable(exc) is False

    def test_response_error_401(self) -> None:
        exc = ollama_sdk.ResponseError("unauthorized")
        exc.status_code = 401
        assert is_retryable(exc) is False

    def test_response_error_403(self) -> None:
        exc = ollama_sdk.ResponseError("forbidden")
        exc.status_code = 403
        assert is_retryable(exc) is False

    def test_response_error_404(self) -> None:
        exc = ollama_sdk.ResponseError("not found")
        exc.status_code = 404
        assert is_retryable(exc) is False

    def test_request_error(self) -> None:
        assert is_retryable(ollama_sdk.RequestError("bad input")) is False

    def test_value_error(self) -> None:
        assert is_retryable(ValueError("bad")) is False

    def test_type_error(self) -> None:
        assert is_retryable(TypeError("wrong type")) is False
