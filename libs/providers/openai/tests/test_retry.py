"""Tests for serapeum.openai.retry — OpenAI exception classification."""

from __future__ import annotations

from unittest.mock import MagicMock

import openai
import pytest

from serapeum.openai.retry import is_retryable


def _make_response(status_code: int) -> MagicMock:
    """Build a mock httpx.Response with the given status code."""
    resp = MagicMock()
    resp.status_code = status_code
    return resp


# ---------------------------------------------------------------------------
# Retryable exceptions
# ---------------------------------------------------------------------------


class TestRetryableExceptions:
    """Exceptions that ``is_retryable`` should classify as transient."""

    @pytest.mark.unit
    def test_api_connection_error(self) -> None:
        exc = openai.APIConnectionError(request=None)
        assert is_retryable(exc) is True

    @pytest.mark.unit
    def test_api_timeout_error(self) -> None:
        exc = openai.APITimeoutError(request=None)
        assert is_retryable(exc) is True

    @pytest.mark.unit
    def test_rate_limit_error(self) -> None:
        resp = _make_response(429)
        exc = openai.RateLimitError("rate limited", response=resp, body=None)
        assert is_retryable(exc) is True

    @pytest.mark.unit
    def test_internal_server_error(self) -> None:
        resp = _make_response(500)
        exc = openai.InternalServerError("server error", response=resp, body=None)
        assert is_retryable(exc) is True

    @pytest.mark.unit
    @pytest.mark.parametrize("status_code", [502, 503, 504])
    def test_server_error_status_codes(self, status_code: int) -> None:
        resp = _make_response(status_code)
        exc = openai.APIStatusError(
            f"error {status_code}",
            response=resp,
            body=None,
        )
        assert is_retryable(exc) is True

    @pytest.mark.unit
    def test_request_timeout_408(self) -> None:
        resp = _make_response(408)
        exc = openai.APIStatusError("timeout", response=resp, body=None)
        assert is_retryable(exc) is True


# ---------------------------------------------------------------------------
# Non-retryable exceptions
# ---------------------------------------------------------------------------


class TestNonRetryableExceptions:
    """Exceptions that ``is_retryable`` should classify as permanent."""

    @pytest.mark.unit
    def test_authentication_error(self) -> None:
        resp = _make_response(401)
        exc = openai.AuthenticationError("bad key", response=resp, body=None)
        assert is_retryable(exc) is False

    @pytest.mark.unit
    def test_bad_request_error(self) -> None:
        resp = _make_response(400)
        exc = openai.BadRequestError("invalid", response=resp, body=None)
        assert is_retryable(exc) is False

    @pytest.mark.unit
    def test_permission_denied_error(self) -> None:
        resp = _make_response(403)
        exc = openai.PermissionDeniedError("forbidden", response=resp, body=None)
        assert is_retryable(exc) is False

    @pytest.mark.unit
    def test_not_found_error(self) -> None:
        resp = _make_response(404)
        exc = openai.NotFoundError("not found", response=resp, body=None)
        assert is_retryable(exc) is False

    @pytest.mark.unit
    def test_conflict_error(self) -> None:
        resp = _make_response(409)
        exc = openai.ConflictError("conflict", response=resp, body=None)
        assert is_retryable(exc) is False

    @pytest.mark.unit
    def test_unprocessable_entity_error(self) -> None:
        resp = _make_response(422)
        exc = openai.UnprocessableEntityError("unprocessable", response=resp, body=None)
        assert is_retryable(exc) is False

    @pytest.mark.unit
    def test_value_error(self) -> None:
        assert is_retryable(ValueError("bad argument")) is False

    @pytest.mark.unit
    def test_runtime_error(self) -> None:
        assert is_retryable(RuntimeError("unexpected state")) is False

    @pytest.mark.unit
    def test_type_error(self) -> None:
        assert is_retryable(TypeError("wrong type")) is False

    @pytest.mark.unit
    def test_key_error(self) -> None:
        assert is_retryable(KeyError("missing")) is False

    @pytest.mark.unit
    def test_generic_api_status_error_403(self) -> None:
        resp = _make_response(403)
        exc = openai.APIStatusError("forbidden", response=resp, body=None)
        assert is_retryable(exc) is False
