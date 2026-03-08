"""OpenAI-specific exception classification for retry.

Consumed by the retry decorators applied to
:class:`~serapeum.openai.llm.OpenAI` and
:class:`~serapeum.openai.responses.OpenAIResponses` methods via
:func:`~serapeum.core.retry.retry`.  Only :func:`is_retryable` needs to be
imported by provider code.

When serapeum's ``@retry`` decorator is active the OpenAI SDK's own
retry mechanism is disabled (``max_retries=0`` on the SDK client) so that
all retry logic is centralised in one place with consistent exponential
back-off and jitter across every provider.
"""

from __future__ import annotations

import openai

# HTTP status codes that indicate a transient server-side problem and are safe
# to retry.  4xx codes other than 408/429 represent permanent client errors.
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def is_retryable(exc: BaseException) -> bool:
    """Return ``True`` if *exc* is transient and the request can be retried.

    Classifies an exception as retryable or permanent based on the type of
    failure.  The classification follows this table:

    +-----------------------------------------------+------------+
    | Exception                                     | Retryable? |
    +===============================================+============+
    | ``openai.APIConnectionError``                 | Yes        |
    +-----------------------------------------------+------------+
    | ``openai.APITimeoutError``                    | Yes        |
    +-----------------------------------------------+------------+
    | ``openai.RateLimitError``                     | Yes        |
    +-----------------------------------------------+------------+
    | ``openai.InternalServerError``                | Yes        |
    +-----------------------------------------------+------------+
    | ``openai.APIStatusError`` with status 408     | Yes        |
    +-----------------------------------------------+------------+
    | ``openai.APIStatusError`` with status 5xx     | Yes        |
    +-----------------------------------------------+------------+
    | ``openai.AuthenticationError``                | No         |
    +-----------------------------------------------+------------+
    | ``openai.BadRequestError``                    | No         |
    +-----------------------------------------------+------------+
    | ``openai.PermissionDeniedError``              | No         |
    +-----------------------------------------------+------------+
    | ``openai.NotFoundError``                      | No         |
    +-----------------------------------------------+------------+
    | Any other exception type                      | No         |
    +-----------------------------------------------+------------+

    Args:
        exc: The exception caught during an OpenAI API call.

    Returns:
        ``True`` when the exception represents a transient network or
        server-side condition that is safe to retry.  ``False`` for permanent
        errors (e.g. bad request, authentication failure, invalid model).

    Examples:
        - Connection errors are retryable
            ```python
            >>> import openai
            >>> from serapeum.openai.retry import is_retryable
            >>> is_retryable(openai.APIConnectionError(request=None))
            True

            ```
        - Timeout errors are retryable
            ```python
            >>> is_retryable(openai.APITimeoutError(request=None))
            True

            ```
        - Rate-limit errors are retryable
            ```python
            >>> from unittest.mock import MagicMock
            >>> resp = MagicMock()
            >>> resp.status_code = 429
            >>> is_retryable(openai.RateLimitError("rate limited", response=resp, body=None))
            True

            ```
        - Internal server errors are retryable
            ```python
            >>> resp = MagicMock()
            >>> resp.status_code = 500
            >>> is_retryable(openai.InternalServerError("server error", response=resp, body=None))
            True

            ```
        - Authentication errors are not retryable
            ```python
            >>> resp = MagicMock()
            >>> resp.status_code = 401
            >>> is_retryable(openai.AuthenticationError("bad key", response=resp, body=None))
            False

            ```
        - Bad request errors are not retryable
            ```python
            >>> resp = MagicMock()
            >>> resp.status_code = 400
            >>> is_retryable(openai.BadRequestError("invalid", response=resp, body=None))
            False

            ```
        - Other exception types are never retryable
            ```python
            >>> is_retryable(ValueError("bad argument"))
            False
            >>> is_retryable(RuntimeError("unexpected state"))
            False

            ```

    See Also:
        serapeum.core.retry.retry: Decorator that calls this predicate.
        serapeum.core.retry.build_retryer: Low-level factory used by the decorators.
    """
    if isinstance(exc, openai.APIConnectionError):
        # Covers APIConnectionError and its subclass APITimeoutError.
        result = True
    elif isinstance(exc, openai.APIStatusError):
        result = exc.status_code in _RETRYABLE_STATUS_CODES
    else:
        result = False
    return result
