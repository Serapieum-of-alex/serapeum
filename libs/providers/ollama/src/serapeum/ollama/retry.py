"""Ollama-specific exception classification for retry.

Consumed by the retry decorators applied to :class:`~serapeum.ollama.llm.Ollama`
methods via :func:`~serapeum.core.retry.retry` and its variants.  Only
:func:`is_retryable` needs to be imported by provider code.
"""

from __future__ import annotations

import httpx
import ollama as ollama_sdk

# HTTP status codes that indicate a transient server-side problem and are safe
# to retry.  4xx codes other than 408/429 represent permanent client errors.
_RETRYABLE_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})


def is_retryable(exc: BaseException) -> bool:
    """Return ``True`` if *exc* is transient and the request can be retried.

    Classifies an exception as retryable or permanent based on the type of
    failure.  The classification follows this table:

    +-------------------------------------------+------------+
    | Exception                                 | Retryable? |
    +===========================================+============+
    | ``ConnectionError``                       | Yes        |
    +-------------------------------------------+------------+
    | ``httpx.TimeoutException``                | Yes        |
    +-------------------------------------------+------------+
    | ``httpx.StreamError``                     | Yes        |
    +-------------------------------------------+------------+
    | ``httpx.ConnectError``                    | Yes        |
    +-------------------------------------------+------------+
    | ``ollama.ResponseError`` with status 408  | Yes        |
    +-------------------------------------------+------------+
    | ``ollama.ResponseError`` with status 429  | Yes        |
    +-------------------------------------------+------------+
    | ``ollama.ResponseError`` with status 5xx  | Yes        |
    +-------------------------------------------+------------+
    | ``ollama.ResponseError`` with status 4xx  | No         |
    +-------------------------------------------+------------+
    | Any other exception type                  | No         |
    +-------------------------------------------+------------+

    Args:
        exc: The exception caught during an Ollama API call.

    Returns:
        ``True`` when the exception represents a transient network or
        server-side condition that is safe to retry.  ``False`` for permanent
        errors (e.g. bad request, authentication failure, unknown model).

    Examples:
        - Network-level errors are retryable
            ```python
            >>> from serapeum.ollama.retry import is_retryable
            >>> is_retryable(ConnectionError("connection refused"))
            True

            ```
        - httpx transport errors are retryable
            ```python
            >>> import httpx
            >>> is_retryable(httpx.ConnectError("timed out"))
            True
            >>> is_retryable(httpx.TimeoutException("read timeout"))
            True

            ```
        - Ollama server errors (5xx) are retryable
            ```python
            >>> import ollama
            >>> err = ollama.ResponseError("internal server error")
            >>> err.status_code = 503
            >>> is_retryable(err)
            True

            ```
        - Ollama client errors (4xx, except 408/429) are permanent
            ```python
            >>> bad_req = ollama.ResponseError("model not found")
            >>> bad_req.status_code = 404
            >>> is_retryable(bad_req)
            False

            ```
        - Unexpected exception types are not retryable
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
    if isinstance(exc, ConnectionError):
        result = True
    elif isinstance(exc, (httpx.TimeoutException, httpx.StreamError, httpx.ConnectError)):
        result = True
    elif isinstance(exc, ollama_sdk.ResponseError):
        result = getattr(exc, "status_code", 0) in _RETRYABLE_STATUS_CODES
    else:
        result = False
    return result
