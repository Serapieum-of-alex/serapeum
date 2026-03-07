"""LlamaCPP-specific exception classification for retry.

Local inference rarely fails transiently, so :attr:`~serapeum.llama_cpp.llm.LlamaCPP.max_retries`
defaults to ``0`` (disabled) in the :class:`~serapeum.llama_cpp.llm.LlamaCPP`
class.  This predicate is provided for consistency with other providers and to
handle rare GPU / system memory exhaustion situations where a brief wait may
allow resources to become available.

Consumed by the retry decorators applied to
:class:`~serapeum.llama_cpp.llm.LlamaCPP` methods via
:func:`~serapeum.core.retry.retry`.
"""

from __future__ import annotations

import errno

# errno values that indicate temporary resource exhaustion rather than a
# permanent programming error.
_TRANSIENT_ERRNOS = frozenset({errno.ENOMEM, errno.EAGAIN})


def is_retryable(exc: BaseException) -> bool:
    """Return ``True`` if *exc* is transient and the operation can be retried.

    Classifies an exception as retryable or permanent.  The classification
    focuses on resource-exhaustion scenarios that may resolve on their own:

    +-----------------------------------------------+------------+
    | Exception                                     | Retryable? |
    +===============================================+============+
    | ``OSError`` with ``errno.ENOMEM``             | Yes        |
    +-----------------------------------------------+------------+
    | ``OSError`` with ``errno.EAGAIN``             | Yes        |
    +-----------------------------------------------+------------+
    | ``RuntimeError`` containing "out of memory"   | Yes        |
    +-----------------------------------------------+------------+
    | Any other exception type                      | No         |
    +-----------------------------------------------+------------+

    Args:
        exc: The exception caught during a llama.cpp inference call.

    Returns:
        ``True`` when the exception represents a transient resource-exhaustion
        condition that may resolve after a brief delay.  ``False`` for all
        other exception types.

    Examples:
        - Out-of-memory OS error is retryable
            ```python
            >>> import errno as _errno
            >>> from serapeum.llama_cpp.retry import is_retryable
            >>> err = OSError(_errno.ENOMEM, "not enough memory")
            >>> is_retryable(err)
            True

            ```
        - Resource-temporarily-unavailable OS error is retryable
            ```python
            >>> err = OSError(_errno.EAGAIN, "resource temporarily unavailable")
            >>> is_retryable(err)
            True

            ```
        - RuntimeError mentioning "out of memory" is retryable
            ```python
            >>> is_retryable(RuntimeError("CUDA out of memory"))
            True
            >>> is_retryable(RuntimeError("Out Of Memory"))
            True

            ```
        - Generic OSError without a transient errno is not retryable
            ```python
            >>> import errno as _errno
            >>> is_retryable(OSError(_errno.EACCES, "permission denied"))
            False

            ```
        - Other exception types are never retryable
            ```python
            >>> is_retryable(ValueError("invalid model path"))
            False
            >>> is_retryable(RuntimeError("model load failed"))
            False

            ```

    See Also:
        serapeum.core.retry.retry: Decorator that calls this predicate.
        serapeum.core.retry.retry: Decorator that calls this predicate (auto-detects generators).
    """
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in _TRANSIENT_ERRNOS:
        result = True
    elif isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        result = True
    else:
        result = False
    return result
