"""Shared retry infrastructure backed by tenacity.

Provides a :class:`Retry` Pydantic mixin (exposing ``max_retries``),
low-level :class:`~tenacity.Retrying` / :class:`~tenacity.AsyncRetrying`
factory functions, and a single :func:`retry` **decorator factory** that
provider methods use to add transparent retry behaviour with exponential
back-off and jitter.

Typical provider usage::

    from serapeum.core.retry import Retry, retry
    from myprovider.retry import is_retryable   # provider-specific predicate

    class MyProvider(Retry, BaseModel):

        @retry(is_retryable)
        def chat(self, prompt: str) -> str: ...

        @retry(is_retryable)
        async def achat(self, prompt: str) -> str: ...

        @retry(is_retryable, stream=True)
        async def astream_chat(self, prompt: str):
            async def gen():
                ...  # yield chunks
            return gen()

The decorator auto-detects sync vs async (via
:func:`inspect.iscoroutinefunction`) and sync generators (via
:func:`inspect.isgeneratorfunction`).  The only case that requires an
explicit flag is **async streaming** (``stream=True``), because the
inner-gen pattern cannot be distinguished from a regular coroutine at
decoration time.

Each provider defines its own ``is_retryable(exc) -> bool`` predicate that
classifies exceptions as transient (retry-able) or permanent (propagate
immediately).  The decorator reads ``self.max_retries`` at *call time*, so
the retry count is instance-configurable without re-decorating.
"""

from __future__ import annotations

import functools
import inspect
import logging
from typing import Any, Callable

from pydantic import BaseModel, Field
from tenacity import (
    AsyncRetrying,
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

__all__ = [
    "Retry",
    "build_retryer",
    "build_async_retryer",
    "retry",
]

DEFAULT_INITIAL_DELAY: float = 0.5
DEFAULT_MAX_DELAY: float = 8.0


class Retry(BaseModel):
    """Pydantic mixin that adds a configurable ``max_retries`` field.

    Intended for multiple inheritance alongside provider client base classes
    (e.g. ``Client``, ``LlamaCPP``).  The retry decorators
    :func:`retry` reads ``self.max_retries`` at *call time*,
    so individual instances can carry different retry budgets without
    re-decorating.

    Examples:
        - Default retry count
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.retry import Retry
            >>> class MyClient(Retry, BaseModel):
            ...     pass
            >>> MyClient().max_retries
            3

            ```
        - Overriding the retry count at construction time
            ```python
            >>> MyClient(max_retries=5).max_retries
            5

            ```
        - Disabling retries entirely
            ```python
            >>> MyClient(max_retries=0).max_retries
            0

            ```
        - Negative values are rejected by Pydantic validation
            ```python
            >>> from pydantic import ValidationError
            >>> try:
            ...     MyClient(max_retries=-1)
            ... except ValidationError:
            ...     print("rejected")
            rejected

            ```

    See Also:
        retry: Decorator that reads ``max_retries`` from ``self``.
        build_retryer: Low-level factory used internally by the decorator.
    """

    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts for transient failures.",
        ge=0,
    )


# ---------------------------------------------------------------------------
# Low-level factories (still useful for custom retry loops)
# ---------------------------------------------------------------------------


def build_retryer(
    max_retries: int,
    is_retryable: Callable[[BaseException], bool],
    logger: logging.Logger | None = None,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> Retrying:
    """Build a configured :class:`~tenacity.Retrying` for synchronous calls.

    Returns a context-manager iterator that drives a ``for`` / ``with attempt``
    retry loop::

        for attempt in build_retryer(max_retries=3, is_retryable=pred):
            with attempt:
                result = do_work()

    The retry strategy is *exponential back-off with jitter*: delays start at
    *initial_delay* seconds and grow towards *max_delay*.  When all attempts
    are exhausted and the exception is still retryable, it is re-raised
    (``reraise=True``).  Non-retryable exceptions propagate immediately
    without consuming any retry budget.

    Args:
        max_retries: Maximum number of *additional* attempts after the first
            failure.  ``0`` disables retries — only one attempt is made.
        is_retryable: Predicate called with the caught exception.  Return
            ``True`` to schedule a retry, ``False`` to re-raise immediately.
        logger: Optional :class:`logging.Logger` used to emit a
            ``WARNING``-level message before each sleep interval.  Pass
            ``None`` to suppress sleep-time logging.
        initial_delay: Seconds to wait before the first retry attempt.
            Defaults to :data:`DEFAULT_INITIAL_DELAY` (0.5 s).
        max_delay: Upper bound on inter-retry delay in seconds.
            Defaults to :data:`DEFAULT_MAX_DELAY` (8.0 s).

    Returns:
        A :class:`~tenacity.Retrying` iterator ready to drive a ``for`` /
        ``with attempt`` retry loop.

    Examples:
        - Succeeds on the first attempt — no retries triggered
            ```python
            >>> from serapeum.core.retry import build_retryer
            >>> def is_transient(exc): return isinstance(exc, ConnectionError)
            >>> for attempt in build_retryer(max_retries=3, is_retryable=is_transient):
            ...     with attempt:
            ...         result = "ok"
            >>> result
            'ok'

            ```
        - Non-retryable exception propagates immediately, retry budget unused
            ```python
            >>> try:
            ...     for attempt in build_retryer(max_retries=3, is_retryable=is_transient):
            ...         with attempt:
            ...             raise ValueError("permanent")
            ... except ValueError as exc:
            ...     print(exc)
            permanent

            ```
        - Retry loop with actual failures (delays make this unsuitable for doctests)
            ```python
            >>> calls = []
            >>> for attempt in build_retryer(  # doctest: +SKIP
            ...         max_retries=2, is_retryable=is_transient,
            ...         initial_delay=0.0, max_delay=0.0):
            ...     with attempt:
            ...         calls.append(1)
            ...         if len(calls) < 2:
            ...             raise ConnectionError("transient")
            >>> len(calls)  # doctest: +SKIP
            2

            ```

    See Also:
        build_async_retryer: Async equivalent returning
            :class:`~tenacity.AsyncRetrying`.
        retry: Decorator factory that uses this function internally.
    """
    kwargs: dict = {
        "stop": stop_after_attempt(max_retries + 1),
        "wait": wait_exponential_jitter(
            initial=initial_delay,
            max=max_delay,
            jitter=initial_delay / 2,
        ),
        "retry": retry_if_exception(is_retryable),
        "reraise": True,
    }

    if logger is not None:
        kwargs["before_sleep"] = before_sleep_log(logger, logging.WARNING)

    return Retrying(**kwargs)


def build_async_retryer(
    max_retries: int,
    is_retryable: Callable[[BaseException], bool],
    logger: logging.Logger | None = None,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
) -> AsyncRetrying:
    """Build a configured :class:`~tenacity.AsyncRetrying` for async calls.

    Returns an async context-manager iterator that drives an ``async for`` /
    ``with attempt`` retry loop::

        async for attempt in build_async_retryer(max_retries=3, is_retryable=pred):
            with attempt:
                result = await do_work()

    All retry parameters and semantics are identical to :func:`build_retryer`;
    the only difference is the async iteration protocol, which allows the
    back-off sleep to yield control to the event loop rather than blocking.

    Args:
        max_retries: Maximum number of *additional* attempts after the first
            failure.  ``0`` disables retries — only one attempt is made.
        is_retryable: Predicate called with the caught exception.  Return
            ``True`` to schedule a retry, ``False`` to re-raise immediately.
        logger: Optional :class:`logging.Logger` used to emit a
            ``WARNING``-level message before each sleep interval.  Pass
            ``None`` to suppress sleep-time logging.
        initial_delay: Seconds to wait before the first retry attempt.
            Defaults to :data:`DEFAULT_INITIAL_DELAY` (0.5 s).
        max_delay: Upper bound on inter-retry delay in seconds.
            Defaults to :data:`DEFAULT_MAX_DELAY` (8.0 s).

    Returns:
        An :class:`~tenacity.AsyncRetrying` async iterator ready to drive an
        ``async for`` / ``with attempt`` retry loop.

    Examples:
        - Async retry loop succeeding on the first attempt
            ```python
            >>> import asyncio
            >>> from serapeum.core.retry import build_async_retryer
            >>> def is_transient(exc): return isinstance(exc, ConnectionError)
            >>> async def run():
            ...     async for attempt in build_async_retryer(
            ...             max_retries=2, is_retryable=is_transient):
            ...         with attempt:
            ...             result = "async_ok"
            ...     return result
            >>> asyncio.run(run())  # doctest: +SKIP
            'async_ok'

            ```

    See Also:
        build_retryer: Synchronous equivalent returning
            :class:`~tenacity.Retrying`.
        retry: Decorator factory that uses this function internally.
    """
    kwargs: dict = {
        "stop": stop_after_attempt(max_retries + 1),
        "wait": wait_exponential_jitter(
            initial=initial_delay,
            max=max_delay,
            jitter=initial_delay / 2,
        ),
        "retry": retry_if_exception(is_retryable),
        "reraise": True,
    }
    if logger is not None:
        kwargs["before_sleep"] = before_sleep_log(logger, logging.WARNING)
    return AsyncRetrying(**kwargs)


def retry(
    is_retryable_fn: Callable[[BaseException], bool],
    retry_logger: logging.Logger | None = None,
    *,
    stream: bool = False,
) -> Callable:
    """Decorator factory that adds retry with exponential back-off to a method.

    Auto-detects sync vs async (via :func:`inspect.iscoroutinefunction`) and
    sync generators (via :func:`inspect.isgeneratorfunction`).  The only case
    requiring an explicit flag is **async streaming** (``stream=True``),
    because the inner-gen pattern cannot be distinguished from a regular
    coroutine at decoration time.

    The four modes are:

    +-------------------+-----------------+-----------------------------------+
    | Method type       | ``stream``      | Behaviour                         |
    +===================+=================+===================================+
    | ``def``           | (ignored)       | Retries the call                  |
    +-------------------+-----------------+-----------------------------------+
    | ``def`` generator | (ignored)       | Retries with a fresh generator    |
    +-------------------+-----------------+-----------------------------------+
    | ``async def``     | ``False``       | Retries the ``await``             |
    +-------------------+-----------------+-----------------------------------+
    | ``async def``     | ``True``        | Retries the async generator iter  |
    +-------------------+-----------------+-----------------------------------+

    The retry count is read from ``self.max_retries`` at *call time*, so
    different instances can carry different retry budgets without
    re-decorating.

    The decorated method must be defined on a class that inherits from
    :class:`Retry`.

    Args:
        is_retryable_fn: Provider-specific predicate.  Called with the caught
            exception; return ``True`` to retry, ``False`` to propagate
            immediately.
        retry_logger: Optional logger for pre-sleep ``WARNING`` messages.
            When ``None``, sleep-time logging is suppressed.
        stream: Set to ``True`` when decorating an ``async def`` method that
            returns an async generator (the *inner-gen pattern*).  Ignored
            for sync methods where generator detection is automatic.

    Returns:
        A decorator that wraps the target method with the appropriate retry
        loop driven by :func:`build_retryer` (sync) or
        :func:`build_async_retryer` (async).

    Examples:
        - Sync regular method
            ```python
            >>> from pydantic import BaseModel
            >>> from serapeum.core.retry import Retry, retry
            >>> def is_transient(exc): return isinstance(exc, ConnectionError)
            >>> class MyClient(Retry, BaseModel):
            ...     @retry(is_transient)
            ...     def call(self) -> str:
            ...         return "response"
            >>> MyClient(max_retries=3).call()
            'response'

            ```
        - Sync generator method (auto-detected)
            ```python
            >>> class StreamClient(Retry, BaseModel):
            ...     @retry(is_transient)
            ...     def stream(self):
            ...         yield "chunk1"
            ...         yield "chunk2"
            >>> list(StreamClient(max_retries=2).stream())
            ['chunk1', 'chunk2']

            ```
        - Async non-streaming (auto-detected)
            ```python
            >>> import asyncio
            >>> class AsyncClient(Retry, BaseModel):
            ...     @retry(is_transient)
            ...     async def call(self) -> str:
            ...         return "async_ok"
            >>> asyncio.run(AsyncClient().call())  # doctest: +SKIP
            'async_ok'

            ```
        - Async streaming (explicit ``stream=True``)
            ```python
            >>> class AsyncStreamClient(Retry, BaseModel):
            ...     @retry(is_transient, stream=True)
            ...     async def stream(self):
            ...         async def gen():
            ...             for item in ["a", "b"]:
            ...                 yield item
            ...         return gen()
            >>> async def run():
            ...     return [c async for c in await AsyncStreamClient().stream()]
            >>> asyncio.run(run())  # doctest: +SKIP
            ['a', 'b']

            ```
        - ``max_retries=0`` disables retry but the method still executes once
            ```python
            >>> MyClient(max_retries=0).call()
            'response'

            ```

    See Also:
        build_retryer: Underlying sync retry loop factory.
        build_async_retryer: Underlying async retry loop factory.
    """

    def decorator(fn: Callable) -> Callable:
        if inspect.iscoroutinefunction(fn):
            if stream:
                wrapper = _wrap_async_stream(fn, is_retryable_fn, retry_logger)
            else:
                wrapper = _wrap_async(fn, is_retryable_fn, retry_logger)
        elif inspect.isgeneratorfunction(fn):
            wrapper = _wrap_sync_stream(fn, is_retryable_fn, retry_logger)
        else:
            wrapper = _wrap_sync(fn, is_retryable_fn, retry_logger)
        return wrapper

    return decorator


def _wrap_sync(
    fn: Callable,
    is_retryable_fn: Callable[[BaseException], bool],
    retry_logger: logging.Logger | None,
) -> Callable:
    """Wrap a sync non-streaming method with retry logic."""

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        for attempt in build_retryer(self.max_retries, is_retryable_fn, retry_logger):
            with attempt:
                result = fn(self, *args, **kwargs)
        return result

    return wrapper


def _wrap_sync_stream(
    fn: Callable,
    is_retryable_fn: Callable[[BaseException], bool],
    retry_logger: logging.Logger | None,
) -> Callable:
    """Wrap a sync generator method with retry logic."""

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        def gen():  # type: ignore[return]
            for attempt in build_retryer(self.max_retries, is_retryable_fn, retry_logger):
                with attempt:
                    yield from fn(self, *args, **kwargs)
                    return

        return gen()

    return wrapper


def _wrap_async(
    fn: Callable,
    is_retryable_fn: Callable[[BaseException], bool],
    retry_logger: logging.Logger | None,
) -> Callable:
    """Wrap an async non-streaming method with retry logic."""

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        async for attempt in build_async_retryer(self.max_retries, is_retryable_fn, retry_logger):
            with attempt:
                result = await fn(self, *args, **kwargs)
        return result

    return wrapper


def _wrap_async_stream(
    fn: Callable,
    is_retryable_fn: Callable[[BaseException], bool],
    retry_logger: logging.Logger | None,
) -> Callable:
    """Wrap an async streaming method (inner-gen pattern) with retry logic."""

    @functools.wraps(fn)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        async def gen():  # type: ignore[return]
            async for attempt in build_async_retryer(self.max_retries, is_retryable_fn, retry_logger):
                with attempt:
                    inner = await fn(self, *args, **kwargs)
                    async for chunk in inner:
                        yield chunk
                    return

        return gen()

    return wrapper
