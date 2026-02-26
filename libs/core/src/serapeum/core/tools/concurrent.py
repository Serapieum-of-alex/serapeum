"""Concurrent execution utilities for running async tasks with batching and progress tracking.

This module provides helpers for executing coroutines with various concurrency
patterns, including simple gathering, batched execution, and optional progress
bar integration using tqdm. It handles event loop management and provides safe
fallbacks for different async execution contexts.
"""

from __future__ import annotations

import asyncio
from itertools import zip_longest
from typing import Any, Coroutine, Iterable


def asyncio_run(coro: Coroutine) -> Any:
    """Run a coroutine in the current or a new event loop with safe fallbacks.

    The function attempts to reuse an existing event loop when possible. If
    the loop exists and is not running, the coroutine is executed via
    ``loop.run_until_complete``. If no loop exists, ``asyncio.run`` is used.

    If this function is called from within a running event loop (nested
    context), Python forbids creating or reusing a loop synchronously for a
    blocking wait; in that case a ``RuntimeError`` is raised with guidance on
    how to handle nested asyncio (e.g., using ``nest_asyncio.apply()`` or
    providing async entrypoints).

    Args:
        coro: The coroutine object to run to completion.

    Returns:
        The value returned by the awaited coroutine.

    Raises:
        RuntimeError: If called from within an already running event loop
            (nested asyncio). The error message explains how to proceed.


    Examples:
        - Run a coroutine when no loop is running
            ```python
            >>> from serapeum.core.tools.concurrent import asyncio_run
            >>> async def add(a, b):
            ...     return a + b
            >>> asyncio_run(add(2, 3))
            5

            ```
        - Detect nested event loop usage and handle the error
            ```python
            >>> import asyncio
            >>> async def inner():
            ...     from serapeum.core.tools.concurrent import asyncio_run
            ...     async def one():
            ...         return 1
            ...
            ...     try:
            ...         asyncio_run(one())  # called within a running loop
            ...     except RuntimeError as e:
            ...         return "nested" if "Detected nested async" in str(e) else "other"
            >>> asyncio.run(inner())
            'nested'

            ```

    See Also:
        run_async_tasks: Convenience to run multiple coroutines and collect results.
        batch_gather: Async helper to gather coroutines in batches.
    """
    try:
        # Check if there's an existing event loop
        loop = asyncio.get_event_loop()

        # If we're here, there's an existing loop but it's not running
        return loop.run_until_complete(coro)

    except RuntimeError:
        # If we can't get the event loop, we're likely in a different thread, or its already running
        try:
            return asyncio.run(coro)
        except RuntimeError as e:
            raise RuntimeError(
                "Detected nested async. Please use nest_asyncio.apply() to allow nested event loops."
                f"Or, use async entry methods like `aquery()`, `aretriever`, `achat`, etc. Error: {e}"
            )


def run_async_tasks(
    tasks: list[Coroutine],
    show_progress: bool = False,
    progress_bar_desc: str = "Running async tasks",
) -> list[Any]:
    """Run a list of coroutines to completion and collect their results.

    This convenience wrapper optionally displays a progress bar via
    ``tqdm.asyncio`` when ``show_progress`` is True. If the environment doesn't
    support that (e.g., ``tqdm`` or ``nest_asyncio`` isn't available, or a
    runtime error occurs), it falls back gracefully to a standard
    ``asyncio.gather`` execution using ``asyncio_run``.

    Args:
        tasks: The coroutines to run.
        show_progress: If True, attempt to use ``tqdm.asyncio.tqdm.gather``
            with a progress bar. If unavailable or incompatible, a silent
            fallback is applied. Defaults to False.
        progress_bar_desc: Optional label shown by the progress bar.
            Defaults to "Running async tasks".

    Returns:
        Results of the completed coroutines in the same order as the input list.

    Raises:
        Exception: Any exception raised by the provided coroutines will
            propagate from ``asyncio.gather``.

    Examples:
        - Run tasks without a progress bar
            ```python
            >>> import asyncio
            >>> from serapeum.core.tools.concurrent import run_async_tasks
            >>> async def f(x):
            ...     await asyncio.sleep(0)
            ...     return x * 2
            >>> tasks = [f(i) for i in range(5)]
            >>> run_async_tasks(tasks, show_progress=False)
            [0, 2, 4, 6, 8]

            ```
        - Request progress display; falls back automatically if unsupported
            ```python
            >>> import asyncio
            >>> async def g(x):
            ...     await asyncio.sleep(0)
            ...     return x + 1
            >>> tasks = [g(i) for i in range(3)]
            >>> # Works regardless of whether tqdm/nest_asyncio are installed
            >>> run_async_tasks(tasks, show_progress=True, progress_bar_desc="Demo")
            [1, 2, 3]

            ```

    See Also:
        asyncio_run: Helper that safely runs a coroutine from sync code.
        batch_gather: Batched variant that controls peak concurrency.
    """
    tasks_to_execute: list[Any] = tasks
    if show_progress:
        try:
            import nest_asyncio
            from tqdm.asyncio import tqdm

            # jupyter notebooks already have an event loop running
            # we need to reuse it instead of creating a new one
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()

            async def _tqdm_gather() -> list[Any]:
                return await tqdm.gather(*tasks_to_execute, desc=progress_bar_desc)

            tqdm_outputs: list[Any] = loop.run_until_complete(_tqdm_gather())
            return tqdm_outputs
        # run the operation w/o tqdm on hitting a fatal
        # may occur in some environments where tqdm.asyncio
        # is not supported
        except Exception:
            pass

    async def _gather() -> list[Any]:
        return await asyncio.gather(*tasks_to_execute)

    outputs: list[Any] = asyncio_run(_gather())
    return outputs


def chunks(iterable: Iterable, size: int) -> Iterable:
    """Group an iterable into fixed-size tuples, padding the last with ``None``.

    Internally this uses :func:`itertools.zip_longest`. It yields tuples of
    length ``size``. If the number of elements in ``iterable`` is not a multiple
    of ``size``, the final tuple is right-padded with ``None`` values.

    Args:
        iterable: The input sequence or iterable to group.
        size: The group size.

    Returns:
        An iterator yielding tuples, each of length ``size``. The last tuple
        may contain trailing ``None`` values as padding.

    Examples:
        - Exact multiple of the size
            ```python
            >>> from serapeum.core.tools.concurrent import chunks
            >>> list(chunks([1, 2, 3, 4, 5, 6], 3))
            [(1, 2, 3), (4, 5, 6)]

            ```
        - Remainder is padded with ``None``
            ```python
            >>> list(chunks([1, 2, 3, 4, 5], 3))
            [(1, 2, 3), (4, 5, None)]

            ```
        - Empty iterable yields no groups
            ```python
            >>> list(chunks([], 4))
            []

            ```

    See Also:
        batch_gather: Consumes chunked coroutines to gather in batches.
    """
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=None)


async def batch_gather(
    tasks: list[Coroutine], batch_size: int = 10, verbose: bool = False
) -> list[Any]:
    """Gather coroutines in sequential batches to control concurrency.

    This helper splits ``tasks`` into chunks of size ``batch_size`` using
    ``chunks`` and awaits each batch with :func:`asyncio.gather`, concatenating
    the results. When ``verbose`` is True, a simple textual progress message is
    printed after each batch.

    Args:
        tasks: The coroutines to run.
        batch_size: Number of tasks to await per batch. Must be a positive
            integer for meaningful batching. Defaults to 10.
        verbose: If True, prints progress after each batch completes.
            Defaults to False.

    Returns:
        The concatenated results from all batches in task order.

    Raises:
        Exception: Any exception raised by the provided coroutines will
            propagate from ``asyncio.gather``.

    Examples:
        - Batch execution to limit peak concurrency
            ```python
            >>> import asyncio
            >>> from serapeum.core.tools.concurrent import batch_gather
            >>> async def f(x):
            ...     await asyncio.sleep(0)
            ...     return x * x
            >>> tasks = [f(i) for i in range(7)]
            >>> asyncio.run(batch_gather(tasks, batch_size=3))
            [0, 1, 4, 9, 16, 25, 36]

            ```
        - Verbose progress messages (output suppressed in this example)
            ```python
            >>> import asyncio
            >>> from serapeum.core.tools.concurrent import batch_gather
            >>> async def g(x):
            ...     await asyncio.sleep(0)
            ...     return x
            >>> tasks = [g(i) for i in range(4)]
            >>> asyncio.run(batch_gather(tasks, batch_size=2, verbose=True))
            Completed 2 out of 4 tasks
            Completed 4 out of 4 tasks
            [0, 1, 2, 3]

            ```

    See Also:
        chunks: Iterator that groups an iterable into fixed-size tuples.
        run_async_tasks: Run all tasks at once (no batching).
    """
    output: list[Any] = []
    for task_chunk in chunks(tasks, batch_size):
        task_chunk = (task for task in task_chunk if task is not None)
        output_chunk = await asyncio.gather(*task_chunk)
        output.extend(output_chunk)
        if verbose:
            print(f"Completed {len(output)} out of {len(tasks)} tasks")
    return output
