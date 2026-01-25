"""Async utils."""

import asyncio
from itertools import zip_longest
from typing import Any, Coroutine, Iterable, List, Optional, TypeVar

DEFAULT_NUM_WORKERS = 4

T = TypeVar("T")


def get_asyncio_module(show_progress: bool = False) -> Any:
    """Return the asyncio-like module to use, optionally with progress support.

    When ``show_progress`` is False, this simply returns Python's built-in
    :mod:`asyncio` module. When ``show_progress`` is True, it attempts to import
    ``tqdm.asyncio.tqdm_asyncio`` and returns that object so that progress-aware
    gathering functions can be used.

    This function is a small utility used by higher-level helpers in this
    module to decide whether to execute tasks with a progress bar. See
    ``run_async_tasks`` and ``run_jobs`` for typical usage.

    Args:
        show_progress (bool):
            If True, return ``tqdm.asyncio.tqdm_asyncio`` (requires the
            optional ``tqdm`` package). If False, return the standard
            :mod:`asyncio` module.

    Returns:
        Any: Either the :mod:`asyncio` module (when ``show_progress`` is False)
        or the ``tqdm.asyncio.tqdm_asyncio`` object (when True).

    Raises:
        ImportError: If ``show_progress`` is True and ``tqdm.asyncio`` cannot be
            imported.

    See Also:
        - ``run_async_tasks``: Runs a list of coroutines with optional progress.
        - ``run_jobs``: Concurrency-limited job runner with optional progress.

    Examples:
        - Return the standard asyncio module
            ```python
            >>> import asyncio as _asyncio
            >>> from serapeum.core.utils.async_utils import get_asyncio_module
            >>> mod = get_asyncio_module(False)
            >>> mod is _asyncio
            True

            ```
        - Return the progress-enabled object by injecting a dummy ``tqdm.asyncio``
          for environments where ``tqdm`` may not be installed.
            ```python
            >>> import sys, types
            >>> dummy = types.ModuleType("tqdm.asyncio")
            >>> class DummyTqdmAsyncio:  # minimal stand-in
            ...     pass
            >>> dummy.tqdm_asyncio = DummyTqdmAsyncio()
            >>> sys.modules["tqdm.asyncio"] = dummy
            >>> from serapeum.core.utils.async_utils import get_asyncio_module
            >>> mod = get_asyncio_module(True)
            >>> isinstance(mod, DummyTqdmAsyncio)
            True
            >>> del sys.modules["tqdm.asyncio"]  # cleanup

            ```
    """
    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        module = tqdm_asyncio
    else:
        module = asyncio

    return module


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
        coro (Coroutine):
            The coroutine object to run to completion.

    Returns:
        Any: The value returned by the awaited coroutine.

    Raises:
        RuntimeError: If called from within an already running event loop
            (nested asyncio), a ``RuntimeError`` is raised with a message
            explaining how to proceed.

    See Also:
        - ``run_async_tasks``: Convenience to run multiple coroutines and
          collect their results.
        - ``batch_gather``: Async helper to gather coroutines in batches.

    Examples:
        - Run a coroutine when no loop is running
            ```python
            >>> from serapeum.core.utils.async_utils import asyncio_run
            >>> async def add(a, b):
            ...     return a + b
            >>> asyncio_run(add(2, 3))
            5

            ```
        - Detect nested event loop usage and handle the error
            ```python
            >>> import asyncio
            >>> async def inner():
            ...     from serapeum.core.utils.async_utils import asyncio_run
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
    tasks: List[Coroutine],
    show_progress: bool = False,
    progress_bar_desc: str = "Running async tasks",
) -> List[Any]:
    """Run a list of coroutines to completion and collect their results.

    This convenience wrapper optionally displays a progress bar via
    ``tqdm.asyncio`` when ``show_progress`` is True. If the environment doesn't
    support that (e.g., ``tqdm`` or ``nest_asyncio`` isn't available, or a
    runtime error occurs), it falls back gracefully to a standard
    ``asyncio.gather`` execution using ``asyncio_run``.

    Args:
        tasks (List[Coroutine]):
            The coroutines to run.
        show_progress (bool):
            If True, attempt to use ``tqdm.asyncio.tqdm.gather`` with a progress
            bar. If unavailable or incompatible, a silent fallback is applied.
        progress_bar_desc (str):
            Optional label shown by the progress bar.

    Returns:
        List[Any]: Results of the completed coroutines in the same order as the
        input list.

    Raises:
        Exception: Any exception raised by the provided coroutines will
            propagate from ``asyncio.gather``.

    See Also:
        - ``asyncio_run``: Helper that safely runs a coroutine from sync code.
        - ``run_jobs``: Concurrency-limited variant suitable for many jobs.

    Examples:
        - Run tasks without a progress bar
            ```python
            >>> import asyncio
            >>> from serapeum.core.utils.async_utils import run_async_tasks
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
    """
    tasks_to_execute: List[Any] = tasks
    if show_progress:
        try:
            import nest_asyncio
            from tqdm.asyncio import tqdm

            # jupyter notebooks already have an event loop running
            # we need to reuse it instead of creating a new one
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()

            async def _tqdm_gather() -> List[Any]:
                return await tqdm.gather(*tasks_to_execute, desc=progress_bar_desc)

            tqdm_outputs: List[Any] = loop.run_until_complete(_tqdm_gather())
            return tqdm_outputs
        # run the operation w/o tqdm on hitting a fatal
        # may occur in some environments where tqdm.asyncio
        # is not supported
        except Exception:
            pass

    async def _gather() -> List[Any]:
        return await asyncio.gather(*tasks_to_execute)

    outputs: List[Any] = asyncio_run(_gather())
    return outputs


def chunks(iterable: Iterable, size: int) -> Iterable:
    """Group an iterable into fixed-size tuples, padding the last with ``None``.

    Internally this uses :func:`itertools.zip_longest`. It yields tuples of
    length ``size``. If the number of elements in ``iterable`` is not a multiple
    of ``size``, the final tuple is right-padded with ``None`` values.

    Args:
        iterable (Iterable): The input sequence or iterable to group.
        size (int): The group size.

    Returns:
        Iterable: An iterator yielding tuples, each of length ``size``. The last
        tuple may contain trailing ``None`` values as padding.

    See Also:
        - ``batch_gather``: Consumes chunked coroutines to gather in batches.

    Examples:
        - Exact multiple of the size
            ```python
            >>> from serapeum.core.utils.async_utils import chunks
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
    """
    args = [iter(iterable)] * size
    return zip_longest(*args, fillvalue=None)


async def batch_gather(
    tasks: List[Coroutine], batch_size: int = 10, verbose: bool = False
) -> List[Any]:
    """Gather coroutines in sequential batches to control concurrency.

    This helper splits ``tasks`` into chunks of size ``batch_size`` using
    ``chunks`` and awaits each batch with :func:`asyncio.gather`, concatenating
    the results. When ``verbose`` is True, a simple textual progress message is
    printed after each batch.

    Args:
        tasks (List[Coroutine]):
            The coroutines to run.
        batch_size (int):
            Number of tasks to await per batch. Must be a positive integer for
            meaningful batching.
        verbose (bool):
            If True, prints progress after each batch completes.

    Returns:
        List[Any]: The concatenated results from all batches in task order.

    Raises:
        Exception: Any exception raised by the provided coroutines will
            propagate from ``asyncio.gather``.

    See Also:
        - ``chunks``: Iterator that groups an iterable into fixed-size tuples.
        - ``run_async_tasks``: Run all tasks at once (no batching).
        - ``run_jobs``: Limit concurrency using a semaphore.

    Examples:
        - Batch execution to limit peak concurrency
            ```python
            >>> import asyncio
            >>> from serapeum.core.utils.async_utils import batch_gather
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
            >>> async def g(x):
            ...     await asyncio.sleep(0)
            ...     return x
            >>> tasks = [g(i) for i in range(4)]
            >>> asyncio.run(batch_gather(tasks, batch_size=2, verbose=True))
            [0, 1, 2, 3]

            ```
    """
    output: List[Any] = []
    for task_chunk in chunks(tasks, batch_size):
        task_chunk = (task for task in task_chunk if task is not None)
        output_chunk = await asyncio.gather(*task_chunk)
        output.extend(output_chunk)
        if verbose:
            print(f"Completed {len(output)} out of {len(tasks)} tasks")
    return output


async def run_jobs(
    jobs: List[Coroutine[Any, Any, T]],
    show_progress: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
    desc: Optional[str] = None,
) -> List[T]:
    """Run a collection of coroutines with limited concurrency.

    Each job is wrapped by a semaphore-guarded worker to ensure that at most
    ``workers`` jobs run concurrently. Results are collected in the same order
    as the input jobs via :func:`asyncio.gather` (or ``tqdm.asyncio`` when
    ``show_progress`` is True).

    Args:
        jobs (List[Coroutine[Any, Any, T]]):
            The coroutines to run.
        show_progress (bool):
            If True, attempts to use ``tqdm.asyncio.tqdm_asyncio.gather`` to
            display a progress bar for the overall job collection.
        workers (int):
            Maximum number of concurrently running jobs. Typical values range
            from 1 (fully serial) to a small multiple of CPU cores or a limit
            suited to the workload and I/O boundness. Defaults to
            ``DEFAULT_NUM_WORKERS``.
        desc (Optional[str]):
            Optional text description for the progress bar (only used when
            ``show_progress`` is True).

    Returns:
        List[T]: The results of the jobs in input order.

    Raises:
        ImportError: If ``show_progress`` is True and ``tqdm.asyncio`` cannot be
            imported.
        Exception: Any exception raised by an individual job will propagate from
            the gather call.

    See Also:
        - ``run_async_tasks``: Synchronous helper to run a list of coroutines.
        - ``batch_gather``: Run coroutines in sequential batches.

    Examples:
        - Limit concurrency without a progress bar
            ```python
            >>> import asyncio
            >>> from serapeum.core.utils.async_utils import run_jobs
            >>> async def job(x):
            ...     await asyncio.sleep(0)
            ...     return x + 1
            >>> jobs = [job(i) for i in range(4)]
            >>> asyncio.run(run_jobs(jobs, show_progress=False, workers=2))
            [1, 2, 3, 4]

            ```
        - Use a dummy ``tqdm.asyncio`` to demonstrate progress-enabled path
            ```python
            >>> import sys, types, asyncio
            >>> dummy_module = types.ModuleType("tqdm.asyncio")
            >>> class DummyTqdmAsyncio:
            ...     @staticmethod
            ...     async def gather(*aws, desc=None):
            ...         return await asyncio.gather(*aws)
            >>> dummy_module.tqdm_asyncio = DummyTqdmAsyncio
            >>> sys.modules["tqdm.asyncio"] = dummy_module
            >>> async def job2(x):
            ...     await asyncio.sleep(0)
            ...     return x
            >>> asyncio.run(run_jobs([job2(i) for i in range(3)], show_progress=True, workers=2, desc="demo"))
            [0, 1, 2]
            >>> del sys.modules["tqdm.asyncio"]  # cleanup

            ```
    """
    # his semaphore is used to limit the number of concurrent tasks that can run simultaneously.
    semaphore = asyncio.Semaphore(workers)

    async def worker(job: Coroutine) -> Any:
        async with semaphore:
            return await job

    pool_jobs = [worker(job) for job in jobs]

    if show_progress:
        from tqdm.asyncio import tqdm_asyncio

        results = await tqdm_asyncio.gather(*pool_jobs, desc=desc)
    else:
        results = await asyncio.gather(*pool_jobs)

    return results
