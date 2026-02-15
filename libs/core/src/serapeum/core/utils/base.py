"""Utilities functions."""

import asyncio
from typing import Iterable, Any, Coroutine, TypeVar


DEFAULT_NUM_WORKERS = 4

T = TypeVar("T")


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def get_tqdm_iterable(
    items: Iterable, show_progress: bool, desc: str, total: int | None = None
) -> Iterable:
    """
    Optionally get a tqdm iterable. Ensures tqdm.auto is used.
    """
    _iterator = items
    if show_progress:
        try:
            from tqdm.auto import tqdm

            return tqdm(items, desc=desc, total=total)
        except ImportError:
            pass
    return _iterator


async def run_jobs(
    jobs: list[Coroutine[Any, Any, T]],
    show_progress: bool = False,
    workers: int = DEFAULT_NUM_WORKERS,
    desc: str | None = None,
) -> list[T]:
    """Run a collection of coroutines with limited concurrency.

    Each job is wrapped by a semaphore-guarded worker to ensure that at most
    ``workers`` jobs run concurrently. Results are collected in the same order
    as the input jobs via :func:`asyncio.gather` (or ``tqdm.asyncio`` when
    ``show_progress`` is True).

    Args:
        jobs (list[Coroutine[Any, Any, T]]):
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
        list[T]: The results of the jobs in input order.

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
            >>> from serapeum.core.utils.base import run_jobs
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
    # This semaphore is used to limit the number of concurrent tasks that can run simultaneously.
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
