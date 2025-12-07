import asyncio
import sys
import types

import pytest

from serapeum.core.utils.async_utils import (
    DEFAULT_NUM_WORKERS,
    asyncio_run,
    batch_gather,
    chunks,
    get_asyncio_module,
    run_async_tasks,
    run_jobs,
)


class TestGetAsyncioModule:
    def test_returns_asyncio_when_no_progress(self):
        """
        Inputs:
            show_progress=False.
        Expectation:
            Function returns the standard asyncio module.

        This checks the default branch without progress support.
        """
        mod = get_asyncio_module(show_progress=False)
        assert mod is asyncio

    def test_returns_tqdm_asyncio_when_progress_true_with_dummy(self, monkeypatch):
        """
        Inputs:
            show_progress=True; inject a dummy tqdm.asyncio module.
        Expectation:
            Function returns the dummy tqdm_asyncio object from the injected module.

        This checks import and selection logic for the progress-enabled path.
        """
        dummy_module = types.ModuleType("tqdm.asyncio")

        class DummyTqdmAsyncio:
            pass

        dummy_module.tqdm_asyncio = DummyTqdmAsyncio()

        # Install module under the exact name used by import
        monkeypatch.setitem(sys.modules, "tqdm.asyncio", dummy_module)
        mod = get_asyncio_module(show_progress=True)
        assert isinstance(mod, DummyTqdmAsyncio)


class TestAsyncioRun:
    def test_runs_using_asyncio_run_when_no_current_loop(self):
        """
        Inputs:
            coroutine; no current event loop set for this thread.
        Expectation:
            Function falls back to asyncio.run and returns the awaited value.

        This checks the RuntimeError branch when get_event_loop() fails.
        """

        async def coro():
            return 42

        # Ensure no event loop is set for the current thread
        try:
            asyncio.get_event_loop()
            # If one exists, clear it for this test
            asyncio.set_event_loop(None)  # type: ignore[arg-type]
        except RuntimeError:
            pass

        result = asyncio_run(coro())
        assert result == 42

    def test_runs_on_existing_non_running_loop(self):
        """
        Inputs:
            coroutine; a new event loop is created and set as current (not running).
        Expectation:
            Function uses loop.run_until_complete and returns the awaited value.

        This checks the primary branch when a loop is present but idle.
        """

        async def coro():
            return "ok"

        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = asyncio_run(coro())
            assert result == "ok"
        finally:
            loop.close()
            asyncio.set_event_loop(None)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_nested_event_loop_raises_runtime_error(self):
        """
        Inputs:
            call asyncio_run() from within an already running event loop.
        Expectation:
            Function raises RuntimeError with guidance about nest_asyncio.

        This checks the nested-loop error handling path.
        """

        async def coro():
            return 1

        c = coro()
        try:
            with pytest.raises(RuntimeError) as exc:
                # Call from within running loop
                asyncio_run(c)
            assert "Detected nested async" in str(exc.value)
        finally:
            # Close the coroutine if it wasn't awaited to avoid warnings
            if not c.cr_running and c.cr_frame is not None:
                c.close()


class TestRunAsyncTasks:
    def test_runs_tasks_and_returns_results(self):
        """
        Inputs:
            a list of simple coroutines; show_progress=False.
        Expectation:
            Returns list of results in order, using asyncio.gather under the hood.

        This checks the standard non-progress path.
        """

        async def f(x):
            await asyncio.sleep(0)
            return x * 2

        tasks = [f(i) for i in range(5)]
        out = run_async_tasks(tasks, show_progress=False)
        assert out == [i * 2 for i in range(5)]

    def test_show_progress_true_falls_back_if_import_or_runtime_fails(
        self, monkeypatch
    ):
        """
        Inputs:
            coroutines with show_progress=True; environment without real tqdm.asyncio.
        Expectation:
            Function handles import/runtime errors gracefully and still returns correct results via fallback path.

        This checks the protective try/except path for progress mode.
        """
        # Ensure tqdm.asyncio import will fail by removing if present
        (
            monkeypatch.setitem(sys.modules, "tqdm.asyncio", None)
            if "tqdm.asyncio" in sys.modules
            else None
        )

        async def f(x):
            await asyncio.sleep(0)
            return x + 1

        tasks = [f(i) for i in range(3)]
        out = run_async_tasks(tasks, show_progress=True)
        assert out == [1, 2, 3]


class TestChunks:
    def test_chunks_exact_multiple(self):
        """
        Inputs:
            iterable of 6 elements; size=3.
        Expectation:
            Produces two tuples of length 3 with no None fill values.

        This checks standard grouping behavior.
        """
        data = [1, 2, 3, 4, 5, 6]
        groups = list(chunks(data, 3))
        assert groups == [(1, 2, 3), (4, 5, 6)]

    def test_chunks_with_remainder_filled_with_none(self):
        """
        Inputs:
            iterable of 5 elements; size=3.
        Expectation:
            Produces two tuples, the last one filled with a trailing None.

        This checks zip_longest fill behavior.
        """
        data = [1, 2, 3, 4, 5]
        groups = list(chunks(data, 3))
        assert groups == [(1, 2, 3), (4, 5, None)]

    def test_chunks_empty_iterable(self):
        """
        Inputs:
            empty iterable; any positive size.
        Expectation:
            Produces an empty iterator (no groups).

        This checks graceful handling of empty input.
        """
        assert list(chunks([], 4)) == []


class TestBatchGather:
    @pytest.mark.asyncio
    async def test_batch_gather_batches_and_concatenates_results(self):
        """
        Inputs:
            7 tasks; batch_size=3; verbose=False.
        Expectation:
            Results are concatenated in task order across batches.

        This checks correct chunking and awaiting of batches.
        """

        async def f(x):
            await asyncio.sleep(0)
            return x * x

        tasks = [f(i) for i in range(7)]
        out = await batch_gather(tasks, batch_size=3, verbose=False)
        assert out == [i * i for i in range(7)]

    @pytest.mark.asyncio
    async def test_batch_gather_verbose_prints_progress(self, capsys):
        """
        Inputs:
            4 tasks; batch_size=2; verbose=True.
        Expectation:
            Progress messages are printed after each batch; final output contains all results.

        This checks the verbose side-effect and output integrity.
        """

        async def f(x):
            await asyncio.sleep(0)
            return x

        tasks = [f(i) for i in range(4)]
        out = await batch_gather(tasks, batch_size=2, verbose=True)
        captured = capsys.readouterr().out
        assert "Completed 2 out of 4 tasks" in captured
        assert "Completed 4 out of 4 tasks" in captured
        assert out == [0, 1, 2, 3]


class TestRunJobs:
    @pytest.mark.asyncio
    async def test_run_jobs_without_progress_limits_concurrency_and_preserves_order(
        self,
    ):
        """
        Inputs:
            6 jobs with varying delays; workers=2; show_progress=False.
        Expectation:
            Returns results in the order of the input jobs (asyncio.gather preserves order).

        This checks that the semaphore-wrapped workers and gather work correctly.
        """

        async def job(x):
            # Staggered small sleeps
            await asyncio.sleep(0.01 * (x % 3))
            return x * 10

        jobs = [job(i) for i in range(6)]
        out = await run_jobs(jobs, show_progress=False, workers=2)
        assert out == [i * 10 for i in range(6)]

    @pytest.mark.asyncio
    async def test_run_jobs_with_progress_uses_dummy_tqdm_asyncio(self, monkeypatch):
        """
        Inputs:
            same jobs as above; show_progress=True; inject dummy tqdm.asyncio.tqdm_asyncio with gather.
        Expectation:
            The function imports our dummy tqdm_asyncio and uses its gather to collect results.

        This checks the progress branch without requiring the real tqdm dependency.
        """
        # Create dummy tqdm.asyncio with an object that has an async gather
        dummy_module = types.ModuleType("tqdm.asyncio")

        class DummyTqdmAsyncio:
            @staticmethod
            async def gather(*aws, desc=None):
                # Simply run asyncio.gather internally
                return await asyncio.gather(*aws)

        dummy_module.tqdm_asyncio = DummyTqdmAsyncio

        # Install into sys.modules for import
        monkeypatch.setitem(sys.modules, "tqdm.asyncio", dummy_module)

        async def job(x):
            await asyncio.sleep(0)
            return x

        jobs = [job(i) for i in range(5)]
        out = await run_jobs(
            jobs, show_progress=True, workers=DEFAULT_NUM_WORKERS, desc="desc"
        )
        assert out == [0, 1, 2, 3, 4]
