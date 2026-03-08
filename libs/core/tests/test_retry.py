"""Unit tests for serapeum.core.retry module."""

from __future__ import annotations

import inspect
import logging
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from serapeum.core.retry import (
    DEFAULT_INITIAL_DELAY,
    DEFAULT_MAX_DELAY,
    Retry,
    build_async_retryer,
    build_retryer,
    retry,
)


def _always_retryable(exc: BaseException) -> bool:
    """Predicate that retries ConnectionError only."""
    return isinstance(exc, ConnectionError)


def _never_retryable(exc: BaseException) -> bool:
    """Predicate that never retries."""
    return False


class _Counter:
    """Mutable call counter for tracking retry attempts.

    Args:
        fail_times: Number of calls that raise before succeeding.
        exc: The exception to raise on failing calls.
    """

    def __init__(self, fail_times: int = 0, exc: BaseException | None = None):
        self.calls = 0
        self.fail_times = fail_times
        self.exc = exc or ConnectionError("transient")

    def __call__(self) -> str:
        """Sync call — raises for the first ``fail_times`` invocations."""
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return "ok"

    async def async_call(self) -> str:
        """Async call — raises for the first ``fail_times`` invocations."""
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return "ok"


class _DummyModel(Retry, BaseModel):
    """Minimal model inheriting from Retry for field tests."""

    name: str = "test"


class _SyncClient(Retry, BaseModel):
    """Test client with sync methods decorated by retry."""

    call_count: int = 0

    @retry(_always_retryable)
    def succeed(self) -> str:
        """Always succeeds on first call."""
        return "ok"

    @retry(_always_retryable)
    def succeed_with_args(self, a: int, b: str, *, kw: bool = False) -> dict[str, Any]:
        """Returns args/kwargs for passthrough verification."""
        return {"a": a, "b": b, "kw": kw}

    @retry(_always_retryable)
    def fail_then_succeed(self) -> str:
        """Fails once then succeeds."""
        self.call_count += 1
        if self.call_count <= 1:
            raise ConnectionError("transient")
        return "recovered"

    @retry(_always_retryable)
    def always_fail_retryable(self) -> str:
        """Always raises a retryable exception."""
        raise ConnectionError("always fails")

    @retry(_always_retryable)
    def always_fail_permanent(self) -> str:
        """Always raises a non-retryable exception."""
        raise ValueError("permanent")


class _AsyncClient(Retry, BaseModel):
    """Test client with async methods decorated by retry."""

    call_count: int = 0

    @retry(_always_retryable)
    async def succeed(self) -> str:
        """Always succeeds on first call."""
        return "async_ok"

    @retry(_always_retryable)
    async def succeed_with_args(self, a: int, b: str, *, kw: bool = False) -> dict[str, Any]:
        """Returns args/kwargs for passthrough verification."""
        return {"a": a, "b": b, "kw": kw}

    @retry(_always_retryable)
    async def fail_then_succeed(self) -> str:
        """Fails once then succeeds."""
        self.call_count += 1
        if self.call_count <= 1:
            raise ConnectionError("transient")
        return "async_recovered"

    @retry(_always_retryable)
    async def always_fail_retryable(self) -> str:
        """Always raises a retryable exception."""
        raise ConnectionError("always fails")

    @retry(_always_retryable)
    async def always_fail_permanent(self) -> str:
        """Always raises a non-retryable exception."""
        raise ValueError("permanent")


class _StreamClient(Retry, BaseModel):
    """Test client with generator methods decorated by retry."""

    call_count: int = 0

    @retry(_always_retryable)
    def stream(self) -> Any:
        """Yields chunks successfully."""
        yield "chunk1"
        yield "chunk2"

    @retry(_always_retryable)
    def stream_with_args(self, prefix: str) -> Any:
        """Yields chunks with args."""
        yield f"{prefix}_1"
        yield f"{prefix}_2"

    @retry(_always_retryable)
    def stream_fail_then_succeed(self) -> Any:
        """Fails mid-stream on first attempt, succeeds fully on second."""
        self.call_count += 1
        yield "partial"
        if self.call_count <= 1:
            raise ConnectionError("mid-stream failure")
        yield "complete"

    @retry(_always_retryable)
    def stream_always_fail(self) -> Any:
        """Always raises retryable after yielding partial data."""
        yield "partial"
        raise ConnectionError("always fails")

    @retry(_always_retryable)
    def stream_permanent_error(self) -> Any:
        """Raises non-retryable exception immediately."""
        raise ValueError("permanent")
        yield  # noqa: unreachable - needed to make this a generator


class _AsyncStreamClient(Retry, BaseModel):
    """Test client with async gen methods decorated by retry(stream=True)."""

    call_count: int = 0

    @retry(_always_retryable, stream=True)
    async def stream(self) -> Any:
        """Returns async gen that yields chunks successfully."""
        async def gen():
            yield "async_chunk1"
            yield "async_chunk2"
        return gen()

    @retry(_always_retryable, stream=True)
    async def stream_with_args(self, prefix: str) -> Any:
        """Returns async gen yielding chunks with args."""
        async def gen():
            yield f"{prefix}_1"
            yield f"{prefix}_2"
        return gen()

    @retry(_always_retryable, stream=True)
    async def stream_fail_then_succeed(self) -> Any:
        """Fails mid-stream on first attempt, succeeds fully on second."""
        self.call_count += 1
        attempt = self.call_count

        async def gen():
            yield "partial"
            if attempt <= 1:
                raise ConnectionError("mid-stream failure")
            yield "complete"
        return gen()

    @retry(_always_retryable, stream=True)
    async def stream_always_fail(self) -> Any:
        """Always raises retryable after yielding partial data."""
        async def gen():
            yield "partial"
            raise ConnectionError("always fails")
        return gen()

    @retry(_always_retryable, stream=True)
    async def stream_permanent_error(self) -> Any:
        """Raises non-retryable during iteration."""
        async def gen():
            raise ValueError("permanent")
            yield  # noqa: unreachable
        return gen()


@pytest.mark.unit
class TestRetry:
    """Tests for the Retry Pydantic mixin class."""

    def test_default_max_retries(self) -> None:
        """Test Retry provides default max_retries of 3."""
        m = _DummyModel()
        assert m.max_retries == 3, f"Expected default 3, got {m.max_retries}"

    def test_custom_max_retries(self) -> None:
        """Test Retry accepts custom max_retries value."""
        m = _DummyModel(max_retries=5)
        assert m.max_retries == 5, f"Expected 5, got {m.max_retries}"

    def test_zero_max_retries(self) -> None:
        """Test Retry accepts zero (disables retries)."""
        m = _DummyModel(max_retries=0)
        assert m.max_retries == 0, f"Expected 0, got {m.max_retries}"

    def test_negative_max_retries_rejected(self) -> None:
        """Test Retry rejects negative max_retries via ge=0 constraint."""
        with pytest.raises(ValidationError) as exc_info:
            _DummyModel(max_retries=-1)
        assert "max_retries" in str(exc_info.value), (
            f"Validation error should reference max_retries: {exc_info.value}"
        )

    def test_large_max_retries(self) -> None:
        """Test Retry accepts large max_retries values without error."""
        m = _DummyModel(max_retries=1000)
        assert m.max_retries == 1000, f"Expected 1000, got {m.max_retries}"

    def test_field_is_serializable(self) -> None:
        """Test max_retries appears in model_dump output."""
        m = _DummyModel(max_retries=7)
        dumped = m.model_dump()
        assert dumped["max_retries"] == 7, f"Expected 7 in dump, got {dumped}"

    def test_inherits_alongside_other_fields(self) -> None:
        """Test Retry mixin coexists with other model fields."""
        m = _DummyModel(name="custom", max_retries=2)
        assert m.name == "custom", f"Expected 'custom', got {m.name}"
        assert m.max_retries == 2, f"Expected 2, got {m.max_retries}"


@pytest.mark.unit
class TestModuleConstants:
    """Tests for module-level constants."""

    def test_default_initial_delay_value(self) -> None:
        """Test DEFAULT_INITIAL_DELAY is 0.5 seconds."""
        assert DEFAULT_INITIAL_DELAY == 0.5, (
            f"Expected 0.5, got {DEFAULT_INITIAL_DELAY}"
        )

    def test_default_max_delay_value(self) -> None:
        """Test DEFAULT_MAX_DELAY is 8.0 seconds."""
        assert DEFAULT_MAX_DELAY == 8.0, f"Expected 8.0, got {DEFAULT_MAX_DELAY}"


@pytest.mark.unit
class TestBuildRetryer:
    """Tests for build_retryer factory function."""

    def test_succeeds_first_try(self) -> None:
        """Test no retries when the operation succeeds immediately."""
        counter = _Counter(fail_times=0)
        for attempt in build_retryer(3, _always_retryable):
            with attempt:
                result = counter()
        assert result == "ok", f"Expected 'ok', got {result}"
        assert counter.calls == 1, f"Expected 1 call, got {counter.calls}"

    def test_succeeds_after_retries(self) -> None:
        """Test successful recovery after transient failures."""
        counter = _Counter(fail_times=2)
        for attempt in build_retryer(3, _always_retryable):
            with attempt:
                result = counter()
        assert result == "ok", f"Expected 'ok', got {result}"
        assert counter.calls == 3, f"Expected 3 calls, got {counter.calls}"

    def test_exhausted_raises(self) -> None:
        """Test original exception re-raised when retries exhausted."""
        counter = _Counter(fail_times=10)
        with pytest.raises(ConnectionError):
            for attempt in build_retryer(2, _always_retryable):
                with attempt:
                    counter()
        assert counter.calls == 3, (
            f"Expected 3 calls (1 initial + 2 retries), got {counter.calls}"
        )

    def test_non_retryable_raises_immediately(self) -> None:
        """Test non-retryable exception propagates without retry."""
        counter = _Counter(fail_times=10, exc=ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            for attempt in build_retryer(3, _always_retryable):
                with attempt:
                    counter()
        assert counter.calls == 1, f"Expected 1 call, got {counter.calls}"

    def test_zero_retries_no_retry(self) -> None:
        """Test max_retries=0 means exactly one attempt, no retries."""
        counter = _Counter(fail_times=1)
        with pytest.raises(ConnectionError):
            for attempt in build_retryer(0, _always_retryable):
                with attempt:
                    counter()
        assert counter.calls == 1, f"Expected 1 call, got {counter.calls}"

    def test_with_logger_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test logger receives WARNING before each sleep when provided."""
        logger = logging.getLogger("test.retry.sync")
        counter = _Counter(fail_times=1)
        with caplog.at_level(logging.WARNING, logger="test.retry.sync"):
            for attempt in build_retryer(
                2, _always_retryable, logger=logger, initial_delay=0.0, max_delay=0.0
            ):
                with attempt:
                    counter()
        assert counter.calls == 2, f"Expected 2 calls, got {counter.calls}"
        assert len(caplog.records) >= 1, "Expected at least one WARNING log record"

    def test_custom_delays(self) -> None:
        """Test custom initial_delay and max_delay are accepted without error."""
        counter = _Counter(fail_times=0)
        for attempt in build_retryer(
            1, _always_retryable, initial_delay=0.01, max_delay=0.02
        ):
            with attempt:
                result = counter()
        assert result == "ok", f"Expected 'ok', got {result}"

    def test_never_retryable_predicate(self) -> None:
        """Test predicate returning False always propagates immediately."""
        counter = _Counter(fail_times=10)
        with pytest.raises(ConnectionError):
            for attempt in build_retryer(5, _never_retryable):
                with attempt:
                    counter()
        assert counter.calls == 1, f"Expected 1 call, got {counter.calls}"


@pytest.mark.unit
class TestBuildAsyncRetryer:
    """Tests for build_async_retryer factory function."""

    @pytest.mark.asyncio
    async def test_succeeds_first_try(self) -> None:
        """Test no retries when async operation succeeds immediately."""
        counter = _Counter(fail_times=0)
        async for attempt in build_async_retryer(3, _always_retryable):
            with attempt:
                result = await counter.async_call()
        assert result == "ok", f"Expected 'ok', got {result}"
        assert counter.calls == 1, f"Expected 1 call, got {counter.calls}"

    @pytest.mark.asyncio
    async def test_succeeds_after_retries(self) -> None:
        """Test async recovery after transient failures."""
        counter = _Counter(fail_times=2)
        async for attempt in build_async_retryer(3, _always_retryable):
            with attempt:
                result = await counter.async_call()
        assert result == "ok", f"Expected 'ok', got {result}"
        assert counter.calls == 3, f"Expected 3 calls, got {counter.calls}"

    @pytest.mark.asyncio
    async def test_exhausted_raises(self) -> None:
        """Test original exception re-raised when async retries exhausted."""
        counter = _Counter(fail_times=10)
        with pytest.raises(ConnectionError):
            async for attempt in build_async_retryer(2, _always_retryable):
                with attempt:
                    await counter.async_call()
        assert counter.calls == 3, f"Expected 3 calls, got {counter.calls}"

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self) -> None:
        """Test non-retryable async exception propagates without retry."""
        counter = _Counter(fail_times=10, exc=ValueError("bad"))
        with pytest.raises(ValueError, match="bad"):
            async for attempt in build_async_retryer(3, _always_retryable):
                with attempt:
                    await counter.async_call()
        assert counter.calls == 1, f"Expected 1 call, got {counter.calls}"

    @pytest.mark.asyncio
    async def test_zero_retries_no_retry(self) -> None:
        """Test max_retries=0 means exactly one async attempt."""
        counter = _Counter(fail_times=1)
        with pytest.raises(ConnectionError):
            async for attempt in build_async_retryer(0, _always_retryable):
                with attempt:
                    await counter.async_call()
        assert counter.calls == 1, f"Expected 1 call, got {counter.calls}"

    @pytest.mark.asyncio
    async def test_with_logger_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test async logger receives WARNING before each sleep."""
        logger = logging.getLogger("test.retry.async")
        counter = _Counter(fail_times=1)
        with caplog.at_level(logging.WARNING, logger="test.retry.async"):
            async for attempt in build_async_retryer(
                2, _always_retryable, logger=logger, initial_delay=0.0, max_delay=0.0
            ):
                with attempt:
                    await counter.async_call()
        assert counter.calls == 2, f"Expected 2 calls, got {counter.calls}"
        assert len(caplog.records) >= 1, "Expected at least one WARNING log record"

    @pytest.mark.asyncio
    async def test_never_retryable_predicate(self) -> None:
        """Test async predicate returning False always propagates immediately."""
        counter = _Counter(fail_times=10)
        with pytest.raises(ConnectionError):
            async for attempt in build_async_retryer(5, _never_retryable):
                with attempt:
                    await counter.async_call()
        assert counter.calls == 1, f"Expected 1 call, got {counter.calls}"


@pytest.mark.unit
class TestRetryDecorator:
    """Tests for retry decorator factory (sync non-streaming)."""

    def test_success_returns_value(self) -> None:
        """Test decorated method returns value on success."""
        client = _SyncClient(max_retries=3)
        result = client.succeed()
        assert result == "ok", f"Expected 'ok', got {result}"

    def test_args_kwargs_passthrough(self) -> None:
        """Test decorated method passes positional and keyword args correctly."""
        client = _SyncClient(max_retries=3)
        result = client.succeed_with_args(42, "hello", kw=True)
        assert result == {"a": 42, "b": "hello", "kw": True}, (
            f"Args not passed through correctly: {result}"
        )

    def test_retries_on_transient_failure(self) -> None:
        """Test decorator retries on retryable exception and returns result."""
        client = _SyncClient(max_retries=3)
        result = client.fail_then_succeed()
        assert result == "recovered", f"Expected 'recovered', got {result}"
        assert client.call_count == 2, f"Expected 2 calls, got {client.call_count}"

    def test_exhausted_retries_raises(self) -> None:
        """Test decorator re-raises when all retries are exhausted."""
        client = _SyncClient(max_retries=2)
        with pytest.raises(ConnectionError, match="always fails"):
            client.always_fail_retryable()

    def test_non_retryable_propagates_immediately(self) -> None:
        """Test non-retryable exception skips retry and propagates."""
        client = _SyncClient(max_retries=3)
        with pytest.raises(ValueError, match="permanent"):
            client.always_fail_permanent()

    def test_zero_retries_executes_once(self) -> None:
        """Test max_retries=0 executes the method exactly once."""
        client = _SyncClient(max_retries=0)
        result = client.succeed()
        assert result == "ok", f"Expected 'ok', got {result}"

    def test_zero_retries_no_retry_on_failure(self) -> None:
        """Test max_retries=0 raises immediately on failure without retry."""
        client = _SyncClient(max_retries=0)
        with pytest.raises(ConnectionError, match="always fails"):
            client.always_fail_retryable()

    def test_different_instances_different_budgets(self) -> None:
        """Test max_retries is read per-instance at call time."""
        zero = _SyncClient(max_retries=0)
        three = _SyncClient(max_retries=3)
        assert zero.succeed() == "ok", "max_retries=0 should still succeed"
        assert three.succeed() == "ok", "max_retries=3 should succeed"

    def test_preserves_function_name(self) -> None:
        """Test functools.wraps preserves the original method name."""
        assert _SyncClient.succeed.__name__ == "succeed", (
            f"Expected 'succeed', got {_SyncClient.succeed.__name__}"
        )

    def test_preserves_docstring(self) -> None:
        """Test functools.wraps preserves the original docstring."""
        assert _SyncClient.succeed.__doc__ is not None, "Docstring should be preserved"
        assert "Always succeeds" in _SyncClient.succeed.__doc__, (
            f"Original docstring lost: {_SyncClient.succeed.__doc__}"
        )

    def test_with_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test decorator with logger emits warnings on retry."""
        logger = logging.getLogger("test.retry")

        class _LogClient(Retry, BaseModel):
            call_count: int = 0

            @retry(_always_retryable, retry_logger=logger)
            def call(self) -> str:
                self.call_count += 1
                if self.call_count <= 1:
                    raise ConnectionError("transient")
                return "ok"

        client = _LogClient(max_retries=3)
        with caplog.at_level(logging.WARNING, logger="test.retry"):
            result = client.call()
        assert result == "ok", f"Expected 'ok', got {result}"
        assert len(caplog.records) >= 1, "Expected at least one WARNING log"


@pytest.mark.unit
class TestAretry:
    """Tests for retry decorator factory (async non-streaming)."""

    @pytest.mark.asyncio
    async def test_success_returns_value(self) -> None:
        """Test decorated async method returns value on success."""
        client = _AsyncClient(max_retries=3)
        result = await client.succeed()
        assert result == "async_ok", f"Expected 'async_ok', got {result}"

    @pytest.mark.asyncio
    async def test_args_kwargs_passthrough(self) -> None:
        """Test decorated async method passes args correctly."""
        client = _AsyncClient(max_retries=3)
        result = await client.succeed_with_args(42, "hello", kw=True)
        assert result == {"a": 42, "b": "hello", "kw": True}, (
            f"Args not passed through correctly: {result}"
        )

    @pytest.mark.asyncio
    async def test_retries_on_transient_failure(self) -> None:
        """Test async decorator retries on retryable exception."""
        client = _AsyncClient(max_retries=3)
        result = await client.fail_then_succeed()
        assert result == "async_recovered", f"Expected 'async_recovered', got {result}"
        assert client.call_count == 2, f"Expected 2 calls, got {client.call_count}"

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self) -> None:
        """Test async decorator re-raises when retries exhausted."""
        client = _AsyncClient(max_retries=2)
        with pytest.raises(ConnectionError, match="always fails"):
            await client.always_fail_retryable()

    @pytest.mark.asyncio
    async def test_non_retryable_propagates_immediately(self) -> None:
        """Test non-retryable async exception propagates without retry."""
        client = _AsyncClient(max_retries=3)
        with pytest.raises(ValueError, match="permanent"):
            await client.always_fail_permanent()

    @pytest.mark.asyncio
    async def test_zero_retries_executes_once(self) -> None:
        """Test max_retries=0 executes async method exactly once."""
        client = _AsyncClient(max_retries=0)
        result = await client.succeed()
        assert result == "async_ok", f"Expected 'async_ok', got {result}"

    @pytest.mark.asyncio
    async def test_preserves_function_name(self) -> None:
        """Test functools.wraps preserves async method name."""
        assert _AsyncClient.succeed.__name__ == "succeed", (
            f"Expected 'succeed', got {_AsyncClient.succeed.__name__}"
        )

    @pytest.mark.asyncio
    async def test_wrapper_is_coroutine_function(self) -> None:
        """Test wrapped method is still recognized as a coroutine function."""
        assert inspect.iscoroutinefunction(_AsyncClient.succeed), (
            "Wrapped method should be a coroutine function"
        )


@pytest.mark.unit
class TestRetryStream:
    """Tests for retry decorator factory (sync streaming)."""

    def test_yields_all_chunks(self) -> None:
        """Test decorated generator yields all chunks on success."""
        client = _StreamClient(max_retries=2)
        chunks = list(client.stream())
        assert chunks == ["chunk1", "chunk2"], f"Expected ['chunk1', 'chunk2'], got {chunks}"

    def test_returns_generator(self) -> None:
        """Test decorated method returns a generator object."""
        client = _StreamClient(max_retries=2)
        result = client.stream()
        assert inspect.isgenerator(result), (
            f"Expected generator, got {type(result).__name__}"
        )

    def test_args_passthrough(self) -> None:
        """Test decorated generator passes args correctly."""
        client = _StreamClient(max_retries=2)
        chunks = list(client.stream_with_args("test"))
        assert chunks == ["test_1", "test_2"], f"Expected ['test_1', 'test_2'], got {chunks}"

    def test_retries_mid_stream_failure(self) -> None:
        """Test generator retries on mid-stream failure.

        Test scenario:
            First attempt yields 'partial' then raises ConnectionError.
            Second attempt yields 'partial' and 'complete'.
            Caller receives all 3 chunks (partial from failed + 2 from retry).
        """
        client = _StreamClient(max_retries=2)
        chunks = list(client.stream_fail_then_succeed())
        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}: {chunks}"
        assert chunks[0] == "partial", f"First chunk should be 'partial', got {chunks[0]}"
        assert chunks[1] == "partial", f"Second chunk should be 'partial', got {chunks[1]}"
        assert chunks[2] == "complete", f"Third chunk should be 'complete', got {chunks[2]}"

    def test_exhausted_retries_raises(self) -> None:
        """Test generator re-raises when all retries exhausted."""
        client = _StreamClient(max_retries=1)
        with pytest.raises(ConnectionError, match="always fails"):
            list(client.stream_always_fail())

    def test_non_retryable_propagates_immediately(self) -> None:
        """Test non-retryable exception in generator propagates without retry."""
        client = _StreamClient(max_retries=3)
        with pytest.raises(ValueError, match="permanent"):
            list(client.stream_permanent_error())

    def test_zero_retries_yields_then_raises(self) -> None:
        """Test max_retries=0 yields partial data then raises on failure."""
        client = _StreamClient(max_retries=0)
        with pytest.raises(ConnectionError, match="always fails"):
            list(client.stream_always_fail())

    def test_preserves_function_name(self) -> None:
        """Test functools.wraps preserves generator method name."""
        assert _StreamClient.stream.__name__ == "stream", (
            f"Expected 'stream', got {_StreamClient.stream.__name__}"
        )

    def test_lazy_evaluation(self) -> None:
        """Test generator is lazily evaluated (no work done until iterated)."""
        client = _StreamClient(max_retries=2)
        gen = client.stream_fail_then_succeed()
        assert client.call_count == 0, (
            f"Generator should not execute until iterated, call_count={client.call_count}"
        )
        next(gen)
        assert client.call_count >= 1, "First next() should trigger execution"


@pytest.mark.unit
class TestAretryStream:
    """Tests for retry(stream=True) decorator factory (async streaming)."""

    @pytest.mark.asyncio
    async def test_yields_all_chunks(self) -> None:
        """Test async gen yields all chunks on success."""
        client = _AsyncStreamClient(max_retries=2)
        chunks = [chunk async for chunk in await client.stream()]
        assert chunks == ["async_chunk1", "async_chunk2"], (
            f"Expected ['async_chunk1', 'async_chunk2'], got {chunks}"
        )

    @pytest.mark.asyncio
    async def test_returns_async_generator(self) -> None:
        """Test decorated method returns an async generator object."""
        client = _AsyncStreamClient(max_retries=2)
        result = await client.stream()
        assert inspect.isasyncgen(result), (
            f"Expected async generator, got {type(result).__name__}"
        )

    @pytest.mark.asyncio
    async def test_args_passthrough(self) -> None:
        """Test async gen passes args correctly."""
        client = _AsyncStreamClient(max_retries=2)
        chunks = [chunk async for chunk in await client.stream_with_args("test")]
        assert chunks == ["test_1", "test_2"], (
            f"Expected ['test_1', 'test_2'], got {chunks}"
        )

    @pytest.mark.asyncio
    async def test_retries_mid_stream_failure(self) -> None:
        """Test async gen retries on mid-stream failure.

        Test scenario:
            First attempt yields 'partial' then raises ConnectionError.
            Second attempt yields 'partial' and 'complete'.
            Caller receives all 3 chunks.
        """
        client = _AsyncStreamClient(max_retries=2)
        chunks = [chunk async for chunk in await client.stream_fail_then_succeed()]
        assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}: {chunks}"
        assert chunks[0] == "partial", f"First chunk should be 'partial', got {chunks[0]}"
        assert chunks[1] == "partial", f"Second chunk should be 'partial', got {chunks[1]}"
        assert chunks[2] == "complete", f"Third chunk should be 'complete', got {chunks[2]}"

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self) -> None:
        """Test async gen re-raises when retries exhausted."""
        client = _AsyncStreamClient(max_retries=1)
        with pytest.raises(ConnectionError, match="always fails"):
            _ = [chunk async for chunk in await client.stream_always_fail()]

    @pytest.mark.asyncio
    async def test_non_retryable_propagates_immediately(self) -> None:
        """Test non-retryable exception in async gen propagates without retry."""
        client = _AsyncStreamClient(max_retries=3)
        with pytest.raises(ValueError, match="permanent"):
            _ = [chunk async for chunk in await client.stream_permanent_error()]

    @pytest.mark.asyncio
    async def test_zero_retries_raises_on_failure(self) -> None:
        """Test max_retries=0 raises on first async failure."""
        client = _AsyncStreamClient(max_retries=0)
        with pytest.raises(ConnectionError, match="always fails"):
            _ = [chunk async for chunk in await client.stream_always_fail()]

    @pytest.mark.asyncio
    async def test_preserves_function_name(self) -> None:
        """Test functools.wraps preserves async gen method name."""
        assert _AsyncStreamClient.stream.__name__ == "stream", (
            f"Expected 'stream', got {_AsyncStreamClient.stream.__name__}"
        )

    @pytest.mark.asyncio
    async def test_wrapper_is_coroutine_function(self) -> None:
        """Test wrapped async gen method is a coroutine function."""
        assert inspect.iscoroutinefunction(_AsyncStreamClient.stream), (
            "Wrapped method should be a coroutine function"
        )
