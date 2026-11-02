from __future__ import annotations

import asyncio

import pytest

from tripmate.resilience import (
    Bulkhead,
    BulkheadFullError,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    InMemorySlidingWindowLimiter,
    InMemoryTTLCache,
    OperationTimeout,
    ResiliencePolicy,
    RetryPolicy,
    SingleFlight,
    get_or_load,
    retry_async,
    with_timeout,
)

pytestmark = pytest.mark.anyio


class Boom(Exception):
    retryable = True


class Fatal(Exception):
    retryable = False


class FakeSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


# --------------------------------------------------------------------------- retry
class TestRetryPolicy:
    def test_exponential_backoff_without_jitter(self):
        p = RetryPolicy(base_delay=1, multiplier=2, max_delay=10, jitter=False)
        assert [p.backoff(n) for n in (1, 2, 3, 4, 5)] == [1, 2, 4, 8, 10]  # capped

    def test_full_jitter_is_between_zero_and_ceiling(self):
        p = RetryPolicy(base_delay=1, multiplier=2, max_delay=10, jitter=True)
        assert p.backoff(3, rng=lambda: 0.0) == 0
        assert p.backoff(3, rng=lambda: 0.5) == 2.0
        assert p.backoff(3, rng=lambda: 1.0) == 4.0

    def test_invalid_policy(self):
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)
        with pytest.raises(ValueError):
            RetryPolicy(multiplier=0.5)


class TestRetry:
    async def test_succeeds_after_transient_failures(self):
        calls = 0
        sleep = FakeSleep()

        async def flaky():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise Boom
            return "ok"

        result = await retry_async(
            flaky, RetryPolicy(max_attempts=3, base_delay=1, jitter=False), sleep=sleep
        )
        assert (result, calls, sleep.delays) == ("ok", 3, [1, 2])

    async def test_is_bounded(self):
        calls = 0

        async def always():
            nonlocal calls
            calls += 1
            raise Boom

        with pytest.raises(Boom):
            await retry_async(always, RetryPolicy(max_attempts=4), sleep=FakeSleep())
        assert calls == 4

    async def test_non_retryable_fails_immediately(self):
        calls = 0

        async def fatal():
            nonlocal calls
            calls += 1
            raise Fatal

        with pytest.raises(Fatal):
            await retry_async(fatal, RetryPolicy(max_attempts=5), sleep=FakeSleep())
        assert calls == 1

    async def test_retry_after_hint_is_honoured_but_capped(self):
        class Limited(Exception):
            retryable = True
            retry_after = 5.0

        sleep = FakeSleep()
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise Limited
            return 1

        await retry_async(fn, RetryPolicy(base_delay=0.1, max_delay=3, jitter=False), sleep=sleep)
        assert sleep.delays == [3]  # server asked 5s, policy caps at 3s

    async def test_on_retry_callback(self):
        seen: list[tuple[int, str]] = []

        async def fn():
            raise Boom

        with pytest.raises(Boom):
            await retry_async(
                fn,
                RetryPolicy(max_attempts=3),
                sleep=FakeSleep(),
                on_retry=lambda n, e, d: seen.append((n, type(e).__name__)),
            )
        assert seen == [(1, "Boom"), (2, "Boom")]

    async def test_cancellation_is_never_swallowed(self):
        async def fn():
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await retry_async(fn, RetryPolicy(max_attempts=3), sleep=FakeSleep())


class TestTimeout:
    async def test_times_out(self):
        async def slow():
            await asyncio.sleep(1)

        with pytest.raises(OperationTimeout):
            await with_timeout(slow, 0.01)

    async def test_none_means_no_deadline_and_fast_calls_pass(self):
        async def fast():
            return 7

        assert await with_timeout(fast, 1) == 7
        assert await with_timeout(fast, None) == 7


# --------------------------------------------------------------------------- breaker
class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class TestCircuitBreaker:
    async def _fail(self, cb):
        async def f():
            raise Boom

        with pytest.raises(Boom):
            await cb.call(f)

    async def test_opens_after_threshold_then_fails_fast(self):
        clock = FakeClock()
        cb = CircuitBreaker("x", failure_threshold=3, recovery_timeout=30, clock=clock)
        for _ in range(3):
            await self._fail(cb)
        assert cb.state is CircuitState.OPEN

        called = False

        async def f():
            nonlocal called
            called = True

        with pytest.raises(CircuitOpenError):
            await cb.call(f)
        assert not called  # the dependency was not touched

    async def test_success_resets_failure_count(self):
        cb = CircuitBreaker("x", failure_threshold=3, clock=FakeClock())
        await self._fail(cb)
        await self._fail(cb)

        async def ok():
            return 1

        await cb.call(ok)
        await self._fail(cb)
        await self._fail(cb)
        assert cb.state is CircuitState.CLOSED  # never reached 3 consecutive

    async def test_half_open_recovers(self):
        clock = FakeClock()
        changes: list[CircuitState] = []
        cb = CircuitBreaker(
            "x",
            failure_threshold=1,
            recovery_timeout=10,
            clock=clock,
            on_state_change=lambda n, s: changes.append(s),
        )
        await self._fail(cb)
        clock.t = 11
        assert cb.state is CircuitState.HALF_OPEN

        async def ok():
            return "pong"

        assert await cb.call(ok) == "pong"
        assert cb.state is CircuitState.CLOSED
        assert changes == [CircuitState.OPEN, CircuitState.HALF_OPEN, CircuitState.CLOSED]

    async def test_half_open_failure_reopens(self):
        clock = FakeClock()
        cb = CircuitBreaker("x", failure_threshold=1, recovery_timeout=10, clock=clock)
        await self._fail(cb)
        clock.t = 11
        await self._fail(cb)
        assert cb.state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            await cb.call(lambda: asyncio.sleep(0))

    async def test_half_open_admits_only_limited_trial_calls(self):
        clock = FakeClock()
        cb = CircuitBreaker(
            "x", failure_threshold=1, recovery_timeout=10, half_open_max_calls=1, clock=clock
        )
        await self._fail(cb)
        clock.t = 11
        gate = asyncio.Event()

        async def slow():
            await gate.wait()
            return 1

        trial = asyncio.create_task(cb.call(slow))
        await asyncio.sleep(0)  # let the trial start
        with pytest.raises(CircuitOpenError):
            await cb.call(slow)
        gate.set()
        assert await trial == 1

    async def test_ignored_failures_do_not_trip(self):
        cb = CircuitBreaker(
            "x",
            failure_threshold=1,
            is_failure=lambda e: not isinstance(e, Fatal),
            clock=FakeClock(),
        )

        async def client_error():
            raise Fatal

        for _ in range(3):
            with pytest.raises(Fatal):
                await cb.call(client_error)
        assert cb.state is CircuitState.CLOSED


# --------------------------------------------------------------------------- bulkhead
class TestBulkhead:
    async def test_limits_concurrency(self):
        bh = Bulkhead("x", max_concurrent=2)
        running = peak = 0

        async def work():
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.01)
            running -= 1

        await asyncio.gather(*(bh.run(work) for _ in range(8)))
        assert peak == 2 and bh.in_flight == 0

    async def test_sheds_load_when_queue_is_full(self):
        bh = Bulkhead("x", max_concurrent=1, max_waiting=1)
        gate = asyncio.Event()

        async def hold():
            await gate.wait()

        t1 = asyncio.create_task(bh.run(hold))
        await asyncio.sleep(0)
        t2 = asyncio.create_task(bh.run(hold))  # waits
        await asyncio.sleep(0)
        assert bh.waiting == 1
        with pytest.raises(BulkheadFullError):
            await bh.run(hold)  # queue full -> shed
        gate.set()
        await asyncio.gather(t1, t2)
        assert bh.waiting == 0 and bh.in_flight == 0

    async def test_acquire_timeout_sheds(self):
        bh = Bulkhead("x", max_concurrent=1, acquire_timeout=0.01)
        gate = asyncio.Event()

        async def hold():
            await gate.wait()

        t1 = asyncio.create_task(bh.run(hold))
        await asyncio.sleep(0)
        with pytest.raises(BulkheadFullError):
            await bh.run(hold)
        gate.set()
        await t1

    async def test_slot_released_on_error(self):
        bh = Bulkhead("x", max_concurrent=1)

        async def bad():
            raise Boom

        with pytest.raises(Boom):
            await bh.run(bad)
        assert await bh.run(lambda: asyncio.sleep(0, "fine")) == "fine"


# --------------------------------------------------------------------------- composed policy
class TestResiliencePolicy:
    async def test_retries_and_reports_events(self):
        events: list[str] = []
        policy = ResiliencePolicy(
            "p",
            timeout=1,
            retry=RetryPolicy(max_attempts=3, jitter=False, base_delay=0.1),
            sink=lambda e, n, f: events.append(e),
        )
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise Boom
            return "done"

        assert await policy.execute(fn, sleep=FakeSleep()) == "done"
        assert events == ["retry", "retry"]

    async def test_timeout_is_per_attempt_and_retried(self):
        events: list[str] = []
        policy = ResiliencePolicy(
            "p",
            timeout=0.01,
            retry=RetryPolicy(max_attempts=2),
            sink=lambda e, n, f: events.append(e),
        )
        calls = 0

        async def slow_then_fast():
            nonlocal calls
            calls += 1
            if calls == 1:
                await asyncio.sleep(1)
            return "ok"

        assert await policy.execute(slow_then_fast, sleep=FakeSleep()) == "ok"
        assert events == ["timeout", "retry"]

    async def test_open_breaker_is_not_retried(self):
        breaker = CircuitBreaker("p", failure_threshold=1, clock=FakeClock())
        policy = ResiliencePolicy(
            "p", timeout=1, retry=RetryPolicy(max_attempts=5), breaker=breaker
        )
        calls = 0

        async def fn():
            nonlocal calls
            calls += 1
            raise Boom

        # attempt 1 fails and trips the breaker; attempt 2 hits the open breaker,
        # and CircuitOpenError is non-retryable, so we fail fast instead of looping.
        with pytest.raises(CircuitOpenError):
            await policy.execute(fn, sleep=FakeSleep())
        assert calls == 1

    async def test_exhausted_retries_surface_the_last_error(self):
        policy = ResiliencePolicy("p", timeout=1, retry=RetryPolicy(max_attempts=2))

        async def fn():
            raise Boom("final")

        with pytest.raises(Boom, match="final"):
            await policy.execute(fn, sleep=FakeSleep())

    async def test_bulkhead_shed_is_reported(self):
        events: list[str] = []
        bh = Bulkhead("p", max_concurrent=1, max_waiting=0)
        policy = ResiliencePolicy(
            "p", timeout=1, bulkhead=bh, sink=lambda e, n, f: events.append(e)
        )
        gate = asyncio.Event()

        async def hold():
            await gate.wait()

        t = asyncio.create_task(policy.execute(hold))
        await asyncio.sleep(0)
        with pytest.raises(BulkheadFullError):
            await policy.execute(hold)
        gate.set()
        await t
        assert events == ["shed"]


# --------------------------------------------------------------------------- cache
class TestCache:
    async def test_ttl_expiry(self):
        clock = FakeClock()
        cache = InMemoryTTLCache(clock=clock)
        await cache.set("k", {"a": 1}, ttl=10)
        assert await cache.get("k") == {"a": 1}
        clock.t = 10
        assert await cache.get("k") is None

    async def test_lru_eviction(self):
        cache = InMemoryTTLCache(max_items=2, clock=FakeClock())
        await cache.set("a", 1, 100)
        await cache.set("b", 2, 100)
        await cache.get("a")  # a becomes most-recent
        await cache.set("c", 3, 100)  # evicts b
        assert [await cache.get(k) for k in "abc"] == [1, None, 3]

    async def test_non_positive_ttl_is_not_stored(self):
        cache = InMemoryTTLCache(clock=FakeClock())
        await cache.set("k", 1, ttl=0)
        assert await cache.get("k") is None

    async def test_get_or_load_hits_after_first_load(self):
        cache = InMemoryTTLCache(clock=FakeClock())
        loads = 0
        results: list[bool] = []

        async def loader():
            nonlocal loads
            loads += 1
            return {"v": loads}

        a, hit_a = await get_or_load(cache, "k", 60, loader, on_result=results.append)
        b, hit_b = await get_or_load(cache, "k", 60, loader, on_result=results.append)
        assert (a, hit_a, b, hit_b, loads, results) == (
            {"v": 1},
            False,
            {"v": 1},
            True,
            1,
            [False, True],
        )

    async def test_single_flight_collapses_concurrent_loads(self):
        cache = InMemoryTTLCache(clock=FakeClock())
        sf = SingleFlight()
        loads = 0

        async def loader():
            nonlocal loads
            loads += 1
            await asyncio.sleep(0.02)
            return "value"

        results = await asyncio.gather(
            *(get_or_load(cache, "k", 60, loader, single_flight=sf) for _ in range(10))
        )
        assert loads == 1 and {r[0] for r in results} == {"value"}

    async def test_single_flight_propagates_errors_to_all_waiters_and_recovers(self):
        sf = SingleFlight()

        async def bad():
            await asyncio.sleep(0.01)
            raise Boom

        outcomes = await asyncio.gather(
            *(sf.do("k", bad) for _ in range(3)), return_exceptions=True
        )
        assert all(isinstance(o, Boom) for o in outcomes)
        assert await sf.do("k", lambda: asyncio.sleep(0, "again")) == "again"  # not stuck


# --------------------------------------------------------------------------- rate limiting
class TestRateLimiter:
    async def test_allows_up_to_limit_then_blocks_with_retry_after(self):
        clock = FakeClock()
        rl = InMemorySlidingWindowLimiter(clock=clock)
        decisions = [await rl.hit("u", 3, 60) for _ in range(4)]
        assert [d.allowed for d in decisions] == [True, True, True, False]
        assert [d.remaining for d in decisions] == [2, 1, 0, 0]
        assert decisions[3].retry_after == 60

    async def test_window_slides(self):
        clock = FakeClock()
        rl = InMemorySlidingWindowLimiter(clock=clock)
        await rl.hit("u", 2, 10)
        clock.t = 6
        await rl.hit("u", 2, 10)
        clock.t = 7
        blocked = await rl.hit("u", 2, 10)
        assert not blocked.allowed and blocked.retry_after == 3  # oldest hit expires at t=10
        clock.t = 10.5
        assert (await rl.hit("u", 2, 10)).allowed  # first hit slid out; second still counts

    async def test_keys_are_independent(self):
        rl = InMemorySlidingWindowLimiter(clock=FakeClock())
        for _ in range(2):
            await rl.hit("a", 2, 60)
        assert not (await rl.hit("a", 2, 60)).allowed
        assert (await rl.hit("b", 2, 60)).allowed

    async def test_memory_is_bounded(self):
        rl = InMemorySlidingWindowLimiter(clock=FakeClock(), max_keys=50)
        for i in range(500):
            await rl.hit(f"k{i}", 1, 60)
        assert len(rl._hits) <= 50
