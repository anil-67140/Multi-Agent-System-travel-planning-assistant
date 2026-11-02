"""Compose bulkhead + retry + circuit breaker + timeout into one call policy.

Execution order (outermost first)::

    bulkhead -> retry -> circuit breaker -> timeout -> operation

* The bulkhead bounds *total* concurrency (one slot held across all retries).
* Each retry attempt passes through the breaker, so repeated failures trip it.
* The timeout bounds a *single* attempt; total time is bounded by
  ``attempts x timeout + backoff`` which callers can reason about.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

from tripmate.resilience.bulkhead import Bulkhead, BulkheadFullError
from tripmate.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from tripmate.resilience.retry import (
    OperationTimeout,
    RetryPolicy,
    default_is_retryable,
    retry_async,
    with_timeout,
)

T = TypeVar("T")

# (event_name, policy_name, fields) -> None. Used to feed Prometheus without
# making this module depend on it.
EventSink = Callable[[str, str, Mapping[str, Any]], None]


def _noop_sink(event: str, name: str, fields: Mapping[str, Any]) -> None:
    return None


@dataclass
class ResiliencePolicy:
    name: str
    timeout: float | None = 10.0
    retry: RetryPolicy | None = None
    breaker: CircuitBreaker | None = None
    bulkhead: Bulkhead | None = None
    sink: EventSink = _noop_sink

    async def execute(
        self,
        fn: Callable[[], Awaitable[T]],
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> T:
        async def attempt() -> T:
            async def timed() -> T:
                try:
                    return await with_timeout(fn, self.timeout)
                except OperationTimeout:
                    self.sink("timeout", self.name, {})
                    raise

            if self.breaker is not None:
                try:
                    return await self.breaker.call(timed)
                except CircuitOpenError:
                    self.sink("breaker_rejected", self.name, {})
                    raise
            return await timed()

        async def with_retry() -> T:
            if self.retry is None:
                return await attempt()

            def on_retry(n: int, exc: BaseException, delay: float) -> None:
                self.sink("retry", self.name, {"attempt": n, "error": type(exc).__name__})

            def retryable(exc: BaseException) -> bool:
                if isinstance(exc, CircuitOpenError | BulkheadFullError):
                    return False
                return default_is_retryable(exc)

            return await retry_async(
                attempt, self.retry, is_retryable=retryable, on_retry=on_retry, sleep=sleep, rng=rng
            )

        if self.bulkhead is not None:
            try:
                return await self.bulkhead.run(with_retry)
            except BulkheadFullError:
                self.sink("shed", self.name, {})
                raise
        return await with_retry()
