"""A small async circuit breaker.

CLOSED  -> normal operation; consecutive failures are counted.
OPEN    -> calls are rejected immediately (fail fast) for ``recovery_timeout``.
HALF_OPEN -> a limited number of trial calls decide whether to close again.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TypeVar

T = TypeVar("T")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    retryable = False  # retrying immediately would just hit the same open breaker

    def __init__(self, name: str, retry_after: float) -> None:
        super().__init__(f"circuit '{name}' is open; retry in {retry_after:.0f}s")
        self.name = name
        self.retry_after = retry_after


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
        clock: Callable[[], float] = time.monotonic,
        is_failure: Callable[[BaseException], bool] = lambda exc: True,
        on_state_change: Callable[[str, CircuitState], None] | None = None,
    ) -> None:
        if failure_threshold < 1 or half_open_max_calls < 1:
            raise ValueError("thresholds must be >= 1")
        self.name = name
        self._threshold = failure_threshold
        self._recovery = recovery_timeout
        self._half_open_max = half_open_max_calls
        self._clock = clock
        self._is_failure = is_failure
        self._on_state_change = on_state_change
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_inflight = 0

    @property
    def state(self) -> CircuitState:
        if self._state is CircuitState.OPEN and self._clock() - self._opened_at >= self._recovery:
            self._set_state(CircuitState.HALF_OPEN)
            self._half_open_inflight = 0
        return self._state

    def _set_state(self, state: CircuitState) -> None:
        if state is not self._state:
            self._state = state
            if self._on_state_change:
                self._on_state_change(self.name, state)

    def _admit(self) -> None:
        state = self.state
        if state is CircuitState.OPEN:
            remaining = self._recovery - (self._clock() - self._opened_at)
            raise CircuitOpenError(self.name, max(0.0, remaining))
        if state is CircuitState.HALF_OPEN:
            if self._half_open_inflight >= self._half_open_max:
                raise CircuitOpenError(self.name, self._recovery)
            self._half_open_inflight += 1

    def _record_success(self) -> None:
        self._failures = 0
        if self._state is CircuitState.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
        self._set_state(CircuitState.CLOSED)

    def _record_failure(self) -> None:
        if self._state is CircuitState.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)
            self._trip()
            return
        self._failures += 1
        if self._failures >= self._threshold:
            self._trip()

    def _trip(self) -> None:
        self._opened_at = self._clock()
        self._failures = 0
        self._set_state(CircuitState.OPEN)

    def _release(self) -> None:
        if self._state is CircuitState.HALF_OPEN:
            self._half_open_inflight = max(0, self._half_open_inflight - 1)

    async def call(self, fn: Callable[[], Awaitable[T]]) -> T:
        self._admit()
        try:
            result = await fn()
        except asyncio.CancelledError:
            self._release()
            raise
        except BaseException as exc:
            if self._is_failure(exc):
                self._record_failure()
            else:
                self._release()  # e.g. a 4xx: the dependency is healthy
            raise
        self._record_success()
        return result
