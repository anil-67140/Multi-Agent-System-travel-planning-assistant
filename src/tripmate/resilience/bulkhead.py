"""Concurrency limiting with load shedding (a 'bulkhead')."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


class BulkheadFullError(Exception):
    retryable = False

    def __init__(self, name: str) -> None:
        super().__init__(f"bulkhead '{name}' is full; request shed")
        self.name = name


class Bulkhead:
    """At most ``max_concurrent`` calls run; at most ``max_waiting`` may queue.

    Anything beyond that is rejected immediately rather than piling up, which
    protects a free-tier LLM quota (and our own memory) under bursts.
    """

    def __init__(
        self,
        name: str,
        max_concurrent: int,
        *,
        max_waiting: int | None = None,
        acquire_timeout: float | None = None,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.name = name
        self.max_concurrent = max_concurrent
        self._max_waiting = max_waiting
        self._acquire_timeout = acquire_timeout
        self._sem = asyncio.Semaphore(max_concurrent)
        self._waiting = 0
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiting(self) -> int:
        return self._waiting

    async def _acquire(self) -> None:
        if not self._sem.locked():
            await self._sem.acquire()
            return
        if self._max_waiting is not None and self._waiting >= self._max_waiting:
            raise BulkheadFullError(self.name)
        self._waiting += 1
        try:
            if self._acquire_timeout is None:
                await self._sem.acquire()
            else:
                try:
                    await asyncio.wait_for(self._sem.acquire(), self._acquire_timeout)
                except TimeoutError as exc:
                    raise BulkheadFullError(self.name) from exc
        finally:
            self._waiting -= 1

    async def run(self, fn: Callable[[], Awaitable[T]]) -> T:
        await self._acquire()
        self._in_flight += 1
        try:
            return await fn()
        finally:
            self._in_flight -= 1
            self._sem.release()
