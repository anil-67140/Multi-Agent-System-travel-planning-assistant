"""TTL caching with single-flight stampede protection.

Two backends share one tiny protocol: an in-process LRU+TTL cache (default,
zero infrastructure) and Redis (shared across API replicas and workers).
Cached values must be JSON-serialisable so both backends behave identically.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class Cache(Protocol):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: float) -> None: ...
    async def delete(self, key: str) -> None: ...


class InMemoryTTLCache:
    def __init__(self, max_items: int = 2048, clock: Callable[[], float] = time.monotonic) -> None:
        self._max = max_items
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    async def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at <= self._clock():
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return value

    async def set(self, key: str, value: Any, ttl: float) -> None:
        if ttl <= 0:
            return
        self._data[key] = (self._clock() + ttl, value)
        self._data.move_to_end(key)
        while len(self._data) > self._max:
            self._data.popitem(last=False)  # evict least-recently-used

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def __len__(self) -> int:
        return len(self._data)


class RedisCache:
    """JSON values in Redis with per-key TTL. ``client`` is a ``redis.asyncio.Redis``."""

    def __init__(self, client: Any, prefix: str = "tripmate:cache:") -> None:
        self._r = client
        self._prefix = prefix

    async def get(self, key: str) -> Any | None:
        raw = await self._r.get(self._prefix + key)
        return None if raw is None else json.loads(raw)

    async def set(self, key: str, value: Any, ttl: float) -> None:
        if ttl <= 0:
            return
        await self._r.set(self._prefix + key, json.dumps(value), ex=max(1, int(ttl)))

    async def delete(self, key: str) -> None:
        await self._r.delete(self._prefix + key)


class SingleFlight:
    """Collapse concurrent identical loads into one (prevents cache stampedes)."""

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[Any]] = {}

    async def do(self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        existing = self._inflight.get(key)
        if existing is not None:
            return await asyncio.shield(existing)
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        # Avoid "exception was never retrieved" when nobody else was waiting.
        future.add_done_callback(lambda f: f.cancelled() or f.exception())
        self._inflight[key] = future
        try:
            result = await fn()
        except asyncio.CancelledError:
            future.cancel()
            raise
        except BaseException as exc:
            future.set_exception(exc)
            raise
        else:
            future.set_result(result)
            return result
        finally:
            self._inflight.pop(key, None)


async def get_or_load(
    cache: Cache,
    key: str,
    ttl: float,
    loader: Callable[[], Awaitable[Any]],
    *,
    single_flight: SingleFlight | None = None,
    on_result: Callable[[bool], None] | None = None,
) -> tuple[Any, bool]:
    """Return ``(value, was_cache_hit)``. ``loader`` must return JSON-serialisable data."""
    cached = await cache.get(key)
    if cached is not None:
        if on_result:
            on_result(True)
        return cached, True

    async def load_and_store() -> Any:
        value = await loader()
        await cache.set(key, value, ttl)
        return value

    value = await (single_flight.do(key, load_and_store) if single_flight else load_and_store())
    if on_result:
        on_result(False)
    return value, False
