"""Rate limiting: an exact in-memory sliding window and a Redis fixed window."""

from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int  # seconds until a request would be allowed (0 if allowed)
    reset_after: int  # seconds until the window fully resets


class RateLimiter(Protocol):
    async def hit(self, key: str, limit: int, window_seconds: int) -> RateLimitDecision: ...


class InMemorySlidingWindowLimiter:
    """Exact sliding-window-log limiter. Per-process: use Redis for multiple replicas."""

    def __init__(
        self, clock: Callable[[], float] = time.monotonic, max_keys: int = 100_000
    ) -> None:
        self._clock = clock
        self._max_keys = max_keys
        self._hits: dict[str, deque[float]] = {}

    async def hit(self, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        now = self._clock()
        window_start = now - window_seconds
        log = self._hits.setdefault(key, deque())
        while log and log[0] <= window_start:
            log.popleft()

        if len(log) >= limit:
            retry_after = max(1, math.ceil(log[0] + window_seconds - now))
            return RateLimitDecision(False, limit, 0, retry_after, retry_after)

        log.append(now)
        self._evict_if_needed()
        reset_after = max(1, math.ceil(log[0] + window_seconds - now))
        return RateLimitDecision(True, limit, limit - len(log), 0, reset_after)

    def _evict_if_needed(self) -> None:
        if len(self._hits) <= self._max_keys:
            return
        # Drop empty/old logs first; as a last resort drop the oldest key.
        for k in [k for k, v in self._hits.items() if not v][: self._max_keys // 10 or 1]:
            self._hits.pop(k, None)
        while len(self._hits) > self._max_keys:
            self._hits.pop(next(iter(self._hits)))


# Atomic INCR + EXPIRE-on-first-hit, returning {count, ttl}.
_REDIS_LUA = """
local c = redis.call('INCR', KEYS[1])
if c == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
local ttl = redis.call('TTL', KEYS[1])
if ttl < 0 then redis.call('EXPIRE', KEYS[1], ARGV[1]); ttl = tonumber(ARGV[1]) end
return {c, ttl}
"""


class RedisFixedWindowLimiter:
    """Shared fixed-window limiter (cheap, atomic, good enough for abuse control)."""

    def __init__(self, client: Any, prefix: str = "tripmate:rl:") -> None:
        self._r = client
        self._prefix = prefix

    async def hit(self, key: str, limit: int, window_seconds: int) -> RateLimitDecision:
        count, ttl = await self._r.eval(
            _REDIS_LUA, 1, f"{self._prefix}{key}:{window_seconds}", window_seconds
        )
        count, ttl = int(count), max(1, int(ttl))
        if count > limit:
            return RateLimitDecision(False, limit, 0, ttl, ttl)
        return RateLimitDecision(True, limit, limit - count, 0, ttl)
