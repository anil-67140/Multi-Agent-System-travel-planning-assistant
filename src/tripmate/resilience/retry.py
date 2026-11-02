"""Bounded retries with exponential backoff and full jitter."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


class OperationTimeout(TimeoutError):
    """An operation exceeded its deadline."""


@dataclass(frozen=True)
class RetryPolicy:
    """``max_attempts`` counts the first try: 3 = one try + two retries."""

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    multiplier: float = 2.0
    jitter: bool = True  # "full jitter": sleep U(0, backoff) to avoid thundering herds

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0 or self.multiplier < 1:
            raise ValueError("invalid backoff parameters")

    def backoff(self, attempt: int, rng: Callable[[], float] = random.random) -> float:
        """Delay before retry number ``attempt`` (1-based)."""
        ceiling = min(self.max_delay, self.base_delay * (self.multiplier ** (attempt - 1)))
        return rng() * ceiling if self.jitter else ceiling


def default_is_retryable(exc: BaseException) -> bool:
    """Retry anything that is not explicitly marked non-retryable."""
    if isinstance(exc, asyncio.CancelledError):
        return False
    return bool(getattr(exc, "retryable", True))


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    is_retryable: Callable[[BaseException], bool] = default_is_retryable,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: Callable[[], float] = random.random,
) -> T:
    """Run ``fn`` up to ``policy.max_attempts`` times, re-raising the last error."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if attempt >= policy.max_attempts or not is_retryable(exc):
                raise
            delay = policy.backoff(attempt, rng)
            hint = getattr(exc, "retry_after", None)  # e.g. HTTP 429 Retry-After
            if isinstance(hint, int | float) and hint > 0:
                delay = max(delay, min(float(hint), policy.max_delay))
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            await sleep(delay)


async def with_timeout(fn: Callable[[], Awaitable[T]], seconds: float | None) -> T:
    if seconds is None:
        return await fn()
    try:
        async with asyncio.timeout(seconds):
            return await fn()
    except TimeoutError as exc:
        raise OperationTimeout(f"operation exceeded {seconds:.1f}s") from exc
