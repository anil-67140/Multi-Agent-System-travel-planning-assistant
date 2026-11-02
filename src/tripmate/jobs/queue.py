"""Queue backends behind one ``enqueue(name, arg)`` signature.

* ``InProcessQueue``: asyncio workers inside the API process. Zero infrastructure; jobs are lost on
  restart, which is acceptable because ``recover_stale_runs`` fails orphaned plans and users can retry.
* ``ArqQueue``: Redis-backed, survives restarts, scales by adding worker containers.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from tripmate.errors import UpstreamUnavailableError

log = logging.getLogger(__name__)
Handler = Callable[[str, str], Awaitable[None]]


class InProcessQueue:
    def __init__(self, handler: Handler, *, workers: int = 4, max_pending: int = 100) -> None:
        self._handler = handler
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=max_pending)
        self._workers = workers
        self._tasks: list[asyncio.Task[None]] = []

    async def start(self) -> None:
        self._tasks = [
            asyncio.create_task(self._worker(i), name=f"job-worker-{i}")
            for i in range(self._workers)
        ]

    async def enqueue(self, name: str, arg: str) -> None:
        try:
            self._queue.put_nowait((name, arg))
        except (
            asyncio.QueueFull
        ) as exc:  # backpressure: say "busy" instead of growing without bound
            raise UpstreamUnavailableError(
                "The service is busy. Please try again shortly."
            ) from exc

    async def _worker(self, index: int) -> None:
        while True:
            name, arg = await self._queue.get()
            try:
                await self._handler(name, arg)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("job failed", extra={"job": name, "worker": index})
            finally:
                self._queue.task_done()

    async def drain(self) -> None:
        await self._queue.join()

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    @property
    def pending(self) -> int:
        return self._queue.qsize()


class ArqQueue:
    """Producer side (API). The consumer is ``tripmate.jobs.worker``."""

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._pool: Any = None

    async def start(self) -> None:
        from arq import create_pool
        from arq.connections import RedisSettings

        self._pool = await create_pool(RedisSettings.from_dsn(self._redis_url))

    async def enqueue(self, name: str, arg: str) -> None:
        # _job_id makes enqueue idempotent: the same plan/job is never queued twice at once.
        job = await self._pool.enqueue_job("dispatch_job", name, arg, _job_id=f"{name}:{arg}")
        if job is None:
            log.info("job already queued", extra={"job": name})

    async def stop(self) -> None:
        if self._pool is not None:
            await self._pool.aclose()
