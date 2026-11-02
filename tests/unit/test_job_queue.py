from __future__ import annotations

import asyncio

import pytest

from tripmate.errors import UpstreamUnavailableError
from tripmate.jobs.queue import InProcessQueue

pytestmark = pytest.mark.anyio


async def test_jobs_run_and_drain():
    seen = []

    async def handler(name, arg):
        seen.append((name, arg))

    q = InProcessQueue(handler, workers=2)
    await q.start()
    for i in range(5):
        await q.enqueue("run_plan", str(i))
    await q.drain()
    await q.stop()
    assert sorted(a for _, a in seen) == ["0", "1", "2", "3", "4"]


async def test_a_failing_job_does_not_kill_the_worker():
    seen = []

    async def handler(name, arg):
        if arg == "bad":
            raise RuntimeError("boom")
        seen.append(arg)

    q = InProcessQueue(handler, workers=1)
    await q.start()
    for a in ("bad", "ok1", "ok2"):
        await q.enqueue("x", a)
    await q.drain()
    await q.stop()
    assert seen == ["ok1", "ok2"]


async def test_backpressure_when_full():
    gate = asyncio.Event()

    async def handler(name, arg):
        await gate.wait()

    q = InProcessQueue(handler, workers=1, max_pending=1)
    await q.start()
    await q.enqueue("x", "1")
    await asyncio.sleep(0)  # worker takes job 1
    await q.enqueue("x", "2")  # fills the queue
    with pytest.raises(UpstreamUnavailableError):
        await q.enqueue("x", "3")
    gate.set()
    await q.drain()
    await q.stop()


async def test_concurrency_is_bounded_by_worker_count():
    running = peak = 0

    async def handler(name, arg):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1

    q = InProcessQueue(handler, workers=3)
    await q.start()
    for i in range(12):
        await q.enqueue("x", str(i))
    await q.drain()
    await q.stop()
    assert peak == 3


async def test_stop_is_clean_with_idle_workers():
    q = InProcessQueue(lambda n, a: asyncio.sleep(0), workers=2)
    await q.start()
    await q.stop()
