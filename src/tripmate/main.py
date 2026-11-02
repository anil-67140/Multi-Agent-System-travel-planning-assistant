"""Process entry points: ``tripmate-api`` and ``tripmate-worker``."""

from __future__ import annotations

import asyncio
import os
import sys


def _windows_loop_policy() -> None:
    # psycopg's async mode cannot run on Windows' default ProactorEventLoop.
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]


def run_api() -> None:
    import uvicorn

    _windows_loop_policy()
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "tripmate.api.app:create_app",
        factory=True,
        host=os.environ.get("HOST", "0.0.0.0"),  # noqa: S104
        port=port,
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
        timeout_graceful_shutdown=30,
        access_log=False,  # access log is emitted (redacted) by our middleware
    )


def run_worker() -> None:
    """ARQ worker process for the Redis queue backend."""
    from arq import run_worker as arq_run

    from tripmate.jobs.worker import WorkerSettings

    _windows_loop_policy()
    arq_run(WorkerSettings)  # type: ignore[arg-type]
