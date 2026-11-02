"""ARQ worker: same composition root as the API, but only consumes jobs.

Scale horizontally by running more copies; any worker can handle any job because workflow state
lives in PostgreSQL and handlers are idempotent.
"""

from __future__ import annotations

import contextlib
from typing import Any

from arq.connections import RedisSettings

from tripmate.config import get_settings
from tripmate.container import build_container
from tripmate.jobs.handlers import dispatch
from tripmate.logging_config import configure_logging


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    stack = contextlib.AsyncExitStack()
    ctx["stack"] = stack
    # The worker must not start the in-process consumers: ARQ is the consumer here.
    ctx["container"] = await stack.enter_async_context(
        build_container(settings, with_queue_consumers=False)
    )


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["stack"].aclose()


async def dispatch_job(ctx: dict[str, Any], name: str, arg: str) -> None:
    await dispatch(ctx["container"].service, name, arg)


class WorkerSettings:
    functions = [dispatch_job]
    on_startup = startup
    on_shutdown = shutdown
    max_jobs = get_settings().worker_concurrency
    job_timeout = int(get_settings().run_timeout_seconds) + 30
    max_tries = 2  # handlers are idempotent; one redelivery covers a worker crash
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url or "redis://localhost:6379")
