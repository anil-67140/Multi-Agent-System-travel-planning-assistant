"""Composition root: builds every long-lived object from ``Settings``.

Nothing else in the codebase reads environment variables or constructs infrastructure.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from tripmate.agents.deps import Deps
from tripmate.config import Settings
from tripmate.domain.clock import Clock, SystemClock
from tripmate.llm.litellm_gateway import GatewayConfig, LiteLLMGateway
from tripmate.observability.metrics import Metrics
from tripmate.resilience import InMemorySlidingWindowLimiter, InMemoryTTLCache
from tripmate.resilience.ratelimit import RateLimiter
from tripmate.services.plan_service import PlanService
from tripmate.services.repository import Repository

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    repo: Repository
    service: PlanService
    limiter: RateLimiter
    metrics: Metrics
    clock: Clock
    queue: Any
    closers: list[Any] = field(default_factory=list)


def gateway_config(s: Settings) -> GatewayConfig:
    secret = lambda v: v.get_secret_value() if v else None  # noqa: E731
    return GatewayConfig(
        smart_models=s.llm_smart_models,
        fast_models=s.llm_fast_models,
        groq_api_key=secret(s.groq_api_key),
        gemini_api_key=secret(s.gemini_api_key),
        timeout_seconds=s.llm_timeout_seconds,
        max_concurrency=s.llm_max_concurrency,
        proxy_url=s.litellm_proxy_url,
        proxy_key=secret(s.litellm_proxy_key),
    )


def configure_tracing(s: Settings) -> None:
    """Langfuse via LiteLLM callbacks (every gateway call becomes a generation, grouped by plan_id)."""
    if not s.langfuse_enabled:
        return
    import os

    import litellm

    os.environ["LANGFUSE_PUBLIC_KEY"] = s.langfuse_public_key.get_secret_value()  # type: ignore[union-attr]
    os.environ["LANGFUSE_SECRET_KEY"] = s.langfuse_secret_key.get_secret_value()  # type: ignore[union-attr]
    if s.langfuse_host:
        os.environ["LANGFUSE_HOST"] = s.langfuse_host
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]
    log.info("langfuse tracing enabled")


def configure_error_reporting(s: Settings) -> None:
    if not s.sentry_dsn:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=s.sentry_dsn.get_secret_value(),
        environment=s.env,
        traces_sample_rate=0.0,
        send_default_pii=False,
    )


@contextlib.asynccontextmanager
async def build_container(
    settings: Settings, *, with_queue_consumers: bool = True
) -> AsyncIterator[Container]:
    from tripmate.db.repositories import SqlRepository
    from tripmate.db.session import create_all_for_dev, make_engine, make_sessionmaker
    from tripmate.graph.builder import build_graph
    from tripmate.graph.checkpointer import open_checkpointer
    from tripmate.jobs.handlers import dispatch
    from tripmate.jobs.queue import ArqQueue, InProcessQueue
    from tripmate.providers.base import AirportDirectory
    from tripmate.providers.flights import AirportsDataDirectory, EstimatedFlightProvider
    from tripmate.providers.fx import ExchangeRateApiClient, FallbackFxProvider, FrankfurterClient
    from tripmate.providers.http import HttpxClient
    from tripmate.providers.open_meteo import OpenMeteoClient
    from tripmate.providers.tavily import TavilySearchClient
    from tripmate.providers.toolbelt import Toolbelt, ToolbeltConfig
    from tripmate.services.workflow import LangGraphWorkflow

    configure_tracing(settings)
    configure_error_reporting(settings)
    clock = SystemClock()
    metrics = Metrics()

    engine = make_engine(settings.database_url)
    if settings.database_url.startswith("sqlite"):
        await create_all_for_dev(engine)
    repo = SqlRepository(make_sessionmaker(engine))

    redis = None
    if settings.redis_url:
        import redis.asyncio as aioredis

        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    if redis is not None:
        from tripmate.resilience import RedisCache, RedisFixedWindowLimiter

        cache: Any = RedisCache(redis)
        limiter: RateLimiter = RedisFixedWindowLimiter(redis)
    else:
        cache, limiter = InMemoryTTLCache(), InMemorySlidingWindowLimiter()

    http = HttpxClient(timeout=settings.tool_timeout_seconds)
    airports: AirportDirectory = AirportsDataDirectory()
    tavily = settings.tavily_api_key.get_secret_value() if settings.tavily_api_key else None
    tools = Toolbelt(
        geocoder=OpenMeteoClient(http, clock),
        weather=OpenMeteoClient(http, clock),
        fx=FallbackFxProvider([ExchangeRateApiClient(http), FrankfurterClient(http)], clock),
        flights=EstimatedFlightProvider(airports, clock),
        web_search=TavilySearchClient(http, tavily) if tavily else None,
        cache=cache,
        clock=clock,
        config=ToolbeltConfig(
            timeout=settings.tool_timeout_seconds,
            max_attempts=settings.tool_max_attempts,
            max_concurrency=settings.tool_max_concurrency,
        ),
        sink=metrics.tool_sink,
    )
    llm = LiteLLMGateway(gateway_config(settings), observer=metrics.observe_llm)

    async with contextlib.AsyncExitStack() as stack:
        saver = await stack.enter_async_context(open_checkpointer(settings.checkpoint_postgres_url))
        holder: dict[str, PlanService] = {}

        async def emit(type: str, node: str | None, message: str, data: dict[str, Any]) -> None:
            await holder["svc"].emit(type, node, message, data)

        deps = Deps(
            llm=llm,
            tools=tools,
            airports=airports,
            clock=clock,
            emit=emit,
            max_revisions=settings.max_revisions,
        )
        graph = build_graph(deps, saver)

        queue: Any
        if settings.queue_backend == "arq" and settings.redis_url:
            queue = ArqQueue(settings.redis_url)
        else:
            queue = InProcessQueue(
                lambda n, a: dispatch(holder["svc"], n, a), workers=settings.worker_concurrency
            )

        service = PlanService(
            repo,
            LangGraphWorkflow(graph),
            clock,
            queue.enqueue,
            run_timeout=settings.run_timeout_seconds,
            stale_after=timedelta(seconds=settings.stale_run_seconds),
            on_status=lambda st: metrics.plans_finished.labels(st).inc(),
        )
        holder["svc"] = service
        if with_queue_consumers or isinstance(queue, ArqQueue):
            await queue.start()
            stack.push_async_callback(queue.stop)
        stack.push_async_callback(http.aclose)
        stack.push_async_callback(engine.dispose)
        if redis is not None:
            stack.push_async_callback(redis.aclose)

        recovered = await service.recover_stale_runs()
        if recovered:
            log.warning("marked interrupted plans as failed", extra={"count": recovered})
        yield Container(
            settings=settings,
            repo=repo,
            service=service,
            limiter=limiter,
            metrics=metrics,
            clock=clock,
            queue=queue,
        )
