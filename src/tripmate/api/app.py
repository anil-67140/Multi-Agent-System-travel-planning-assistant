"""FastAPI application factory."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tripmate import __version__
from tripmate.api.errors import install_error_handlers
from tripmate.api.middleware import (
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from tripmate.api.routes import auth, health, plans
from tripmate.config import Settings, get_settings
from tripmate.container import Container, build_container
from tripmate.logging_config import configure_logging


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    """``container`` can be injected (tests); otherwise one is built in the lifespan."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if container is not None:
            app.state.container = container
            yield
            return
        async with build_container(settings) as built:
            app.state.container = built
            yield

    app = FastAPI(
        title="TripMate API",
        version=__version__,
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(plans.router)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "PATCH"],
            allow_headers=["Authorization", "Content-Type", "Idempotency-Key", "Last-Event-ID"],
            expose_headers=["X-Request-ID", "Retry-After"],
            max_age=600,
        )
    # Added last = outermost: request id wraps everything, then headers, then the size cap.
    app.add_middleware(BodySizeLimitMiddleware, max_bytes=settings.max_body_bytes)
    app.add_middleware(SecurityHeadersMiddleware, hsts=settings.is_production)

    def on_request(method: str, route: str, status: int, seconds: float) -> None:
        c = getattr(app.state, "container", None)
        if c is not None:
            c.metrics.observe_http(method, route, status, seconds)

    app.add_middleware(RequestContextMiddleware, on_request=on_request)

    # Optional UI mounted into the same process (separate deployment is also supported).
    try:
        from tripmate.ui.gradio_app import mount_ui

        mount_ui(app, api_base_url=None)
    except ImportError:  # gradio not installed: API-only
        pass
    return app
