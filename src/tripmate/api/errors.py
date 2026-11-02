"""Uniform RFC 9457 problem+json errors. Internals never reach the client."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from tripmate.errors import RateLimitedError, TripMateError

log = logging.getLogger(__name__)
MEDIA = "application/problem+json"


def problem(
    request: Request,
    status: int,
    title: str,
    code: str,
    detail: str,
    headers: dict[str, str] | None = None,
    extra: dict[str, Any] | None = None,
) -> JSONResponse:
    body = {
        "type": "about:blank",
        "title": title,
        "status": status,
        "code": code,
        "detail": detail,
        "request_id": getattr(request.state, "request_id", None),
        **(extra or {}),
    }
    return JSONResponse(body, status_code=status, media_type=MEDIA, headers=headers)


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(TripMateError)
    async def _domain(request: Request, exc: TripMateError) -> JSONResponse:
        headers = (
            {"Retry-After": str(exc.retry_after)}
            if isinstance(exc, RateLimitedError) and getattr(exc, "retry_after", None)
            else None
        )
        if exc.http_status >= 500:
            log.error("domain error", extra={"code": exc.code, "detail": exc.detail})
        return problem(
            request, exc.http_status, exc.__class__.__name__, exc.code, exc.message, headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Report *where* the problem is, never echo submitted values back (they may be secrets).
        errors = [
            {"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
            for e in exc.errors()
        ][:20]
        return problem(
            request,
            422,
            "Validation failed",
            "validation_error",
            "The request is invalid.",
            extra={"errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return problem(
            request,
            exc.status_code,
            "HTTP error",
            f"http_{exc.status_code}",
            str(exc.detail),
            dict(exc.headers) if exc.headers else None,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return problem(
            request,
            500,
            "Internal Server Error",
            "internal_error",
            "Something went wrong on our side. Please try again.",
        )
