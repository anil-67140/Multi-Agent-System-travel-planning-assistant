"""FastAPI dependencies: container access, authentication, rate limiting, idempotency keys."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Annotated

from fastapi import Depends, Header, Request

from tripmate.container import Container
from tripmate.errors import RateLimitedError, UnauthorizedError, ValidationFailure
from tripmate.security.tokens import decode_token
from tripmate.services.records import UserRecord

IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9._:\-]{8,80}$")


def get_container(request: Request) -> Container:
    return request.app.state.container  # type: ignore[no-any-return]


ContainerDep = Annotated[Container, Depends(get_container)]


def client_ip(request: Request) -> str:
    # With ``uvicorn --proxy-headers --forwarded-allow-ips`` set to the trusted proxy,
    # request.client already reflects X-Forwarded-For. We never parse the header ourselves.
    return request.client.host if request.client else "unknown"


async def current_user(request: Request, c: ContainerDep) -> UserRecord:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise UnauthorizedError("Authentication is required.")
    claims = decode_token(token, c.settings.jwt_secret.get_secret_value(), c.clock.now())
    user = await c.repo.get_user(claims.user_id)
    if user is None or not user.is_active:
        raise UnauthorizedError("Invalid authentication token.")
    request.state.user_id = user.id
    return user


CurrentUser = Annotated[UserRecord, Depends(current_user)]


async def _enforce(
    request: Request, c: Container, scope: str, limit: int, window: int, subject: str | None
) -> None:
    key = f"{scope}:{subject or client_ip(request)}"
    decision = await c.limiter.hit(key, limit, window)
    if not decision.allowed:
        c.metrics.rate_limited.labels(scope).inc()
        raise RateLimitedError(
            "Too many requests. Please slow down.", retry_after=decision.retry_after
        )


def rate_limit(
    scope: str, *, per: int | Callable[[Container], int], window: int, by_user: bool = False
) -> Callable[..., Awaitable[None]]:
    """Dependency factory. ``by_user`` keys the limit on the authenticated user (and so also
    enforces authentication *before* any work); otherwise on the client IP."""

    def limit_for(c: Container) -> int:
        return per(c) if callable(per) else per

    if by_user:

        async def per_user(request: Request, c: ContainerDep, user: CurrentUser) -> None:
            await _enforce(request, c, scope, limit_for(c), window, user.id)

        return per_user

    async def per_ip(request: Request, c: ContainerDep) -> None:
        await _enforce(request, c, scope, limit_for(c), window, None)

    return per_ip


def idempotency_key(key: Annotated[str | None, Header(alias="Idempotency-Key")] = None) -> str:
    if key is None:
        raise ValidationFailure("The Idempotency-Key header is required for this request.")
    if not IDEMPOTENCY_RE.match(key):
        raise ValidationFailure("Idempotency-Key must be 8-80 characters: letters, digits, . _ : -")
    return key


IdempotencyKey = Annotated[str, Depends(idempotency_key)]
LOCKOUT_THRESHOLD = 5
LOCKOUT = timedelta(minutes=15)
