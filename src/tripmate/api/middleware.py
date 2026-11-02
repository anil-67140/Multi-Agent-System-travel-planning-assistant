"""Pure-ASGI middleware (no framework dependency, so it is unit-tested directly).

* RequestContextMiddleware: request id in/out, log correlation, access log, latency.
* SecurityHeadersMiddleware: conservative headers on every response.
* BodySizeLimitMiddleware: rejects oversized bodies early (declared or streamed).
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from tripmate.logging_config import request_id_var

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

log = logging.getLogger("tripmate.access")
_SAFE_ID = re.compile(r"^[A-Za-z0-9._\-]{8,64}$")


def _header(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers", []):
        if key.lower() == name:
            return value.decode("latin-1")
    return None


class RequestContextMiddleware:
    def __init__(
        self, app: ASGIApp, on_request: Callable[[str, str, int, float], None] | None = None
    ) -> None:
        self.app, self._on_request = app, on_request

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        incoming = _header(scope, b"x-request-id")
        # Only accept well-formed ids: a client-chosen value must not be able to forge log lines.
        request_id = incoming if incoming and _SAFE_ID.match(incoming) else uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status = 500

        async def send_wrapper(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], (b"x-request-id", request_id.encode())]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed = time.perf_counter() - started
            route = scope.get("route")
            path_template = getattr(route, "path", None) or "unmatched"  # low-cardinality label
            log.info(
                "request",
                extra={
                    "method": scope["method"],
                    "path": scope["path"],
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 1),
                },
            )
            if self._on_request:
                self._on_request(scope["method"], path_template, status, elapsed)
            request_id_var.reset(token)


SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"permissions-policy", b"geolocation=(), microphone=(), camera=()"),
    (b"cross-origin-opener-policy", b"same-origin"),
    (b"cache-control", b"no-store"),
]


class SecurityHeadersMiddleware:
    def __init__(self, app: ASGIApp, hsts: bool = False) -> None:
        self.app = app
        self._extra = (
            [(b"strict-transport-security", b"max-age=31536000; includeSubDomains")] if hsts else []
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                present = {k.lower() for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in [*SECURITY_HEADERS, *self._extra] if k not in present]
                message["headers"] = [*message.get("headers", []), *extra]
            await send(message)

        await self.app(scope, receive, send_wrapper)


class BodySizeLimitMiddleware:
    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = _header(scope, b"content-length")
        if declared is not None:
            try:
                too_big = int(declared) > self.max_bytes
            except ValueError:
                too_big = True
            if too_big:
                await self._reject(scope, send)
                return
        seen = 0
        responded = False

        async def limited_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:  # chunked upload without content-length
                    raise _TooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal responded
            responded = responded or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _TooLarge:
            if not responded:
                await self._reject(scope, send)

    async def _reject(self, scope: Scope, send: Send) -> None:
        body = json.dumps(
            {
                "type": "about:blank",
                "title": "Payload Too Large",
                "status": 413,
                "code": "payload_too_large",
                "detail": f"Request body must be at most {self.max_bytes} bytes.",
                "request_id": scope.get("state", {}).get("request_id"),
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/problem+json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class _TooLarge(Exception):
    pass
