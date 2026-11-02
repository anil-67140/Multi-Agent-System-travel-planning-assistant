from __future__ import annotations

import json

import pytest

from tripmate.api.middleware import (
    BodySizeLimitMiddleware,
    RequestContextMiddleware,
    SecurityHeadersMiddleware,
)
from tripmate.logging_config import request_id_var

pytestmark = pytest.mark.anyio


async def call(app, headers=None, body_chunks=(b"",), method="GET"):
    sent = []
    chunks = list(body_chunks)

    async def receive():
        if chunks:
            c = chunks.pop(0)
            return {"type": "http.request", "body": c, "more_body": bool(chunks)}
        return {"type": "http.disconnect"}

    async def send(m):
        sent.append(m)

    scope = {"type": "http", "method": method, "path": "/x", "headers": headers or []}
    await app(scope, receive, send)
    return scope, sent


async def ok_app(scope, receive, send):
    msg = await receive()
    while msg["type"] == "http.request" and msg.get("more_body"):
        msg = await receive()
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def hdrs(sent):
    return {k: v for k, v in sent[0]["headers"]}


class TestRequestContext:
    async def test_generates_id_and_returns_it(self):
        _, sent = await call(RequestContextMiddleware(ok_app))
        assert len(hdrs(sent)[b"x-request-id"]) == 32

    async def test_accepts_a_well_formed_incoming_id(self):
        _, sent = await call(
            RequestContextMiddleware(ok_app), headers=[(b"x-request-id", b"abc-12345678")]
        )
        assert hdrs(sent)[b"x-request-id"] == b"abc-12345678"

    @pytest.mark.parametrize(
        "bad", [b"short", b"has space in it!!", b"x" * 200, b"new\nline-injection"]
    )
    async def test_rejects_forgeable_ids(self, bad):
        _, sent = await call(RequestContextMiddleware(ok_app), headers=[(b"x-request-id", bad)])
        assert hdrs(sent)[b"x-request-id"] != bad

    async def test_context_var_is_set_during_and_cleared_after(self):
        seen = []

        async def app(scope, receive, send):
            seen.append(request_id_var.get())
            await ok_app(scope, receive, send)

        await call(RequestContextMiddleware(app))
        assert seen[0] and request_id_var.get() is None

    async def test_metrics_callback_gets_status_even_on_crash(self):
        got = []

        async def boom(scope, receive, send):
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            await call(RequestContextMiddleware(boom, on_request=lambda *a: got.append(a)))
        assert got and got[0][2] == 500


class TestSecurityHeaders:
    async def test_adds_headers_without_overriding(self):
        async def app(scope, receive, send):
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"cache-control", b"max-age=5")],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        _, sent = await call(SecurityHeadersMiddleware(app))
        h = hdrs(sent)
        assert h[b"x-content-type-options"] == b"nosniff" and h[b"x-frame-options"] == b"DENY"
        assert h[b"cache-control"] == b"max-age=5" and b"strict-transport-security" not in h

    async def test_hsts_opt_in(self):
        _, sent = await call(SecurityHeadersMiddleware(ok_app, hsts=True))
        assert b"strict-transport-security" in hdrs(sent)


class TestBodyLimit:
    async def test_small_body_passes(self):
        _, sent = await call(
            BodySizeLimitMiddleware(ok_app, 100),
            headers=[(b"content-length", b"10")],
            body_chunks=[b"x" * 10],
        )
        assert sent[0]["status"] == 200

    async def test_declared_oversize_rejected_before_reading(self):
        _, sent = await call(
            BodySizeLimitMiddleware(ok_app, 100), headers=[(b"content-length", b"101")]
        )
        assert (
            sent[0]["status"] == 413 and json.loads(sent[1]["body"])["code"] == "payload_too_large"
        )

    async def test_streamed_oversize_rejected_without_content_length(self):
        _, sent = await call(
            BodySizeLimitMiddleware(ok_app, 100), body_chunks=[b"x" * 60, b"x" * 60]
        )
        assert sent[0]["status"] == 413

    async def test_garbage_content_length_rejected(self):
        _, sent = await call(
            BodySizeLimitMiddleware(ok_app, 100), headers=[(b"content-length", b"abc")]
        )
        assert sent[0]["status"] == 413
