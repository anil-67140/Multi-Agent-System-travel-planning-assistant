"""HTTP-level tests. Need fastapi/httpx/prometheus_client/pydantic-settings (always installed in CI)."""

from __future__ import annotations

import pytest

from tests.helpers import NOW, require_dep

for _m in ("fastapi", "httpx", "prometheus_client", "pydantic_settings"):
    require_dep(_m)

import httpx  # noqa: E402

from tests.services.test_plan_service import Rig  # noqa: E402
from tripmate.api.app import create_app  # noqa: E402
from tripmate.config import Settings  # noqa: E402
from tripmate.container import Container  # noqa: E402
from tripmate.observability.metrics import Metrics  # noqa: E402
from tripmate.resilience import InMemorySlidingWindowLimiter  # noqa: E402

pytestmark = pytest.mark.anyio
PW = "a-good-passphrase-42"


class App:
    def __init__(self, **settings):
        self.rig = Rig()
        self.settings = Settings(_env_file=None, jwt_secret="j" * 40, groq_api_key="k", **settings)

        class Q:
            pending = 0

            async def enqueue(_, n, a):
                await self.rig._enqueue(n, a)

        self.container = Container(
            settings=self.settings,
            repo=self.rig.repo,
            service=self.rig.svc,
            limiter=InMemorySlidingWindowLimiter(),
            metrics=Metrics(),
            clock=self.rig.clock,
            queue=Q(),
        )
        self.app = create_app(self.settings, self.container)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://t"
        )

    async def register(self, email="u@example.com"):
        self.app.state.container = self.container  # lifespan is not run by ASGITransport
        r = await self.client.post("/v1/auth/register", json={"email": email, "password": PW})
        assert r.status_code == 201, r.text
        return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
async def t():
    a = App()
    a.app.state.container = a.container
    yield a
    await a.client.aclose()


class TestAuth:
    async def test_register_login_me(self, t):
        h = await t.register()
        assert (await t.client.get("/v1/auth/me", headers=h)).json()["email"] == "u@example.com"
        r = await t.client.post("/v1/auth/login", json={"email": "U@example.com", "password": PW})
        assert r.status_code == 200 and r.json()["token_type"] == "bearer"

    async def test_weak_password_and_duplicate(self, t):
        assert (
            await t.client.post("/v1/auth/register", json={"email": "a@b.co", "password": "short"})
        ).status_code == 422
        await t.register("d@example.com")
        assert (
            await t.client.post(
                "/v1/auth/register", json={"email": "d@example.com", "password": PW}
            )
        ).status_code == 409

    async def test_wrong_password_and_unknown_user_look_the_same(self, t):
        await t.register()
        a = await t.client.post(
            "/v1/auth/login", json={"email": "u@example.com", "password": "wrong-password-1"}
        )
        b = await t.client.post(
            "/v1/auth/login", json={"email": "nobody@example.com", "password": "wrong-password-1"}
        )
        assert a.status_code == b.status_code == 401 and a.json()["detail"] == b.json()["detail"]

    async def test_account_locks_after_repeated_failures(self, t):
        await t.register()
        for _ in range(5):
            await t.client.post(
                "/v1/auth/login", json={"email": "u@example.com", "password": "wrong-password-1"}
            )
        r = await t.client.post("/v1/auth/login", json={"email": "u@example.com", "password": PW})
        assert r.status_code == 401  # locked even with the right password

    async def test_protected_routes_need_a_valid_token(self, t):
        for headers in ({}, {"Authorization": "Bearer junk"}, {"Authorization": "Basic abc"}):
            assert (await t.client.get("/v1/plans", headers=headers)).status_code == 401

    async def test_login_is_rate_limited(self, t):
        codes = [
            (
                await t.client.post(
                    "/v1/auth/login",
                    json={"email": "x@example.com", "password": "wrong-password-1"},
                )
            ).status_code
            for _ in range(12)
        ]
        assert 429 in codes


class TestPlansFlow:
    async def test_full_flow_create_poll_approve(self, t):
        h = await t.register()
        r = await t.client.post(
            "/v1/plans", headers=h, json={"message": "5 days in Tokyo from Karachi"}
        )
        assert r.status_code == 202
        pid = r.json()["id"]
        await t.rig.work()
        d = (await t.client.get(f"/v1/plans/{pid}", headers=h)).json()
        assert (
            d["status"] == "awaiting_approval" and d["itinerary"]["version"] == 1 and d["bundles"]
        )
        ev = (await t.client.get(f"/v1/plans/{pid}/events", headers=h)).json()
        assert ev and ev[0]["seq"] == 1
        ok = await t.client.post(
            f"/v1/plans/{pid}/decision",
            headers={**h, "Idempotency-Key": "key-0000001"},
            json={"itinerary_version": 1, "approved": True},
        )
        assert ok.status_code == 202
        await t.rig.work()
        assert (await t.client.get(f"/v1/plans/{pid}", headers=h)).json()["status"] == "completed"

    async def test_decision_requires_idempotency_key(self, t):
        h = await t.register()
        pid = (await t.client.post("/v1/plans", headers=h, json={"message": "Tokyo trip"})).json()[
            "id"
        ]
        r = await t.client.post(
            f"/v1/plans/{pid}/decision", headers=h, json={"itinerary_version": 1, "approved": True}
        )
        assert r.status_code == 422
        r = await t.client.post(
            f"/v1/plans/{pid}/decision",
            headers={**h, "Idempotency-Key": "bad key!"},
            json={"itinerary_version": 1, "approved": True},
        )
        assert r.status_code == 422

    async def test_other_users_cannot_see_or_decide(self, t):
        a = await t.register("a@example.com")
        b = await t.register("b@example.com")
        pid = (await t.client.post("/v1/plans", headers=a, json={"message": "Tokyo trip"})).json()[
            "id"
        ]
        await t.rig.work()
        assert (await t.client.get(f"/v1/plans/{pid}", headers=b)).status_code == 404
        r = await t.client.post(
            f"/v1/plans/{pid}/decision",
            headers={**b, "Idempotency-Key": "key-0000002"},
            json={"itinerary_version": 1, "approved": True},
        )
        assert r.status_code == 404
        assert (await t.client.get(f"/v1/plans/{pid}/stream", headers=b)).status_code == 404
        assert (await t.client.get("/v1/plans", headers=b)).json()["items"] == []

    async def test_validation_errors_are_problem_json_without_echoing_input(self, t):
        h = await t.register()
        r = await t.client.post(
            "/v1/plans", headers=h, json={"message": "x", "evil": "SECRET-VALUE"}
        )
        assert r.status_code == 422 and r.headers["content-type"].startswith(
            "application/problem+json"
        )
        assert "SECRET-VALUE" not in r.text and r.json()["code"] == "validation_error"

    async def test_unknown_route_and_method_are_problem_json(self, t):
        r = await t.client.get("/v1/nope")
        assert r.status_code == 404 and r.json()["code"] == "http_404"

    async def test_edit_constraints_and_alternative(self, t):
        h = await t.register()
        pid = (await t.client.post("/v1/plans", headers=h, json={"message": "Tokyo trip"})).json()[
            "id"
        ]
        await t.rig.work()
        r = await t.client.patch(f"/v1/plans/{pid}/constraints", headers=h, json={"travelers": 3})
        assert r.status_code == 202
        assert (
            await t.client.patch(f"/v1/plans/{pid}/constraints", headers=h, json={})
        ).status_code == 422
        assert (
            await t.client.patch(
                f"/v1/plans/{pid}/constraints", headers=h, json={"destination": "Paris"}
            )
        ).status_code == 422

    async def test_rate_limit_on_plan_creation(self):
        a = App(plan_rate_limit_per_hour=2)
        h = await a.register()
        codes = []
        for _ in range(4):
            r = await a.client.post("/v1/plans", headers=h, json={"message": "Tokyo trip"})
            codes.append(r.status_code)
            await a.rig.work()
        assert codes[:2] == [202, 202] and codes[-1] == 429
        assert int(r.headers["retry-after"]) >= 1
        await a.client.aclose()

    async def test_oversized_body_rejected(self):
        a = App(max_body_bytes=2048)
        h = await a.register()
        r = await a.client.post("/v1/plans", headers=h, content=b"x" * 5000)
        assert r.status_code == 413
        await a.client.aclose()


class TestOps:
    async def test_health_and_ready(self, t):
        assert (await t.client.get("/healthz")).json() == {"status": "ok"}
        r = await t.client.get("/readyz")
        assert r.status_code == 200 and r.json()["checks"]["database"] == "ok"

    async def test_security_headers_and_request_id_present(self, t):
        r = await t.client.get("/healthz", headers={"X-Request-ID": "client-req-12345"})
        assert (
            r.headers["x-request-id"] == "client-req-12345"
            and r.headers["x-content-type-options"] == "nosniff"
        )

    async def test_metrics_endpoint_exposes_counters(self, t):
        await t.client.get("/healthz")
        body = (await t.client.get("/metrics")).text
        assert "tripmate_http_requests_total" in body

    async def test_metrics_require_token_when_configured(self):
        a = App(metrics_token="m" * 20)
        assert (await a.client.get("/metrics")).status_code == 401
        assert (
            await a.client.get("/metrics", headers={"Authorization": "Bearer " + "m" * 20})
        ).status_code == 200
        await a.client.aclose()

    async def test_docs_hidden_in_production(self):
        s = Settings(
            _env_file=None,
            env="production",
            jwt_secret="j" * 40,
            database_url="postgresql+psycopg://u:p@h/d",
            checkpoint_postgres_url="postgresql://u:p@h/d",
            groq_api_key="k",
        )
        app = create_app(s, App().container)
        assert app.docs_url is None and app.openapi_url is None


_ = NOW
