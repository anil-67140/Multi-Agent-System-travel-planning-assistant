from __future__ import annotations

from fastapi import APIRouter, Request, Response

from tripmate.api.deps import ContainerDep
from tripmate.errors import UnauthorizedError

router = APIRouter(tags=["ops"])


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    """Liveness: the process is up. Must not touch dependencies."""
    return {"status": "ok"}


@router.get("/readyz")
async def readyz(c: ContainerDep, response: Response) -> dict[str, object]:
    """Readiness: can we serve traffic? Checks the database; reports queue depth."""
    checks: dict[str, str] = {}
    try:
        await c.repo.get_user("readiness-probe")
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "unavailable"
    checks["llm"] = (
        "configured"
        if (c.settings.groq_api_key or c.settings.gemini_api_key or c.settings.litellm_proxy_url)
        else "missing_keys"
    )
    ready = checks["database"] == "ok" and checks["llm"] == "configured"
    if not ready:
        response.status_code = 503
    return {"ready": ready, "checks": checks}


@router.get("/metrics", include_in_schema=False)
async def metrics(request: Request, c: ContainerDep) -> Response:
    token = c.settings.metrics_token
    if token is not None:
        header = request.headers.get("authorization", "")
        import hmac

        if not hmac.compare_digest(header, f"Bearer {token.get_secret_value()}"):
            raise UnauthorizedError("Metrics require a bearer token.")
    if hasattr(c.queue, "pending"):
        c.metrics.jobs_pending.set(c.queue.pending)
    body, content_type = c.metrics.render()
    return Response(body, media_type=content_type)
