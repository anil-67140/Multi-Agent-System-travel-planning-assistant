from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, status
from fastapi.responses import StreamingResponse

from tripmate.api.deps import ContainerDep, CurrentUser, IdempotencyKey, rate_limit
from tripmate.api.schemas import (
    AnswerRequest,
    ConstraintEdit,
    CreatePlanRequest,
    DecisionRequest,
    EventOut,
    PlanDetail,
    PlanList,
    PlanSummary,
    PreferencesBody,
    PreferencesOut,
    SelectAlternativeRequest,
)
from tripmate.errors import ValidationFailure
from tripmate.schemas.plan import PlanStatus
from tripmate.services.records import PreferenceRecord

router = APIRouter(prefix="/v1", tags=["plans"])

_create_limit = rate_limit(
    "plans", per=lambda c: c.settings.plan_rate_limit_per_hour, window=3600, by_user=True
)
_general_limit = rate_limit(
    "api", per=lambda c: c.settings.rate_limit_per_minute, window=60, by_user=True
)
# States where nothing more will happen until the person acts.
SETTLED = {s.value for s in PlanStatus if s.is_terminal} | {
    PlanStatus.AWAITING_APPROVAL.value,
    PlanStatus.NEEDS_CLARIFICATION.value,
}
STREAM_POLL_SECONDS = 0.7
STREAM_MAX_SECONDS = 600
HEARTBEAT_EVERY = 15


@router.post(
    "/plans",
    response_model=PlanSummary,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_create_limit)],
)
async def create_plan(body: CreatePlanRequest, user: CurrentUser, c: ContainerDep) -> PlanSummary:
    return PlanSummary.of(await c.service.create_plan(user.id, body.message))


@router.get("/plans", response_model=PlanList, dependencies=[Depends(_general_limit)])
async def list_plans(
    user: CurrentUser,
    c: ContainerDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0, le=10_000)] = 0,
) -> PlanList:
    plans = await c.service.list_plans(user.id, limit, offset)
    return PlanList(items=[PlanSummary.of(p) for p in plans], limit=limit, offset=offset)


@router.get("/plans/{plan_id}", response_model=PlanDetail, dependencies=[Depends(_general_limit)])
async def get_plan(plan_id: str, user: CurrentUser, c: ContainerDep) -> PlanDetail:
    return PlanDetail.of(await c.service.get_owned(user.id, plan_id))


@router.get("/plans/{plan_id}/versions/{version}")
async def get_version(plan_id: str, version: int, user: CurrentUser, c: ContainerDep) -> dict:
    plan = await c.service.get_owned(user.id, plan_id)
    found = next((v for v in plan.versions if v.version == version), None)
    if found is None:
        from tripmate.errors import NotFoundError

        raise NotFoundError("Version not found.")
    return found.model_dump(mode="json")


@router.get(
    "/plans/{plan_id}/events", response_model=list[EventOut], dependencies=[Depends(_general_limit)]
)
async def list_events(
    plan_id: str, user: CurrentUser, c: ContainerDep, after: Annotated[int, Query(ge=0)] = 0
) -> list[EventOut]:
    await c.service.get_owned(user.id, plan_id)
    return [EventOut(**e.model_dump()) for e in await c.repo.events_after(plan_id, after)]


@router.get("/plans/{plan_id}/stream")
async def stream_events(
    plan_id: str, request: Request, user: CurrentUser, c: ContainerDep
) -> StreamingResponse:
    """Server-Sent Events. Events live in the database, so this works across replicas and
    resumes with ``Last-Event-ID`` after a dropped connection. No pub/sub infrastructure needed."""
    await c.service.get_owned(user.id, plan_id)  # authorise before streaming anything
    try:
        last = max(0, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        last = 0

    async def generate() -> AsyncIterator[str]:
        seq, idle, waited = last, 0.0, 0.0
        while waited < STREAM_MAX_SECONDS:
            if await request.is_disconnected():
                return
            events = await c.repo.events_after(plan_id, seq)
            for e in events:
                seq = e.seq
                yield f"id: {e.seq}\nevent: {e.type}\ndata: {json.dumps(EventOut(**e.model_dump()).model_dump(mode='json'))}\n\n"
            if events:
                idle = 0.0
            plan = await c.repo.get_plan(plan_id)
            if plan and not events and plan.status in SETTLED:
                yield f"event: done\ndata: {json.dumps({'status': plan.status})}\n\n"
                return
            await asyncio.sleep(STREAM_POLL_SECONDS)
            idle += STREAM_POLL_SECONDS
            waited += STREAM_POLL_SECONDS
            if idle >= HEARTBEAT_EVERY:
                idle = 0.0
                yield ": keep-alive\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/plans/{plan_id}/decision",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_general_limit)],
)
async def decide(
    plan_id: str, body: DecisionRequest, user: CurrentUser, key: IdempotencyKey, c: ContainerDep
) -> dict[str, object]:
    """Approve or request changes. Bound to ``itinerary_version`` and idempotent per ``Idempotency-Key``."""
    record = await c.service.decide(
        user.id,
        plan_id,
        itinerary_version=body.itinerary_version,
        approved=body.approved,
        feedback=body.feedback,
        idempotency_key=key,
    )
    return {
        "plan_id": plan_id,
        "itinerary_version": record.itinerary_version,
        "approved": record.approved,
        "status": "accepted",
    }


@router.post(
    "/plans/{plan_id}/answer",
    response_model=PlanSummary,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_create_limit)],
)
async def answer(
    plan_id: str, body: AnswerRequest, user: CurrentUser, c: ContainerDep
) -> PlanSummary:
    return PlanSummary.of(await c.service.answer_clarification(user.id, plan_id, body.answer))


@router.patch(
    "/plans/{plan_id}/constraints",
    response_model=PlanSummary,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_create_limit)],
)
async def edit_constraints(
    plan_id: str, body: ConstraintEdit, user: CurrentUser, c: ContainerDep
) -> PlanSummary:
    changes = body.changes()
    if not changes:
        raise ValidationFailure("Nothing to change.")
    return PlanSummary.of(await c.service.update_constraints(user.id, plan_id, changes))


@router.post(
    "/plans/{plan_id}/alternative",
    response_model=PlanSummary,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_create_limit)],
)
async def select_alternative(
    plan_id: str, body: SelectAlternativeRequest, user: CurrentUser, c: ContainerDep
) -> PlanSummary:
    return PlanSummary.of(await c.service.select_alternative(user.id, plan_id, body.label))


@router.post(
    "/plans/{plan_id}/retry",
    response_model=PlanSummary,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(_create_limit)],
)
async def retry(plan_id: str, user: CurrentUser, c: ContainerDep) -> PlanSummary:
    return PlanSummary.of(await c.service.retry(user.id, plan_id))


@router.get("/preferences", response_model=PreferencesOut)
async def get_preferences(user: CurrentUser, c: ContainerDep) -> PreferencesOut:
    p = await c.service.get_preferences(user.id)
    return PreferencesOut(**p.model_dump(exclude={"user_id"}))


@router.put("/preferences", response_model=PreferencesOut)
async def put_preferences(
    body: PreferencesBody, user: CurrentUser, c: ContainerDep
) -> PreferencesOut:
    saved = await c.service.save_preferences(PreferenceRecord(user_id=user.id, **body.model_dump()))
    return PreferencesOut(**saved.model_dump(exclude={"user_id"}))
