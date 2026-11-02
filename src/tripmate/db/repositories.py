"""SQLAlchemy implementation of the ``Repository`` contract.

Every method opens its own short transaction, so no session is held across an LLM call.
Mapping between ORM rows and the pydantic ``*Record`` types happens here only.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from tripmate.db import models as m
from tripmate.errors import ConflictError
from tripmate.services.records import (
    ApprovalRecord,
    EventRecord,
    PlanRecord,
    PreferenceRecord,
    SourceEntry,
    UserRecord,
    VersionRecord,
)


def _aware(value: datetime | None) -> datetime | None:
    """SQLite returns naive datetimes; treat stored values as UTC."""
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _user(row: m.User) -> UserRecord:
    return UserRecord(
        id=row.id,
        email=row.email,
        password_hash=row.password_hash,
        is_active=row.is_active,
        failed_logins=row.failed_logins,
        locked_until=_aware(row.locked_until),
    )


def _plan(row: m.Plan) -> PlanRecord:
    return PlanRecord(
        id=row.id,
        user_id=row.user_id,
        status=row.status,
        message=row.message,
        run_no=row.run_no,
        destination=row.destination,
        start_date=row.start_date,
        end_date=row.end_date,
        travelers=row.travelers,
        currency=row.currency,
        budget_amount=row.budget_amount,
        trip=row.trip,
        budget=row.budget,
        bundles=row.bundles,
        selected_label=row.selected_label,
        notes=row.notes or [],
        tool_errors=row.tool_errors or [],
        issues=row.issues or [],
        clarification=row.clarification,
        error=row.error,
        heartbeat_at=_aware(row.heartbeat_at),
        created_at=_aware(row.created_at),  # type: ignore[arg-type]
        updated_at=_aware(row.updated_at),  # type: ignore[arg-type]
        versions=[
            VersionRecord(
                version=v.version,
                itinerary=v.itinerary,
                feedback=v.feedback,
                created_at=_aware(v.created_at),
            )
            for v in row.versions
        ],  # type: ignore[arg-type]
        sources=[
            SourceEntry(
                provider=s.provider,
                kind=s.kind,
                origin=s.origin,
                url=s.url,
                title=s.title,
                retrieved_at=_aware(s.retrieved_at),
            )
            for s in row.sources
        ],  # type: ignore[arg-type]
    )


_PLAN_SCALARS = (
    "status",
    "message",
    "run_no",
    "destination",
    "start_date",
    "end_date",
    "travelers",
    "currency",
    "budget_amount",
    "trip",
    "budget",
    "bundles",
    "selected_label",
    "notes",
    "tool_errors",
    "issues",
    "clarification",
    "error",
    "heartbeat_at",
)


def _coerce(name: str, value: Any) -> Any:
    from datetime import date

    if name in {"start_date", "end_date"} and isinstance(value, str):
        return date.fromisoformat(value)
    if name == "budget_amount" and value is not None:
        return Decimal(str(value))
    return value


class SqlRepository:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._s = sessions

    # users ------------------------------------------------------------------
    async def create_user(self, email: str, password_hash: str) -> UserRecord:
        async with self._s() as s, s.begin():
            row = m.User(email=email.lower(), password_hash=password_hash)
            s.add(row)
            try:
                await s.flush()
            except IntegrityError as exc:
                raise ConflictError("An account with this email already exists.") from exc
            return _user(row)

    async def get_user(self, user_id: str) -> UserRecord | None:
        async with self._s() as s:
            row = await s.get(m.User, user_id)
            return _user(row) if row else None

    async def get_user_by_email(self, email: str) -> UserRecord | None:
        async with self._s() as s:
            row = (
                await s.execute(select(m.User).where(m.User.email == email.lower()))
            ).scalar_one_or_none()
            return _user(row) if row else None

    async def save_user(self, user: UserRecord) -> None:
        async with self._s() as s, s.begin():
            row = await s.get(m.User, user.id)
            if row:
                row.is_active, row.failed_logins, row.locked_until = (
                    user.is_active,
                    user.failed_logins,
                    user.locked_until,
                )

    # preferences ---------------------------------------------------------------
    async def get_preferences(self, user_id: str) -> PreferenceRecord:
        async with self._s() as s:
            row = await s.get(m.UserPreference, user_id)
            if row is None:
                return PreferenceRecord(user_id=user_id)
            return PreferenceRecord(
                user_id=user_id,
                home_city=row.home_city,
                currency=row.currency,
                style=row.style,
                pace=row.pace,
                dietary=row.dietary or [],
                interests=row.interests or [],
                bundle_affinity=row.bundle_affinity or {},
            )

    async def save_preferences(self, prefs: PreferenceRecord) -> None:
        async with self._s() as s, s.begin():
            row = await s.get(m.UserPreference, prefs.user_id)
            if row is None:
                row = m.UserPreference(user_id=prefs.user_id)
                s.add(row)
            for name in (
                "home_city",
                "currency",
                "style",
                "pace",
                "dietary",
                "interests",
                "bundle_affinity",
            ):
                setattr(row, name, getattr(prefs, name))

    # plans ----------------------------------------------------------------------
    async def create_plan(self, user_id: str, message: str) -> PlanRecord:
        async with self._s() as s, s.begin():
            row = m.Plan(user_id=user_id, message=message, notes=[], tool_errors=[], issues=[])
            s.add(row)
            await s.flush()
            await s.refresh(row, ["versions", "sources"])
            return _plan(row)

    async def _load(self, s: AsyncSession, plan_id: str) -> m.Plan | None:
        stmt = (
            select(m.Plan)
            .where(m.Plan.id == plan_id)
            .options(selectinload(m.Plan.versions), selectinload(m.Plan.sources))
        )
        return (await s.execute(stmt)).scalar_one_or_none()

    async def get_plan(self, plan_id: str) -> PlanRecord | None:
        async with self._s() as s:
            row = await self._load(s, plan_id)
            return _plan(row) if row else None

    async def save_plan(self, plan: PlanRecord) -> None:
        async with self._s() as s, s.begin():
            row = await s.get(m.Plan, plan.id)
            if row is None:
                return
            for name in _PLAN_SCALARS:
                setattr(row, name, _coerce(name, getattr(plan, name)))

    async def list_plans(self, user_id: str, limit: int = 20, offset: int = 0) -> list[PlanRecord]:
        async with self._s() as s:
            stmt = (
                select(m.Plan)
                .where(m.Plan.user_id == user_id)
                .order_by(m.Plan.created_at.desc())
                .limit(limit)
                .offset(offset)
                .options(selectinload(m.Plan.versions), selectinload(m.Plan.sources))
            )
            return [_plan(r) for r in (await s.execute(stmt)).scalars()]

    async def count_active_plans(self, user_id: str) -> int:
        async with self._s() as s:
            stmt = (
                select(func.count())
                .select_from(m.Plan)
                .where(m.Plan.user_id == user_id, m.Plan.status.in_(["queued", "running"]))
            )
            return int((await s.execute(stmt)).scalar_one())

    async def find_stale_running(self, older_than: datetime) -> list[PlanRecord]:
        async with self._s() as s:
            stmt = (
                select(m.Plan)
                .where(m.Plan.status == "running")
                .options(selectinload(m.Plan.versions), selectinload(m.Plan.sources))
            )
            rows = (await s.execute(stmt)).scalars()
            return [
                _plan(r)
                for r in rows
                if (_aware(r.heartbeat_at) or _aware(r.updated_at)) < older_than
            ]  # type: ignore[operator]

    async def add_version(
        self, plan_id: str, version: int, itinerary: dict, feedback: str | None
    ) -> None:
        async with self._s() as s:
            try:
                async with s.begin():
                    s.add(
                        m.ItineraryVersion(
                            plan_id=plan_id, version=version, itinerary=itinerary, feedback=feedback
                        )
                    )
            except IntegrityError:
                return  # already stored: idempotent replay

    async def replace_sources(self, plan_id: str, sources: list[SourceEntry]) -> None:
        async with self._s() as s, s.begin():
            existing = (
                await s.execute(select(m.SourceRecord).where(m.SourceRecord.plan_id == plan_id))
            ).scalars()
            for row in existing:
                await s.delete(row)
            for src in sources:
                s.add(
                    m.SourceRecord(
                        plan_id=plan_id,
                        provider=src.provider,
                        kind=src.kind,
                        origin=src.origin,
                        url=src.url,
                        title=src.title,
                        retrieved_at=src.retrieved_at,
                    )
                )

    # approvals -----------------------------------------------------------------
    async def add_approval(self, approval: ApprovalRecord) -> ApprovalRecord:
        async with self._s() as s:
            try:
                async with s.begin():
                    s.add(
                        m.Approval(
                            id=m._uuid(),
                            plan_id=approval.plan_id,
                            user_id=approval.user_id,
                            itinerary_version=approval.itinerary_version,
                            approved=approval.approved,
                            feedback=approval.feedback,
                            idempotency_key=approval.idempotency_key,
                        )
                    )
            except IntegrityError as exc:
                raise ConflictError("Duplicate approval request.") from exc
        return approval

    async def get_approval(self, plan_id: str, idempotency_key: str) -> ApprovalRecord | None:
        async with self._s() as s:
            stmt = select(m.Approval).where(
                m.Approval.plan_id == plan_id, m.Approval.idempotency_key == idempotency_key
            )
            row = (await s.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            return ApprovalRecord(
                id=row.idempotency_key,
                plan_id=row.plan_id,
                user_id=row.user_id,
                itinerary_version=row.itinerary_version,
                approved=row.approved,
                feedback=row.feedback,
                idempotency_key=row.idempotency_key,
                created_at=_aware(row.created_at),
            )  # type: ignore[arg-type]

    # events --------------------------------------------------------------------
    async def add_event(
        self, plan_id: str, run_no: int, type: str, node: str | None, message: str, data: dict
    ) -> EventRecord:
        # seq is allocated inside the transaction; the unique (plan_id, seq) constraint plus a retry
        # makes concurrent emitters (parallel graph nodes) safe.
        for _ in range(5):
            try:
                async with self._s() as s, s.begin():
                    last = (
                        await s.execute(
                            select(func.max(m.RunEvent.seq)).where(m.RunEvent.plan_id == plan_id)
                        )
                    ).scalar()
                    row = m.RunEvent(
                        plan_id=plan_id,
                        seq=(last or 0) + 1,
                        run_no=run_no,
                        type=type,
                        node=node,
                        message=message[:400],
                        data=data,
                    )
                    s.add(row)
                    await s.flush()
                    return EventRecord(
                        seq=row.seq,
                        run_no=run_no,
                        type=type,
                        node=node,
                        message=row.message,
                        data=data,
                        created_at=_aware(row.created_at),
                    )  # type: ignore[arg-type]
            except IntegrityError:
                continue
        raise ConflictError("Could not record progress event.")

    async def events_after(
        self, plan_id: str, after_seq: int, limit: int = 200
    ) -> list[EventRecord]:
        async with self._s() as s:
            stmt = (
                select(m.RunEvent)
                .where(m.RunEvent.plan_id == plan_id, m.RunEvent.seq > after_seq)
                .order_by(m.RunEvent.seq)
                .limit(limit)
            )
            return [
                EventRecord(
                    seq=r.seq,
                    run_no=r.run_no,
                    type=r.type,
                    node=r.node,
                    message=r.message,
                    data=r.data or {},
                    created_at=_aware(r.created_at),
                )  # type: ignore[arg-type]
                for r in (await s.execute(stmt)).scalars()
            ]
