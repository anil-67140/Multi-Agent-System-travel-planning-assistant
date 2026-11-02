"""Repository contract + an in-memory implementation.

The in-memory repository is used by tests and quick local demos; the SQLAlchemy
one (``db/repositories.py``) is used in production. Both must pass the same
contract tests (``tests/contract``).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

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


class Repository(Protocol):
    async def create_user(self, email: str, password_hash: str) -> UserRecord: ...
    async def get_user(self, user_id: str) -> UserRecord | None: ...
    async def get_user_by_email(self, email: str) -> UserRecord | None: ...
    async def save_user(self, user: UserRecord) -> None: ...

    async def get_preferences(self, user_id: str) -> PreferenceRecord: ...
    async def save_preferences(self, prefs: PreferenceRecord) -> None: ...

    async def create_plan(self, user_id: str, message: str) -> PlanRecord: ...
    async def get_plan(self, plan_id: str) -> PlanRecord | None: ...
    async def save_plan(self, plan: PlanRecord) -> None: ...
    async def list_plans(
        self, user_id: str, limit: int = 20, offset: int = 0
    ) -> list[PlanRecord]: ...
    async def count_active_plans(self, user_id: str) -> int: ...
    async def find_stale_running(self, older_than: datetime) -> list[PlanRecord]: ...
    async def add_version(
        self, plan_id: str, version: int, itinerary: dict[str, Any], feedback: str | None
    ) -> None: ...
    async def replace_sources(self, plan_id: str, sources: list[SourceEntry]) -> None: ...

    async def add_approval(self, approval: ApprovalRecord) -> ApprovalRecord: ...
    async def get_approval(self, plan_id: str, idempotency_key: str) -> ApprovalRecord | None: ...

    async def add_event(
        self, plan_id: str, run_no: int, type: str, node: str | None, message: str, data: dict
    ) -> EventRecord: ...
    async def events_after(
        self, plan_id: str, after_seq: int, limit: int = 200
    ) -> list[EventRecord]: ...


def _id() -> str:
    return uuid.uuid4().hex


class InMemoryRepository:
    def __init__(self, now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        self._now = now
        self.users: dict[str, UserRecord] = {}
        self.prefs: dict[str, PreferenceRecord] = {}
        self.plans: dict[str, PlanRecord] = {}
        self.approvals: dict[tuple[str, str], ApprovalRecord] = {}
        self.events: dict[str, list[EventRecord]] = {}

    # users
    async def create_user(self, email: str, password_hash: str) -> UserRecord:
        email = email.lower()
        if any(u.email == email for u in self.users.values()):
            raise ConflictError("An account with this email already exists.")
        user = UserRecord(id=_id(), email=email, password_hash=password_hash)
        self.users[user.id] = user
        return user.model_copy(deep=True)

    async def get_user(self, user_id: str) -> UserRecord | None:
        u = self.users.get(user_id)
        return u.model_copy(deep=True) if u else None

    async def get_user_by_email(self, email: str) -> UserRecord | None:
        u = next((u for u in self.users.values() if u.email == email.lower()), None)
        return u.model_copy(deep=True) if u else None

    async def save_user(self, user: UserRecord) -> None:
        self.users[user.id] = user.model_copy(deep=True)

    # preferences
    async def get_preferences(self, user_id: str) -> PreferenceRecord:
        return self.prefs.get(user_id, PreferenceRecord(user_id=user_id)).model_copy(deep=True)

    async def save_preferences(self, prefs: PreferenceRecord) -> None:
        self.prefs[prefs.user_id] = prefs.model_copy(deep=True)

    # plans
    async def create_plan(self, user_id: str, message: str) -> PlanRecord:
        now = self._now()
        plan = PlanRecord(
            id=_id(), user_id=user_id, message=message, created_at=now, updated_at=now
        )
        self.plans[plan.id] = plan
        return plan.model_copy(deep=True)

    async def get_plan(self, plan_id: str) -> PlanRecord | None:
        p = self.plans.get(plan_id)
        return p.model_copy(deep=True) if p else None

    async def save_plan(self, plan: PlanRecord) -> None:
        plan.updated_at = self._now()
        stored = self.plans[plan.id]
        copy = plan.model_copy(deep=True)
        copy.versions, copy.sources = (
            stored.versions,
            stored.sources,
        )  # children have their own methods
        self.plans[plan.id] = copy

    async def list_plans(self, user_id: str, limit: int = 20, offset: int = 0) -> list[PlanRecord]:
        mine = sorted(
            (p for p in self.plans.values() if p.user_id == user_id),
            key=lambda p: p.created_at,
            reverse=True,
        )
        return [p.model_copy(deep=True) for p in mine[offset : offset + limit]]

    async def count_active_plans(self, user_id: str) -> int:
        return sum(
            1
            for p in self.plans.values()
            if p.user_id == user_id and p.status in {"queued", "running"}
        )

    async def find_stale_running(self, older_than: datetime) -> list[PlanRecord]:
        return [
            p.model_copy(deep=True)
            for p in self.plans.values()
            if p.status == "running" and (p.heartbeat_at or p.updated_at) < older_than
        ]

    async def add_version(
        self, plan_id: str, version: int, itinerary: dict[str, Any], feedback: str | None
    ) -> None:
        plan = self.plans[plan_id]
        if any(v.version == version for v in plan.versions):
            return  # idempotent: replaying the same result must not duplicate versions
        plan.versions.append(
            VersionRecord(
                version=version, itinerary=itinerary, feedback=feedback, created_at=self._now()
            )
        )

    async def replace_sources(self, plan_id: str, sources: list[SourceEntry]) -> None:
        self.plans[plan_id].sources = [s.model_copy(deep=True) for s in sources]

    # approvals
    async def add_approval(self, approval: ApprovalRecord) -> ApprovalRecord:
        key = (approval.plan_id, approval.idempotency_key)
        if key in self.approvals:
            raise ConflictError("Duplicate approval request.")
        self.approvals[key] = approval.model_copy(deep=True)
        return approval

    async def get_approval(self, plan_id: str, idempotency_key: str) -> ApprovalRecord | None:
        a = self.approvals.get((plan_id, idempotency_key))
        return a.model_copy(deep=True) if a else None

    # events
    async def add_event(
        self, plan_id: str, run_no: int, type: str, node: str | None, message: str, data: dict
    ) -> EventRecord:
        events = self.events.setdefault(plan_id, [])
        ev = EventRecord(
            seq=len(events) + 1,
            run_no=run_no,
            type=type,
            node=node,
            message=message[:400],
            data=data,
            created_at=self._now(),
        )
        events.append(ev)
        return ev

    async def events_after(
        self, plan_id: str, after_seq: int, limit: int = 200
    ) -> list[EventRecord]:
        return [e.model_copy(deep=True) for e in self.events.get(plan_id, []) if e.seq > after_seq][
            :limit
        ]
