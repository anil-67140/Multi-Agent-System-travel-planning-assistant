"""Persistence-agnostic records exchanged between services, repositories and the API."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UserRecord(BaseModel):
    id: str
    email: str
    password_hash: str
    is_active: bool = True
    failed_logins: int = 0
    locked_until: datetime | None = None


class PreferenceRecord(BaseModel):
    user_id: str
    home_city: str | None = None
    currency: str | None = None
    style: str | None = None
    pace: str | None = None
    dietary: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    bundle_affinity: dict[str, int] = Field(default_factory=dict)


class VersionRecord(BaseModel):
    version: int
    itinerary: dict[str, Any]
    feedback: str | None = None
    created_at: datetime


class SourceEntry(BaseModel):
    provider: str
    kind: str
    origin: str
    url: str | None = None
    title: str | None = None
    retrieved_at: datetime


class ApprovalRecord(BaseModel):
    id: str
    plan_id: str
    user_id: str
    itinerary_version: int
    approved: bool
    feedback: str | None = None
    idempotency_key: str
    created_at: datetime


class EventRecord(BaseModel):
    seq: int
    run_no: int = 0
    type: str
    node: str | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class PlanRecord(BaseModel):
    model_config = ConfigDict(validate_assignment=False)

    id: str
    user_id: str
    status: str = "queued"
    message: str = ""
    run_no: int = 0
    destination: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    travelers: int | None = None
    currency: str | None = None
    budget_amount: Decimal | None = None
    trip: dict[str, Any] | None = None
    budget: dict[str, Any] | None = None
    bundles: list[dict[str, Any]] | None = None
    selected_label: str | None = None
    notes: list[str] = Field(default_factory=list)
    tool_errors: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    clarification: str | None = None
    error: str | None = None
    heartbeat_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    versions: list[VersionRecord] = Field(default_factory=list)
    sources: list[SourceEntry] = Field(default_factory=list)

    @property
    def thread_id(self) -> str:
        return f"{self.id}:{self.run_no}"

    @property
    def latest_version(self) -> VersionRecord | None:
        return self.versions[-1] if self.versions else None
