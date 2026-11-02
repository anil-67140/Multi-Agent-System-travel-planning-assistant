"""Relational schema.

Design notes
* Money is stored as ``Numeric(18, 4)`` + ISO currency, never float.
* ``itinerary_versions`` is append-only: every generation/revision is kept, so approvals
  can be bound to the exact version the person saw.
* ``approvals`` has a unique (plan_id, idempotency_key): a retried click is a no-op.
* ``source_records`` keeps provenance (provider, kind, origin, retrieved_at, url) per plan.
* ``run_events`` is the progress log the UI streams (SSE polls by ``seq``).
* JSON columns hold validated snapshots (trip, budget, bundles); queryable facts
  (status, destination, dates) are real columns.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    failed_logins: Mapped[int] = mapped_column(Integer, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    preferences: Mapped[UserPreference | None] = relationship(
        back_populates="user", uselist=False, cascade="all, delete-orphan"
    )
    plans: Mapped[list[Plan]] = relationship(back_populates="user", cascade="all, delete-orphan")


class UserPreference(TimestampMixin, Base):
    __tablename__ = "user_preferences"
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    home_city: Mapped[str | None] = mapped_column(String(80), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    style: Mapped[str | None] = mapped_column(String(16), nullable=True)
    pace: Mapped[str | None] = mapped_column(String(16), nullable=True)
    dietary: Mapped[list[str]] = mapped_column(JSON, default=list)
    interests: Mapped[list[str]] = mapped_column(JSON, default=list)
    # How often the person chose each bundle label; nudges optimiser weights over time.
    bundle_affinity: Mapped[dict[str, int]] = mapped_column(JSON, default=dict)
    user: Mapped[User] = relationship(back_populates="preferences")


class Plan(TimestampMixin, Base):
    """A trip plan (the aggregate root). One plan can have several workflow runs."""

    __tablename__ = "plans"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    message: Mapped[str] = mapped_column(Text, default="")
    run_no: Mapped[int] = mapped_column(Integer, default=0)  # thread_id = f"{id}:{run_no}"
    destination: Mapped[str | None] = mapped_column(String(120), nullable=True)
    start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    travelers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(3), nullable=True)
    budget_amount: Mapped[Decimal | None] = mapped_column(Numeric(18, 4), nullable=True)
    trip: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    budget: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    bundles: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    selected_label: Mapped[str | None] = mapped_column(String(24), nullable=True)
    notes: Mapped[list[str]] = mapped_column(JSON, default=list)
    tool_errors: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    issues: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    clarification: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped[User] = relationship(back_populates="plans")
    versions: Mapped[list[ItineraryVersion]] = relationship(
        back_populates="plan", cascade="all, delete-orphan", order_by="ItineraryVersion.version"
    )
    approvals: Mapped[list[Approval]] = relationship(
        back_populates="plan", cascade="all, delete-orphan"
    )
    sources: Mapped[list[SourceRecord]] = relationship(
        back_populates="plan", cascade="all, delete-orphan"
    )
    events: Mapped[list[RunEvent]] = relationship(
        back_populates="plan", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_plans_user_created", "user_id", "created_at"),)


class ItineraryVersion(Base):
    __tablename__ = "itinerary_versions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer)
    itinerary: Mapped[dict[str, Any]] = mapped_column(JSON)
    feedback: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # feedback that produced this version
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    plan: Mapped[Plan] = relationship(back_populates="versions")
    __table_args__ = (UniqueConstraint("plan_id", "version", name="uq_plan_version"),)


class Approval(Base):
    __tablename__ = "approvals"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    itinerary_version: Mapped[int] = mapped_column(Integer)
    approved: Mapped[bool] = mapped_column(Boolean)
    feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    idempotency_key: Mapped[str] = mapped_column(String(80))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    plan: Mapped[Plan] = relationship(back_populates="approvals")
    __table_args__ = (UniqueConstraint("plan_id", "idempotency_key", name="uq_approval_idem"),)


class SourceRecord(Base):
    __tablename__ = "source_records"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id", ondelete="CASCADE"), index=True)
    provider: Mapped[str] = mapped_column(String(60))
    kind: Mapped[str] = mapped_column(String(32))
    origin: Mapped[str] = mapped_column(String(16))
    url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    plan: Mapped[Plan] = relationship(back_populates="sources")


class RunEvent(Base):
    __tablename__ = "run_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id", ondelete="CASCADE"))
    seq: Mapped[int] = mapped_column(Integer)
    run_no: Mapped[int] = mapped_column(Integer, default=0)
    type: Mapped[str] = mapped_column(String(32))
    node: Mapped[str | None] = mapped_column(String(32), nullable=True)
    message: Mapped[str] = mapped_column(String(400), default="")
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    plan: Mapped[Plan] = relationship(back_populates="events")
    __table_args__ = (UniqueConstraint("plan_id", "seq", name="uq_event_seq"),)
