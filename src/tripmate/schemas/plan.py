"""Plan-level schemas: lifecycle status, alternative bundles, progress events."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tripmate.domain.money import Money
from tripmate.domain.provenance import UtcDatetime


class PlanStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    NEEDS_CLARIFICATION = "needs_clarification"
    AWAITING_APPROVAL = "awaiting_approval"
    COMPLETED = "completed"
    BLOCKED = "blocked"  # guardrail rejected the request
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in {PlanStatus.COMPLETED, PlanStatus.BLOCKED, PlanStatus.FAILED}


class BundleLabel(StrEnum):
    BUDGET_SAVER = "budget_saver"
    BALANCED = "balanced"
    COMFORT = "comfort"


class PlanBundle(BaseModel):
    """One flight + hotel combination offered as an alternative."""

    model_config = ConfigDict(frozen=True)

    label: BundleLabel
    flight_id: str | None = None
    hotel_id: str | None = None
    total_cost: Money
    travel_minutes: int = Field(ge=0)
    quality: float = Field(ge=0, le=1)
    utility: float = Field(ge=0, le=1)
    within_budget: bool = True
    pareto_optimal: bool = True
    tradeoffs: list[str] = Field(default_factory=list)


class PlanEvent(BaseModel):
    """A progress event persisted for the UI / SSE stream."""

    seq: int
    type: str
    node: str | None = None
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: UtcDatetime
