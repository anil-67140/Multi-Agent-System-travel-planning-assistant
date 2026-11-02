"""HTTP request/response models. Requests forbid unknown fields and bound every size."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from tripmate.domain.money import Money
from tripmate.schemas.trip import Pace, TravelStyle
from tripmate.services.records import PlanRecord

Tag = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ auth
class Credentials(_Strict):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.strip().lower()


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"  # noqa: S105
    expires_in: int


# ------------------------------------------------------------------ plans
class CreatePlanRequest(_Strict):
    message: str = Field(min_length=3, max_length=2000)


class ConstraintEdit(_Strict):
    """Only these fields can be edited after planning; all are re-validated server-side."""

    start_date: date | None = None
    end_date: date | None = None
    travelers: int | None = Field(default=None, ge=1, le=12)
    budget: Money | None = None
    display_currency: str | None = Field(default=None, min_length=3, max_length=3)
    style: TravelStyle | None = None
    pace: Pace | None = None
    interests: list[Tag] | None = Field(default=None, max_length=12)
    dietary: list[Tag] | None = Field(default=None, max_length=8)
    constraints: (
        list[Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)]] | None
    ) = Field(default=None, max_length=10)
    clear_budget: bool = False

    def changes(self) -> dict[str, Any]:
        data = self.model_dump(exclude_none=True, exclude={"clear_budget"})
        if self.clear_budget:
            data["budget"] = None
        return data


class DecisionRequest(_Strict):
    itinerary_version: int = Field(ge=1)
    approved: bool
    feedback: str | None = Field(default=None, max_length=1000)


class AnswerRequest(_Strict):
    answer: str = Field(min_length=1, max_length=500)


class SelectAlternativeRequest(_Strict):
    label: str = Field(min_length=1, max_length=24)


class PreferencesBody(_Strict):
    home_city: str | None = Field(default=None, max_length=80)
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    style: TravelStyle | None = None
    pace: Pace | None = None
    dietary: list[Tag] = Field(default_factory=list, max_length=8)
    interests: list[Tag] = Field(default_factory=list, max_length=12)


class PreferencesOut(PreferencesBody):
    bundle_affinity: dict[str, int] = Field(default_factory=dict)


class SourceOut(BaseModel):
    provider: str
    kind: str
    origin: str
    url: str | None = None
    title: str | None = None
    retrieved_at: datetime


class VersionOut(BaseModel):
    version: int
    itinerary: dict[str, Any]
    feedback: str | None = None
    created_at: datetime


class PlanSummary(BaseModel):
    id: str
    status: str
    destination: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    travelers: int | None = None
    currency: str | None = None
    created_at: datetime

    @classmethod
    def of(cls, p: PlanRecord) -> PlanSummary:
        return cls(
            id=p.id,
            status=p.status,
            destination=p.destination,
            start_date=p.start_date,
            end_date=p.end_date,
            travelers=p.travelers,
            currency=p.currency,
            created_at=p.created_at,
        )


class PlanDetail(PlanSummary):
    message: str
    run_no: int
    trip: dict[str, Any] | None = None
    budget: dict[str, Any] | None = None
    bundles: list[dict[str, Any]] | None = None
    selected_label: str | None = None
    itinerary: VersionOut | None = None
    versions: list[int] = Field(default_factory=list)
    sources: list[SourceOut] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    tool_errors: list[dict[str, Any]] = Field(default_factory=list)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    clarification: str | None = None
    error: str | None = None
    budget_limit: Decimal | None = None

    @classmethod
    def of(cls, p: PlanRecord) -> PlanDetail:  # type: ignore[override]
        latest = p.latest_version
        base = PlanSummary.of(p).model_dump()
        return cls(
            **base,
            message=p.message,
            run_no=p.run_no,
            trip=p.trip,
            budget=p.budget,
            bundles=p.bundles,
            selected_label=p.selected_label,
            itinerary=VersionOut(**latest.model_dump()) if latest else None,
            versions=[v.version for v in p.versions],
            sources=[SourceOut(**s.model_dump()) for s in p.sources],
            notes=p.notes,
            tool_errors=p.tool_errors,
            # internal validator codes are shown to the person as plain warnings
            issues=[i for i in p.issues if i.get("severity") in {"warning", "info"}],
            clarification=p.clarification,
            error=p.error,
            budget_limit=p.budget_amount,
        )


class PlanList(BaseModel):
    items: list[PlanSummary]
    limit: int
    offset: int


class EventOut(BaseModel):
    seq: int
    type: str
    node: str | None = None
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class Problem(BaseModel):
    """RFC 9457 problem details."""

    type: str = "about:blank"
    title: str
    status: int
    code: str
    detail: str
    request_id: str | None = None
