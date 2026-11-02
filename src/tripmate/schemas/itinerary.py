"""Itinerary: the validated domain model and the looser LLM draft model."""

from __future__ import annotations

from datetime import date, time, timedelta
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tripmate.domain.money import Money


class ActivityCategory(StrEnum):
    SIGHTSEEING = "sightseeing"
    CULTURE = "culture"
    NATURE = "nature"
    FOOD = "food"
    TRANSPORT = "transport"
    LEISURE = "leisure"
    SHOPPING = "shopping"
    NIGHTLIFE = "nightlife"
    REST = "rest"
    OTHER = "other"


class Activity(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: time
    end: time
    title: str = Field(min_length=1, max_length=140)
    description: str = Field(default="", max_length=600)
    category: ActivityCategory = ActivityCategory.OTHER
    location: str | None = Field(default=None, max_length=160)
    estimated_cost: Money | None = None  # for the whole party
    outdoor: bool = False
    source_urls: list[str] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def _order(self) -> Activity:
        if self.end <= self.start:
            raise ValueError(f"activity '{self.title}' must end after it starts")
        return self

    @property
    def minutes(self) -> int:
        return (self.end.hour * 60 + self.end.minute) - (self.start.hour * 60 + self.start.minute)


class DayPlan(BaseModel):
    model_config = ConfigDict(frozen=True)

    day_number: int = Field(ge=1, le=60)
    date: date
    theme: str = Field(default="", max_length=140)
    activities: list[Activity] = Field(min_length=1, max_length=14)
    weather_note: str | None = Field(default=None, max_length=300)


class Itinerary(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = Field(min_length=1, max_length=160)
    currency: str
    summary: str = Field(default="", max_length=1200)
    days: list[DayPlan] = Field(min_length=1, max_length=30)
    tips: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def _sequence(self) -> Itinerary:
        for index, day in enumerate(self.days, start=1):
            if day.day_number != index:
                raise ValueError("day_number values must run 1, 2, 3 ... without gaps")
        for earlier, later in zip(self.days, self.days[1:], strict=False):
            if later.date != earlier.date + timedelta(days=1):
                raise ValueError("itinerary dates must be consecutive")
        return self


# ---------------------------------------------------------------------------
# LLM draft models: plain numbers instead of Money, so the JSON schema we send
# to the model stays small and easy to satisfy. `draft_to_itinerary` converts.
# ---------------------------------------------------------------------------
class ActivityDraft(BaseModel):
    start: time = Field(description="24h local time HH:MM")
    end: time = Field(description="24h local time HH:MM, after start")
    title: str
    description: str = ""
    category: ActivityCategory = ActivityCategory.OTHER
    location: str | None = None
    estimated_cost: float | None = Field(
        default=None, ge=0, description="Cost for the whole party in the plan currency"
    )
    outdoor: bool = False

    @field_validator("estimated_cost")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and value != value:  # NaN
            raise ValueError("estimated_cost must be a number")
        return value


class DayDraft(BaseModel):
    day_number: int
    date: date
    theme: str = ""
    activities: list[ActivityDraft]
    weather_note: str | None = None


class ItineraryDraft(BaseModel):
    title: str
    summary: str = ""
    days: list[DayDraft]
    tips: list[str] = Field(default_factory=list)


def draft_to_itinerary(draft: ItineraryDraft, currency: str) -> Itinerary:
    """Convert a loose LLM draft into the strict domain model (may raise)."""
    days: list[DayPlan] = []
    for day in draft.days:
        activities = [
            Activity(
                start=a.start,
                end=a.end,
                title=a.title.strip()[:140],
                description=a.description.strip()[:600],
                category=a.category,
                location=(a.location or None),
                estimated_cost=(
                    Money(amount=Decimal(str(a.estimated_cost)), currency=currency)
                    if a.estimated_cost is not None
                    else None
                ),
                outdoor=a.outdoor,
            )
            for a in day.activities
        ]
        days.append(
            DayPlan(
                day_number=day.day_number,
                date=day.date,
                theme=day.theme.strip()[:140],
                activities=activities,
                weather_note=day.weather_note,
            )
        )
    return Itinerary(
        title=draft.title.strip()[:160],
        currency=currency,
        summary=draft.summary.strip()[:1200],
        days=days,
        tips=[tip.strip()[:300] for tip in draft.tips][:12],
    )
