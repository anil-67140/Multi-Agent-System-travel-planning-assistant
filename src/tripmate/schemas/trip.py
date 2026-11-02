"""Trip details: where, when, who, how much."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Annotated
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

from tripmate.domain.money import Money, normalize_currency

MAX_TRIP_DAYS = 30

ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=60)]


class TravelStyle(StrEnum):
    BUDGET = "budget"
    MID_RANGE = "mid_range"
    LUXURY = "luxury"


class Pace(StrEnum):
    RELAXED = "relaxed"
    BALANCED = "balanced"
    PACKED = "packed"


def validate_iana_timezone(value: str) -> str:
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise ValueError(f"Unknown IANA time zone: {value!r}") from exc
    return value


class Place(BaseModel):
    """A geocoded location."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=120)
    country_code: str | None = Field(default=None, pattern=r"^[A-Za-z]{2}$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str | None = None
    iata: str | None = Field(default=None, pattern=r"^[A-Za-z]{3}$")

    @field_validator("country_code", "iata")
    @classmethod
    def _upper(cls, value: str | None) -> str | None:
        return value.upper() if value else value

    @field_validator("timezone")
    @classmethod
    def _tz(cls, value: str | None) -> str | None:
        return validate_iana_timezone(value) if value else value


class TripRequest(BaseModel):
    """A fully-resolved trip. Immutable: edits create a new, re-validated copy."""

    model_config = ConfigDict(frozen=True)

    origin: Place | None = None
    destination: Place
    start_date: date
    end_date: date
    travelers: int = Field(default=1, ge=1, le=12)
    budget: Money | None = None  # total for the whole party
    display_currency: str = "USD"
    style: TravelStyle = TravelStyle.MID_RANGE
    pace: Pace = Pace.BALANCED
    interests: list[ShortText] = Field(default_factory=list, max_length=12)
    dietary: list[ShortText] = Field(default_factory=list, max_length=8)
    constraints: list[Annotated[str, StringConstraints(strip_whitespace=True, max_length=200)]] = (
        Field(default_factory=list, max_length=10)
    )
    assumptions: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("display_currency", mode="before")
    @classmethod
    def _currency(cls, value: str) -> str:
        return normalize_currency(value)

    @model_validator(mode="after")
    def _dates(self) -> TripRequest:
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        if self.duration_days > MAX_TRIP_DAYS:
            raise ValueError(f"trips longer than {MAX_TRIP_DAYS} days are not supported")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def duration_days(self) -> int:
        return (self.end_date - self.start_date).days + 1

    @computed_field  # type: ignore[prop-decorator]
    @property
    def nights(self) -> int:
        return self.duration_days - 1

    @property
    def currency(self) -> str:
        """The single currency every amount in the plan is expressed in."""
        return self.budget.currency if self.budget else self.display_currency

    def edited(self, **changes: object) -> TripRequest:
        """Return a re-validated copy with ``changes`` applied."""
        data = self.model_dump(mode="python", exclude={"duration_days", "nights"})
        data.update(changes)
        return TripRequest.model_validate(data)
