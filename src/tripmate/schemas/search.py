"""Structured results returned by the data tools (flights, hotels, weather ...)."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from tripmate.domain.money import Money
from tripmate.domain.provenance import SourceRef, UtcDatetime


class ToolError(BaseModel):
    """A tool failure, recorded instead of crashing the workflow."""

    model_config = ConfigDict(frozen=True)

    provider: str
    code: str
    message: str = Field(max_length=500)
    retryable: bool = False
    occurred_at: UtcDatetime


class Cabin(StrEnum):
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"


class FlightOption(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=40)
    origin_iata: str = Field(pattern=r"^[A-Z]{3}$")
    destination_iata: str = Field(pattern=r"^[A-Z]{3}$")
    airline: str | None = None
    depart_date: date
    return_date: date | None = None
    departure: UtcDatetime | None = None
    arrival: UtcDatetime | None = None
    duration_minutes: int = Field(ge=30, le=2400)
    stops: int = Field(default=0, ge=0, le=3)
    cabin: Cabin = Cabin.ECONOMY
    travelers: int = Field(default=1, ge=1, le=12)
    price_per_traveler: Money
    original_price_per_traveler: Money | None = None  # before FX conversion
    price_is_estimate: bool = True
    schedule_is_indicative: bool = True
    source: SourceRef
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _times(self) -> FlightOption:
        if self.departure and self.arrival and self.arrival <= self.departure:
            raise ValueError("arrival must be after departure")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_price(self) -> Money:
        return self.price_per_traveler.scaled(self.travelers)


class HotelOption(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1, max_length=40)
    name: str = Field(min_length=1, max_length=160)
    area: str | None = None
    stars: float | None = Field(default=None, ge=0, le=5)
    rating: float | None = Field(default=None, ge=0, le=10)
    nightly_price: Money  # per room per night
    original_nightly_price: Money | None = None
    rooms: int = Field(default=1, ge=1, le=10)
    nights: int = Field(default=1, ge=1, le=60)
    amenities: list[str] = Field(default_factory=list)
    price_is_estimate: bool = True
    source: SourceRef

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_price(self) -> Money:
        return self.nightly_price.scaled(self.rooms * self.nights)


class DailyWeather(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: date
    temp_min_c: float = Field(ge=-90, le=65)
    temp_max_c: float = Field(ge=-90, le=65)
    precipitation_mm: float = Field(default=0, ge=0, le=2000)
    precipitation_probability_pct: float | None = Field(default=None, ge=0, le=100)
    wind_kph: float | None = Field(default=None, ge=0, le=500)
    weather_code: int | None = None
    condition: str = "unknown"

    @model_validator(mode="after")
    def _range(self) -> DailyWeather:
        if self.temp_min_c > self.temp_max_c:
            raise ValueError("temp_min_c must not exceed temp_max_c")
        return self

    @property
    def is_wet(self) -> bool:
        heavy_code = self.weather_code is not None and self.weather_code >= 61
        likely = (self.precipitation_probability_pct or 0) >= 60
        return heavy_code or likely or self.precipitation_mm >= 8


class WeatherReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    place_name: str
    timezone: str | None = None
    days: list[DailyWeather]
    is_climatology: bool = False  # True when beyond the forecast horizon
    source: SourceRef


T = TypeVar("T")


class SearchResult(BaseModel, Generic[T]):
    """A tool outcome: results, plus any errors, plus how degraded we are."""

    items: list[T] = Field(default_factory=list)
    errors: list[ToolError] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    degraded: bool = False
    note: str | None = None
