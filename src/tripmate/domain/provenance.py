"""Where a piece of data came from, how old it is, and how far to trust it."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, field_validator


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


# A datetime that MUST carry a timezone and is normalised to UTC.
UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]


class DataOrigin(StrEnum):
    LIVE = "live"  # fetched from a provider for this request
    CACHED = "cached"  # a previous live fetch served from cache
    ESTIMATE = "estimate"  # computed by a documented heuristic
    LLM_KNOWLEDGE = "llm_knowledge"  # model recall, unverified


class DataKind(StrEnum):
    WEATHER_FORECAST = "weather_forecast"
    WEATHER_CLIMATOLOGY = "weather_climatology"
    FX_RATE = "fx_rate"
    FLIGHT_PRICE = "flight_price"
    HOTEL_PRICE = "hotel_price"
    WEB_SOURCE = "web_source"
    GEOCODING = "geocoding"


class SourceRef(BaseModel):
    """A citation for a fact shown to the traveller."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1, max_length=60)
    kind: DataKind
    origin: DataOrigin
    retrieved_at: UtcDatetime
    url: str | None = Field(default=None, max_length=2000)
    title: str | None = Field(default=None, max_length=300)
    note: str | None = Field(default=None, max_length=300)

    @field_validator("url")
    @classmethod
    def _http_only(cls, value: str | None) -> str | None:
        if value is None:
            return None
        lowered = value.lower()
        if not lowered.startswith(("http://", "https://")) or any(c.isspace() for c in value):
            raise ValueError("source url must be an absolute http(s) URL without whitespace")
        return value
