"""LLM-facing schemas for the guardrail and the supervisor/intake step.

These intentionally use loose, JSON-friendly types. The deterministic resolver
(`agents/resolver.py`) turns a :class:`ParsedTripIntent` into a validated
:class:`~tripmate.schemas.trip.TripRequest` or into a clarification question.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field

from tripmate.schemas.trip import Pace, TravelStyle


class GuardrailCategory(StrEnum):
    TRAVEL = "travel"
    OFF_TOPIC = "off_topic"
    HARMFUL = "harmful"
    PROMPT_INJECTION = "prompt_injection"


class GuardrailVerdict(BaseModel):
    allowed: bool
    category: GuardrailCategory = GuardrailCategory.TRAVEL
    reason: str = Field(default="", max_length=300)


class ParsedTripIntent(BaseModel):
    """Everything the model could extract from the traveller's message."""

    destination: str | None = Field(default=None, description="City or country to visit")
    destination_iata: str | None = Field(
        default=None, description="IATA code of the main airport for the destination, if known"
    )
    origin: str | None = Field(default=None, description="City the traveller departs from")
    origin_iata: str | None = Field(default=None, description="IATA code of the origin airport")
    start_date: date | None = Field(default=None, description="ISO date YYYY-MM-DD")
    end_date: date | None = Field(default=None, description="ISO date YYYY-MM-DD")
    duration_days: int | None = Field(default=None, ge=1, le=60)
    travelers: int | None = Field(default=None, ge=1, le=50)
    budget_amount: Decimal | None = Field(
        default=None,
        ge=0,
        description="Total budget as a plain number. Expand 'lakh'=100000, 'crore'=10000000, 'k'=1000",
    )
    budget_currency: str | None = Field(default=None, description="ISO 4217 code, e.g. USD, PKR")
    style: TravelStyle | None = None
    pace: Pace | None = None
    interests: list[str] = Field(default_factory=list, max_length=20)
    dietary: list[str] = Field(default_factory=list, max_length=10)
    constraints: list[str] = Field(default_factory=list, max_length=10)
    wants_flights: bool = True
    wants_hotels: bool = True
    wants_weather: bool = True
    clarifying_question: str | None = Field(
        default=None, description="Only if the destination is missing or ambiguous"
    )
