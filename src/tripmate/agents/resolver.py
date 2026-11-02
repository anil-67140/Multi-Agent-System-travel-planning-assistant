"""Deterministic step turning the model's loose extraction into a validated TripRequest.

The model extracts; code decides. Every guess we make for a missing field is
recorded in ``assumptions`` so the traveller can see and edit it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation

from tripmate.domain.money import Money, UnknownCurrencyError, normalize_currency
from tripmate.errors import ProviderError
from tripmate.providers.toolbelt import Toolbelt
from tripmate.schemas.intake import ParsedTripIntent
from tripmate.schemas.trip import MAX_TRIP_DAYS, Pace, Place, TravelStyle, TripRequest
from tripmate.validation import check_trip_dates, default_start_date, errors, format_issues

DEFAULT_TRIP_DAYS = 5


@dataclass(frozen=True)
class Clarification:
    question: str


@dataclass(frozen=True)
class UserDefaults:
    """Persisted preferences used when the message does not say."""

    home_city: str | None = None
    currency: str | None = None
    style: TravelStyle | None = None
    pace: Pace | None = None
    dietary: tuple[str, ...] = ()
    interests: tuple[str, ...] = ()


async def _geocode(tools: Toolbelt, name: str, iata: str | None) -> Place:
    place = await tools.geocode(name)
    if iata and len(iata) == 3 and iata.isalpha():
        place = place.model_copy(
            update={"iata": iata.upper()}
        )  # verified later against the airport directory
    return place


def _dedupe(items: list[str], limit: int) -> list[str]:
    seen: dict[str, str] = {}
    for item in items:
        text = item.strip()[:60]
        if text:
            seen.setdefault(text.lower(), text)
    return list(seen.values())[:limit]


async def resolve_intent(
    intent: ParsedTripIntent,
    tools: Toolbelt,
    *,
    today: date,
    defaults: UserDefaults | None = None,
) -> TripRequest | Clarification:
    defaults = defaults or UserDefaults()
    assumptions: list[str] = []

    if not intent.destination:
        return Clarification(intent.clarifying_question or "Where would you like to travel?")

    try:
        destination = await _geocode(tools, intent.destination, intent.destination_iata)
    except ProviderError:
        return Clarification(
            f"I couldn't find a place called '{intent.destination}'. Could you give a city or country?"
        )

    origin: Place | None = None
    origin_name = intent.origin or defaults.home_city
    if origin_name:
        try:
            origin = await _geocode(tools, origin_name, intent.origin_iata)
            if not intent.origin:
                assumptions.append(f"Departing from your saved home city, {origin_name}.")
        except ProviderError:
            assumptions.append(f"Could not locate '{origin_name}', so flights are left out.")
    else:
        assumptions.append("No departure city given, so flights are left out.")

    # ---- dates ----------------------------------------------------------
    start, end = intent.start_date, intent.end_date
    if start is None:
        start = default_start_date(today)
        assumptions.append(
            f"No start date given; assumed {start.isoformat()} (about six weeks ahead)."
        )
    if end is None:
        days = intent.duration_days or DEFAULT_TRIP_DAYS
        end = start + timedelta(days=days - 1)
        if not intent.duration_days:
            assumptions.append(f"No length given; assumed {DEFAULT_TRIP_DAYS} days.")
    if start < today:
        # A past date almost always means a model/year slip, so say so instead of silently moving it.
        return Clarification(
            f"{start.isoformat()} is in the past (today is {today.isoformat()}). Which dates did you mean?"
        )
    if (end - start).days + 1 > MAX_TRIP_DAYS:
        return Clarification(
            f"I can plan up to {MAX_TRIP_DAYS} days at a time. Could you shorten the trip?"
        )
    date_issues = errors(check_trip_dates(start, end, today=today))
    if date_issues:
        return Clarification(format_issues(date_issues))

    # ---- money ----------------------------------------------------------
    budget: Money | None = None
    display = defaults.currency or "USD"
    if intent.budget_currency:
        try:
            display = normalize_currency(intent.budget_currency)
        except UnknownCurrencyError:
            assumptions.append(f"Unknown currency '{intent.budget_currency}'; using {display}.")
    if intent.budget_amount is not None and intent.budget_amount > 0:
        try:
            budget = Money.of(Decimal(intent.budget_amount), display)
        except (InvalidOperation, ValueError):
            assumptions.append("The budget could not be read and was ignored.")
    if budget is None and not intent.budget_currency and not defaults.currency:
        assumptions.append("No budget given; costs are shown in USD.")

    style = intent.style or defaults.style or TravelStyle.MID_RANGE
    if intent.style is None and defaults.style is not None:
        assumptions.append(f"Using your saved travel style: {defaults.style.value}.")

    return TripRequest(
        origin=origin,
        destination=destination,
        start_date=start,
        end_date=end,
        travelers=intent.travelers or 1,
        budget=budget,
        display_currency=display,
        style=style,
        pace=intent.pace or defaults.pace or Pace.BALANCED,
        interests=_dedupe([*intent.interests, *defaults.interests], 12),
        dietary=_dedupe([*intent.dietary, *defaults.dietary], 8),
        constraints=[c.strip()[:200] for c in intent.constraints if c.strip()][:10],
        assumptions=assumptions[:10],
    )
