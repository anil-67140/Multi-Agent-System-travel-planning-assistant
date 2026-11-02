"""Deterministic budget builder.

Money math is never delegated to an LLM: models are good at prose and bad at
arithmetic. Every line records its *basis* so the traveller can see whether it
is a live quote or an estimate.
"""

from __future__ import annotations

from decimal import Decimal

from tripmate.domain.cost_model import (
    BUFFER_RATE,
    DAILY_SPLIT,
    DAILY_USD_PER_PERSON,
    HOTEL_USD_PER_NIGHT,
    cost_index,
    rooms_needed,
)
from tripmate.domain.money import FxRates, Money, sum_money
from tripmate.schemas.budget import (
    BudgetBreakdown,
    BudgetCategory,
    BudgetLine,
    BudgetVerdict,
)
from tripmate.schemas.search import FlightOption, HotelOption
from tripmate.schemas.trip import TripRequest

TIGHT_HEADROOM = Decimal("0.10")


def _usd(amount: Decimal, currency: str, fx: FxRates) -> Money:
    return fx.convert(Money.of(amount, "USD"), currency)


def build_budget(
    trip: TripRequest,
    flight: FlightOption | None,
    hotel: HotelOption | None,
    fx: FxRates,
) -> BudgetBreakdown:
    """Raises :class:`~tripmate.domain.money.MissingRateError` if FX cannot convert."""
    cur = trip.currency
    lines: list[BudgetLine] = []
    assumptions: list[str] = []
    index = cost_index(trip.destination.country_code)

    # ---- flights ---------------------------------------------------------
    if flight is not None:
        lines.append(
            BudgetLine(
                category=BudgetCategory.FLIGHTS,
                amount=fx.convert(flight.total_price, cur),
                basis=f"{flight.travelers} traveller(s) x {flight.price_per_traveler} "
                f"({'estimate' if flight.price_is_estimate else 'quoted'})",
                is_estimate=flight.price_is_estimate,
            )
        )
    else:
        assumptions.append("Flights are not included in this budget.")

    # ---- lodging ---------------------------------------------------------
    rooms = rooms_needed(trip.travelers)
    if hotel is not None:
        lodging = fx.convert(hotel.nightly_price, cur).scaled(rooms * trip.nights)
        basis = f"{rooms} room(s) x {trip.nights} night(s) x {hotel.nightly_price} at {hotel.name}"
        is_est = hotel.price_is_estimate
    else:
        nightly = HOTEL_USD_PER_NIGHT[trip.style] * index
        lodging = _usd(nightly, cur, fx).scaled(rooms * trip.nights)
        basis = f"{rooms} room(s) x {trip.nights} night(s) at a typical {trip.style.value} rate"
        is_est = True
    lines.append(
        BudgetLine(category=BudgetCategory.LODGING, amount=lodging, basis=basis, is_estimate=is_est)
    )

    # ---- daily spend -----------------------------------------------------
    daily_usd = DAILY_USD_PER_PERSON[trip.style] * index
    person_days = trip.travelers * trip.duration_days
    for key, category in (
        ("food", BudgetCategory.FOOD),
        ("local_transport", BudgetCategory.LOCAL_TRANSPORT),
        ("activities", BudgetCategory.ACTIVITIES),
    ):
        amount = _usd(daily_usd * DAILY_SPLIT[key], cur, fx).scaled(person_days)
        lines.append(
            BudgetLine(
                category=category,
                amount=amount,
                basis=f"{person_days} person-days at a regional {trip.style.value} {key.replace('_', ' ')} estimate",
                is_estimate=True,
            )
        )
    assumptions.append("Food, transport and activities are regional estimates, not live prices.")

    # ---- buffer ----------------------------------------------------------
    subtotal = sum_money((ln.amount for ln in lines), cur)
    lines.append(
        BudgetLine(
            category=BudgetCategory.BUFFER,
            amount=subtotal.scaled(BUFFER_RATE),
            basis="10% contingency for fees, tips and price changes",
            is_estimate=True,
        )
    )
    total = sum_money((ln.amount for ln in lines), cur)

    if fx.origin.value != "live":
        assumptions.append(f"Currency conversion used {fx.origin.value} rates from {fx.source}.")

    # ---- verdict ---------------------------------------------------------
    limit = trip.budget
    if limit is None:
        verdict, remaining, utilization = BudgetVerdict.NO_LIMIT, None, None
    else:
        limit = fx.convert(limit, cur)
        remaining = limit.amount - total.amount
        utilization = float(total.amount / limit.amount * 100) if limit.amount > 0 else None
        if total.amount > limit.amount:
            verdict = BudgetVerdict.OVER_BUDGET
        elif remaining < limit.amount * TIGHT_HEADROOM:
            verdict = BudgetVerdict.TIGHT
        else:
            verdict = BudgetVerdict.WITHIN_BUDGET

    return BudgetBreakdown(
        currency=cur,
        lines=lines,
        total=total,
        limit=limit,
        verdict=verdict,
        remaining=remaining,
        utilization_pct=utilization,
        assumptions=assumptions,
    )
