"""Price sanity checks.

These are *plausibility envelopes*, deliberately wide. They catch unit errors
(cents vs dollars, wrong currency, a missing digit) rather than judge deals.
"""

from __future__ import annotations

from decimal import Decimal

from tripmate.domain.money import FxRates, Money
from tripmate.schemas.search import Cabin, FlightOption, HotelOption
from tripmate.validation.issues import Issue, Severity

# USD per traveller per great-circle km (economy), plus a flat minimum.
FLIGHT_USD_PER_KM = (Decimal("0.02"), Decimal("1.50"))
FLIGHT_MIN_USD = Decimal("20")
CABIN_MULTIPLIER = {
    Cabin.ECONOMY: Decimal(1),
    Cabin.PREMIUM_ECONOMY: Decimal(2),
    Cabin.BUSINESS: Decimal(6),
}

HOTEL_USD_PER_NIGHT = (Decimal("5"), Decimal("3500"))


def to_usd(money: Money, fx: FxRates) -> Decimal | None:
    """USD value of ``money``, or ``None`` if the FX table cannot convert it."""
    try:
        return fx.convert(money, "USD").amount
    except (KeyError, ValueError):
        return None


def check_flight_price(option: FlightOption, distance_km: float, fx: FxRates) -> list[Issue]:
    usd = to_usd(option.price_per_traveler, fx)
    path = f"flight[{option.id}]"
    if usd is None:
        return [
            Issue(
                code="PRICE_UNCHECKED",
                message="Flight price could not be checked: no exchange rate for its currency.",
                severity=Severity.WARNING,
                path=path,
            )
        ]
    km = Decimal(str(max(distance_km, 1.0)))
    mult = CABIN_MULTIPLIER[option.cabin]
    low = max(FLIGHT_MIN_USD, FLIGHT_USD_PER_KM[0] * km)
    high = FLIGHT_USD_PER_KM[1] * km * mult + Decimal(200)
    if usd < low:
        return [
            Issue(
                code="PRICE_SUSPICIOUSLY_LOW",
                message=f"Flight price ~{usd:.0f} USD/person is implausibly low for {distance_km:.0f} km.",
                path=path,
            )
        ]
    if usd > high:
        return [
            Issue(
                code="PRICE_SUSPICIOUSLY_HIGH",
                message=f"Flight price ~{usd:.0f} USD/person is implausibly high for {distance_km:.0f} km.",
                severity=Severity.WARNING,
                path=path,
            )
        ]
    return []


def check_hotel_price(option: HotelOption, fx: FxRates) -> list[Issue]:
    usd = to_usd(option.nightly_price, fx)
    path = f"hotel[{option.id}]"
    if usd is None:
        return [
            Issue(
                code="PRICE_UNCHECKED",
                message="Hotel price could not be checked: no exchange rate for its currency.",
                severity=Severity.WARNING,
                path=path,
            )
        ]
    low, high = HOTEL_USD_PER_NIGHT
    if usd < low:
        return [
            Issue(
                code="PRICE_SUSPICIOUSLY_LOW",
                message=f"~{usd:.2f} USD/night is implausibly low.",
                path=path,
            )
        ]
    if usd > high:
        return [
            Issue(
                code="PRICE_SUSPICIOUSLY_HIGH",
                message=f"~{usd:.0f} USD/night is implausibly high.",
                severity=Severity.WARNING,
                path=path,
            )
        ]
    return []
