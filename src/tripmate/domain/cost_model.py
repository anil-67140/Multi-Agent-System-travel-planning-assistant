"""A small, transparent cost model used when no live price exists.

Everything here is an *estimate* and is labelled as one wherever it reaches the
traveller. The figures are coarse regional multipliers, deliberately simple so
they can be explained (and replaced by a real data source) without surprises.
"""

from __future__ import annotations

from decimal import Decimal

from tripmate.schemas.trip import TravelStyle

# Cost of living relative to a global average of 1.0 (rough, for scaling only).
COUNTRY_COST_INDEX: dict[str, Decimal] = {
    code: Decimal(str(v))
    for code, v in {
        "CH": 1.9,
        "NO": 1.8,
        "IS": 1.8,
        "SG": 1.5,
        "US": 1.5,
        "AU": 1.5,
        "GB": 1.4,
        "AE": 1.3,
        "FR": 1.3,
        "DE": 1.2,
        "IT": 1.2,
        "NL": 1.3,
        "JP": 1.15,
        "CA": 1.3,
        "KR": 1.0,
        "ES": 1.0,
        "PT": 0.9,
        "GR": 0.9,
        "CN": 0.75,
        "BR": 0.75,
        "TR": 0.7,
        "MX": 0.7,
        "MY": 0.6,
        "ZA": 0.6,
        "KE": 0.6,
        "MA": 0.55,
        "ID": 0.55,
        "TH": 0.65,
        "LK": 0.5,
        "VN": 0.5,
        "EG": 0.45,
        "NP": 0.45,
        "IN": 0.45,
        "PK": 0.4,
        "BD": 0.4,
    }.items()
}

# Per person per day, in USD at index 1.0: food + local transport + activities.
DAILY_USD_PER_PERSON: dict[TravelStyle, Decimal] = {
    TravelStyle.BUDGET: Decimal(35),
    TravelStyle.MID_RANGE: Decimal(85),
    TravelStyle.LUXURY: Decimal(220),
}
DAILY_SPLIT = {
    "food": Decimal("0.40"),
    "local_transport": Decimal("0.15"),
    "activities": Decimal("0.45"),
}

# Per room per night, in USD at index 1.0.
HOTEL_USD_PER_NIGHT: dict[TravelStyle, Decimal] = {
    TravelStyle.BUDGET: Decimal(40),
    TravelStyle.MID_RANGE: Decimal(110),
    TravelStyle.LUXURY: Decimal(300),
}

BUFFER_RATE = Decimal("0.10")


def cost_index(country_code: str | None) -> Decimal:
    if not country_code:
        return Decimal(1)
    return COUNTRY_COST_INDEX.get(country_code.upper(), Decimal(1))


def rooms_needed(travelers: int) -> int:
    return (travelers + 1) // 2  # two per room
