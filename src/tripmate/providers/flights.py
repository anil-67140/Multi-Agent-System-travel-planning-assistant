"""Flight options.

There is no free, reliable, *priced* flight API, so the default provider is an
honest **estimator**: distance-based fares and durations, every option flagged
``price_is_estimate=True`` with ``origin=ESTIMATE``. It sits behind the
``FlightProvider`` protocol, so a real source (Duffel, Amadeus, Skyscanner ...)
can replace it without touching any agent.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from decimal import Decimal

from tripmate.domain.clock import Clock
from tripmate.domain.money import Money
from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef
from tripmate.errors import ProviderError
from tripmate.providers.base import Airport, AirportDirectory
from tripmate.schemas.search import Cabin, FlightOption
from tripmate.schemas.trip import Place
from tripmate.validation.durations import great_circle_km, typical_minutes
from tripmate.validation.timezones import local_to_utc

PROVIDER = "tripmate-estimator"
HINT_MAX_DISTANCE_KM = 150.0  # an LLM-suggested IATA code must be this close to the place
SEARCH_RADIUS_KM = (60.0, 150.0, 300.0)
CABIN_MULTIPLIER = {
    Cabin.ECONOMY: Decimal(1),
    Cabin.PREMIUM_ECONOMY: Decimal("1.7"),
    Cabin.BUSINESS: Decimal("3.2"),
}
ROUND_TRIP_FACTOR = Decimal("1.9")  # a return ticket is cheaper than two one-ways
PEAK_MONTHS = {6, 7, 8, 12}
LOW_MONTHS = {1, 2, 11}


def resolve_airport(place: Place, airports: AirportDirectory) -> Airport | None:
    """Pick the airport for a place.

    The LLM may suggest an IATA code (``place.iata``); we *verify* it exists and
    is near the place, because models confidently invent airport codes.
    """
    if place.iata:
        candidate = airports.get(place.iata)
        if candidate is not None:
            km = great_circle_km(
                place.latitude, place.longitude, candidate.latitude, candidate.longitude
            )
            if km <= HINT_MAX_DISTANCE_KM:
                return candidate
    for radius in SEARCH_RADIUS_KM:
        nearby = airports.nearby(place.latitude, place.longitude, radius)
        if nearby:
            same_city = [a for a in nearby if a[0].city.lower() == place.name.lower()]
            pool = same_city or nearby
            international = [a for a in pool if "international" in a[0].name.lower()]
            return (international or pool)[0][0]
    return None


def _per_km_rate(distance_km: float) -> Decimal:
    if distance_km < 1500:
        return Decimal("0.12")
    if distance_km < 4000:
        return Decimal("0.085")
    if distance_km < 9000:
        return Decimal("0.075")
    return Decimal("0.070")


def _season_factor(day: date) -> Decimal:
    if day.month in PEAK_MONTHS:
        return Decimal("1.20")
    if day.month in LOW_MONTHS:
        return Decimal("0.92")
    return Decimal(1)


class EstimatedFlightProvider:
    def __init__(self, airports: AirportDirectory, clock: Clock) -> None:
        self._airports = airports
        self._clock = clock

    async def search(
        self,
        origin: Place,
        destination: Place,
        depart: date,
        return_date: date | None,
        travelers: int,
        cabin: Cabin,
    ) -> list[FlightOption]:
        src_ap = resolve_airport(origin, self._airports)
        dst_ap = resolve_airport(destination, self._airports)
        if src_ap is None or dst_ap is None:
            missing = origin.name if src_ap is None else destination.name
            raise ProviderError(PROVIDER, f"no airport found near {missing}", retryable=False)
        if src_ap.iata == dst_ap.iata:
            raise ProviderError(
                PROVIDER, "origin and destination use the same airport", retryable=False
            )

        km = great_circle_km(src_ap.latitude, src_ap.longitude, dst_ap.latitude, dst_ap.longitude)
        one_way = (
            (Decimal(45) + Decimal(str(km)) * _per_km_rate(km))
            * _season_factor(depart)
            * CABIN_MULTIPLIER[cabin]
        )
        base = one_way * ROUND_TRIP_FACTOR if return_date else one_way

        # (label, stops, extra layover minutes, price factor)
        shapes: list[tuple[str, int, int, Decimal]] = (
            [
                ("direct", 0, 0, Decimal("1.10")),
                ("1-stop", 1, 0, Decimal("0.82")),
                ("1-stop-long-layover", 1, 240, Decimal("0.72")),
            ]
            if km <= 9500
            else [("1-stop", 1, 0, Decimal("0.95")), ("2-stop", 2, 0, Decimal("0.78"))]
        )

        now = self._clock.now()
        source = SourceRef(
            provider=PROVIDER,
            kind=DataKind.FLIGHT_PRICE,
            origin=DataOrigin.ESTIMATE,
            retrieved_at=now,
            title="Distance-based fare estimate",
            note="Indicative only. Check airlines for live fares and schedules.",
        )
        departure_utc = local_to_utc(
            datetime.combine(depart, datetime.min.time()).replace(hour=9), src_ap.timezone
        )

        options: list[FlightOption] = []
        for n, (label, stops, extra, factor) in enumerate(shapes, start=1):
            minutes = typical_minutes(km, stops) + extra
            options.append(
                FlightOption(
                    id=f"{src_ap.iata}-{dst_ap.iata}-{depart:%Y%m%d}-{n}",
                    origin_iata=src_ap.iata,
                    destination_iata=dst_ap.iata,
                    depart_date=depart,
                    return_date=return_date,
                    departure=departure_utc,
                    arrival=departure_utc + timedelta(minutes=minutes),
                    duration_minutes=minutes,
                    stops=stops,
                    cabin=cabin,
                    travelers=travelers,
                    price_per_traveler=Money.of(base * factor, "USD"),
                    price_is_estimate=True,
                    schedule_is_indicative=True,
                    source=source,
                    notes=[f"{label}; ~{km:,.0f} km"],
                )
            )
        return options


class AirportsDataDirectory:
    """``airportsdata``-backed directory (IATA-keyed, offline, ~7k airports)."""

    def __init__(self) -> None:
        import airportsdata

        self._by_iata: dict[str, Airport] = {}
        for code, row in airportsdata.load("IATA").items():
            if not row.get("tz") or row.get("lat") is None or row.get("lon") is None:
                continue
            self._by_iata[code] = Airport(
                iata=code,
                name=row.get("name", ""),
                city=row.get("city", ""),
                country=row.get("country", ""),
                latitude=float(row["lat"]),
                longitude=float(row["lon"]),
                timezone=row["tz"],
            )

    def get(self, iata: str) -> Airport | None:
        return self._by_iata.get(iata.upper())

    def nearby(
        self, latitude: float, longitude: float, radius_km: float
    ) -> list[tuple[Airport, float]]:
        # Cheap bounding-box prefilter (1 deg latitude ~ 111 km) before haversine.
        dlat = radius_km / 111.0
        out: list[tuple[Airport, float]] = []
        for a in self._by_iata.values():
            if abs(a.latitude - latitude) > dlat:
                continue
            km = great_circle_km(latitude, longitude, a.latitude, a.longitude)
            if km <= radius_km and not math.isnan(km):
                out.append((a, km))
        return sorted(out, key=lambda pair: pair[1])
