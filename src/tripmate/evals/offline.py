"""Deterministic stand-ins so evals run with no network, keys or quota (CI-friendly).

Offline mode measures *our* pipeline (resolution, validation, optimisation, budgeting, scoring) against a
reference planner. Live mode (``--live``) swaps in the real gateway and providers to measure the model.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from typing import Any

from tripmate.domain.clock import FixedClock
from tripmate.errors import ProviderError
from tripmate.llm.base import BaseGateway, Completion
from tripmate.providers.base import Airport
from tripmate.providers.flights import EstimatedFlightProvider
from tripmate.providers.fx import static_fx_rates
from tripmate.providers.toolbelt import Toolbelt, ToolbeltConfig
from tripmate.resilience import InMemoryTTLCache
from tripmate.schemas.trip import Place
from tripmate.validation import great_circle_km

EVAL_NOW = datetime(2026, 10, 22, 4, 0, tzinfo=UTC)

_CITIES = {
    # name: (country, lat, lon, tz, iata)
    "tokyo": ("JP", 35.6762, 139.6503, "Asia/Tokyo", "HND"),
    "karachi": ("PK", 24.8607, 67.0011, "Asia/Karachi", "KHI"),
    "lahore": ("PK", 31.5204, 74.3587, "Asia/Karachi", "LHE"),
    "dubai": ("AE", 25.2048, 55.2708, "Asia/Dubai", "DXB"),
    "london": ("GB", 51.5072, -0.1276, "Europe/London", "LHR"),
    "istanbul": ("TR", 41.0082, 28.9784, "Europe/Istanbul", "IST"),
    "lisbon": ("PT", 38.7223, -9.1393, "Europe/Lisbon", "LIS"),
    "paris": ("FR", 48.8566, 2.3522, "Europe/Paris", "CDG"),
    "new york": ("US", 40.7128, -74.0060, "America/New_York", "JFK"),
    "singapore": ("SG", 1.3521, 103.8198, "Asia/Singapore", "SIN"),
    "bangkok": ("TH", 13.7563, 100.5018, "Asia/Bangkok", "BKK"),
    "rome": ("IT", 41.9028, 12.4964, "Europe/Rome", "FCO"),
}
_AIRPORTS = {
    "HND": (35.5494, 139.7798),
    "KHI": (24.9065, 67.1608),
    "LHE": (31.5216, 74.4036),
    "DXB": (25.2532, 55.3657),
    "LHR": (51.47, -0.4543),
    "IST": (41.2753, 28.7519),
    "LIS": (38.7742, -9.1342),
    "CDG": (49.0097, 2.5479),
    "JFK": (40.6413, -73.7781),
    "SIN": (1.3644, 103.9915),
    "BKK": (13.69, 100.7501),
    "FCO": (41.8003, 12.2389),
}


class StaticAirports:
    def __init__(self) -> None:
        self._all = [
            Airport(i, f"{i} Airport", i, "", lat, lon, "UTC")
            for i, (lat, lon) in _AIRPORTS.items()
        ]

    def get(self, iata: str) -> Airport | None:
        return next((a for a in self._all if a.iata == iata.upper()), None)

    def nearby(
        self, latitude: float, longitude: float, radius_km: float
    ) -> list[tuple[Airport, float]]:
        pairs = [
            (a, great_circle_km(latitude, longitude, a.latitude, a.longitude)) for a in self._all
        ]
        return sorted([p for p in pairs if p[1] <= radius_km], key=lambda p: p[1])


class StaticGeocoder:
    async def geocode(self, query: str) -> Place:
        row = _CITIES.get(query.strip().lower())
        if row is None:
            raise ProviderError("static-geo", f"no location found for {query!r}", retryable=False)
        cc, lat, lon, tz, iata = row
        return Place(
            name=query.strip().title(),
            country_code=cc,
            latitude=lat,
            longitude=lon,
            timezone=tz,
            iata=iata,
        )


class _NoWeather:
    async def forecast(self, *_: Any, **__: Any) -> Any:
        raise ProviderError("static-weather", "offline eval has no weather", retryable=False)


class _StaticFx:
    async def rates(self, base: str) -> Any:
        return static_fx_rates()


async def _no_sleep(_: float) -> None:
    return None


def offline_toolbelt() -> tuple[Toolbelt, StaticAirports]:
    clock = FixedClock(EVAL_NOW)
    airports = StaticAirports()
    return Toolbelt(
        geocoder=StaticGeocoder(),
        weather=_NoWeather(),
        fx=_StaticFx(),
        flights=EstimatedFlightProvider(airports, clock),
        web_search=None,
        cache=InMemoryTTLCache(),
        clock=clock,
        config=ToolbeltConfig(timeout=2.0, max_attempts=1),
        sleep=_no_sleep,
    ), airports


# --------------------------------------------------------------------------- reference planner
_BY_INTEREST = {
    "food": ("food", "Local food tasting in {city}", False),
    "temples": ("culture", "Visit a historic temple in {city}", False),
    "museums": ("culture", "Museum visit in {city}", False),
    "history": ("culture", "Historic quarter walk in {city}", True),
    "nature": ("nature", "Nature walk near {city}", True),
    "shopping": ("shopping", "Market and shopping stroll in {city}", False),
    "beaches": ("nature", "Beach afternoon near {city}", True),
    "nightlife": ("nightlife", "Evening out in {city}", False),
}
_FILLER = [
    ("sightseeing", "Landmarks tour of {city}", True),
    ("leisure", "Cafe and people-watching in {city}", False),
    ("culture", "Neighbourhood exploration in {city}", True),
    ("rest", "Rest at the hotel", False),
]


def reference_itinerary(trip: dict[str, Any]) -> dict[str, Any]:
    """A deliberately plain planner that always satisfies the hard rules. It is the eval baseline."""
    city = trip["destination"]["name"]
    start = date.fromisoformat(trip["start_date"])
    n = (date.fromisoformat(trip["end_date"]) - start).days + 1
    interests = [i.lower() for i in trip.get("interests", [])]
    per_day = {"relaxed": 3, "balanced": 4, "packed": 6}[trip.get("pace", "balanced")]
    slots = [
        ("09:00", "10:30"),
        ("11:00", "12:00"),
        ("14:00", "15:30"),
        ("16:00", "17:30"),
        ("18:00", "19:30"),
        ("20:00", "21:00"),
    ]
    days = []
    for d in range(n):
        picks: list[tuple[str, str, bool]] = []
        for k, interest in enumerate(interests):
            due = (d + k) % max(len(interests), 1) == 0 or len(interests) <= 2
            if due and interest in _BY_INTEREST:
                picks.append(_BY_INTEREST[interest])
        picks += [_FILLER[(d + j) % len(_FILLER)] for j in range(per_day)]
        acts: list[dict[str, Any]] = []
        meal_done = False
        for idx in range(per_day):
            cat, title, outdoor = picks[idx]
            if idx == 1 and not any(a["category"] == "food" for a in acts):
                cat, title, outdoor = "food", f"Lunch at a local restaurant in {city}", False
            meal_done = meal_done or cat == "food"
            s, e = slots[idx]
            acts.append(
                {
                    "start": s,
                    "end": e,
                    "title": title.format(city=city) + (f" (day {d + 1})" if idx > 1 else ""),
                    "description": "",
                    "category": cat,
                    "outdoor": outdoor,
                    "estimated_cost": None,
                }
            )
        if not meal_done:
            acts[1]["category"], acts[1]["title"] = "food", f"Lunch in {city}"
        days.append(
            {
                "day_number": d + 1,
                "date": (start + timedelta(days=d)).isoformat(),
                "theme": f"Day {d + 1} in {city}",
                "activities": acts,
            }
        )
    return {"title": f"{n} days in {city}", "summary": "Reference plan", "days": days, "tips": []}


class ReferenceGateway(BaseGateway):
    """Answers each task deterministically from the case fixture (no model involved)."""

    def __init__(self, intent: dict[str, Any], guard_allowed: bool = True) -> None:
        self.intent, self.guard_allowed = intent, guard_allowed
        self.trip: dict[str, Any] | None = None
        self.calls = 0

    async def _complete(  # type: ignore[no-untyped-def]
        self, messages, *, task, tier, temperature, max_tokens, json_mode, metadata
    ) -> Completion:
        self.calls += 1
        if task == "guardrail":
            body: Any = {
                "allowed": self.guard_allowed,
                "category": "travel" if self.guard_allowed else "off_topic",
                "reason": "" if self.guard_allowed else "off topic",
            }
        elif task == "intake":
            body = self.intent
        elif task == "hotel_extract":
            body = {"hotels": []}
        elif task == "itinerary":
            brief = messages[-1]["content"]
            body = reference_itinerary(_brief_to_trip(brief, self.trip))
        else:
            raise AssertionError(f"unexpected task {task}")
        return Completion(text=json.dumps(body), model="reference")


def _brief_to_trip(brief: str, trip: dict[str, Any] | None) -> dict[str, Any]:
    assert trip is not None, "pipeline must publish the trip to the reference gateway"
    return trip
