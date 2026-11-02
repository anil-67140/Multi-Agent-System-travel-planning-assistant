"""Hand-written test doubles (no mocking framework: they read like documentation)."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from typing import Any

from tests.helpers import NOW
from tripmate.domain.clock import FixedClock
from tripmate.domain.money import FxRates
from tripmate.errors import ProviderError
from tripmate.providers.base import Airport, WebHit
from tripmate.providers.flights import EstimatedFlightProvider
from tripmate.providers.toolbelt import Toolbelt, ToolbeltConfig
from tripmate.resilience import InMemoryTTLCache
from tripmate.schemas.search import Cabin, FlightOption, WeatherReport
from tripmate.schemas.trip import Place
from tripmate.validation.durations import great_circle_km


class FakeHttp:
    """Maps a URL substring to a canned response, an exception, or a sequence of them."""

    def __init__(self, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def _respond(self, url: str, payload: dict[str, Any] | None) -> Any:
        self.calls.append((url, payload))
        for needle, response in self.routes.items():
            if needle in url:
                if isinstance(response, list):
                    response = response.pop(0) if len(response) > 1 else response[0]
                if isinstance(response, BaseException):
                    raise response
                return response() if callable(response) else json.loads(json.dumps(response))
        raise AssertionError(f"unexpected request to {url}")

    async def get_json(
        self, url: str, *, provider: str, params: dict[str, Any] | None = None, headers: Any = None
    ) -> Any:
        return self._respond(url, params)

    async def post_json(
        self, url: str, *, provider: str, json: dict[str, Any], headers: Any = None
    ) -> Any:
        return self._respond(url, json)


AIRPORTS = [
    Airport(
        "KHI", "Jinnah International Airport", "Karachi", "PK", 24.9065, 67.1608, "Asia/Karachi"
    ),
    Airport(
        "HND", "Tokyo Haneda International Airport", "Tokyo", "JP", 35.5494, 139.7798, "Asia/Tokyo"
    ),
    Airport("NRT", "Narita International Airport", "Narita", "JP", 35.7647, 140.3864, "Asia/Tokyo"),
    Airport("DXB", "Dubai International Airport", "Dubai", "AE", 25.2532, 55.3657, "Asia/Dubai"),
    Airport("LHR", "Heathrow Airport", "London", "GB", 51.4700, -0.4543, "Europe/London"),
    Airport(
        "JFK",
        "John F Kennedy International Airport",
        "New York",
        "US",
        40.6413,
        -73.7781,
        "America/New_York",
    ),
    Airport(
        "SYD",
        "Sydney Kingsford Smith Airport",
        "Sydney",
        "AU",
        -33.9399,
        151.1753,
        "Australia/Sydney",
    ),
    Airport(
        "LHE",
        "Allama Iqbal International Airport",
        "Lahore",
        "PK",
        31.5216,
        74.4036,
        "Asia/Karachi",
    ),
]


class FakeAirports:
    def __init__(self, airports: list[Airport] | None = None) -> None:
        self._all = airports if airports is not None else AIRPORTS

    def get(self, iata: str) -> Airport | None:
        return next((a for a in self._all if a.iata == iata.upper()), None)

    def nearby(
        self, latitude: float, longitude: float, radius_km: float
    ) -> list[tuple[Airport, float]]:
        out = [
            (a, great_circle_km(latitude, longitude, a.latitude, a.longitude)) for a in self._all
        ]
        return sorted([p for p in out if p[1] <= radius_km], key=lambda p: p[1])


class ScriptedGeocoder:
    def __init__(self, places: dict[str, Place]) -> None:
        self.places = {k.lower(): v for k, v in places.items()}
        self.calls = 0

    async def geocode(self, query: str) -> Place:
        self.calls += 1
        try:
            return self.places[query.strip().lower()]
        except KeyError:
            raise ProviderError(
                "fake-geo", f"no location found for {query!r}", retryable=False
            ) from None


class ScriptedWeather:
    def __init__(self, report: WeatherReport | None = None, error: Exception | None = None) -> None:
        self.report, self.error, self.calls = report, error, 0

    async def forecast(self, place: Place, start: date, end: date, today: date) -> WeatherReport:
        self.calls += 1
        if self.error:
            raise self.error
        assert self.report is not None
        return self.report


class ScriptedFx:
    def __init__(self, rates: FxRates | None = None, error: Exception | None = None) -> None:
        self.rates_obj, self.error, self.calls = rates, error, 0

    async def rates(self, base: str) -> FxRates:
        self.calls += 1
        if self.error:
            raise self.error
        assert self.rates_obj is not None
        return self.rates_obj


class ScriptedFlights:
    def __init__(
        self, options: list[FlightOption] | None = None, error: Exception | None = None
    ) -> None:
        self.options, self.error, self.calls = options or [], error, 0

    async def search(
        self,
        origin: Place,
        destination: Place,
        depart: date,
        return_date: date | None,
        travelers: int,
        cabin: Cabin,
    ) -> list[FlightOption]:
        self.calls += 1
        if self.error:
            raise self.error
        return list(self.options)


class ScriptedSearch:
    def __init__(self, hits: list[WebHit] | None = None, error: Exception | None = None) -> None:
        self.hits, self.error, self.calls = hits or [], error, 0

    async def search(self, query: str, max_results: int = 6) -> list[WebHit]:
        self.calls += 1
        if self.error:
            raise self.error
        return list(self.hits)


FX = FxRates(
    base="USD",
    rates={"EUR": "0.92", "PKR": 280, "JPY": 150, "GBP": "0.79"},
    as_of=NOW,
    source="test-fx",
)


async def _no_sleep(_: float) -> None:
    return None


def make_toolbelt(
    *,
    geocoder: Any = None,
    weather: Any = None,
    fx: Any = None,
    flights: Any = None,
    web_search: Any = None,
    config: ToolbeltConfig | None = None,
    cache: InMemoryTTLCache | None = None,
    clock: FixedClock | None = None,
    sink: Callable[[str, str, Any], None] | None = None,
) -> Toolbelt:
    clock = clock or FixedClock(NOW)
    return Toolbelt(
        geocoder=geocoder or ScriptedGeocoder({}),
        weather=weather or ScriptedWeather(error=ProviderError("w", "down")),
        fx=fx or ScriptedFx(FX),
        flights=flights or EstimatedFlightProvider(FakeAirports(), clock),
        web_search=web_search,
        cache=cache or InMemoryTTLCache(),
        clock=clock,
        config=config or ToolbeltConfig(timeout=1.0, max_attempts=3, base_delay=0.01),
        sink=sink,
        sleep=_no_sleep,
        rng=lambda: 0.5,
    )


# --------------------------------------------------------------------------- LLM
from tripmate.llm.base import BaseGateway, Completion, LLMCallRecord  # noqa: E402


class ScriptedGateway(BaseGateway):
    """Returns canned replies in order (strings), or raises if an item is an Exception.

    An item may also be a callable ``(messages, task) -> str`` so workflow tests can
    answer by task name. ``calls`` and ``records`` let tests assert on prompts/metrics.
    """

    def __init__(
        self, replies: list[Any] | Callable[[list[dict[str, str]], str], str] | None = None
    ) -> None:
        self._replies = replies if replies is not None else []
        self.calls: list[dict[str, Any]] = []
        self.records: list[LLMCallRecord] = []

    def _observe(self, record: LLMCallRecord) -> None:
        self.records.append(record)

    async def _complete(
        self, messages, *, task, tier, temperature, max_tokens, json_mode, metadata
    ) -> Completion:
        self.calls.append(
            {
                "task": task,
                "tier": tier,
                "messages": list(messages),
                "json_mode": json_mode,
                "metadata": metadata,
            }
        )
        if callable(self._replies):
            text = self._replies(messages, task)
        else:
            if not self._replies:
                raise AssertionError(f"ScriptedGateway ran out of replies (task={task})")
            item = self._replies.pop(0)
            if isinstance(item, Exception):
                raise item
            text = item
        if not isinstance(text, str):
            text = json.dumps(text)
        return Completion(text=text, model="scripted")
