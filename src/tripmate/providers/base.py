"""Provider interfaces. Everything external hides behind one of these Protocols,
so agents and tests never depend on a specific vendor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from tripmate.domain.money import FxRates
from tripmate.schemas.search import Cabin, FlightOption, WeatherReport
from tripmate.schemas.trip import Place


class HttpClient(Protocol):
    """Minimal async JSON-over-HTTP client. Implementations raise ``ProviderError``."""

    async def get_json(
        self,
        url: str,
        *,
        provider: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any: ...

    async def post_json(
        self,
        url: str,
        *,
        provider: str,
        json: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> Any: ...


class WebHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    title: str = Field(max_length=300)
    url: str = Field(max_length=2000)
    content: str = Field(default="", max_length=4000)
    score: float | None = None


@dataclass(frozen=True)
class Airport:
    iata: str
    name: str
    city: str
    country: str
    latitude: float
    longitude: float
    timezone: str


class AirportDirectory(Protocol):
    def get(self, iata: str) -> Airport | None: ...

    def nearby(
        self, latitude: float, longitude: float, radius_km: float
    ) -> list[tuple[Airport, float]]:
        """Airports within ``radius_km``, as ``(airport, distance_km)`` nearest-first."""
        ...


class Geocoder(Protocol):
    async def geocode(self, query: str) -> Place: ...


class WeatherProvider(Protocol):
    async def forecast(
        self, place: Place, start: date, end: date, today: date
    ) -> WeatherReport: ...


class FxProvider(Protocol):
    async def rates(self, base: str) -> FxRates: ...


class FlightProvider(Protocol):
    async def search(
        self,
        origin: Place,
        destination: Place,
        depart: date,
        return_date: date | None,
        travelers: int,
        cabin: Cabin,
    ) -> list[FlightOption]: ...


class WebSearchProvider(Protocol):
    async def search(self, query: str, max_results: int = 6) -> list[WebHit]: ...
