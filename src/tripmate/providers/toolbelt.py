"""The resilient facade that agents use to reach the outside world.

Agents never call a provider directly. The toolbelt adds, uniformly:

* bounded concurrency (bulkhead, sheds load rather than queueing forever)
* per-provider circuit breakers (stop hammering a dead dependency)
* bounded retries with exponential backoff + jitter, honouring ``Retry-After``
* a per-attempt timeout
* TTL caching with single-flight (no thundering herd on a cold key)
* graceful degradation: failures become :class:`ToolError` records inside a
  :class:`SearchResult`, so one dead provider never kills the whole plan
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from tripmate.domain.clock import Clock
from tripmate.domain.money import FxRates
from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef
from tripmate.errors import ProviderError
from tripmate.providers.base import (
    FlightProvider,
    FxProvider,
    Geocoder,
    WeatherProvider,
    WebHit,
    WebSearchProvider,
)
from tripmate.resilience import (
    Bulkhead,
    BulkheadFullError,
    Cache,
    CircuitBreaker,
    CircuitOpenError,
    OperationTimeout,
    ResiliencePolicy,
    RetryPolicy,
    SingleFlight,
    get_or_load,
)
from tripmate.schemas.search import Cabin, FlightOption, SearchResult, ToolError, WeatherReport
from tripmate.schemas.trip import Place

log = logging.getLogger(__name__)
T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)
Sink = Callable[[str, str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class ToolbeltConfig:
    timeout: float = 10.0
    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 6.0
    max_concurrency: int = 8
    max_waiting: int = 32
    breaker_failures: int = 5
    breaker_recovery: float = 30.0
    geocode_ttl: float = 30 * 24 * 3600
    weather_ttl: float = 3 * 3600
    fx_ttl: float = 6 * 3600
    flights_ttl: float = 30 * 60
    search_ttl: float = 24 * 3600


def _counts_against_breaker(exc: BaseException) -> bool:
    """Only dependency health problems trip a breaker; a 404 'no such city' does not."""
    if isinstance(exc, OperationTimeout):
        return True
    if isinstance(exc, ProviderError):
        return exc.retryable
    return not isinstance(exc, ValidationError | ValueError)


def to_tool_error(provider: str, exc: BaseException, now: Any) -> ToolError:
    """Translate any failure into a safe, structured record (no stack traces)."""
    if isinstance(exc, OperationTimeout):
        code, retryable, msg = "timeout", True, "The provider took too long to respond."
    elif isinstance(exc, CircuitOpenError):
        code, retryable, msg = "circuit_open", True, "The provider is temporarily unavailable."
    elif isinstance(exc, BulkheadFullError):
        code, retryable, msg = "overloaded", True, "Too many concurrent lookups; try again shortly."
    elif isinstance(exc, ProviderError):
        code, retryable, msg = (
            "provider_error",
            exc.retryable,
            str(exc.detail or "Provider request failed.")[:300],
        )
    elif isinstance(exc, ValidationError):
        code, retryable, msg = (
            "invalid_response",
            False,
            "The provider returned data we could not use.",
        )
    else:
        code, retryable, msg = (
            "unexpected",
            False,
            "An unexpected error occurred while fetching data.",
        )
        log.exception("unexpected tool failure", extra={"provider": provider})
    return ToolError(
        provider=provider, code=code, message=msg, retryable=retryable, occurred_at=now
    )


def _cached_copy(source: SourceRef) -> SourceRef:
    return source.model_copy(update={"origin": DataOrigin.CACHED})


class Toolbelt:
    def __init__(
        self,
        *,
        geocoder: Geocoder,
        weather: WeatherProvider,
        fx: FxProvider,
        flights: FlightProvider,
        web_search: WebSearchProvider | None,
        cache: Cache,
        clock: Clock,
        config: ToolbeltConfig | None = None,
        sink: Sink | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        self._geocoder, self._weather, self._fx, self._flights, self._search = (
            geocoder,
            weather,
            fx,
            flights,
            web_search,
        )
        self._cache, self._clock = cache, clock
        self._cfg = config or ToolbeltConfig()
        self._sink: Sink = sink or (lambda event, name, fields: None)
        self._sleep, self._rng = sleep, rng
        self._single_flight = SingleFlight()
        self._bulkhead = Bulkhead(
            "tools", self._cfg.max_concurrency, max_waiting=self._cfg.max_waiting
        )
        self._breakers: dict[str, CircuitBreaker] = {}

    # ------------------------------------------------------------------ plumbing
    def breaker(self, name: str) -> CircuitBreaker:
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name,
                failure_threshold=self._cfg.breaker_failures,
                recovery_timeout=self._cfg.breaker_recovery,
                is_failure=_counts_against_breaker,
                on_state_change=lambda n, s: self._sink("breaker_state", n, {"state": s.value}),
            )
        return self._breakers[name]

    def _policy(self, name: str) -> ResiliencePolicy:
        return ResiliencePolicy(
            name,
            timeout=self._cfg.timeout,
            retry=RetryPolicy(self._cfg.max_attempts, self._cfg.base_delay, self._cfg.max_delay),
            breaker=self.breaker(name),
            bulkhead=self._bulkhead,
            sink=self._sink,
        )

    async def _call(self, name: str, fn: Callable[[], Awaitable[T]]) -> T:
        return await self._policy(name).execute(fn, sleep=self._sleep, rng=self._rng)

    async def _cached(
        self, name: str, key: str, ttl: float, loader: Callable[[], Awaitable[Any]]
    ) -> tuple[Any, bool]:
        value, hit = await get_or_load(
            self._cache,
            key,
            ttl,
            loader,
            single_flight=self._single_flight,
            on_result=lambda h: self._sink("cache_hit" if h else "cache_miss", name, {}),
        )
        return value, hit

    # ------------------------------------------------------------------- geocode
    async def geocode(self, query: str) -> Place:
        """Resolve a place name. Raises ``ProviderError`` if it cannot be found."""
        key = f"geo:{query.strip().lower()}"

        async def load() -> Any:
            place = await self._call("geocoding", lambda: self._geocoder.geocode(query))
            return place.model_dump(mode="json")

        data, _ = await self._cached("geocoding", key, self._cfg.geocode_ttl, load)
        return Place.model_validate(data)

    # ------------------------------------------------------------------- weather
    async def weather(self, place: Place, start: date, end: date) -> SearchResult[WeatherReport]:
        today = self._clock.now().date()
        key = f"wx:{place.latitude:.2f},{place.longitude:.2f}:{start}:{end}:{today}"

        async def load() -> Any:
            report = await self._call(
                "weather", lambda: self._weather.forecast(place, start, end, today)
            )
            return report.model_dump(mode="json")

        try:
            data, hit = await self._cached("weather", key, self._cfg.weather_ttl, load)
            report = WeatherReport.model_validate(data)
            if hit:
                report = report.model_copy(update={"source": _cached_copy(report.source)})
        except Exception as exc:
            return self._failed(
                "open-meteo", exc, "Weather unavailable; check a live forecast before you travel."
            )
        return SearchResult[WeatherReport](
            items=[report], sources=[report.source], degraded=report.is_climatology
        )

    # ------------------------------------------------------------------------ fx
    async def fx_rates(self, base: str = "USD") -> FxRates:
        """Never raises: falls back to a static (flagged-as-estimate) table."""

        async def load() -> Any:
            rates = await self._fx.rates(base)
            return rates.model_dump(mode="json")

        try:
            data, _ = await self._cached("fx", f"fx:{base}", self._cfg.fx_ttl, load)
            rates = FxRates.model_validate(data)
        except Exception:
            log.exception("fx lookup failed; using static table")
            from tripmate.providers.fx import static_fx_rates

            return static_fx_rates()
        if rates.origin is DataOrigin.ESTIMATE:
            await self._cache.delete(
                f"fx:{base}"
            )  # do not let a fallback table linger in the cache
        return rates

    # ------------------------------------------------------------------- flights
    async def flights(
        self,
        origin: Place,
        destination: Place,
        depart: date,
        return_date: date | None,
        travelers: int,
        cabin: Cabin = Cabin.ECONOMY,
    ) -> SearchResult[FlightOption]:
        key = f"fl:{origin.iata or origin.name}:{destination.iata or destination.name}:{depart}:{return_date}:{travelers}:{cabin.value}"

        async def load() -> Any:
            options = await self._call(
                "flights",
                lambda: self._flights.search(
                    origin, destination, depart, return_date, travelers, cabin
                ),
            )
            return [o.model_dump(mode="json") for o in options]

        try:
            data, hit = await self._cached("flights", key, self._cfg.flights_ttl, load)
            options = [FlightOption.model_validate(d) for d in data]
            if hit:
                options = [o.model_copy(update={"source": _cached_copy(o.source)}) for o in options]
        except Exception as exc:
            return self._failed("flights", exc, "Flight options unavailable.")
        if not options:
            return SearchResult[FlightOption](
                degraded=True, note="No flight options were found for this route."
            )
        return SearchResult[FlightOption](items=options, sources=[options[0].source])

    # --------------------------------------------------------------- web search
    async def web_search(self, query: str, max_results: int = 6) -> SearchResult[WebHit]:
        if self._search is None:
            return SearchResult[WebHit](degraded=True, note="Web search is not configured.")
        key = f"ws:{query.strip().lower()}:{max_results}"

        async def load() -> Any:
            hits = await self._call("web_search", lambda: self._search.search(query, max_results))  # type: ignore[union-attr]
            return [h.model_dump(mode="json") for h in hits]

        try:
            data, _ = await self._cached("web_search", key, self._cfg.search_ttl, load)
            hits = [WebHit.model_validate(d) for d in data]
        except Exception as exc:
            return self._failed("tavily", exc, "Live search unavailable.")
        now = self._clock.now()
        sources = [
            SourceRef(
                provider="tavily",
                kind=DataKind.WEB_SOURCE,
                origin=DataOrigin.LIVE,
                retrieved_at=now,
                url=h.url,
                title=h.title[:300],
            )
            for h in hits
        ]
        return SearchResult[WebHit](items=hits, sources=sources)

    # ------------------------------------------------------------------- errors
    def _failed(self, provider: str, exc: BaseException, note: str) -> SearchResult[Any]:
        err = to_tool_error(provider, exc, self._clock.now())
        log.warning("tool degraded", extra={"provider": provider, "code": err.code})
        return SearchResult(errors=[err], degraded=True, note=note)
