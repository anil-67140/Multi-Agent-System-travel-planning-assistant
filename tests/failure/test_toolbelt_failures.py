"""Failure-path tests: what happens when dependencies misbehave."""

from __future__ import annotations

import asyncio
from datetime import date

import pytest

from tests.fakes import (
    FX,
    ScriptedFlights,
    ScriptedFx,
    ScriptedGeocoder,
    ScriptedSearch,
    ScriptedWeather,
    make_toolbelt,
)
from tests.helpers import NOW, make_origin, make_place
from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef
from tripmate.errors import ProviderError
from tripmate.providers.base import WebHit
from tripmate.providers.toolbelt import ToolbeltConfig
from tripmate.resilience import InMemoryTTLCache
from tripmate.schemas.search import DailyWeather, WeatherReport

pytestmark = pytest.mark.anyio

SRC = SourceRef(
    provider="open-meteo", kind=DataKind.WEATHER_FORECAST, origin=DataOrigin.LIVE, retrieved_at=NOW
)
REPORT = WeatherReport(
    place_name="Tokyo",
    source=SRC,
    days=[DailyWeather(date=date(2026, 10, 25), temp_min_c=8, temp_max_c=16)],
)
TOKYO = make_place()


class Flaky:
    """Fails ``n`` times with ``exc`` then succeeds."""

    def __init__(self, n: int, exc: Exception) -> None:
        self.n, self.exc, self.calls = n, exc, 0

    async def forecast(self, place, start, end, today):
        self.calls += 1
        if self.calls <= self.n:
            raise self.exc
        return REPORT


class Hangs:
    calls = 0

    async def forecast(self, place, start, end, today):
        Hangs.calls += 1
        await asyncio.sleep(10)


class TestGeocoding:
    async def test_cached_after_first_lookup(self):
        geo = ScriptedGeocoder({"Tokyo": TOKYO})
        belt = make_toolbelt(geocoder=geo)
        assert (await belt.geocode("Tokyo")) == (await belt.geocode("  tokyo ")) == TOKYO
        assert geo.calls == 1

    async def test_unknown_place_raises_and_does_not_trip_the_breaker(self):
        geo = ScriptedGeocoder({})
        belt = make_toolbelt(
            geocoder=geo,
            config=ToolbeltConfig(timeout=1, max_attempts=3, base_delay=0, breaker_failures=2),
        )
        for _ in range(5):
            with pytest.raises(ProviderError):
                await belt.geocode("Atlantis")
        assert geo.calls == 5  # not retried (non-retryable) and never short-circuited
        assert belt.breaker("geocoding").state.value == "closed"


class TestWeather:
    async def test_transient_failures_are_retried(self):
        provider = Flaky(2, ProviderError("om", "503", retryable=True))
        result = await make_toolbelt(weather=provider).weather(
            TOKYO, date(2026, 10, 25), date(2026, 10, 25)
        )
        assert result.items == [REPORT] and not result.errors and provider.calls == 3

    async def test_retries_are_bounded_and_failure_degrades_gracefully(self):
        provider = Flaky(99, ProviderError("om", "503", retryable=True))
        result = await make_toolbelt(weather=provider).weather(
            TOKYO, date(2026, 10, 25), date(2026, 10, 25)
        )
        assert provider.calls == 3  # max_attempts, then give up
        assert result.degraded and result.items == []
        assert result.errors[0].provider == "open-meteo" and result.errors[0].retryable
        assert "live forecast" in result.note  # tells the traveller what to do instead

    async def test_non_retryable_error_is_not_retried(self):
        provider = Flaky(99, ProviderError("om", "bad request", retryable=False))
        result = await make_toolbelt(weather=provider).weather(
            TOKYO, date(2026, 10, 25), date(2026, 10, 25)
        )
        assert (
            provider.calls == 1
            and result.errors[0].code == "provider_error"
            and not result.errors[0].retryable
        )

    async def test_timeout_is_enforced_and_reported(self):
        Hangs.calls = 0
        belt = make_toolbelt(
            weather=Hangs(), config=ToolbeltConfig(timeout=0.01, max_attempts=2, base_delay=0)
        )
        result = await belt.weather(TOKYO, date(2026, 10, 25), date(2026, 10, 25))
        assert result.errors[0].code == "timeout" and Hangs.calls == 2

    async def test_breaker_opens_and_stops_calling_the_dead_provider(self):
        provider = Flaky(10_000, ProviderError("om", "503", retryable=True))
        belt = make_toolbelt(
            weather=provider,
            cache=InMemoryTTLCache(),
            config=ToolbeltConfig(
                timeout=1, max_attempts=1, base_delay=0, breaker_failures=3, breaker_recovery=60
            ),
        )
        for day in range(3):  # distinct keys so the cache cannot hide the calls
            await belt.weather(TOKYO, date(2026, 10, 25 + day), date(2026, 10, 25 + day))
        assert provider.calls == 3
        result = await belt.weather(TOKYO, date(2026, 10, 30), date(2026, 10, 30))
        assert provider.calls == 3  # breaker is open: the provider was NOT called
        assert result.errors[0].code == "circuit_open"

    async def test_success_is_cached_and_marked_as_cached(self):
        provider = Flaky(0, Exception("never"))
        belt = make_toolbelt(weather=provider)
        first = await belt.weather(TOKYO, date(2026, 10, 25), date(2026, 10, 25))
        second = await belt.weather(TOKYO, date(2026, 10, 25), date(2026, 10, 25))
        assert provider.calls == 1
        assert (
            first.sources[0].origin is DataOrigin.LIVE
            and second.sources[0].origin is DataOrigin.CACHED
        )
        assert (
            second.sources[0].retrieved_at == first.sources[0].retrieved_at
        )  # original timestamp kept

    async def test_concurrent_identical_requests_make_one_upstream_call(self):
        provider = Flaky(0, Exception("never"))
        belt = make_toolbelt(weather=provider)
        await asyncio.gather(
            *(belt.weather(TOKYO, date(2026, 10, 25), date(2026, 10, 25)) for _ in range(20))
        )
        assert provider.calls == 1

    async def test_climatology_marks_result_degraded(self):
        clim = REPORT.model_copy(update={"is_climatology": True})
        result = await make_toolbelt(weather=ScriptedWeather(clim)).weather(
            TOKYO, date(2027, 1, 1), date(2027, 1, 3)
        )
        assert result.degraded and result.items[0].is_climatology

    async def test_unexpected_exception_never_leaks_internals(self):
        result = await make_toolbelt(
            weather=ScriptedWeather(error=RuntimeError("secret db password=hunter2"))
        ).weather(TOKYO, date(2026, 10, 25), date(2026, 10, 25))
        assert "hunter2" not in result.errors[0].message and result.errors[0].code == "unexpected"


class TestFlightsAndSearch:
    async def test_flight_provider_failure_degrades(self):
        belt = make_toolbelt(
            flights=ScriptedFlights(error=ProviderError("f", "down", retryable=False))
        )
        r = await belt.flights(make_origin(), TOKYO, date(2026, 12, 1), None, 2)
        assert r.degraded and not r.items and r.errors[0].provider == "flights"

    async def test_flights_estimated_by_default_and_cached(self):
        belt = make_toolbelt()
        a = await belt.flights(make_origin(), TOKYO, date(2026, 12, 1), date(2026, 12, 8), 2)
        b = await belt.flights(make_origin(), TOKYO, date(2026, 12, 1), date(2026, 12, 8), 2)
        assert len(a.items) == 3 and not a.degraded
        assert a.items[0].source.origin is DataOrigin.ESTIMATE
        assert b.items[0].source.origin is DataOrigin.CACHED

    async def test_search_not_configured_is_a_degraded_not_an_error(self):
        r = await make_toolbelt(web_search=None).web_search("hotels in tokyo")
        assert r.degraded and not r.errors and "not configured" in r.note

    async def test_search_results_become_cited_sources(self):
        hits = [WebHit(title="Hotel A", url="https://a.example/h", content="from $90")]
        r = await make_toolbelt(web_search=ScriptedSearch(hits)).web_search("hotels")
        assert (
            r.items == hits
            and r.sources[0].url == "https://a.example/h"
            and r.sources[0].kind is DataKind.WEB_SOURCE
        )

    async def test_search_outage_degrades(self):
        r = await make_toolbelt(
            web_search=ScriptedSearch(error=ProviderError("t", "429", retryable=True))
        ).web_search("x")
        assert r.degraded and r.errors[0].provider == "tavily"


class TestFx:
    async def test_never_raises_and_falls_back_to_flagged_static_rates(self):
        belt = make_toolbelt(fx=ScriptedFx(error=RuntimeError("down")))
        rates = await belt.fx_rates()
        assert rates.origin is DataOrigin.ESTIMATE and rates.supports("PKR")

    async def test_live_rates_are_cached(self):
        fx = ScriptedFx(FX)
        belt = make_toolbelt(fx=fx)
        await belt.fx_rates()
        await belt.fx_rates()
        assert fx.calls == 1

    async def test_estimated_rates_are_not_left_in_the_cache(self):
        from tripmate.providers.fx import static_fx_rates

        fx = ScriptedFx(static_fx_rates())
        belt = make_toolbelt(fx=fx)
        await belt.fx_rates()
        await belt.fx_rates()
        assert fx.calls == 2  # retried next time instead of serving a stale fallback for 6 hours


class TestObservabilityHooks:
    async def test_events_reach_the_sink(self):
        events: list[tuple[str, str]] = []
        belt = make_toolbelt(
            weather=Flaky(1, ProviderError("om", "503")), sink=lambda e, n, f: events.append((e, n))
        )
        await belt.weather(TOKYO, date(2026, 10, 25), date(2026, 10, 25))
        assert ("retry", "weather") in events and ("cache_miss", "weather") in events
