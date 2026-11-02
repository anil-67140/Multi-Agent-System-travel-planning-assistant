from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import pytest

from tests.fakes import FX, FakeAirports, FakeHttp
from tests.helpers import NOW, TODAY, make_place
from tripmate.domain.clock import FixedClock
from tripmate.domain.provenance import DataKind, DataOrigin
from tripmate.errors import ProviderError
from tripmate.providers.flights import EstimatedFlightProvider, resolve_airport
from tripmate.providers.fx import (
    ExchangeRateApiClient,
    FallbackFxProvider,
    FrankfurterClient,
    static_fx_rates,
)
from tripmate.providers.open_meteo import OpenMeteoClient
from tripmate.providers.tavily import TavilySearchClient
from tripmate.schemas.search import Cabin
from tripmate.validation import (
    check_flight_duration,
    check_flight_price,
    check_flight_times,
    great_circle_km,
    has_errors,
)
from tripmate.validation.freshness import Freshness, assess_freshness

pytestmark = pytest.mark.anyio


# --------------------------------------------------------------------------- Open-Meteo
GEO_OK = {
    "results": [
        {
            "name": "Tokyo",
            "latitude": 35.6895,
            "longitude": 139.69171,
            "country_code": "JP",
            "timezone": "Asia/Tokyo",
            "admin1": "Tokyo",
        }
    ]
}


def daily(days, **cols):
    base = {
        "time": days,
        "temperature_2m_max": [12.0] * len(days),
        "temperature_2m_min": [4.0] * len(days),
        "precipitation_sum": [0.0] * len(days),
        "precipitation_probability_max": [10] * len(days),
        "weather_code": [1] * len(days),
        "wind_speed_10m_max": [14.0] * len(days),
    }
    base.update(cols)
    return {"timezone": "Asia/Tokyo", "daily": base}


class TestOpenMeteo:
    def client(self, routes):
        return OpenMeteoClient(FakeHttp(routes), FixedClock(NOW)), routes

    async def test_geocode(self):
        c, _ = self.client({"geocoding-api": GEO_OK})
        p = await c.geocode(" Tokyo ")
        assert (p.name, p.country_code, p.timezone) == ("Tokyo", "JP", "Asia/Tokyo")

    async def test_geocode_not_found_is_non_retryable(self):
        c, _ = self.client({"geocoding-api": {"generationtime_ms": 0.3}})
        with pytest.raises(ProviderError) as e:
            await c.geocode("Atlantis")
        assert e.value.retryable is False

    async def test_geocode_garbled_response(self):
        c, _ = self.client({"geocoding-api": {"results": [{"name": "X"}]}})
        with pytest.raises(ProviderError):
            await c.geocode("x")

    async def test_live_forecast_inside_horizon(self):
        days = ["2026-10-25", "2026-10-26"]
        payload = daily(days, weather_code=[63, 0], precipitation_sum=[12.5, 0.0])
        c, routes = self.client({"api.open-meteo.com/v1/forecast": payload})
        r = await c.forecast(make_place(), date(2026, 10, 25), date(2026, 10, 26), TODAY)
        assert not r.is_climatology and r.source.kind is DataKind.WEATHER_FORECAST
        assert [d.condition for d in r.days] == ["rain", "clear sky"]
        assert r.days[0].is_wet and not r.days[1].is_wet
        url, params = c._http.calls[0]
        assert params["start_date"] == "2026-10-25" and params["timezone"] == "auto"

    async def test_far_future_uses_last_years_observations_and_says_so(self):
        payload = daily(["2025-12-01", "2025-12-02"])
        c, _ = self.client({"archive-api": payload})
        r = await c.forecast(make_place(), date(2026, 12, 1), date(2026, 12, 2), TODAY)
        assert r.is_climatology and r.source.kind is DataKind.WEATHER_CLIMATOLOGY
        assert [d.date for d in r.days] == [
            date(2026, 12, 1),
            date(2026, 12, 2),
        ]  # shifted back to this year
        assert "last year" in r.source.note
        _, params = c._http.calls[0]
        assert params["start_date"] == "2025-12-01"
        assert "precipitation_probability_max" not in params["daily"]  # not offered by the archive

    async def test_partially_in_horizon_is_treated_as_climatology(self):
        c, _ = self.client({"archive-api": daily(["2026-10-30"])})
        r = await c.forecast(make_place(), date(2026, 10, 30), date(2026, 11, 20), TODAY)
        assert r.is_climatology

    async def test_leap_day_shift_does_not_crash(self):
        c, _ = self.client(
            {"archive-api": daily(["2027-02-28"])}
        )  # trip starts 2028-02-29 -> 2027-02-28
        r = await c.forecast(make_place(), date(2028, 2, 29), date(2028, 2, 29), date(2028, 2, 1))
        assert r.days[0].date == date(2028, 2, 28)

    async def test_days_without_temperatures_are_skipped_not_invented(self):
        payload = daily(
            ["2026-10-25", "2026-10-26"],
            temperature_2m_max=[None, 15.0],
            temperature_2m_min=[None, 7.0],
        )
        c, _ = self.client({"api.open-meteo.com/v1/forecast": payload})
        r = await c.forecast(make_place(), date(2026, 10, 25), date(2026, 10, 26), TODAY)
        assert [d.date for d in r.days] == [date(2026, 10, 26)]

    async def test_all_days_missing_is_an_error(self):
        payload = daily(["2026-10-25"], temperature_2m_max=[None], temperature_2m_min=[None])
        c, _ = self.client({"api.open-meteo.com/v1/forecast": payload})
        with pytest.raises(ProviderError):
            await c.forecast(make_place(), date(2026, 10, 25), date(2026, 10, 25), TODAY)

    async def test_error_payload(self):
        c, _ = self.client({"api.open-meteo.com/v1/forecast": {"error": True, "reason": "bad"}})
        with pytest.raises(ProviderError):
            await c.forecast(make_place(), date(2026, 10, 25), date(2026, 10, 25), TODAY)


# --------------------------------------------------------------------------- FX
ER_OK = {
    "result": "success",
    "base_code": "USD",
    "time_last_update_unix": 1792627200,
    "rates": {"USD": 1, "PKR": 281.5, "EUR": 0.91, "XXX": 5, "ZERO": 0},
}


class TestFx:
    async def test_er_api_parsed_and_unknown_currencies_dropped(self):
        fx = await ExchangeRateApiClient(FakeHttp({"open.er-api.com": ER_OK})).rates("USD")
        assert fx.rate("PKR") == Decimal("281.5") and "XXX" not in fx.rates
        assert (
            fx.as_of == datetime.fromtimestamp(1792627200, tz=UTC) and fx.origin is DataOrigin.LIVE
        )

    async def test_er_api_error_result(self):
        with pytest.raises(ProviderError):
            await ExchangeRateApiClient(
                FakeHttp({"open.er-api.com": {"result": "error", "error-type": "x"}})
            ).rates("USD")

    async def test_frankfurter(self):
        fx = await FrankfurterClient(
            FakeHttp({"frankfurter": {"base": "USD", "date": "2026-10-21", "rates": {"EUR": 0.9}}})
        ).rates("USD")
        assert fx.rate("EUR") == Decimal("0.9") and fx.as_of == datetime(2026, 10, 21, tzinfo=UTC)

    async def test_fallback_chain_uses_first_healthy_provider(self):
        primary = ExchangeRateApiClient(FakeHttp({"open.er-api.com": ProviderError("er", "boom")}))
        secondary = FrankfurterClient(
            FakeHttp({"frankfurter": {"base": "USD", "date": "2026-10-21", "rates": {"EUR": 0.9}}})
        )
        fx = await FallbackFxProvider([primary, secondary], FixedClock(NOW)).rates("USD")
        assert fx.source == "frankfurter.dev"

    async def test_all_failing_returns_flagged_static_table(self):
        primary = ExchangeRateApiClient(FakeHttp({"open.er-api.com": ProviderError("er", "boom")}))
        fx = await FallbackFxProvider([primary], FixedClock(NOW)).rates("USD")
        assert fx.origin is DataOrigin.ESTIMATE and fx.supports("PKR")
        # and the freshness policy refuses to call it current
        assert assess_freshness(fx.as_of, DataKind.FX_RATE, NOW) is Freshness.EXPIRED

    def test_static_table_covers_every_supported_currency_except_base(self):
        from tripmate.domain.money import CURRENCY_EXPONENT

        missing = set(CURRENCY_EXPONENT) - set(static_fx_rates().rates) - {"USD"}
        assert missing == set()


# --------------------------------------------------------------------------- flights
class TestAirportResolution:
    airports = FakeAirports()

    def test_valid_hint_is_used(self):
        assert resolve_airport(make_place("Tokyo", iata="NRT"), self.airports).iata == "NRT"

    def test_hallucinated_hint_far_from_the_place_is_ignored(self):
        # The model claims Tokyo's airport is JFK. It is 10,000 km away -> rejected, nearest used.
        assert resolve_airport(make_place("Tokyo", iata="JFK"), self.airports).iata == "HND"

    def test_unknown_hint_falls_back_to_nearest(self):
        assert resolve_airport(make_place("Tokyo", iata="ZZZ"), self.airports).iata == "HND"

    def test_prefers_airport_matching_the_city(self):
        p = make_place(
            "Lahore",
            country_code="PK",
            latitude=31.55,
            longitude=74.34,
            timezone="Asia/Karachi",
            iata=None,
        )
        assert resolve_airport(p, self.airports).iata == "LHE"

    def test_nothing_nearby(self):
        mid_ocean = make_place(
            "Nowhere", latitude=-45, longitude=-130, timezone="Pacific/Pitcairn", iata=None
        )
        assert resolve_airport(mid_ocean, self.airports) is None


class TestFlightEstimator:
    def provider(self):
        return EstimatedFlightProvider(FakeAirports(), FixedClock(NOW))

    async def search(self, o="Karachi", d="Tokyo", **kw):
        from tests.helpers import make_origin

        origin = make_origin() if o == "Karachi" else make_place(o, **kw.pop("o_kw", {}))
        return await self.provider().search(
            origin,
            make_place(d, **kw.pop("d_kw", {})),
            date(2026, 12, 1),
            date(2026, 12, 8),
            2,
            kw.pop("cabin", Cabin.ECONOMY),
        )

    async def test_offers_direct_and_connecting_options(self):
        opts = await self.search()
        assert [o.stops for o in opts] == [0, 1, 1]
        assert opts[0].price_per_traveler > opts[1].price_per_traveler > opts[2].price_per_traveler
        assert opts[0].duration_minutes < opts[1].duration_minutes < opts[2].duration_minutes

    async def test_everything_is_labelled_as_an_estimate(self):
        for o in await self.search():
            assert o.price_is_estimate and o.schedule_is_indicative
            assert o.source.origin is DataOrigin.ESTIMATE and "Indicative" in o.source.note

    async def test_estimates_pass_our_own_validators(self):
        km = great_circle_km(24.9065, 67.1608, 35.5494, 139.7798)
        for o in await self.search():
            issues = (
                check_flight_duration(o.duration_minutes, km, o.stops)
                + check_flight_price(o, km, FX)
                + check_flight_times(o)
            )
            assert not has_errors(issues), issues

    async def test_ids_are_unique(self):
        ids = [o.id for o in await self.search()]
        assert len(ids) == len(set(ids))

    async def test_cabin_and_party_size(self):
        eco = (await self.search())[0]
        biz = (await self.search(cabin=Cabin.BUSINESS))[0]
        assert float(
            biz.price_per_traveler.amount / eco.price_per_traveler.amount
        ) == pytest.approx(3.2, rel=0.01)
        assert eco.total_price == eco.price_per_traveler.scaled(2)

    async def test_longer_routes_cost_more(self):
        short = (
            await self.provider().search(
                make_place(
                    "Dubai",
                    country_code="AE",
                    latitude=25.2,
                    longitude=55.27,
                    timezone="Asia/Dubai",
                    iata="DXB",
                ),
                make_place(
                    "Karachi",
                    country_code="PK",
                    latitude=24.86,
                    longitude=67.0,
                    timezone="Asia/Karachi",
                    iata="KHI",
                ),
                date(2026, 12, 1),
                None,
                1,
                Cabin.ECONOMY,
            )
        )[0]
        long = await self.provider().search(
            make_place(
                "London",
                country_code="GB",
                latitude=51.5,
                longitude=-0.12,
                timezone="Europe/London",
                iata="LHR",
            ),
            make_place(
                "Sydney",
                country_code="AU",
                latitude=-33.87,
                longitude=151.2,
                timezone="Australia/Sydney",
                iata="SYD",
            ),
            date(2026, 12, 1),
            None,
            1,
            Cabin.ECONOMY,
        )
        assert long[0].price_per_traveler > short.price_per_traveler
        assert all(o.stops >= 1 for o in long)  # ~17,000 km: no nonstop offered

    async def test_peak_season_costs_more_than_shoulder(self):
        from tests.helpers import make_origin

        p = self.provider()
        peak = (
            await p.search(make_origin(), make_place(), date(2026, 12, 1), None, 1, Cabin.ECONOMY)
        )[0]
        shoulder = (
            await p.search(make_origin(), make_place(), date(2026, 9, 15), None, 1, Cabin.ECONOMY)
        )[0]
        assert peak.price_per_traveler > shoulder.price_per_traveler

    async def test_one_way_is_cheaper_than_return(self):
        from tests.helpers import make_origin

        p = self.provider()
        ow = (
            await p.search(make_origin(), make_place(), date(2026, 12, 1), None, 1, Cabin.ECONOMY)
        )[0]
        rt = (
            await p.search(
                make_origin(), make_place(), date(2026, 12, 1), date(2026, 12, 8), 1, Cabin.ECONOMY
            )
        )[0]
        assert ow.price_per_traveler < rt.price_per_traveler

    async def test_no_airport_is_a_non_retryable_error(self):
        from tests.helpers import make_origin

        with pytest.raises(ProviderError) as e:
            await self.provider().search(
                make_origin(),
                make_place(
                    "Nowhere", latitude=-45, longitude=-130, timezone="Pacific/Pitcairn", iata=None
                ),
                date(2026, 12, 1),
                None,
                1,
                Cabin.ECONOMY,
            )
        assert e.value.retryable is False

    async def test_same_airport_rejected(self):
        with pytest.raises(ProviderError):
            await self.provider().search(
                make_place("Tokyo"), make_place("Tokyo"), date(2026, 12, 1), None, 1, Cabin.ECONOMY
            )


# --------------------------------------------------------------------------- Tavily
class TestTavily:
    async def test_parses_and_skips_malformed_or_unsafe_entries(self):
        http = FakeHttp(
            {
                "tavily": {
                    "results": [
                        {
                            "title": "Hotel A",
                            "url": "https://a.example/h",
                            "content": "from $90",
                            "score": 0.9,
                        },
                        {"title": "No url"},
                        {"title": "XSS", "url": "javascript:alert(1)"},
                        {"title": "Hotel B", "url": "https://b.example", "score": "oops"},
                    ]
                }
            }
        )
        hits = await TavilySearchClient(http, "key").search("hotels tokyo")
        assert [h.url for h in hits] == ["https://a.example/h"]
        url, body = http.calls[0]
        assert body["query"] == "hotels tokyo" and body["max_results"] == 6

    async def test_unexpected_shape(self):
        with pytest.raises(ProviderError):
            await TavilySearchClient(FakeHttp({"tavily": {"nope": 1}}), "key").search("x")

    def test_requires_key(self):
        with pytest.raises(ValueError):
            TavilySearchClient(FakeHttp({}), "")

    async def test_max_results_is_clamped(self):
        http = FakeHttp({"tavily": {"results": []}})
        await TavilySearchClient(http, "k").search("x", max_results=500)
        assert http.calls[0][1]["max_results"] == 10
