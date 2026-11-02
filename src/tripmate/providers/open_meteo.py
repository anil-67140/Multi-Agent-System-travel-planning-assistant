"""Open-Meteo: free, keyless geocoding + weather (includes the IANA time zone)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from tripmate.domain.clock import Clock
from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef
from tripmate.errors import ProviderError
from tripmate.providers.base import HttpClient
from tripmate.schemas.search import DailyWeather, WeatherReport
from tripmate.schemas.trip import Place

PROVIDER = "open-meteo"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_HORIZON_DAYS = 15  # the API allows up to 16 days including today

WMO_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "freezing drizzle",
    61: "slight rain",
    63: "rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "freezing rain",
    71: "slight snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "severe thunderstorm with hail",
}

_FORECAST_VARS = "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code,wind_speed_10m_max"
_ARCHIVE_VARS = (
    "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code,wind_speed_10m_max"
)


def _shift_year(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 Feb in a non-leap year
        return d.replace(year=d.year + years, day=28)


class OpenMeteoClient:
    def __init__(self, http: HttpClient, clock: Clock) -> None:
        self._http = http
        self._clock = clock

    # ------------------------------------------------------------------ geocoding
    async def geocode(self, query: str) -> Place:
        data = await self._http.get_json(
            GEOCODE_URL,
            provider=PROVIDER,
            params={"name": query.strip(), "count": 1, "language": "en", "format": "json"},
        )
        results = (data or {}).get("results") or []
        if not results:
            raise ProviderError(PROVIDER, f"no location found for {query!r}", retryable=False)
        top = results[0]
        try:
            return Place(
                name=str(top["name"]),
                country_code=top.get("country_code"),
                latitude=float(top["latitude"]),
                longitude=float(top["longitude"]),
                timezone=top.get("timezone"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderError(PROVIDER, "unexpected geocoding response", retryable=False) from exc

    # ------------------------------------------------------------------- weather
    async def forecast(self, place: Place, start: date, end: date, today: date) -> WeatherReport:
        """Real forecast when the whole trip is inside the horizon, otherwise the
        previous year's observed weather for the same dates (clearly labelled)."""
        if start >= today and end <= today + timedelta(days=FORECAST_HORIZON_DAYS):
            return await self._live_forecast(place, start, end)
        return await self._climatology(place, start, end)

    async def _live_forecast(self, place: Place, start: date, end: date) -> WeatherReport:
        data = await self._http.get_json(
            FORECAST_URL,
            provider=PROVIDER,
            params={
                "latitude": place.latitude,
                "longitude": place.longitude,
                "daily": _FORECAST_VARS,
                "timezone": "auto",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
        )
        days = self._parse_daily(data, shift_years=0)
        return self._report(place, data, days, climatology=False, note=None)

    async def _climatology(self, place: Place, start: date, end: date) -> WeatherReport:
        data = await self._http.get_json(
            ARCHIVE_URL,
            provider=PROVIDER,
            params={
                "latitude": place.latitude,
                "longitude": place.longitude,
                "daily": _ARCHIVE_VARS,
                "timezone": "auto",
                "start_date": _shift_year(start, -1).isoformat(),
                "end_date": _shift_year(end, -1).isoformat(),
            },
        )
        days = self._parse_daily(data, shift_years=1)
        note = (
            "Beyond the forecast window: showing last year's observed weather for the same dates."
        )
        return self._report(place, data, days, climatology=True, note=note)

    def _report(
        self,
        place: Place,
        data: Any,
        days: list[DailyWeather],
        *,
        climatology: bool,
        note: str | None,
    ) -> WeatherReport:
        if not days:
            raise ProviderError(PROVIDER, "weather response contained no days", retryable=False)
        return WeatherReport(
            place_name=place.name,
            timezone=(data or {}).get("timezone") or place.timezone,
            days=days,
            is_climatology=climatology,
            source=SourceRef(
                provider=PROVIDER,
                kind=DataKind.WEATHER_CLIMATOLOGY if climatology else DataKind.WEATHER_FORECAST,
                origin=DataOrigin.LIVE,
                retrieved_at=self._clock.now(),
                url="https://open-meteo.com/",
                title="Open-Meteo weather data",
                note=note,
            ),
        )

    @staticmethod
    def _parse_daily(data: Any, *, shift_years: int) -> list[DailyWeather]:
        daily = (data or {}).get("daily")
        if not isinstance(daily, dict) or not daily.get("time"):
            raise ProviderError(PROVIDER, "unexpected weather response", retryable=False)

        def col(name: str, i: int) -> Any:
            values = daily.get(name)
            return values[i] if isinstance(values, list) and i < len(values) else None

        out: list[DailyWeather] = []
        for i, raw_day in enumerate(daily["time"]):
            tmax, tmin = col("temperature_2m_max", i), col("temperature_2m_min", i)
            if tmax is None or tmin is None:
                continue  # the model has no value for this day: skip rather than invent
            code = col("weather_code", i)
            try:
                out.append(
                    DailyWeather(
                        date=_shift_year(date.fromisoformat(raw_day), shift_years),
                        temp_min_c=float(min(tmin, tmax)),
                        temp_max_c=float(max(tmin, tmax)),
                        precipitation_mm=max(0.0, float(col("precipitation_sum", i) or 0)),
                        precipitation_probability_pct=(
                            float(p)
                            if (p := col("precipitation_probability_max", i)) is not None
                            else None
                        ),
                        wind_kph=(
                            float(w) if (w := col("wind_speed_10m_max", i)) is not None else None
                        ),
                        weather_code=int(code) if code is not None else None,
                        condition=WMO_CODES.get(int(code), "unknown")
                        if code is not None
                        else "unknown",
                    )
                )
            except (ValueError, TypeError):
                continue
        return out
