"""Exchange rates: two free keyless feeds, then a clearly-labelled static fallback.

Attribution (required by the ExchangeRate-API open endpoint): rates by
https://www.exchangerate-api.com - shown in the UI footer.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from tripmate.domain.clock import Clock
from tripmate.domain.money import STATIC_USD_RATES, FxRates
from tripmate.domain.provenance import DataOrigin
from tripmate.errors import ProviderError
from tripmate.providers.base import HttpClient

log = logging.getLogger(__name__)

ER_API = "https://open.er-api.com/v6/latest/{base}"
FRANKFURTER = "https://api.frankfurter.dev/v1/latest"


class ExchangeRateApiClient:
    """~160 currencies including PKR/BDT/LKR/NPR. Updates once a day."""

    provider = "open.er-api.com"

    def __init__(self, http: HttpClient) -> None:
        self._http = http

    async def rates(self, base: str) -> FxRates:
        data: Any = await self._http.get_json(ER_API.format(base=base), provider=self.provider)
        if not isinstance(data, dict) or data.get("result") != "success" or not data.get("rates"):
            raise ProviderError(self.provider, "no rates in response", retryable=True)
        try:
            as_of = datetime.fromtimestamp(int(data["time_last_update_unix"]), tz=UTC)
            return FxRates(
                base=data.get("base_code", base),
                rates=_known(data["rates"]),
                as_of=as_of,
                source=self.provider,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderError(self.provider, "malformed rates response", retryable=False) from exc


class FrankfurterClient:
    """ECB reference rates (~30 major currencies, business days only)."""

    provider = "frankfurter.dev"

    def __init__(self, http: HttpClient) -> None:
        self._http = http

    async def rates(self, base: str) -> FxRates:
        data: Any = await self._http.get_json(
            FRANKFURTER, provider=self.provider, params={"base": base}
        )
        if not isinstance(data, dict) or not data.get("rates"):
            raise ProviderError(self.provider, "no rates in response", retryable=True)
        try:
            as_of = datetime.fromisoformat(data["date"]).replace(tzinfo=UTC)
            return FxRates(
                base=data.get("base", base),
                rates=_known(data["rates"]),
                as_of=as_of,
                source=self.provider,
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise ProviderError(self.provider, "malformed rates response", retryable=False) from exc


def _known(rates: dict[str, Any]) -> dict[str, Any]:
    """Keep only currencies we can quantise; drop anything unexpected."""
    from tripmate.domain.money import CURRENCY_EXPONENT

    return {k: v for k, v in rates.items() if k in CURRENCY_EXPONENT and v}


def static_fx_rates() -> FxRates:
    """Last resort. ``as_of`` is deliberately old so freshness checks flag it."""
    return FxRates(
        base="USD",
        rates=dict(STATIC_USD_RATES),
        as_of=datetime(2025, 1, 1, tzinfo=UTC),
        source="built-in approximate table",
        origin=DataOrigin.ESTIMATE,
    )


class FallbackFxProvider:
    """Primary -> secondary -> static. Always returns *something*, never raises."""

    def __init__(self, providers: list[Any], clock: Clock) -> None:
        self._providers = providers
        self._clock = clock

    async def rates(self, base: str = "USD") -> FxRates:
        for provider in self._providers:
            try:
                return await provider.rates(base)
            except ProviderError as exc:
                log.warning(
                    "fx provider failed",
                    extra={"provider": getattr(provider, "provider", "?"), "error": str(exc)},
                )
        log.error("all live fx providers failed; using static table")
        return static_fx_rates()
