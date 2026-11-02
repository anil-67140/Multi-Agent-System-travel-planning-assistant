"""Money and foreign-exchange primitives.

Design rules (these are the classic travel-app bugs):

* Money is ``Decimal``, never ``float``.
* Every amount is quantised to the currency's ISO-4217 minor-unit exponent
  (JPY has 0 decimals, USD 2, KWD 3).
* Arithmetic between different currencies is an *error*, not a silent bug.
* Conversion needs an explicit :class:`FxRates` object that knows how old it
  is and where it came from.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tripmate.domain.provenance import DataOrigin, UtcDatetime

# ISO-4217 minor-unit exponents for currencies a traveller is likely to meet.
CURRENCY_EXPONENT: dict[str, int] = {
    **dict.fromkeys(
        [
            "USD",
            "EUR",
            "GBP",
            "CNY",
            "INR",
            "PKR",
            "BDT",
            "LKR",
            "NPR",
            "AED",
            "SAR",
            "QAR",
            "TRY",
            "EGP",
            "MAD",
            "ZAR",
            "NGN",
            "KES",
            "THB",
            "MYR",
            "SGD",
            "IDR",
            "PHP",
            "HKD",
            "TWD",
            "AUD",
            "NZD",
            "CAD",
            "MXN",
            "BRL",
            "ARS",
            "COP",
            "PEN",
            "CHF",
            "SEK",
            "NOK",
            "DKK",
            "PLN",
            "CZK",
            "HUF",
            "RON",
            "BGN",
            "RUB",
            "UAH",
            "ILS",
            "MVR",
            "BTN",
            "MMK",
            "KHR",
            "LAK",
        ],
        2,
    ),
    **dict.fromkeys(["JPY", "KRW", "VND", "CLP", "ISK", "XOF", "XAF", "PYG", "UGX", "RWF"], 0),
    **dict.fromkeys(["KWD", "BHD", "OMR", "JOD", "TND"], 3),
}


class UnknownCurrencyError(ValueError):
    pass


class CurrencyMismatchError(ValueError):
    pass


class MissingRateError(KeyError):
    pass


def normalize_currency(code: str) -> str:
    cleaned = code.strip().upper()
    if cleaned not in CURRENCY_EXPONENT:
        raise UnknownCurrencyError(f"Unsupported or unknown currency code: {code!r}")
    return cleaned


def quantize(amount: Decimal, currency: str) -> Decimal:
    exp = CURRENCY_EXPONENT[currency]
    return amount.quantize(Decimal(1).scaleb(-exp), rounding=ROUND_HALF_UP)


def _to_decimal(value: Any) -> Any:
    # Go through ``str`` so 0.1 stays 0.1 instead of 0.1000000000000000055...
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid money amount")
    if isinstance(value, float):
        return Decimal(str(value))
    return value


class Money(BaseModel):
    """An immutable amount of money in a specific ISO-4217 currency."""

    model_config = ConfigDict(frozen=True)

    # ``currency`` is declared first so the ``amount`` validator can read it.
    currency: str
    amount: Decimal = Field(ge=0, allow_inf_nan=False, max_digits=16)

    @field_validator("currency", mode="before")
    @classmethod
    def _check_currency(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("currency must be a string")
        return normalize_currency(value)

    @field_validator("amount", mode="before")
    @classmethod
    def _coerce_and_quantize(cls, value: Any, info: Any) -> Any:
        # Quantise *before* pydantic applies the digit/range constraints, so the
        # long tail of a division (FX conversion) never trips ``max_digits``.
        value = _to_decimal(value)
        try:
            dec = value if isinstance(value, Decimal) else Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError("amount must be a number") from exc
        if not dec.is_finite():
            raise ValueError("amount must be finite")
        currency = info.data.get("currency")
        if not currency:
            return dec
        try:
            return quantize(dec, currency)
        except InvalidOperation as exc:
            raise ValueError("amount is too large") from exc

    # -- construction helpers -------------------------------------------------
    @classmethod
    def of(cls, amount: Any, currency: str) -> Money:
        return cls(amount=amount, currency=currency)

    @classmethod
    def zero(cls, currency: str) -> Money:
        return cls(amount=Decimal(0), currency=currency)

    # -- arithmetic -----------------------------------------------------------
    def _require_same(self, other: Money) -> None:
        if self.currency != other.currency:
            raise CurrencyMismatchError(f"Cannot combine {self.currency} with {other.currency}")

    def __add__(self, other: Money) -> Money:
        self._require_same(other)
        return Money(amount=self.amount + other.amount, currency=self.currency)

    def scaled(self, factor: Any) -> Money:
        """Multiply by a number (travellers, nights, percentage ...)."""
        return Money(amount=self.amount * Decimal(str(factor)), currency=self.currency)

    def __lt__(self, other: Money) -> bool:
        self._require_same(other)
        return self.amount < other.amount

    def __le__(self, other: Money) -> bool:
        self._require_same(other)
        return self.amount <= other.amount

    def __gt__(self, other: Money) -> bool:
        self._require_same(other)
        return self.amount > other.amount

    def __ge__(self, other: Money) -> bool:
        self._require_same(other)
        return self.amount >= other.amount

    def __str__(self) -> str:
        exp = CURRENCY_EXPONENT[self.currency]
        return f"{self.amount:,.{exp}f} {self.currency}"


def sum_money(items: Iterable[Money], currency: str) -> Money:
    total = Money.zero(currency)
    for item in items:
        total = total + item
    return total


class FxRates(BaseModel):
    """A snapshot of exchange rates: ``rates[X]`` = units of X per 1 ``base``."""

    model_config = ConfigDict(frozen=True)

    base: str
    rates: dict[str, Decimal]
    as_of: UtcDatetime
    source: str
    origin: DataOrigin = DataOrigin.LIVE

    @field_validator("base", mode="before")
    @classmethod
    def _base(cls, value: str) -> str:
        return normalize_currency(value)

    @field_validator("rates", mode="before")
    @classmethod
    def _rates(cls, value: dict[str, Any]) -> dict[str, Decimal]:
        out: dict[str, Decimal] = {}
        for code, rate in value.items():
            dec = Decimal(str(rate))
            if dec <= 0:
                raise ValueError(f"Non-positive FX rate for {code}")
            out[str(code).upper()] = dec
        return out

    def rate(self, currency: str) -> Decimal:
        currency = normalize_currency(currency)
        if currency == self.base:
            return Decimal(1)
        try:
            return self.rates[currency]
        except KeyError as exc:
            raise MissingRateError(f"No FX rate for {currency} (base {self.base})") from exc

    def supports(self, currency: str) -> bool:
        try:
            self.rate(currency)
        except (MissingRateError, UnknownCurrencyError):
            return False
        return True

    def convert(self, money: Money, to: str) -> Money:
        to = normalize_currency(to)
        if money.currency == to:
            return money
        # Cross rate through the base currency.
        value = money.amount / self.rate(money.currency) * self.rate(to)
        return Money(amount=value, currency=to)

    def age_seconds(self, now: datetime) -> float:
        return (now - self.as_of).total_seconds()


# Last-resort rates used *only* when every live source fails. They are
# deliberately approximate and carry an old ``as_of`` so freshness checks flag
# them as expired and the UI can warn the traveller.
STATIC_USD_RATES: dict[str, str] = {
    "EUR": "0.92",
    "GBP": "0.79",
    "JPY": "150",
    "INR": "84",
    "PKR": "280",
    "BDT": "120",
    "AED": "3.6725",
    "SAR": "3.75",
    "QAR": "3.64",
    "KWD": "0.307",
    "OMR": "0.385",
    "BHD": "0.376",
    "JOD": "0.709",
    "THB": "35",
    "MYR": "4.4",
    "SGD": "1.34",
    "AUD": "1.5",
    "NZD": "1.65",
    "CAD": "1.38",
    "CNY": "7.2",
    "KRW": "1350",
    "TRY": "35",
    "IDR": "15800",
    "VND": "25000",
    "PHP": "58",
    "LKR": "300",
    "NPR": "135",
    "EGP": "49",
    "ZAR": "18",
    "NGN": "1500",
    "KES": "129",
    "MXN": "20",
    "BRL": "5.7",
    "CHF": "0.88",
    "SEK": "10.8",
    "NOK": "11",
    "DKK": "6.9",
    "PLN": "4.0",
    "CZK": "23",
    "HUF": "370",
    "ILS": "3.7",
    "HKD": "7.8",
    "TWD": "32",
    "MAD": "10",
    "COP": "4100",
    "PEN": "3.8",
    "ARS": "1000",
    "CLP": "950",
    "RON": "4.6",
    "BGN": "1.8",
    "ISK": "138",
    "RUB": "92",
    "UAH": "41",
    "MVR": "15.4",
    "BTN": "84",
    "MMK": "2100",
    "KHR": "4100",
    "LAK": "21000",
    "TND": "3.1",
    "XOF": "610",
    "XAF": "610",
    "PYG": "7800",
    "UGX": "3700",
    "RWF": "1400",
}
