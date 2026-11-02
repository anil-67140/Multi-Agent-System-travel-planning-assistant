"""Currency consistency checks."""

from __future__ import annotations

from collections.abc import Iterable

from tripmate.domain.money import FxRates, Money
from tripmate.validation.issues import Issue


def check_single_currency(
    amounts: Iterable[tuple[str, Money | None]], expected: str
) -> list[Issue]:
    """Every amount shown to the traveller must be in the plan currency."""
    issues: list[Issue] = []
    for path, money in amounts:
        if money is not None and money.currency != expected:
            issues.append(
                Issue(
                    code="CURRENCY_MISMATCH",
                    message=f"{path} is in {money.currency} but the plan currency is {expected}.",
                    path=path,
                )
            )
    return issues


def check_fx_covers(fx: FxRates, currencies: Iterable[str]) -> list[Issue]:
    issues: list[Issue] = []
    for code in sorted(set(currencies)):
        if not fx.supports(code):
            issues.append(
                Issue(
                    code="FX_RATE_MISSING",
                    message=f"No exchange rate available for {code}; amounts cannot be converted.",
                )
            )
    return issues
