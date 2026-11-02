"""Source-freshness policy: how old may a fact be before we distrust it?"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from enum import StrEnum

from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef
from tripmate.validation.issues import Issue, Severity

DEFAULT_MAX_AGE: dict[DataKind, timedelta] = {
    DataKind.WEATHER_FORECAST: timedelta(hours=6),
    DataKind.WEATHER_CLIMATOLOGY: timedelta(days=30),
    DataKind.FX_RATE: timedelta(hours=36),  # most free feeds refresh daily
    DataKind.FLIGHT_PRICE: timedelta(hours=1),
    DataKind.HOTEL_PRICE: timedelta(hours=12),
    DataKind.WEB_SOURCE: timedelta(days=7),
    DataKind.GEOCODING: timedelta(days=90),
}


class Freshness(StrEnum):
    FRESH = "fresh"  # within the policy window
    STALE = "stale"  # up to 2x the window: usable with a warning
    EXPIRED = "expired"  # older: do not present as current


def assess_freshness(
    retrieved_at: datetime,
    kind: DataKind,
    now: datetime,
    policy: dict[DataKind, timedelta] | None = None,
) -> Freshness:
    max_age = (policy or DEFAULT_MAX_AGE)[kind]
    age = now - retrieved_at
    if age <= max_age:
        return Freshness.FRESH
    if age <= max_age * 2:
        return Freshness.STALE
    return Freshness.EXPIRED


def check_sources(sources: Iterable[SourceRef], now: datetime) -> list[Issue]:
    """One issue per source that is stale/expired; estimates get an explicit note."""
    issues: list[Issue] = []
    for src in sources:
        label = f"{src.provider} ({src.kind.value})"
        if src.origin is DataOrigin.ESTIMATE:
            issues.append(
                Issue(
                    code="SOURCE_ESTIMATE",
                    message=f"{label} is an estimate, not a live quote.",
                    severity=Severity.INFO,
                )
            )
            continue
        state = assess_freshness(src.retrieved_at, src.kind, now)
        if state is Freshness.STALE:
            issues.append(
                Issue(
                    code="SOURCE_STALE",
                    message=f"{label} was retrieved {_age(now - src.retrieved_at)} ago.",
                    severity=Severity.WARNING,
                )
            )
        elif state is Freshness.EXPIRED:
            issues.append(
                Issue(
                    code="SOURCE_EXPIRED",
                    message=f"{label} is too old ({_age(now - src.retrieved_at)}) to present as current.",
                    severity=Severity.WARNING,
                )
            )
    return issues


def _age(delta: timedelta) -> str:
    hours = delta.total_seconds() / 3600
    return f"{hours:.0f}h" if hours < 48 else f"{hours / 24:.0f}d"
