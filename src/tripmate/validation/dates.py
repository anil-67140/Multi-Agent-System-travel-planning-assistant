"""Trip-date correctness."""

from __future__ import annotations

from datetime import date, timedelta

from tripmate.validation.issues import Issue, Severity

DEFAULT_MAX_HORIZON_DAYS = 365
DEFAULT_MAX_LENGTH_DAYS = 30
SHORT_NOTICE_DAYS = 3


def check_trip_dates(
    start: date,
    end: date,
    *,
    today: date,
    max_horizon_days: int = DEFAULT_MAX_HORIZON_DAYS,
    max_length_days: int = DEFAULT_MAX_LENGTH_DAYS,
) -> list[Issue]:
    """Validate a date range against ``today`` (injected, so this is testable)."""
    issues: list[Issue] = []

    if end < start:
        issues.append(Issue(code="DATE_ORDER", message="The end date is before the start date."))
        return issues  # everything below assumes a sane range

    if start < today:
        issues.append(Issue(code="DATE_IN_PAST", message=f"The start date {start} is in the past."))

    if start > today + timedelta(days=max_horizon_days):
        issues.append(
            Issue(
                code="DATE_TOO_FAR",
                message=f"Trips can be planned at most {max_horizon_days} days ahead.",
            )
        )

    length = (end - start).days + 1
    if length > max_length_days:
        issues.append(
            Issue(
                code="TRIP_TOO_LONG",
                message=f"A {length}-day trip exceeds the {max_length_days}-day limit.",
            )
        )

    if today <= start <= today + timedelta(days=SHORT_NOTICE_DAYS):
        issues.append(
            Issue(
                code="SHORT_NOTICE",
                message="The trip starts very soon: check visa, vaccination and booking lead times.",
                severity=Severity.WARNING,
            )
        )
    return issues


def default_start_date(today: date, lead_days: int = 45) -> date:
    """A sensible placeholder start date when the traveller gives none."""
    return today + timedelta(days=lead_days)
