"""Time-zone-correct arithmetic for flights and itineraries.

Rule: store and compare instants in UTC; convert to a place's IANA zone only
for display and for "is this activity before the traveller landed?" checks.
Never subtract naive datetimes across zones.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from tripmate.schemas.search import FlightOption
from tripmate.validation.issues import Issue

DURATION_TOLERANCE_MIN = 5


def local_to_utc(local: datetime, tz: str) -> datetime:
    """Interpret a *naive* local wall-clock time in ``tz`` and return aware UTC."""
    if local.tzinfo is not None:
        raise ValueError("local_to_utc expects a naive datetime")
    return local.replace(tzinfo=ZoneInfo(tz)).astimezone(UTC)


def to_local(instant: datetime, tz: str) -> datetime:
    if instant.tzinfo is None:
        raise ValueError("to_local expects an aware datetime")
    return instant.astimezone(ZoneInfo(tz))


def estimate_arrival(
    depart_local: datetime, origin_tz: str, dest_tz: str, duration_minutes: int
) -> datetime:
    """Local arrival time at the destination for a given local departure."""
    departure_utc = local_to_utc(depart_local, origin_tz)
    return to_local(departure_utc + timedelta(minutes=duration_minutes), dest_tz)


def utc_offset_hours(tz: str, at: datetime) -> float:
    offset = at.astimezone(ZoneInfo(tz)).utcoffset()
    return (offset.total_seconds() / 3600) if offset else 0.0


def check_flight_times(option: FlightOption) -> list[Issue]:
    """If both instants are present, they must agree with ``duration_minutes``."""
    if option.departure is None or option.arrival is None:
        return []
    actual = (option.arrival - option.departure).total_seconds() / 60
    if abs(actual - option.duration_minutes) > DURATION_TOLERANCE_MIN:
        return [
            Issue(
                code="FLIGHT_DURATION_MISMATCH",
                message=(
                    f"Stated duration {option.duration_minutes} min differs from "
                    f"arrival-departure {actual:.0f} min."
                ),
                path=f"flight[{option.id}]",
            )
        ]
    return []
