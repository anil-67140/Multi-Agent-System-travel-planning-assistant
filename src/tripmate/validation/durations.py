"""Travel-duration plausibility, derived from great-circle geometry."""

from __future__ import annotations

import math

from tripmate.validation.issues import Issue, Severity

EARTH_RADIUS_KM = 6371.0088

# Tuned to be *lenient*: they exist to catch absurd values (a 2-hour flight
# from New York to London), not to second-guess real schedules.
FASTEST_GROUND_SPEED_KMH = 1100.0  # jet-stream tailwind, best case
FASTEST_OVERHEAD_MIN = 15
TYPICAL_SPEED_KMH = 800.0  # effective block speed
TYPICAL_OVERHEAD_MIN = 35
LAYOVER_ALLOWANCE_MIN = 480  # per stop


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance between two points, in kilometres."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def min_plausible_minutes(distance_km: float) -> int:
    return round(distance_km / FASTEST_GROUND_SPEED_KMH * 60 + FASTEST_OVERHEAD_MIN)


def typical_minutes(distance_km: float, stops: int = 0) -> int:
    base = distance_km / TYPICAL_SPEED_KMH * 60 + TYPICAL_OVERHEAD_MIN
    # A connection adds a detour plus a layover; ~3h is a fair planning figure.
    return round(base * (1 + 0.08 * stops) + stops * 180)


def max_plausible_minutes(distance_km: float, stops: int = 0) -> int:
    base = distance_km / TYPICAL_SPEED_KMH * 60 + TYPICAL_OVERHEAD_MIN
    return round(base * 1.5 + stops * LAYOVER_ALLOWANCE_MIN)


def check_flight_duration(
    duration_minutes: int, distance_km: float, stops: int = 0, *, path: str = "flight"
) -> list[Issue]:
    issues: list[Issue] = []
    floor = min_plausible_minutes(distance_km)
    ceiling = max_plausible_minutes(distance_km, stops)
    if duration_minutes < floor:
        issues.append(
            Issue(
                code="FLIGHT_TOO_FAST",
                message=(
                    f"{duration_minutes} min is faster than physically plausible for "
                    f"{distance_km:.0f} km (minimum about {floor} min)."
                ),
                path=path,
            )
        )
    elif duration_minutes > ceiling:
        issues.append(
            Issue(
                code="FLIGHT_TOO_SLOW",
                message=(
                    f"{duration_minutes} min is unusually long for {distance_km:.0f} km with "
                    f"{stops} stop(s) (expected under about {ceiling} min)."
                ),
                severity=Severity.WARNING,
                path=path,
            )
        )
    return issues
