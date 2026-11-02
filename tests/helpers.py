"""Test data builders shared across suites."""

from __future__ import annotations

import importlib
import os
from datetime import UTC, date, datetime

import pytest

from tripmate.domain.money import Money
from tripmate.schemas.trip import Place, TripRequest

NOW = datetime(2026, 10, 22, 4, 0, tzinfo=UTC)
TODAY = NOW.date()


def require_dep(name: str):
    """Import an optional heavy dependency, or skip the module.

    In CI ``REQUIRE_ALL_DEPS=1`` turns a missing dependency into a hard
    failure, so the framework-level tests can never be silently skipped there.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        if os.environ.get("REQUIRE_ALL_DEPS") == "1":
            raise
        pytest.skip(f"optional dependency '{name}' is not installed", allow_module_level=True)


def make_place(name: str = "Tokyo", **overrides) -> Place:
    base = {
        "name": name,
        "country_code": "JP",
        "latitude": 35.6762,
        "longitude": 139.6503,
        "timezone": "Asia/Tokyo",
        "iata": "HND",
    }
    base.update(overrides)
    return Place(**base)


def make_origin() -> Place:
    return make_place(
        "Karachi",
        country_code="PK",
        latitude=24.86,
        longitude=67.0,
        timezone="Asia/Karachi",
        iata="KHI",
    )


def make_trip(**overrides) -> TripRequest:
    base = {
        "origin": make_origin(),
        "destination": make_place(),
        "start_date": date(2026, 12, 1),
        "end_date": date(2026, 12, 7),
        "travelers": 2,
        "budget": Money.of(3000, "USD"),
    }
    base.update(overrides)
    return TripRequest(**base)
