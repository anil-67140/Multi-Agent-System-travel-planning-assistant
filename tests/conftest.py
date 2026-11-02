"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from tests.helpers import NOW
from tripmate.domain.clock import FixedClock


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def clock() -> FixedClock:
    return FixedClock(NOW)
