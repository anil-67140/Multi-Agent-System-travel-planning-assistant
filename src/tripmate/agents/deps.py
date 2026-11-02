"""Everything a node may touch, passed in explicitly so tests can swap any of it."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from tripmate.domain.clock import Clock, SystemClock
from tripmate.llm import LLMGateway
from tripmate.providers.base import AirportDirectory
from tripmate.providers.toolbelt import Toolbelt
from tripmate.validation.itinerary import ItineraryRules

# (event_type, node, message, data) - persisted by the service as run_events for SSE/UI.
Emit = Callable[[str, str | None, str, dict[str, Any]], Awaitable[None]]


async def _noop_emit(event: str, node: str | None, message: str, data: dict[str, Any]) -> None:
    return None


@dataclass
class Deps:
    llm: LLMGateway
    tools: Toolbelt
    airports: AirportDirectory
    clock: Clock = field(default_factory=SystemClock)
    emit: Emit = _noop_emit
    rules: ItineraryRules = field(default_factory=ItineraryRules)
    max_itinerary_repairs: int = 2
    max_revisions: int = 3
