"""Sequential execution of the real node functions (same order as the LangGraph topology)."""

from __future__ import annotations

import asyncio
from typing import Any

from tripmate.agents import nodes
from tripmate.agents.deps import Deps
from tripmate.graph.state import PlanState

_FANOUT = {
    "flights": nodes.flights_node,
    "hotels": nodes.hotels_node,
    "weather": nodes.weather_node,
    "fx": nodes.fx_node,
}
_ADDITIVE = ("tool_errors", "sources", "notes")


def _merge(state: PlanState, update: dict[str, Any]) -> None:
    for k, v in update.items():
        state[k] = [*state.get(k, []), *v] if k in _ADDITIVE else v  # type: ignore[literal-required]


async def run_pipeline(state: PlanState, deps: Deps, on_trip: Any = None) -> PlanState:
    _merge(state, await nodes.guardrail_node(state, deps))
    if nodes.route_after_guardrail(state) == "stop":
        return state
    _merge(state, await nodes.intake_node(state, deps))
    targets = nodes.route_after_intake(state)
    if targets == ["__end__"]:
        return state
    if on_trip:
        on_trip(state["trip"])
    for update in await asyncio.gather(*(_FANOUT[t](state, deps) for t in targets)):
        _merge(state, update)
    for fn in (nodes.assemble_node, nodes.optimize_node, nodes.budget_node, nodes.itinerary_node):
        _merge(state, await fn(state, deps))
        if fn is nodes.itinerary_node and nodes.route_after_itinerary(state) == "stop":
            return state
    state["status"] = "awaiting_approval"
    return state
