"""Drives the real node functions in the same order as the LangGraph topology.

LangGraph itself is exercised separately (tests/workflow/test_graph.py, run in CI where it
is installed). This runner lets the business logic be tested anywhere.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tripmate.agents import nodes
from tripmate.agents.deps import Deps
from tripmate.graph.state import PlanState


def merge(state: PlanState, update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key in ("tool_errors", "sources", "notes"):  # reducer = operator.add
            state[key] = [*state.get(key, []), *value]  # type: ignore[literal-required]
        else:
            state[key] = value  # type: ignore[literal-required]


async def run_to_approval(state: PlanState, deps: Deps) -> PlanState:
    merge(state, await nodes.guardrail_node(state, deps))
    if nodes.route_after_guardrail(state) == "stop":
        return state
    merge(state, await nodes.intake_node(state, deps))
    targets = nodes.route_after_intake(state)
    if targets == ["__end__"]:
        return state
    fns = {
        "flights": nodes.flights_node,
        "hotels": nodes.hotels_node,
        "weather": nodes.weather_node,
        "fx": nodes.fx_node,
    }
    results = await asyncio.gather(*(fns[t](state, deps) for t in targets))
    for r in results:
        merge(state, r)
    for fn in (nodes.assemble_node, nodes.optimize_node, nodes.budget_node, nodes.itinerary_node):
        merge(state, await fn(state, deps))
        if fn is nodes.itinerary_node and nodes.route_after_itinerary(state) == "stop":
            return state
    state["status"] = "awaiting_approval"
    return state


async def decide(
    state: PlanState, deps: Deps, *, approved: bool, feedback: str | None = None
) -> PlanState:
    state["approval"] = {"approved": approved, "feedback": feedback}
    merge(state, await nodes.finalize_node(state, deps))
    if nodes.route_after_finalize(state) == "revise":
        merge(state, await nodes.itinerary_node(state, deps))
        state["status"] = (
            "awaiting_approval"
            if nodes.route_after_itinerary(state) == "approval"
            else state["status"]
        )
    return state
