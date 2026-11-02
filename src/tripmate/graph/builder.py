"""Assemble the LangGraph workflow.

guardrail -> intake -> [flights | hotels | weather | fx] -> assemble -> optimize
-> budget -> itinerary -> human_approval (interrupt) -> finalize -> (revise -> itinerary)

The approval node contains *only* ``interrupt()``. On resume LangGraph re-runs the
node from its first line, so keeping side effects out of it makes resume idempotent.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from tripmate.agents import nodes
from tripmate.agents.deps import Deps
from tripmate.graph.state import PlanState

NodeFn = Callable[[PlanState, Deps], Awaitable[dict[str, Any]]]

FAN_OUT = nodes.FAN_OUT

# Execution order, also used by tests that drive nodes without LangGraph installed.
PIPELINE: tuple[tuple[str, NodeFn], ...] = (
    ("assemble", nodes.assemble_node),
    ("optimize", nodes.optimize_node),
    ("budget", nodes.budget_node),
    ("itinerary", nodes.itinerary_node),
)


def human_approval(state: PlanState) -> dict[str, Any]:
    decision = interrupt(
        {
            "type": "approval_required",
            "itinerary_version": state.get("itinerary_version"),
            "plan_id": state.get("plan_id"),
        }
    )
    return {"approval": decision, "status": "awaiting_approval"}


def build_graph(deps: Deps, checkpointer: Any) -> Any:
    g: StateGraph = StateGraph(PlanState)

    def bind(fn: NodeFn) -> Callable[[PlanState], Awaitable[dict[str, Any]]]:
        async def node(state: PlanState) -> dict[str, Any]:
            return await fn(state, deps)

        node.__name__ = fn.__name__
        return node

    g.add_node("guardrail", bind(nodes.guardrail_node))
    g.add_node("intake", bind(nodes.intake_node))
    g.add_node("flights", bind(nodes.flights_node))
    g.add_node("hotels", bind(nodes.hotels_node))
    g.add_node("weather", bind(nodes.weather_node))
    g.add_node("fx", bind(nodes.fx_node))
    g.add_node("assemble", bind(nodes.assemble_node))
    g.add_node("optimize", bind(nodes.optimize_node))
    g.add_node("budget", bind(nodes.budget_node))
    g.add_node("itinerary", bind(nodes.itinerary_node))
    g.add_node("human_approval", human_approval)
    g.add_node("finalize", bind(nodes.finalize_node))

    g.add_edge(START, "guardrail")
    g.add_conditional_edges(
        "guardrail", nodes.route_after_guardrail, {"intake": "intake", "stop": END}
    )
    # Returns the four data nodes (run in parallel) or END; no unconditional edges, so a
    # clarification/blocked run never triggers any provider call.
    g.add_conditional_edges("intake", nodes.route_after_intake, [*FAN_OUT, END])
    g.add_edge(list(FAN_OUT), "assemble")  # waits for all four
    g.add_edge("assemble", "optimize")
    g.add_edge("optimize", "budget")
    g.add_edge("budget", "itinerary")
    g.add_conditional_edges(
        "itinerary", nodes.route_after_itinerary, {"approval": "human_approval", "stop": END}
    )
    g.add_edge("human_approval", "finalize")
    g.add_conditional_edges(
        "finalize", nodes.route_after_finalize, {"done": END, "revise": "itinerary"}
    )
    return g.compile(checkpointer=checkpointer)
