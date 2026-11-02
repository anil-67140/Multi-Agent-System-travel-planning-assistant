"""Workflow state.

Everything is stored as JSON-compatible dicts (``model_dump(mode="json")``) so
any checkpointer can serialise it and old checkpoints stay loadable when a model
gains a field. Typed access goes through the helpers below.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict, TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


class PlanState(TypedDict, total=False):
    # inputs
    plan_id: str
    user_id: str
    message: str
    prefs: dict[str, Any]  # affinity counts + defaults loaded from the DB
    trip: dict[str, Any]  # present on constraint-edit runs: skips intake
    # control
    status: str
    guardrail: dict[str, Any]
    clarification: str | None
    # parallel fan-out results (each node owns its own key)
    flights: list[dict[str, Any]]
    hotels: list[dict[str, Any]]
    weather: dict[str, Any] | None
    fx: dict[str, Any]
    # merged by reducers because several parallel nodes append
    tool_errors: Annotated[list[dict[str, Any]], operator.add]
    sources: Annotated[list[dict[str, Any]], operator.add]
    notes: Annotated[list[str], operator.add]
    # derived
    bundles: list[dict[str, Any]]
    selected_label: str | None
    budget: dict[str, Any] | None
    itinerary: dict[str, Any] | None
    itinerary_version: int
    issues: list[dict[str, Any]]
    # human in the loop
    approval: dict[str, Any] | None
    feedback: str | None
    revisions: int


def load(model: type[M], data: dict[str, Any] | None) -> M | None:
    return model.model_validate(data) if data else None


def dump(model: BaseModel | None) -> dict[str, Any] | None:
    return model.model_dump(mode="json") if model is not None else None


def dump_all(models: list[Any]) -> list[dict[str, Any]]:
    return [m.model_dump(mode="json") for m in models]
