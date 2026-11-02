"""The workflow port used by the service layer.

``LangGraphWorkflow`` is the production adapter (PostgreSQL checkpoints). Tests use a
double so service logic is verified without LangGraph.
"""

from __future__ import annotations

from typing import Any, Protocol

from tripmate.graph.state import PlanState


class Workflow(Protocol):
    async def start(self, thread_id: str, initial: PlanState) -> PlanState:
        """Run until the workflow finishes or pauses for approval. Returns the latest state."""

    async def resume(self, thread_id: str, decision: dict[str, Any]) -> PlanState:
        """Continue a paused workflow with the human decision."""

    async def is_waiting_for_approval(self, thread_id: str) -> bool: ...

    async def snapshot(self, thread_id: str) -> PlanState | None: ...


class LangGraphWorkflow:
    def __init__(self, graph: Any) -> None:
        self._graph = graph

    @staticmethod
    def _cfg(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def start(self, thread_id: str, initial: PlanState) -> PlanState:
        cfg = self._cfg(thread_id)
        if await self.snapshot(thread_id):  # same thread already has state: never start over
            raise RuntimeError(f"thread {thread_id} already exists")
        return await self._graph.ainvoke(initial, cfg)

    async def resume(self, thread_id: str, decision: dict[str, Any]) -> PlanState:
        from langgraph.types import Command

        if not await self.is_waiting_for_approval(thread_id):
            raise RuntimeError("workflow is not waiting for approval")
        return await self._graph.ainvoke(Command(resume=decision), self._cfg(thread_id))

    async def is_waiting_for_approval(self, thread_id: str) -> bool:
        state = await self._graph.aget_state(self._cfg(thread_id))
        return bool(state and state.next and "human_approval" in state.next)

    async def snapshot(self, thread_id: str) -> PlanState | None:
        state = await self._graph.aget_state(self._cfg(thread_id))
        return dict(state.values) if state and state.values else None
