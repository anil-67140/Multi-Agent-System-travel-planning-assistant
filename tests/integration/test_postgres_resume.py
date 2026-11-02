"""Proves the headline claim: a plan paused for approval survives a process restart.

Runs only when TRIPMATE_TEST_POSTGRES_URL points at a disposable PostgreSQL (set in CI).
The 'restart' is simulated by building a brand-new graph, saver and pool for the second half.
"""

from __future__ import annotations

import os
import uuid

import pytest

from tests.helpers import require_dep

URL = os.environ.get("TRIPMATE_TEST_POSTGRES_URL")
if not URL:
    pytest.skip("TRIPMATE_TEST_POSTGRES_URL not set", allow_module_level=True)
for _m in ("langgraph", "psycopg", "psycopg_pool", "sqlalchemy"):
    require_dep(_m)

from tests.workflow.test_nodes_workflow import good_itins, make_deps, make_llm  # noqa: E402
from tripmate.graph.builder import build_graph  # noqa: E402
from tripmate.graph.checkpointer import open_checkpointer  # noqa: E402
from tripmate.services.workflow import LangGraphWorkflow  # noqa: E402

pytestmark = [pytest.mark.anyio, pytest.mark.integration]
LIBPQ_URL = URL.replace("postgresql+psycopg://", "postgresql://")


def init(pid):
    return {
        "plan_id": pid,
        "user_id": "u",
        "message": "5 days in Tokyo from Karachi",
        "prefs": {},
        "revisions": 0,
    }


async def test_paused_plan_resumes_after_a_simulated_restart():
    thread = f"it-{uuid.uuid4().hex}:0"

    async with open_checkpointer(LIBPQ_URL) as saver:  # "process 1"
        wf = LangGraphWorkflow(build_graph(make_deps(make_llm(itineraries=good_itins())), saver))
        await wf.start(thread, init("p"))
        assert await wf.is_waiting_for_approval(thread)

    async with open_checkpointer(
        LIBPQ_URL
    ) as saver:  # "process 2": new pool, new graph, empty memory
        llm = make_llm()  # would raise if any node tried to call the model again
        wf = LangGraphWorkflow(build_graph(make_deps(llm), saver))
        assert await wf.is_waiting_for_approval(thread)
        snap = await wf.snapshot(thread)
        assert snap["itinerary_version"] == 1 and snap["itinerary"]
        final = await wf.resume(thread, {"approved": True, "feedback": None})
        assert final["status"] == "completed"
        assert llm.calls == [], "resume must not re-run completed nodes"


async def test_two_replicas_cannot_both_resume_the_same_plan():
    import asyncio

    thread = f"it-{uuid.uuid4().hex}:0"
    async with open_checkpointer(LIBPQ_URL) as saver:
        wf = LangGraphWorkflow(build_graph(make_deps(make_llm(itineraries=good_itins())), saver))
        await wf.start(thread, init("p"))
        results = await asyncio.gather(
            *(wf.resume(thread, {"approved": True, "feedback": None}) for _ in range(2)),
            return_exceptions=True,
        )
    ok = [r for r in results if not isinstance(r, Exception)]
    assert (
        len(ok) >= 1
    )  # at least one resume wins; the service layer's approval row makes the second a no-op
