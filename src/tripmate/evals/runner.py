"""Run cases, score them, compare with a stored baseline, and gate regressions."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tripmate.agents.deps import Deps
from tripmate.domain.clock import FixedClock
from tripmate.evals.cases import EvalCase, Expect
from tripmate.evals.offline import EVAL_NOW, ReferenceGateway, offline_toolbelt
from tripmate.evals.pipeline import run_pipeline
from tripmate.evals.scorers import score_itinerary
from tripmate.graph.state import PlanState, load
from tripmate.schemas.budget import BudgetBreakdown
from tripmate.schemas.itinerary import Itinerary
from tripmate.schemas.search import WeatherReport
from tripmate.schemas.trip import TripRequest

SCORE_TOLERANCE = 2.0  # points a case may drop before it counts as a regression


@dataclass
class CaseResult:
    id: str
    passed: bool
    score: float | None
    status: str
    failures: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0


def check_expectations(expect: Expect, state: PlanState, score: float | None) -> list[str]:
    """Pure: compares a finished state with the ground truth. Returns human-readable failures."""
    fails: list[str] = []
    if state.get("status") != expect.status:
        fails.append(f"status: expected {expect.status}, got {state.get('status')}")
    trip = load(TripRequest, state.get("trip"))
    if trip is not None:
        if expect.destination and trip.destination.name.lower() != expect.destination.lower():
            fails.append(f"destination: expected {expect.destination}, got {trip.destination.name}")
        if expect.days is not None and trip.duration_days != expect.days:
            fails.append(f"days: expected {expect.days}, got {trip.duration_days}")
        if expect.travelers is not None and trip.travelers != expect.travelers:
            fails.append(f"travelers: expected {expect.travelers}, got {trip.travelers}")
        if expect.currency and trip.currency != expect.currency:
            fails.append(f"currency: expected {expect.currency}, got {trip.currency}")
    elif expect.destination or expect.days is not None:
        fails.append("no trip was resolved")
    budget = load(BudgetBreakdown, state.get("budget"))
    if expect.budget_verdicts and (
        budget is None or budget.verdict.value not in expect.budget_verdicts
    ):
        fails.append(
            f"budget verdict: expected one of {expect.budget_verdicts}, got {budget.verdict.value if budget else None}"
        )
    notes = " ".join(state.get("notes", [])).lower()
    for needle in expect.note_contains:
        if needle.lower() not in notes:
            fails.append(f"missing disclosure containing '{needle}'")
    if (
        expect.clarification_contains
        and expect.clarification_contains.lower() not in (state.get("clarification") or "").lower()
    ):
        fails.append(f"clarification should mention '{expect.clarification_contains}'")
    if expect.status == "awaiting_approval":
        if score is None:
            fails.append("no itinerary to score")
        elif score < expect.min_score:
            fails.append(f"quality score {score} below minimum {expect.min_score}")
    return fails


async def run_case_offline(case: EvalCase) -> CaseResult:
    started = time.perf_counter()
    tools, airports = offline_toolbelt()
    llm = ReferenceGateway(case.intent, case.guard_allowed)
    deps = Deps(llm=llm, tools=tools, airports=airports, clock=FixedClock(EVAL_NOW))
    state: PlanState = {
        "plan_id": f"eval-{case.id}",
        "message": case.message,
        "prefs": {},
        "revisions": 0,
    }

    def remember(trip: dict[str, Any]) -> None:
        llm.trip = trip

    state = await run_pipeline(state, deps, on_trip=remember)
    return finish(case, state, started)


def finish(case: EvalCase, state: PlanState, started: float) -> CaseResult:
    score: float | None = None
    comps: dict[str, float] = {}
    itin, trip = load(Itinerary, state.get("itinerary")), load(TripRequest, state.get("trip"))
    if itin is not None and trip is not None:
        report = score_itinerary(
            itin,
            trip,
            load(BudgetBreakdown, state.get("budget")),
            load(WeatherReport, state.get("weather")),
        )
        score, comps = report.total, dict(report.as_dict()["components"])  # type: ignore[call-overload]
    failures = check_expectations(case.expect, state, score)
    return CaseResult(
        case.id,
        not failures,
        score,
        str(state.get("status")),
        failures,
        comps,
        round(time.perf_counter() - started, 3),
    )


async def run_all(cases: list[EvalCase]) -> list[CaseResult]:
    return [await run_case_offline(c) for c in cases]


# --------------------------------------------------------------------------- reporting / regression
def summarize(results: list[CaseResult]) -> dict[str, Any]:
    scored = [r.score for r in results if r.score is not None]
    return {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "pass_rate": round(sum(r.passed for r in results) / max(len(results), 1), 4),
        "mean_score": round(sum(scored) / len(scored), 2) if scored else None,
        "scores": {r.id: r.score for r in results if r.score is not None},
    }


def compare_to_baseline(summary: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Return regression messages (empty = OK). Improvements never fail the gate."""
    problems: list[str] = []
    if summary["pass_rate"] < baseline["pass_rate"]:
        problems.append(f"pass rate fell {baseline['pass_rate']:.2%} -> {summary['pass_rate']:.2%}")
    if (
        baseline.get("mean_score")
        and summary.get("mean_score") is not None
        and summary["mean_score"] < baseline["mean_score"] - SCORE_TOLERANCE
    ):
        problems.append(f"mean score fell {baseline['mean_score']} -> {summary['mean_score']}")
    for cid, old in baseline.get("scores", {}).items():
        new = summary["scores"].get(cid)
        if new is None:
            problems.append(f"{cid}: no longer produces a scored itinerary (was {old})")
        elif new < old - SCORE_TOLERANCE:
            problems.append(f"{cid}: score fell {old} -> {new}")
    return problems


def load_baseline(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def to_json(results: list[CaseResult], summary: dict[str, Any]) -> str:
    return json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2)
