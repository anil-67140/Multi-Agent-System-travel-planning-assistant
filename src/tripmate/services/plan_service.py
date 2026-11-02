"""Plan lifecycle: create, run, approve/revise, edit constraints, pick alternatives, recover.

Invariants this class protects
* A person can only see/act on their own plans (others get NotFound, never Forbidden - no id probing).
* An approval is bound to the itinerary *version* the person saw; approving a stale version fails.
* Approvals are idempotent per ``Idempotency-Key``; a double click never resumes the graph twice.
* A workflow thread is never restarted: edits/alternatives start a *new* run (``run_no + 1``).
* A crashed worker leaves a plan ``running``; ``recover_stale_runs`` marks it failed so the UI is truthful.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from datetime import timedelta
from decimal import Decimal
from typing import Any

from pydantic import ValidationError

from tripmate.domain.clock import Clock
from tripmate.errors import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    RateLimitedError,
    TripMateError,
    ValidationFailure,
)
from tripmate.graph.state import PlanState
from tripmate.schemas.plan import BundleLabel, PlanStatus
from tripmate.schemas.trip import TripRequest
from tripmate.services.records import ApprovalRecord, PlanRecord, PreferenceRecord, SourceEntry
from tripmate.services.repository import Repository
from tripmate.services.workflow import Workflow

log = logging.getLogger(__name__)

_current_run: ContextVar[tuple[str, int] | None] = ContextVar("tripmate_current_run", default=None)
Enqueue = Callable[[str, str], Awaitable[None]]  # (job_name, plan_id)
MAX_ACTIVE_PLANS_PER_USER = 3
EDITABLE_FIELDS = {
    "start_date",
    "end_date",
    "travelers",
    "budget",
    "display_currency",
    "style",
    "pace",
    "interests",
    "dietary",
    "constraints",
}
MAX_FEEDBACK_CHARS = 1000


class PlanService:
    def __init__(
        self,
        repo: Repository,
        workflow: Workflow,
        clock: Clock,
        enqueue: Enqueue,
        *,
        run_timeout: float = 180.0,
        stale_after: timedelta = timedelta(minutes=10),
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self.repo, self.workflow, self.clock, self.enqueue = repo, workflow, clock, enqueue
        self._on_status = on_status or (lambda status: None)  # metrics hook
        self.run_timeout, self.stale_after = run_timeout, stale_after

    # ------------------------------------------------------------------ access
    async def get_owned(self, user_id: str, plan_id: str) -> PlanRecord:
        plan = await self.repo.get_plan(plan_id)
        if plan is None or plan.user_id != user_id:
            raise NotFoundError("Plan not found.")
        return plan

    async def list_plans(self, user_id: str, limit: int = 20, offset: int = 0) -> list[PlanRecord]:
        return await self.repo.list_plans(user_id, min(max(limit, 1), 100), max(offset, 0))

    # ------------------------------------------------------------------ create
    async def create_plan(self, user_id: str, message: str) -> PlanRecord:
        message = message.strip()
        if not message:
            raise BadRequestError("Please describe the trip you want.")
        if await self.repo.count_active_plans(user_id) >= MAX_ACTIVE_PLANS_PER_USER:
            raise RateLimitedError("You already have plans being prepared. Wait for one to finish.")
        plan = await self.repo.create_plan(user_id, message)
        await self._event(plan, "queued", None, "Plan queued", {})
        await self.enqueue("run_plan", plan.id)
        return plan

    # ------------------------------------------------------------------ run (worker side)
    async def run_plan(self, plan_id: str) -> None:
        plan = await self.repo.get_plan(plan_id)
        if plan is None or plan.status in {PlanStatus.RUNNING.value, PlanStatus.COMPLETED.value}:
            return  # duplicate delivery of the job: do nothing
        prefs = await self.repo.get_preferences(plan.user_id)
        plan.status, plan.heartbeat_at, plan.error = (
            PlanStatus.RUNNING.value,
            self.clock.now(),
            None,
        )
        await self.repo.save_plan(plan)
        initial: PlanState = {
            "plan_id": plan.id,
            "user_id": plan.user_id,
            "message": plan.message,
            "prefs": prefs.model_dump(mode="json"),
            "revisions": 0,
            "itinerary_version": plan.latest_version.version if plan.latest_version else 0,
        }
        if plan.run_no > 0 and plan.trip:
            # A follow-up run (edited constraints / chosen alternative / retry): the vetted trip and
            # choice were persisted by the API process, so any worker can pick the job up.
            initial["trip"] = plan.trip
            if plan.selected_label:
                initial["selected_label"] = plan.selected_label
        await self._execute(plan, lambda: self.workflow.start(plan.thread_id, initial))

    async def resume_plan(self, plan_id: str, approval_id: str) -> None:
        plan = await self.repo.get_plan(plan_id)
        if plan is None:
            return
        if not await self.workflow.is_waiting_for_approval(plan.thread_id):
            log.info("resume skipped: not waiting", extra={"plan_id": plan_id})
            return  # already resumed by an earlier delivery
        approval = await self.repo.get_approval(plan.id, approval_id)  # id == idempotency key
        if approval is None:
            return
        decision = {"approved": approval.approved, "feedback": approval.feedback}
        plan.status, plan.heartbeat_at = PlanStatus.RUNNING.value, self.clock.now()
        await self.repo.save_plan(plan)
        await self._execute(
            plan, lambda: self.workflow.resume(plan.thread_id, decision), feedback=approval.feedback
        )

    async def _execute(
        self,
        plan: PlanRecord,
        call: Callable[[], Awaitable[PlanState]],
        *,
        feedback: str | None = None,
    ) -> None:
        token = _current_run.set((plan.id, plan.run_no))
        try:
            state = await asyncio.wait_for(call(), timeout=self.run_timeout)
            await self._apply(plan, state, feedback)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            await self._fail(plan, "The plan took too long to prepare. Please try again.")
        except TripMateError as exc:
            log.warning("run failed", extra={"plan_id": plan.id, "code": exc.code})
            await self._fail(plan, exc.message)
        except Exception:
            log.exception("run crashed", extra={"plan_id": plan.id})
            await self._fail(plan, "Something went wrong while preparing the plan.")
        finally:
            _current_run.reset(token)

    async def _fail(self, plan: PlanRecord, message: str) -> None:
        fresh = await self.repo.get_plan(plan.id) or plan
        fresh.status, fresh.error = PlanStatus.FAILED.value, message
        await self.repo.save_plan(fresh)
        self._on_status(PlanStatus.FAILED.value)
        await self._event(fresh, "failed", None, message, {})

    async def _apply(self, plan: PlanRecord, state: PlanState, feedback: str | None) -> None:
        fresh = await self.repo.get_plan(plan.id) or plan
        trip = state.get("trip")
        if trip:
            fresh.trip = trip
            fresh.destination = (trip.get("destination") or {}).get("name")
            fresh.start_date, fresh.end_date = trip.get("start_date"), trip.get("end_date")
            fresh.travelers = trip.get("travelers")
            fresh.currency = (trip.get("budget") or {}).get("currency") or trip.get(
                "display_currency"
            )
            budget = trip.get("budget")
            fresh.budget_amount = Decimal(str(budget["amount"])) if budget else None
        fresh.budget = state.get("budget") or fresh.budget
        fresh.bundles = state.get("bundles") or fresh.bundles
        fresh.selected_label = state.get("selected_label") or fresh.selected_label
        fresh.notes = list(dict.fromkeys([*fresh.notes, *state.get("notes", [])]))[:30]
        fresh.tool_errors = state.get("tool_errors", fresh.tool_errors)
        fresh.issues = state.get("issues", fresh.issues)
        fresh.clarification = state.get("clarification")
        fresh.heartbeat_at = self.clock.now()

        itinerary, version = state.get("itinerary"), int(state.get("itinerary_version", 0))
        if itinerary and version:
            await self.repo.add_version(fresh.id, version, itinerary, feedback)
        sources = _dedupe_sources(state.get("sources", []))
        if sources:
            await self.repo.replace_sources(fresh.id, sources)

        status = state.get("status") or PlanStatus.RUNNING.value
        if await self.workflow.is_waiting_for_approval(fresh.thread_id):
            status = PlanStatus.AWAITING_APPROVAL.value
        elif status == PlanStatus.RUNNING.value:
            status = (
                PlanStatus.FAILED.value
            )  # finished without a terminal status: never leave "running"
            fresh.error = fresh.error or "The workflow ended unexpectedly."
        if status == PlanStatus.BLOCKED.value:
            fresh.error = (state.get("guardrail") or {}).get(
                "reason"
            ) or "This request can't be handled here."
        if status == PlanStatus.FAILED.value and not fresh.error:
            fresh.error = (state.get("notes") or ["The plan could not be completed."])[-1]
        fresh.status = status
        await self.repo.save_plan(fresh)
        if status == PlanStatus.COMPLETED.value:
            await self._learn(fresh)
        self._on_status(status)
        await self._event(fresh, status, None, f"Plan {status.replace('_', ' ')}", {})

    async def _learn(self, plan: PlanRecord) -> None:
        if not plan.selected_label:
            return
        prefs = await self.repo.get_preferences(plan.user_id)
        prefs.bundle_affinity[plan.selected_label] = (
            prefs.bundle_affinity.get(plan.selected_label, 0) + 1
        )
        await self.repo.save_preferences(prefs)

    # ------------------------------------------------------------------ approval (API side)
    async def decide(
        self,
        user_id: str,
        plan_id: str,
        *,
        itinerary_version: int,
        approved: bool,
        feedback: str | None,
        idempotency_key: str,
    ) -> ApprovalRecord:
        plan = await self.get_owned(user_id, plan_id)
        existing = await self.repo.get_approval(plan.id, idempotency_key)
        if existing is not None:  # a retry of the same request: same answer, no second resume
            if (existing.itinerary_version, existing.approved) != (itinerary_version, approved):
                raise ConflictError(
                    "This Idempotency-Key was already used for a different decision."
                )
            return existing
        if plan.status != PlanStatus.AWAITING_APPROVAL.value:
            raise ConflictError("This plan is not waiting for approval.")
        latest = plan.latest_version
        if latest is None or latest.version != itinerary_version:
            raise ConflictError(
                "The itinerary changed since you viewed it. Reload to review the latest version."
            )
        if not approved and not (feedback or "").strip():
            raise ValidationFailure("Please say what to change when requesting a revision.")
        record = ApprovalRecord(
            id=idempotency_key,
            plan_id=plan.id,
            user_id=user_id,
            itinerary_version=itinerary_version,
            approved=approved,
            feedback=(feedback or "").strip()[:MAX_FEEDBACK_CHARS] or None,
            idempotency_key=idempotency_key,
            created_at=self.clock.now(),
        )
        record = await self.repo.add_approval(record)
        plan.status = (
            PlanStatus.RUNNING.value
        )  # blocks a second decision while the resume job is queued
        plan.heartbeat_at = self.clock.now()
        await self.repo.save_plan(plan)
        await self._event(
            plan,
            "decision",
            None,
            "Approved" if approved else "Revision requested",
            {"version": itinerary_version},
        )
        await self.enqueue("resume_plan", f"{plan.id}:{record.idempotency_key}")
        return record

    # ------------------------------------------------------------------ edits / alternatives
    async def _new_run(self, plan: PlanRecord) -> PlanRecord:
        if plan.status in {PlanStatus.RUNNING.value, PlanStatus.QUEUED.value}:
            raise ConflictError("The plan is still being prepared.")
        plan.run_no += 1
        plan.status, plan.error, plan.clarification = PlanStatus.QUEUED.value, None, None
        await self.repo.save_plan(plan)
        return plan

    async def update_constraints(
        self, user_id: str, plan_id: str, changes: dict[str, Any]
    ) -> PlanRecord:
        plan = await self.get_owned(user_id, plan_id)
        if not plan.trip:
            raise ConflictError("The trip details are not ready yet.")
        unknown = set(changes) - EDITABLE_FIELDS
        if unknown:
            raise ValidationFailure(f"Cannot edit: {', '.join(sorted(unknown))}")
        try:
            new_trip = TripRequest.model_validate(plan.trip).edited(**changes)
        except (ValidationError, ValueError) as exc:
            raise ValidationFailure("Those changes are not valid: " + _first_error(exc)) from exc
        plan.trip = new_trip.model_dump(mode="json")
        plan = await self._new_run(plan)
        await self._event(
            plan,
            "constraints_edited",
            None,
            "Trip constraints changed",
            {"fields": sorted(changes)},
        )
        await self.enqueue("run_plan", plan.id)
        return plan

    async def select_alternative(self, user_id: str, plan_id: str, label: str) -> PlanRecord:
        plan = await self.get_owned(user_id, plan_id)
        try:
            BundleLabel(label)
        except ValueError as exc:
            raise ValidationFailure("Unknown alternative.") from exc
        if not plan.trip or not any(b.get("label") == label for b in plan.bundles or []):
            raise NotFoundError("That alternative is not available for this plan.")
        plan.selected_label = label
        plan = await self._new_run(plan)
        await self._event(
            plan, "alternative_selected", None, f"Switched to {label.replace('_', ' ')}", {}
        )
        await self.enqueue("run_plan", plan.id)
        return plan

    async def retry(self, user_id: str, plan_id: str) -> PlanRecord:
        plan = await self.get_owned(user_id, plan_id)
        if plan.status not in {PlanStatus.FAILED.value, PlanStatus.NEEDS_CLARIFICATION.value}:
            raise ConflictError("Only failed plans can be retried.")
        plan = await self._new_run(plan)
        await self.enqueue("run_plan", plan.id)
        return plan

    async def answer_clarification(self, user_id: str, plan_id: str, answer: str) -> PlanRecord:
        plan = await self.get_owned(user_id, plan_id)
        if plan.status != PlanStatus.NEEDS_CLARIFICATION.value:
            raise ConflictError("This plan is not waiting for more information.")
        answer = answer.strip()
        if not answer:
            raise BadRequestError("Please answer the question.")
        plan.message = f"{plan.message}\n{plan.clarification}\n{answer}"[:2000]
        plan.trip = None  # intake must run again
        plan = await self._new_run(plan)
        await self.enqueue("run_plan", plan.id)
        return plan

    # ------------------------------------------------------------------ preferences
    async def get_preferences(self, user_id: str) -> PreferenceRecord:
        return await self.repo.get_preferences(user_id)

    async def save_preferences(self, prefs: PreferenceRecord) -> PreferenceRecord:
        current = await self.repo.get_preferences(prefs.user_id)
        prefs.bundle_affinity = current.bundle_affinity  # learned data is not user-editable
        await self.repo.save_preferences(prefs)
        return prefs

    # ------------------------------------------------------------------ recovery
    async def recover_stale_runs(self) -> int:
        cutoff = self.clock.now() - self.stale_after
        stale = await self.repo.find_stale_running(cutoff)
        for plan in stale:
            await self._fail(plan, "This plan was interrupted. Please try again.")
        return len(stale)

    async def _event(
        self, plan: PlanRecord, type: str, node: str | None, message: str, data: dict[str, Any]
    ) -> None:
        await self.repo.add_event(plan.id, plan.run_no, type, node, message, data)

    async def emit(self, type: str, node: str | None, message: str, data: dict[str, Any]) -> None:
        """``Deps.emit`` target. The active run is read from a context variable, so one shared
        Deps object serves concurrent plans (context propagates into LangGraph's node tasks)."""
        current = _current_run.get()
        if current is None:
            return
        plan_id, run_no = current
        await self.repo.add_event(plan_id, run_no, type, node, message, data)
        plan = await self.repo.get_plan(plan_id)
        if plan and plan.status == PlanStatus.RUNNING.value:
            plan.heartbeat_at = self.clock.now()  # progress doubles as the liveness heartbeat
            await self.repo.save_plan(plan)


def _first_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError) and exc.errors():
        e = exc.errors()[0]
        return f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
    return str(exc)


def _dedupe_sources(raw: list[dict[str, Any]]) -> list[SourceEntry]:
    seen: dict[tuple[str, str, str | None], SourceEntry] = {}
    for item in raw:
        entry = SourceEntry(
            provider=item["provider"],
            kind=item["kind"],
            origin=item["origin"],
            url=item.get("url"),
            title=item.get("title"),
            retrieved_at=item["retrieved_at"],
        )
        seen.setdefault((entry.provider, entry.kind, entry.url), entry)
    return list(seen.values())
