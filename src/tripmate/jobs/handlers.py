"""Job names -> service calls. Shared by every queue backend."""

from __future__ import annotations

import logging

from tripmate.logging_config import plan_id_var
from tripmate.services.plan_service import PlanService

log = logging.getLogger(__name__)

RUN_PLAN = "run_plan"
RESUME_PLAN = "resume_plan"
KNOWN_JOBS = {RUN_PLAN, RESUME_PLAN}


async def dispatch(service: PlanService, name: str, arg: str) -> None:
    """Handlers are idempotent, so at-least-once delivery (retries, redelivery) is safe."""
    if name not in KNOWN_JOBS:
        log.error("unknown job", extra={"job": name})
        return
    if name == RUN_PLAN:
        plan_id_var.set(arg)
        await service.run_plan(arg)
    else:
        plan_id, _, key = arg.partition(":")
        plan_id_var.set(plan_id)
        await service.resume_plan(plan_id, key)
