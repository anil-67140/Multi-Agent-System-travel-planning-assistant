# Architecture

## Layers (dependencies point downward)

```
api/ ui/ jobs/           transport: HTTP, Gradio, queue consumers
services/                use cases: PlanService (state machine, idempotency, ownership)
graph/ agents/           orchestration: LangGraph topology, node functions
llm/ providers/          adapters: LiteLLM gateway, Open-Meteo, FX, Tavily, estimator
optimizer/ validation/   pure logic: Pareto, budget math, travel-data validators
resilience/              pure infrastructure patterns: retry, breaker, bulkhead, cache, rate limit
schemas/ domain/         pure model: Money, FxRates, TripRequest, Itinerary, SourceRef
```

`schemas/`, `domain/`, `validation/`, `optimizer/`, `resilience/`, `agents/` and `services/` import no web, DB or
LangGraph framework, so ~390 tests run in under two seconds with no network.

## Workflow

State is a `TypedDict` of JSON-compatible dicts (checkpoint-safe). Parallel nodes write disjoint keys; the
shared lists (`tool_errors`, `sources`, `notes`) use an additive reducer. The `human_approval` node contains only
`interrupt()`, because LangGraph re-executes an interrupted node from its first line on resume.

A plan has many *runs*. `thread_id = "{plan_id}:{run_no}"`. Editing constraints, picking an alternative or retrying
starts a new run rather than mutating a checkpoint, so history is never rewritten and a thread is never restarted.

## Plan lifecycle

`queued -> running -> awaiting_approval -> (running -> awaiting_approval)* -> completed`,
with `needs_clarification`, `blocked`, `failed` as side exits. `decide()` flips the plan to `running` before the
resume job is queued, so a second click is rejected, and the approval row is unique on `(plan_id, Idempotency-Key)`.

## Data honesty

Every figure carries a `SourceRef` (`live | cached | estimate | llm_knowledge`, retrieved-at, URL). The budget is
computed by code, never by the model; the model only writes prose and activity suggestions, and its output must
pass `validate_itinerary` (overlaps, day count, dates, pace, meals, currency, arrival buffer, weather) or it is sent
back with the error list (bounded repairs).

## Evaluation

Offline mode replays each case's *ground-truth extraction* through the real pipeline with a reference planner. That
measures the system around the model (resolution, validation, optimisation, budget, scoring) deterministically,
for free, in CI. Quality score (0-100) = constraints 30, budget 15, interests 15, pacing 10, variety 10, meals 10,
weather 5, cost realism 5. The gate fails on a lower pass rate, a lower mean (tolerance 2 points) or any case
regressing more than 2 points. Measuring the *live* model (extraction accuracy, itinerary quality) is the next
step: the cases already carry the expectations; only a live runner is missing.

## Scaling

The API is stateless. Work goes through a queue; with `TRIPMATE_QUEUE_BACKEND=arq` any number of worker containers
consume it (jobs are idempotent, ARQ job ids dedupe). SSE reads events from the database by `seq`, so any replica
can serve any stream and `Last-Event-ID` resumes after a drop. Rate limits and the cache move to Redis when
`TRIPMATE_REDIS_URL` is set. The LLM bulkhead bounds in-flight model calls per process; free-tier quota is the
real ceiling, which is why overload sheds with a 503 instead of queueing without limit.

## Known limitations (deliberate, documented)

* No free real-time flight-price API exists; flights are distance-based estimates behind a `FlightProvider` protocol.
* Hotels depend on Tavily (free tier) to find real prices; without a key only labelled regional averages are shown.
* The in-process queue loses queued jobs on restart (stale runs are failed by recovery); use ARQ for durability.
* Offline evals do not measure model quality.
