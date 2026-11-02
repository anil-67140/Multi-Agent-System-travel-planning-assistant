# ADR 0003: PostgreSQL for checkpoints and progress events; SSE by polling

**Decision:** LangGraph `AsyncPostgresSaver` stores workflow state; our own tables store trips, versions,
approvals, sources and `run_events`. SSE streams by polling `run_events` for `seq > Last-Event-ID`.
**Why:** one datastore to operate on a free tier; no pub/sub needed; works across replicas and reconnects.
**Trade-off:** ~0.7 s latency floor and one cheap indexed query per open stream. Revisit with LISTEN/NOTIFY or Redis
streams if streams reach the thousands.
