# ADR 0004: Approvals are bound to an itinerary version and are idempotent

**Problem:** a double click, a retry or a stale tab must never approve something the person did not see.
**Decision:** `POST /decision` requires `itinerary_version` and an `Idempotency-Key`. A stale version -> 409. The same
key + same decision returns the stored result; the same key + different decision -> 409. The plan flips to `running`
before enqueueing the resume job, which also checks the graph is actually waiting.
**Consequence:** exactly-once resume without distributed locks.
