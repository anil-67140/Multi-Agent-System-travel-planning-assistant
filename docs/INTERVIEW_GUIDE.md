# Interview guide

## 60-second pitch
"TripMate is a multi-agent travel planner. A LangGraph workflow fans out to flight, hotel, weather and FX tools in
parallel, validates every number, ranks alternatives with a Pareto optimiser, drafts an itinerary that must pass
hard constraint checks, and then pauses for human approval with state in PostgreSQL, so it survives restarts.
The model only writes prose and suggestions; money, dates and ranking are deterministic code. It has auth, rate
limits, idempotent approvals, a LiteLLM gateway with Groq-to-Gemini fallback, tracing, metrics, CI with an
evaluation regression gate, and it runs entirely on free tiers."

## Three-minute demo
1. Plan "5 days in Tokyo from Karachi, 3000 USD, food and temples" - show live progress and the partial results.
2. Open **Sources & notes**: flights are labelled *estimate*, hotels cite a URL or say "typical regional rate".
3. **Edit trip**: change travelers to 4 - a new run starts, version history is kept.
4. **Alternatives**: switch from balanced to budget saver; the budget changes.
5. Request changes ("more street food") -> version 2; approve.
6. Show `/metrics`, a Langfuse trace (one session per plan), and the green CI run with the eval gate.

## Design decisions you should be able to defend
| Decision | Why | Trade-off |
|---|---|---|
| Code does arithmetic, model writes prose | LLM maths is unreliable and unauditable | less "magic", more modelling work |
| Flights are estimates, labelled | no free real-time API; faking quotes is worse than saying so | lower realism; protocol makes a real API a drop-in |
| Interrupt node holds only `interrupt()` | resume re-runs the node from line 1 | one extra tiny node |
| New run per edit (`thread_id=plan:run`) | never rewrite checkpoint history | more threads to garbage-collect |
| Approval bound to version + Idempotency-Key | stale tabs / double clicks must not approve unseen content | client must send two extra fields |
| SSE by polling DB | works across replicas with no pub/sub | ~0.7 s latency floor |
| Deterministic evals, no LLM judge | reproducible regressions, free in CI | does not measure live model quality (yet) |
| Own retry/breaker/bulkhead (small) | needed per-event metrics and exact semantics | more code to maintain than tenacity |
| Structured output via schema-in-prompt + repair | provider-independent | one extra call on failures |

## Likely questions
* **How do you stop the model inventing hotels?** Extraction may only return hotels that cite a URL from the search hits; entries without a price or a valid URL are dropped; if nothing survives, labelled regional averages are used.
* **What if Groq is down or rate limited?** Router retries, then falls back to Gemini per tier; a circuit/bulkhead sheds load with 503; metrics + alert on fallback share.
* **How is resume safe?** Checkpoints in Postgres; `is_waiting_for_approval` guard; approval row unique per key; job handlers are idempotent; tested across a simulated restart.
* **How do you know a change did not make plans worse?** Golden cases + scorers; CI fails if pass rate, mean score or any case drops past tolerance.
* **Prompt injection?** Deterministic gate, fenced untrusted text, no model-controlled side effects or numbers, strict schemas.
* **How would you scale 100x?** Switch to ARQ workers (already supported), Redis cache/limiter, read replicas, raise provider quotas or move to paid models behind the same gateway; the LLM quota is the real limit.
* **What would you do next?** A live-model eval runner, a real flight API, LISTEN/NOTIFY for streams, email verification, per-user cost budgets.

## Be upfront about
* Flight prices are estimates and some framework-heavy paths (LangGraph/FastAPI/Gradio wiring) were written in an environment without those packages installed; CI is where they are first executed. Say you ran `make test` and `make eval` locally and in CI before presenting.
