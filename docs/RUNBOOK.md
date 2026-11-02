# Runbook

| Symptom | Likely cause | Check | Action |
|---|---|---|---|
| Plans stuck in `running` | worker died | `tripmate_jobs_pending`, worker logs by `plan_id` | restart worker; `recover_stale_runs` fails orphans after `TRIPMATE_STALE_RUN_SECONDS`; user retries |
| `LLMFailures` alert | provider outage / quota | `tripmate_llm_calls_total{outcome}`; Langfuse traces | confirm fallback active (`tripmate_llm_fallbacks_total`); add a key for the other provider |
| `LLMFallbackStorm` | primary throttled | Groq console rate limits | lower `TRIPMATE_LLM_MAX_CONCURRENCY`; accept Gemini as primary |
| Many `needs_clarification` | model extraction regressed | run `tripmate-eval --live` (planned) / check intake traces | roll back model list |
| Hotel prices all "typical regional" | no/blocked `TAVILY_API_KEY` | `tool_errors` on a plan | set key; this is a labelled degradation, not an outage |
| 429s for a normal user | rate limits too tight | `tripmate_rate_limited_total{scope}` | raise `TRIPMATE_PLAN_RATE_LIMIT_PER_HOUR` |
| 503 `readyz` | DB down | `/readyz` body shows which check | restore DB; API keeps serving `/healthz` |

Deploy: CI builds and scans the image; `deploy.yml` triggers the host after CI passes. Migrations run as a release
step (`alembic upgrade head`), never at app start. Rollback: redeploy the previous image tag
(`ghcr.io/<repo>:sha-<commit>`); migrations are additive, `alembic downgrade -1` is tested in CI.
