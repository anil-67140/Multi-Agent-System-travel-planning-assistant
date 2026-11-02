# Security

**Reporting:** open a private security advisory on GitHub; do not file public issues for vulnerabilities.

## Controls
* Auth: scrypt (n=2^14) hashes, constant-time verification incl. unknown users, 5-failure/15-minute lockout, HS256 JWT with pinned algorithm, issuer and required claims, 60-minute default TTL.
* Authorisation: every plan access goes through `PlanService.get_owned`; foreign and missing plans are indistinguishable (404).
* Input: all request models forbid unknown fields and bound sizes; 64 KiB body cap (declared and streamed); validation errors never echo submitted values.
* Prompt injection: deterministic pattern gate before any model; user text is fenced as untrusted data in every prompt; model output is never trusted for numbers (budget is code) and hotel facts must cite a URL from the search results.
* Abuse: per-IP and per-user rate limits (Redis-backed when configured); concurrent-plan cap; bounded queue.
* Secrets: environment only, `SecretStr` everywhere, production refuses to start with the default/short JWT secret, SQLite, wildcard CORS or no checkpoint DB; log redaction for key shapes, bearer tokens, JWTs, DB URL passwords and e-mails.
* Transport/headers: nosniff, frame deny, no-referrer, HSTS in production, docs/OpenAPI disabled in production, `/metrics` optionally token-protected.
* Supply chain: pinned-range deps, Dependabot, pip-audit, bandit, gitleaks, trivy, CodeQL, non-root container.

## Not covered
No e-mail verification or password reset, no MFA, no per-tenant encryption at rest. Treat as a portfolio-grade
baseline, not a regulated-data system.
