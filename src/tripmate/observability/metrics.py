"""Prometheus metrics. One registry per app instance (tests create many apps)."""

from __future__ import annotations

from typing import Any

from tripmate.llm.base import LLMCallRecord


class Metrics:
    def __init__(self) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        self.registry = CollectorRegistry()
        self.http_requests = Counter(
            "tripmate_http_requests_total",
            "HTTP requests",
            ["method", "route", "status"],
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "tripmate_http_request_seconds",
            "HTTP latency",
            ["route"],
            registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
        )
        self.llm_calls = Counter(
            "tripmate_llm_calls_total",
            "LLM calls",
            ["task", "model", "outcome"],
            registry=self.registry,
        )
        self.llm_latency = Histogram(
            "tripmate_llm_call_seconds",
            "LLM latency",
            ["task"],
            registry=self.registry,
            buckets=(0.25, 0.5, 1, 2, 4, 8, 16, 32, 64),
        )
        self.llm_tokens = Counter(
            "tripmate_llm_tokens_total", "LLM tokens", ["direction"], registry=self.registry
        )
        self.llm_fallbacks = Counter(
            "tripmate_llm_fallbacks_total",
            "Calls served by a fallback model",
            registry=self.registry,
        )
        self.tool_events = Counter(
            "tripmate_tool_events_total",
            "Provider resilience events",
            ["provider", "event"],
            registry=self.registry,
        )
        self.plans_finished = Counter(
            "tripmate_plans_finished_total",
            "Plans reaching a terminal/paused state",
            ["status"],
            registry=self.registry,
        )
        self.jobs_pending = Gauge("tripmate_jobs_pending", "Queued jobs", registry=self.registry)
        self.rate_limited = Counter(
            "tripmate_rate_limited_total",
            "Requests rejected by rate limit",
            ["scope"],
            registry=self.registry,
        )

    # ---- sinks wired into the rest of the app -----------------------------
    def observe_http(self, method: str, route: str, status: int, seconds: float) -> None:
        self.http_requests.labels(method, route, str(status)).inc()
        self.http_latency.labels(route).observe(seconds)

    def observe_llm(self, rec: LLMCallRecord) -> None:
        self.llm_calls.labels(rec.task, rec.model, rec.outcome).inc()
        if rec.outcome == "ok":
            self.llm_latency.labels(rec.task).observe(rec.latency_ms / 1000)
            self.llm_tokens.labels("prompt").inc(rec.prompt_tokens)
            self.llm_tokens.labels("completion").inc(rec.completion_tokens)
            if rec.fallback_used:
                self.llm_fallbacks.inc()

    def tool_sink(self, event: str, name: str, fields: Any) -> None:
        self.tool_events.labels(name, event).inc()

    def render(self) -> tuple[bytes, str]:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(self.registry), CONTENT_TYPE_LATEST
