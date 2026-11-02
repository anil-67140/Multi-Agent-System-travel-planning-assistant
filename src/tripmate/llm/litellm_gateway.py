"""LiteLLM-backed gateway: model routing, fallbacks and key wiring.

Two deployment modes, chosen by configuration:

* **SDK mode (default)** - an in-process ``litellm.Router`` with ordered
  fallbacks, e.g. Groq (fast, free) -> Gemini (free) when Groq is rate limited.
* **Proxy mode** - set ``litellm_proxy_url`` to call a central LiteLLM Proxy
  (see ``deploy/litellm/config.yaml``) that owns keys, budgets and fallbacks.

``build_deployments`` is a pure function so the routing config is unit-testable
without importing litellm.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from tripmate.errors import LLMError, UpstreamUnavailableError
from tripmate.llm.base import BaseGateway, Completion, LLMCallRecord, Observer, Tier
from tripmate.resilience import Bulkhead, BulkheadFullError

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GatewayConfig:
    # Ordered: first = primary, rest = fallbacks. ``provider/model`` as LiteLLM expects.
    smart_models: tuple[str, ...] = ("groq/openai/gpt-oss-120b", "gemini/gemini-3.8-flash")
    fast_models: tuple[str, ...] = ("groq/openai/gpt-oss-20b", "gemini/gemini-3.5-flash-lite")
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    timeout_seconds: float = 45.0
    num_retries: int = 2
    cooldown_seconds: int = 30
    allowed_fails: int = 3
    max_concurrency: int = 4
    max_waiting: int = 16
    reasoning_effort: str | None = "low"  # only forwarded to gpt-oss models
    proxy_url: str | None = None
    proxy_key: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _api_key_for(model: str, cfg: GatewayConfig) -> str | None:
    if model.startswith("groq/"):
        return cfg.groq_api_key
    if model.startswith("gemini/"):
        return cfg.gemini_api_key
    return None


def build_deployments(
    cfg: GatewayConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, list[str]]], dict[Tier, str]]:
    """Return ``(model_list, fallbacks, entry_names_by_tier)`` for ``litellm.Router``.

    Each model becomes its own named deployment (``smart-0``, ``smart-1`` ...) and
    a fallback chain links them in order. Models without an API key are skipped,
    so a student with only a Groq key still gets a working (fallback-less) gateway.
    """
    model_list: list[dict[str, Any]] = []
    fallbacks: list[dict[str, list[str]]] = []
    entries: dict[Tier, str] = {}

    for tier, models in (("smart", cfg.smart_models), ("fast", cfg.fast_models)):
        names: list[str] = []
        for model in models:
            key = _api_key_for(model, cfg)
            if not key:
                continue
            name = f"{tier}-{len(names)}"
            params: dict[str, Any] = {"model": model, "api_key": key}
            if cfg.reasoning_effort and "gpt-oss" in model:
                params["reasoning_effort"] = cfg.reasoning_effort
            model_list.append({"model_name": name, "litellm_params": params})
            names.append(name)
        if not names:
            continue
        entries[tier] = names[0]  # type: ignore[index]
        if len(names) > 1:
            fallbacks.append({names[0]: names[1:]})
            # Let the second model also fall back to the rest if it fails too.
            for i in range(1, len(names) - 1):
                fallbacks.append({names[i]: names[i + 1 :]})
    return model_list, fallbacks, entries  # type: ignore[return-value]


def build_proxy_entries(cfg: GatewayConfig) -> dict[Tier, str]:
    """In proxy mode the proxy owns routing; we only name the logical model groups."""
    return {"smart": "openai/smart", "fast": "openai/fast"}


class LiteLLMGateway(BaseGateway):
    def __init__(
        self, cfg: GatewayConfig, observer: Observer | None = None, router: Any = None
    ) -> None:
        self._cfg = cfg
        self._observer = observer
        self._bulkhead = Bulkhead("llm", cfg.max_concurrency, max_waiting=cfg.max_waiting)

        if cfg.proxy_url:
            self._router = None
            self._entries = build_proxy_entries(cfg)
            self._primary = dict(self._entries)
        else:
            model_list, fallbacks, entries = build_deployments(cfg)
            if not entries:
                raise LLMError(
                    "No LLM is configured. Set GROQ_API_KEY and/or GEMINI_API_KEY (both are free)."
                )
            self._entries = entries
            self._primary = {
                t: (cfg.smart_models if t == "smart" else cfg.fast_models)[0] for t in entries
            }
            if router is not None:
                self._router = router
            else:
                import litellm
                from litellm import Router

                litellm.drop_params = True  # e.g. reasoning_effort is not valid for Gemini
                self._router = Router(
                    model_list=model_list,
                    fallbacks=fallbacks,
                    num_retries=cfg.num_retries,
                    timeout=cfg.timeout_seconds,
                    allowed_fails=cfg.allowed_fails,
                    cooldown_time=cfg.cooldown_seconds,
                    routing_strategy="simple-shuffle",
                )

    def _observe(self, record: LLMCallRecord) -> None:
        if self._observer:
            try:
                self._observer(record)
            except Exception:  # an observer must never break a request
                log.exception("llm observer failed")

    def _route_for(self, tier: Tier) -> str:
        # If the requested tier has no deployment (e.g. only "smart" is configured) use what exists.
        return self._entries.get(tier) or next(iter(self._entries.values()))

    async def _complete(
        self,
        messages: list[dict[str, str]],
        *,
        task: str,
        tier: Tier,
        temperature: float,
        max_tokens: int | None,
        json_mode: bool,
        metadata: dict[str, Any] | None,
    ) -> Completion:
        route = self._route_for(tier)
        kwargs: dict[str, Any] = {
            "model": route,
            "messages": messages,
            "temperature": temperature,
            "timeout": self._cfg.timeout_seconds,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        # Picked up by the Langfuse callback: groups every call of one plan into a session.
        kwargs["metadata"] = {"generation_name": task, **(metadata or {})}

        async def call() -> Any:
            if self._router is not None:
                return await self._router.acompletion(**kwargs)
            import litellm

            return await litellm.acompletion(
                **{
                    **kwargs,
                    "api_base": self._cfg.proxy_url,
                    "api_key": self._cfg.proxy_key or "sk-none",
                }
            )

        started = time.perf_counter()
        try:
            response = await self._bulkhead.run(call)
        except BulkheadFullError as exc:
            self._observe(LLMCallRecord(task, tier, route, "shed", 0.0, metadata=metadata or {}))
            raise UpstreamUnavailableError(
                "The AI service is busy. Please retry in a moment."
            ) from exc
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            self._observe(
                LLMCallRecord(task, tier, route, "error", latency, metadata=metadata or {})
            )
            log.warning(
                "llm call failed", extra={"task": task, "tier": tier, "error": type(exc).__name__}
            )
            raise LLMError(
                "The language model could not be reached.", detail=f"{type(exc).__name__}: {exc}"
            ) from exc

        latency = (time.perf_counter() - started) * 1000
        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMError("The language model returned an empty response.") from exc
        usage = getattr(response, "usage", None)
        served = str(getattr(response, "model", None) or route)
        primary = self._primary.get(tier, "")
        fallback_used = bool(primary) and primary.split("/")[-1] not in served
        completion = Completion(
            text=text,
            model=served,
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=latency,
            fallback_used=fallback_used,
        )
        self._observe(
            LLMCallRecord(
                task,
                tier,
                served,
                "ok",
                latency,
                completion.prompt_tokens,
                completion.completion_tokens,
                fallback_used,
                metadata or {},
            )
        )
        return completion
