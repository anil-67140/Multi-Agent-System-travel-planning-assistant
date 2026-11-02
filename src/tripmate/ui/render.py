"""Turns API payloads into display text. Pure functions (unit-tested; no Gradio import)."""

from __future__ import annotations

from typing import Any

ORIGIN_LABEL = {
    "live": "live",
    "cached": "cached",
    "estimate": "estimate",
    "llm_knowledge": "model knowledge",
}
STATUS_LABEL = {
    "queued": "Queued",
    "running": "Planning...",
    "needs_clarification": "Needs more information",
    "awaiting_approval": "Ready for your review",
    "completed": "Approved",
    "blocked": "Request not allowed",
    "failed": "Failed",
}
NODE_LABEL = {
    "guardrail": "Safety check",
    "intake": "Understanding request",
    "flights": "Flights",
    "hotels": "Hotels",
    "weather": "Weather",
    "assemble": "Checking data",
    "optimize": "Comparing options",
    "budget": "Budget",
    "itinerary": "Itinerary",
    "finalize": "Finishing",
}


def _money(m: dict[str, Any] | None) -> str:
    if not m:
        return "-"
    return f"{float(m['amount']):,.2f} {m['currency']}"


def status_text(status: str) -> str:
    return STATUS_LABEL.get(status, status)


def progress_markdown(events: list[dict[str, Any]], status: str) -> str:
    lines = [f"**{status_text(status)}**", ""]
    for e in events[-12:]:
        label = NODE_LABEL.get(e.get("node") or "", "")
        mark = "warning" if e.get("data", {}).get("degraded") else "ok"
        prefix = f"[{mark}] " if e["type"] in {"node_done", "repair"} else ""
        lines.append(f"- {prefix}{label + ': ' if label else ''}{e['message']}")
    return "\n".join(lines)


def itinerary_markdown(itinerary: dict[str, Any] | None) -> str:
    if not itinerary:
        return "_No itinerary yet._"
    out = [f"## {itinerary['title']}", itinerary.get("summary", ""), ""]
    for day in itinerary["days"]:
        out.append(
            f"### Day {day['day_number']} - {day['date']}"
            + (f": {day['theme']}" if day.get("theme") else "")
        )
        if day.get("weather_note"):
            out.append(f"_{day['weather_note']}_")
        for a in day["activities"]:
            cost = f" - {_money(a['estimated_cost'])}" if a.get("estimated_cost") else ""
            out.append(f"- **{a['start'][:5]}-{a['end'][:5]}** {a['title']}{cost}")
            if a.get("description"):
                out.append(f"  {a['description']}")
        out.append("")
    if itinerary.get("tips"):
        out += ["### Tips", *[f"- {t}" for t in itinerary["tips"]]]
    return "\n".join(out)


def budget_rows(budget: dict[str, Any] | None) -> list[list[str]]:
    if not budget:
        return []
    rows = [
        [
            ln["category"].replace("_", " ").title(),
            _money(ln["amount"]),
            "estimate" if ln["is_estimate"] else "quoted",
            ln["basis"],
        ]
        for ln in budget["lines"]
    ]
    rows.append(["TOTAL", _money(budget["total"]), "", budget["verdict"].replace("_", " ")])
    return rows


def budget_summary(budget: dict[str, Any] | None) -> str:
    if not budget:
        return ""
    if budget.get("limit"):
        rem = float(budget["remaining"] or 0)
        word = "left" if rem >= 0 else "over"
        return (
            f"Total **{_money(budget['total'])}** of {_money(budget['limit'])} budget "
            f"({abs(rem):,.0f} {budget['currency']} {word}, {budget.get('utilization_pct', 0):.0f}% used)."
        )
    return f"Total **{_money(budget['total'])}** (no budget set)."


def bundle_choices(bundles: list[dict[str, Any]] | None) -> list[tuple[str, str]]:
    out = []
    for b in bundles or []:
        hours = b["travel_minutes"] / 60
        flag = "" if b.get("within_budget", True) else " (over budget)"
        out.append(
            (
                f"{b['label'].replace('_', ' ').title()}: {_money(b['total_cost'])}, {hours:.1f}h travel, "
                f"quality {b['quality']:.0%}{flag}",
                b["label"],
            )
        )
    return out


def sources_markdown(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "_No external sources were used._"
    seen, lines = set(), []
    for s in sources:
        key = (s["provider"], s.get("url"))
        if key in seen:
            continue
        seen.add(key)
        label = s.get("title") or s["provider"]
        link = f"[{label}]({s['url']})" if s.get("url") else label
        lines.append(
            f"- {link} - {s['kind'].replace('_', ' ')}, {ORIGIN_LABEL.get(s['origin'], s['origin'])}, "
            f"retrieved {str(s['retrieved_at'])[:16].replace('T', ' ')} UTC"
        )
    return "\n".join(lines)


def problems_markdown(plan: dict[str, Any]) -> str:
    lines: list[str] = []
    for n in plan.get("notes", []):
        lines.append(f"- {n}")
    for e in plan.get("tool_errors", []):
        retry = " (can be retried)" if e.get("retryable") else ""
        lines.append(f"- {e['provider']}: {e['message']}{retry}")
    for i in plan.get("issues", []):
        lines.append(f"- {i['message']}")
    for a in (plan.get("trip") or {}).get("assumptions", []):
        lines.append(f"- Assumed: {a}")
    return "\n".join(lines) if lines else "_Nothing to flag._"


def problem_message(status: int, body: Any) -> str:
    """Human message from a problem+json body (or a fallback)."""
    if isinstance(body, dict):
        detail = body.get("detail") or "Request failed."
        if body.get("errors"):
            detail += " " + "; ".join(f"{e['field']}: {e['message']}" for e in body["errors"][:3])
        if status == 429:
            detail += " Please wait a moment and try again."
        return detail
    return f"Request failed ({status})."
