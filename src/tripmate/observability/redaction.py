"""Scrub secrets and personal data from structured log events (pure stdlib)."""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[REDACTED]"
SENSITIVE_KEYS = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|authorization|cookie|jwt|credential|dsn)",
    re.IGNORECASE,
)
_VALUE_PATTERNS = [
    re.compile(r"(?i)bearer\s+[a-z0-9._\-]{8,}"),
    re.compile(r"\bgsk_[A-Za-z0-9]{16,}\b"),  # Groq
    re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b"),  # Google
    re.compile(r"\btvly-[A-Za-z0-9\-]{12,}\b"),  # Tavily
    re.compile(r"\bsk-[A-Za-z0-9\-_]{12,}\b"),
    re.compile(r"\bey[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"),  # JWT
    re.compile(r"(?i)(postgres(?:ql)?(?:\+\w+)?://[^:/\s]+:)[^@\s]+(@)"),  # DB URL password
]
_EMAIL = re.compile(r"\b([A-Za-z0-9._%+\-])[A-Za-z0-9._%+\-]*@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")
MAX_DEPTH = 6


def redact_text(text: str) -> str:
    for pattern in _VALUE_PATTERNS:
        text = pattern.sub(
            lambda m: (m.group(1) + REDACTED + m.group(2)) if m.lastindex == 2 else REDACTED, text
        )
    return _EMAIL.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", text)


def redact(value: Any, _depth: int = 0) -> Any:
    if _depth > MAX_DEPTH:
        return REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {
            k: (
                REDACTED
                if isinstance(k, str) and SENSITIVE_KEYS.search(k)
                else redact(v, _depth + 1)
            )
            for k, v in value.items()
        }
    if isinstance(value, list | tuple | set):
        return [redact(v, _depth + 1) for v in value]
    return value


def redaction_processor(_logger: Any, _name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor."""
    return redact(event_dict)
