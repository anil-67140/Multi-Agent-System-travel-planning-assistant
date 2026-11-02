"""A uniform way to report validation problems without raising."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Severity(StrEnum):
    ERROR = "error"  # the data is wrong / unusable
    WARNING = "warning"  # usable, but the traveller should know
    INFO = "info"


class Issue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    severity: Severity = Severity.ERROR
    path: str | None = None  # e.g. "days[2].activities[1]"


def errors(issues: Iterable[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity == Severity.ERROR]


def warnings(issues: Iterable[Issue]) -> list[Issue]:
    return [i for i in issues if i.severity == Severity.WARNING]


def has_errors(issues: Iterable[Issue]) -> bool:
    return any(i.severity == Severity.ERROR for i in issues)


def format_issues(issues: Iterable[Issue]) -> str:
    """Compact, model-friendly rendering used when asking the LLM to repair output."""
    return "\n".join(f"- [{i.severity.value}] {i.code}: {i.message}" for i in issues)
