"""Ground-truth cases (JSON Lines). One line = one scenario with expectations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_DATASET = Path(__file__).resolve().parents[3] / "evals" / "datasets" / "golden.jsonl"


class Expect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "awaiting_approval"
    destination: str | None = None
    days: int | None = None
    travelers: int | None = None
    currency: str | None = None
    budget_verdicts: list[str] | None = None  # allowed verdicts
    interests: list[str] = Field(
        default_factory=list
    )  # keywords that should appear in the itinerary
    min_score: float = 70.0
    note_contains: list[str] = Field(
        default_factory=list
    )  # honesty: degradations must be disclosed
    clarification_contains: str | None = None


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    message: str
    # Offline mode replays this as the model's extraction so the pipeline (not the LLM) is measured.
    intent: dict[str, Any] = Field(default_factory=dict)
    guard_allowed: bool = True
    tags: list[str] = Field(default_factory=list)
    expect: Expect = Field(default_factory=Expect)


def load_cases(path: Path | None = None, tags: set[str] | None = None) -> list[EvalCase]:
    path = path or DEFAULT_DATASET
    cases: list[EvalCase] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            cases.append(EvalCase.model_validate(json.loads(line)))
        except ValueError as exc:
            raise ValueError(f"{path.name} line {number}: {exc}") from exc
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case ids in dataset")
    return [c for c in cases if not tags or tags & set(c.tags)]
