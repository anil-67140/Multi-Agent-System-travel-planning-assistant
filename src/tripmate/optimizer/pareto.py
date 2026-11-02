"""Pareto dominance, generic over any list of objective vectors (all minimised)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """``a`` dominates ``b`` if it is no worse everywhere and strictly better somewhere."""
    return all(x <= y for x, y in zip(a, b, strict=True)) and any(
        x < y for x, y in zip(a, b, strict=True)
    )


def pareto_front(items: Sequence[T], objectives: Callable[[T], Sequence[float]]) -> list[T]:
    """Return the non-dominated items, preserving input order. O(n^2): n is tiny here."""
    vectors = [tuple(objectives(item)) for item in items]
    front: list[T] = []
    for i, item in enumerate(items):
        if not any(dominates(vectors[j], vectors[i]) for j in range(len(items)) if j != i):
            front.append(item)
    return front
