"""Deterministic, explainable itinerary-quality scoring (0-100).

No LLM-as-judge: scores must be reproducible so a regression means the system changed, not the judge.
Each component returns 0..1 plus a reason; the weights sum to 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from tripmate.schemas.budget import BudgetBreakdown, BudgetVerdict
from tripmate.schemas.itinerary import ActivityCategory, Itinerary
from tripmate.schemas.search import WeatherReport
from tripmate.schemas.trip import Pace, TripRequest
from tripmate.validation import Issue, has_errors, validate_itinerary

WEIGHTS = {
    "constraints": 0.30,
    "budget": 0.15,
    "interests": 0.15,
    "pacing": 0.10,
    "variety": 0.10,
    "meals": 0.10,
    "weather": 0.05,
    "cost_realism": 0.05,
}
assert abs(sum(WEIGHTS.values()) - 1) < 1e-9

TARGET_ACTIVITIES = {Pace.RELAXED: (2, 3), Pace.BALANCED: (3, 5), Pace.PACKED: (5, 8)}


@dataclass(frozen=True)
class Component:
    name: str
    score: float
    reason: str


@dataclass
class QualityReport:
    components: list[Component]
    issues: list[Issue] = field(default_factory=list)

    @property
    def total(self) -> float:
        return round(100 * sum(WEIGHTS[c.name] * c.score for c in self.components), 1)

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "components": {c.name: round(c.score, 3) for c in self.components},
        }


def _text(it: Itinerary) -> str:
    parts = []
    for d in it.days:
        parts.append(d.theme)
        for a in d.activities:
            parts += [a.title, a.description, a.location or ""]
    return " ".join(parts).lower()


def score_constraints(issues: list[Issue]) -> Component:
    errs = [i for i in issues if i.severity.value == "error"]
    warns = [i for i in issues if i.severity.value == "warning"]
    score = 0.0 if errs else max(0.0, 1.0 - 0.1 * len(warns))
    return Component("constraints", score, f"{len(errs)} error(s), {len(warns)} warning(s)")


def score_budget(budget: BudgetBreakdown | None) -> Component:
    if budget is None:
        return Component("budget", 0.0, "no budget computed")
    return Component(
        "budget",
        {
            BudgetVerdict.WITHIN_BUDGET: 1.0,
            BudgetVerdict.NO_LIMIT: 1.0,
            BudgetVerdict.TIGHT: 0.8,
            BudgetVerdict.OVER_BUDGET: 0.2,
        }[budget.verdict],
        f"verdict {budget.verdict.value}",
    )


def score_interests(it: Itinerary, trip: TripRequest) -> Component:
    if not trip.interests:
        return Component("interests", 1.0, "no interests stated")
    text = _text(it)
    hit = [i for i in trip.interests if i.lower() in text]
    return Component(
        "interests",
        len(hit) / len(trip.interests),
        f"{len(hit)}/{len(trip.interests)} interests covered",
    )


def score_pacing(it: Itinerary, trip: TripRequest) -> Component:
    lo, hi = TARGET_ACTIVITIES[trip.pace]
    ok = sum(1 for d in it.days if lo <= len(d.activities) <= hi)
    return Component(
        "pacing", ok / len(it.days), f"{ok}/{len(it.days)} days match {trip.pace.value} pace"
    )


def score_variety(it: Itinerary) -> Component:
    cats = {a.category for d in it.days for a in d.activities} - {ActivityCategory.OTHER}
    titles = [a.title.strip().lower() for d in it.days for a in d.activities]
    unique = len(set(titles)) / len(titles)
    return Component(
        "variety",
        min(1.0, 0.5 * min(len(cats) / 4, 1) + 0.5 * unique),
        f"{len(cats)} categories, {unique:.0%} unique activities",
    )


def score_meals(it: Itinerary) -> Component:
    ok = sum(1 for d in it.days if any(a.category is ActivityCategory.FOOD for a in d.activities))
    return Component("meals", ok / len(it.days), f"meals on {ok}/{len(it.days)} days")


def score_weather(it: Itinerary, weather: WeatherReport | None) -> Component:
    if weather is None:
        return Component("weather", 0.7, "no weather data (partial credit; disclosed to traveller)")
    wet = {d.date for d in weather.days if d.is_wet}
    bad = sum(1 for d in it.days if d.date in wet for a in d.activities if a.outdoor)
    return Component("weather", 0.0 if bad else 1.0, f"{bad} outdoor activity(ies) on wet days")


def score_cost_realism(
    it: Itinerary, trip: TripRequest, budget: BudgetBreakdown | None
) -> Component:
    planned = sum(
        (a.estimated_cost.amount for d in it.days for a in d.activities if a.estimated_cost),
        Decimal(0),
    )
    if budget is None or planned == 0:
        return Component("cost_realism", 0.8, "no activity costs to compare")
    allowance = next(
        (ln.amount.amount for ln in budget.lines if ln.category.value == "activities"), Decimal(0)
    )
    if allowance == 0:
        return Component("cost_realism", 0.8, "no activity allowance")
    ratio = float(planned / allowance)
    return Component(
        "cost_realism",
        1.0 if ratio <= 1.5 else max(0.0, 1 - (ratio - 1.5) / 2),
        f"planned/allowance={ratio:.2f}",
    )


def score_itinerary(
    it: Itinerary, trip: TripRequest, budget: BudgetBreakdown | None, weather: WeatherReport | None
) -> QualityReport:
    issues = validate_itinerary(it, trip, budget=budget, weather=weather)
    comps = [
        score_constraints(issues),
        score_budget(budget),
        score_interests(it, trip),
        score_pacing(it, trip),
        score_variety(it),
        score_meals(it),
        score_weather(it, weather),
        score_cost_realism(it, trip, budget),
    ]
    report = QualityReport(comps, issues)
    if has_errors(issues):
        report.components[0] = Component("constraints", 0.0, "hard constraint violated")
    return report
