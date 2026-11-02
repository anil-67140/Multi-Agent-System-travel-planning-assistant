"""Multi-objective selection of flight + hotel bundles.

Three objectives: minimise cost, minimise travel time, maximise quality.
We keep only Pareto-optimal combinations, then pick three *alternatives* with
different priorities so the traveller can see real trade-offs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from itertools import product

from tripmate.domain.money import Money
from tripmate.optimizer.pareto import pareto_front
from tripmate.schemas.plan import BundleLabel, PlanBundle
from tripmate.schemas.search import Cabin, FlightOption, HotelOption
from tripmate.schemas.trip import TravelStyle


@dataclass(frozen=True)
class Weights:
    cost: float
    time: float
    quality: float

    def normalised(self) -> Weights:
        total = self.cost + self.time + self.quality
        return Weights(self.cost / total, self.time / total, self.quality / total)


STYLE_WEIGHTS: dict[TravelStyle, Weights] = {
    TravelStyle.BUDGET: Weights(0.60, 0.20, 0.20),
    TravelStyle.MID_RANGE: Weights(0.40, 0.25, 0.35),
    TravelStyle.LUXURY: Weights(0.20, 0.20, 0.60),
}

CABIN_BONUS = {Cabin.ECONOMY: 0.0, Cabin.PREMIUM_ECONOMY: 0.10, Cabin.BUSINESS: 0.25}


def adjust_weights(base: Weights, affinity: dict[str, int]) -> Weights:
    """Shift weights toward what this traveller has chosen before.

    ``affinity`` counts past approved bundles by label. Choosing budget-saver
    bundles raises the cost weight; choosing comfort raises the quality weight.
    """
    total = sum(affinity.values())
    if total <= 0:
        return base
    saver = affinity.get(BundleLabel.BUDGET_SAVER.value, 0) / total
    comfort = affinity.get(BundleLabel.COMFORT.value, 0) / total
    cost = max(0.05, base.cost * (1 + 0.5 * (saver - comfort)))
    quality = max(0.05, base.quality * (1 + 0.5 * (comfort - saver)))
    return Weights(cost, base.time, quality).normalised()


def flight_quality(f: FlightOption) -> float:
    return max(0.0, min(1.0, 1.0 - 0.25 * f.stops + CABIN_BONUS[f.cabin]))


def hotel_quality(h: HotelOption) -> float:
    stars = (h.stars if h.stars is not None else 3.0) / 5.0
    rating = (h.rating if h.rating is not None else 7.5) / 10.0
    return max(0.0, min(1.0, 0.6 * stars + 0.4 * rating))


@dataclass(frozen=True)
class Candidate:
    flight: FlightOption | None
    hotel: HotelOption | None
    cost: Decimal
    travel_minutes: int
    quality: float

    @property
    def key(self) -> tuple[str | None, str | None]:
        return (self.flight.id if self.flight else None, self.hotel.id if self.hotel else None)


@dataclass
class OptimizationResult:
    bundles: list[PlanBundle]
    considered: int
    front_size: int
    notes: list[str] = field(default_factory=list)

    @property
    def recommended(self) -> PlanBundle | None:
        return self.bundles[0] if self.bundles else None

    def get(self, label: BundleLabel) -> PlanBundle | None:
        return next((b for b in self.bundles if b.label == label), None)


def _normalise(values: list[float], v: float) -> float:
    lo, hi = min(values), max(values)
    return 0.0 if hi == lo else (v - lo) / (hi - lo)


def optimize(
    flights: list[FlightOption],
    hotels: list[HotelOption],
    *,
    currency: str,
    other_costs: Decimal = Decimal(0),
    budget: Money | None = None,
    weights: Weights,
) -> OptimizationResult:
    """All prices must already be in ``currency`` (the assemble step converts)."""
    for f in flights:
        if f.price_per_traveler.currency != currency:
            raise ValueError(f"flight {f.id} is not in {currency}")
    for h in hotels:
        if h.nightly_price.currency != currency:
            raise ValueError(f"hotel {h.id} is not in {currency}")

    if not flights and not hotels:
        return OptimizationResult([], 0, 0, ["No flight or hotel options to optimise."])

    candidates = [
        Candidate(
            flight=f,
            hotel=h,
            cost=(f.total_price.amount if f else Decimal(0))
            + (h.total_price.amount if h else Decimal(0))
            + other_costs,
            travel_minutes=f.duration_minutes if f else 0,
            quality=((flight_quality(f) if f else 0.5) + (hotel_quality(h) if h else 0.5)) / 2,
        )
        for f, h in product(flights or [None], hotels or [None])
    ]

    front = pareto_front(candidates, lambda c: (float(c.cost), c.travel_minutes, -c.quality))
    w = weights.normalised()
    costs = [float(c.cost) for c in candidates]
    times = [float(c.travel_minutes) for c in candidates]
    quals = [c.quality for c in candidates]

    def utility(c: Candidate) -> float:
        return (
            w.cost * (1 - _normalise(costs, float(c.cost)))
            + w.time * (1 - _normalise(times, float(c.travel_minutes)))
            + w.quality * _normalise(quals, c.quality)
        )

    notes: list[str] = []
    feasible = [c for c in front if budget is None or c.cost <= budget.amount]
    if budget is not None and not feasible:
        notes.append("No combination fits the budget; showing the closest options.")
        feasible = front

    balanced = max(feasible, key=lambda c: (utility(c), -float(c.cost)))
    saver = min(feasible, key=lambda c: (c.cost, c.travel_minutes))
    comfort = max(feasible, key=lambda c: (c.quality, -float(c.cost)))

    chosen: list[tuple[BundleLabel, Candidate]] = []
    seen: set[tuple[str | None, str | None]] = set()
    for label, cand in (
        (BundleLabel.BALANCED, balanced),
        (BundleLabel.BUDGET_SAVER, saver),
        (BundleLabel.COMFORT, comfort),
    ):
        if cand.key not in seen:
            seen.add(cand.key)
            chosen.append((label, cand))

    bundles = [
        PlanBundle(
            label=label,
            flight_id=c.flight.id if c.flight else None,
            hotel_id=c.hotel.id if c.hotel else None,
            total_cost=Money(amount=c.cost, currency=currency),
            travel_minutes=c.travel_minutes,
            quality=round(c.quality, 3),
            utility=round(max(0.0, min(1.0, utility(c))), 3),
            within_budget=budget is None or c.cost <= budget.amount,
            pareto_optimal=True,
            tradeoffs=_tradeoffs(label, c, balanced, currency),
        )
        for label, c in chosen
    ]
    return OptimizationResult(
        bundles, considered=len(candidates), front_size=len(front), notes=notes
    )


def _tradeoffs(label: BundleLabel, c: Candidate, ref: Candidate, currency: str) -> list[str]:
    if label is BundleLabel.BALANCED:
        return ["Best overall mix of cost, travel time and quality for your style."]
    out: list[str] = []
    saved = ref.cost - c.cost
    if saved > 0:
        out.append(f"Saves about {saved:,.0f} {currency} versus the balanced option.")
    elif saved < 0:
        out.append(f"Costs about {-saved:,.0f} {currency} more than the balanced option.")
    dt = c.travel_minutes - ref.travel_minutes
    if dt > 0:
        out.append(f"Adds about {dt / 60:.1f} h of travel time.")
    elif dt < 0:
        out.append(f"Saves about {-dt / 60:.1f} h of travel time.")
    dq = c.quality - ref.quality
    if dq > 0.01:
        out.append("Higher comfort: better hotel and/or fewer stops.")
    elif dq < -0.01:
        out.append("Lower comfort: simpler hotel and/or more stops.")
    return out
