from __future__ import annotations

import random
from datetime import date
from decimal import Decimal

import pytest

from tests.helpers import NOW, make_place, make_trip
from tripmate.domain.money import FxRates, Money
from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef
from tripmate.optimizer import (
    STYLE_WEIGHTS,
    Weights,
    adjust_weights,
    build_budget,
    dominates,
    optimize,
    pareto_front,
)
from tripmate.schemas import FlightOption, HotelOption
from tripmate.schemas.budget import BudgetCategory, BudgetVerdict
from tripmate.schemas.plan import BundleLabel
from tripmate.schemas.search import Cabin
from tripmate.schemas.trip import TravelStyle

SRC = SourceRef(
    provider="t", kind=DataKind.FLIGHT_PRICE, origin=DataOrigin.ESTIMATE, retrieved_at=NOW
)
FX = FxRates(base="USD", rates={"EUR": "0.92", "PKR": 280, "JPY": 150}, as_of=NOW, source="t")


def flight(id_, price, minutes, stops=0, cabin=Cabin.ECONOMY, travelers=2) -> FlightOption:
    return FlightOption(
        id=id_,
        origin_iata="KHI",
        destination_iata="HND",
        depart_date=date(2026, 12, 1),
        duration_minutes=minutes,
        stops=stops,
        cabin=cabin,
        travelers=travelers,
        price_per_traveler=Money.of(price, "USD"),
        source=SRC,
    )


def hotel(id_, nightly, stars=3.0, rating=7.5, nights=6) -> HotelOption:
    return HotelOption(
        id=id_,
        name=id_,
        stars=stars,
        rating=rating,
        nights=nights,
        nightly_price=Money.of(nightly, "USD"),
        source=SRC,
    )


# --------------------------------------------------------------------------- pareto
class TestPareto:
    def test_dominates(self):
        assert dominates((1, 1), (2, 2))
        assert dominates((1, 2), (1, 3))
        assert not dominates((1, 1), (1, 1))  # equal does not dominate
        assert not dominates((1, 3), (2, 2))  # incomparable

    def test_front_basic(self):
        pts = [(1, 5), (2, 2), (5, 1), (3, 3), (6, 6)]
        assert pareto_front(pts, lambda p: p) == [(1, 5), (2, 2), (5, 1)]

    def test_duplicates_are_both_kept(self):
        pts = [(1, 1), (1, 1)]
        assert pareto_front(pts, lambda p: p) == pts

    def test_empty(self):
        assert pareto_front([], lambda p: p) == []

    def test_property_front_is_exactly_the_undominated_set(self):
        rng = random.Random(7)
        for _ in range(50):
            pts = [tuple(rng.randint(0, 9) for _ in range(3)) for _ in range(rng.randint(1, 15))]
            front = pareto_front(pts, lambda p: p)
            assert front, "a non-empty set always has a non-empty front"
            for p in front:  # nothing dominates a front member
                assert not any(dominates(q, p) for q in pts)
            for p in pts:  # every excluded point is dominated by a front member
                if p not in front:
                    assert any(dominates(q, p) for q in front)


# --------------------------------------------------------------------------- optimiser
class TestOptimize:
    def setup_method(self):
        self.flights = [
            flight("cheap", 300, 900, stops=1),
            flight("fast", 700, 540, stops=0),
            flight("mid", 450, 660, stops=1),
            flight(
                "bad", 800, 950, stops=2
            ),  # dominated: pricier AND slower AND worse than "cheap"
        ]
        self.hotels = [
            hotel("hostel", 40, 2.0, 6.5),
            hotel("inn", 100, 3.5, 8.0),
            hotel("grand", 300, 5.0, 9.2),
        ]

    def run(self, **kw):
        args = {
            "currency": "USD",
            "weights": STYLE_WEIGHTS[TravelStyle.MID_RANGE],
            "other_costs": Decimal(1000),
        }
        args.update(kw)
        return optimize(self.flights, self.hotels, **args)

    def test_considers_all_pairs_and_prunes_dominated(self):
        r = self.run()
        assert r.considered == 12 and 0 < r.front_size < 12

    def test_alternatives_have_the_expected_character(self):
        r = self.run()
        saver, comfort = r.get(BundleLabel.BUDGET_SAVER), r.get(BundleLabel.COMFORT)
        assert saver and comfort
        assert saver.total_cost < comfort.total_cost
        assert comfort.quality > saver.quality
        assert (saver.flight_id, saver.hotel_id) == ("cheap", "hostel")  # global cost minimum
        assert (comfort.flight_id, comfort.hotel_id) == ("fast", "grand")  # global quality maximum

    def test_bundles_are_distinct_and_balanced_comes_first(self):
        r = self.run()
        keys = [(b.flight_id, b.hotel_id) for b in r.bundles]
        assert len(keys) == len(set(keys))
        assert r.bundles[0].label is BundleLabel.BALANCED and r.recommended is r.bundles[0]

    def test_dominated_flight_is_never_offered(self):
        r = self.run()
        assert all(b.flight_id != "bad" for b in r.bundles)

    def test_budget_filters_candidates(self):
        r = self.run(budget=Money.of(3500, "USD"))
        assert all(b.within_budget for b in r.bundles)
        assert all(b.total_cost.amount <= 3500 for b in r.bundles)

    def test_impossible_budget_falls_back_with_a_note(self):
        r = self.run(budget=Money.of(100, "USD"))
        assert r.bundles and not any(b.within_budget for b in r.bundles)
        assert any("budget" in n.lower() for n in r.notes)

    def test_tradeoffs_are_described(self):
        r = self.run()
        saver = r.get(BundleLabel.BUDGET_SAVER)
        assert any("Saves" in t for t in saver.tradeoffs) or saver is r.bundles[0]

    def test_single_option_yields_single_bundle(self):
        r = optimize(
            [flight("only", 500, 600)],
            [hotel("h", 100)],
            currency="USD",
            weights=STYLE_WEIGHTS[TravelStyle.BUDGET],
        )
        assert len(r.bundles) == 1 and r.bundles[0].label is BundleLabel.BALANCED

    def test_flights_only_or_hotels_only(self):
        only_f = optimize(
            [flight("a", 500, 600), flight("b", 400, 700)],
            [],
            currency="USD",
            weights=STYLE_WEIGHTS[TravelStyle.BUDGET],
        )
        assert only_f.bundles and all(b.hotel_id is None for b in only_f.bundles)
        only_h = optimize(
            [],
            [hotel("a", 50), hotel("b", 200, 5, 9)],
            currency="USD",
            weights=STYLE_WEIGHTS[TravelStyle.BUDGET],
        )
        assert only_h.bundles and all(b.flight_id is None for b in only_h.bundles)

    def test_nothing_to_optimise(self):
        r = optimize([], [], currency="USD", weights=STYLE_WEIGHTS[TravelStyle.BUDGET])
        assert r.bundles == [] and r.notes

    def test_rejects_unconverted_currency(self):
        eur = HotelOption(id="e", name="e", nightly_price=Money.of(90, "EUR"), source=SRC)
        with pytest.raises(ValueError):
            optimize([], [eur], currency="USD", weights=STYLE_WEIGHTS[TravelStyle.BUDGET])

    def test_luxury_weights_prefer_quality_budget_weights_prefer_cost(self):
        lux = self.run(weights=STYLE_WEIGHTS[TravelStyle.LUXURY]).recommended
        cheap = self.run(weights=STYLE_WEIGHTS[TravelStyle.BUDGET]).recommended
        assert lux.quality >= cheap.quality and lux.total_cost >= cheap.total_cost

    def test_deterministic(self):
        assert self.run() == self.run()

    def test_scores_are_bounded(self):
        for b in self.run().bundles:
            assert 0 <= b.utility <= 1 and 0 <= b.quality <= 1


class TestWeights:
    def test_normalised_sums_to_one(self):
        w = Weights(2, 1, 1).normalised()
        assert w.cost + w.time + w.quality == pytest.approx(1)

    def test_no_history_leaves_weights_alone(self):
        base = STYLE_WEIGHTS[TravelStyle.MID_RANGE]
        assert adjust_weights(base, {}) == base

    def test_learning_from_history(self):
        base = STYLE_WEIGHTS[TravelStyle.MID_RANGE].normalised()
        saver = adjust_weights(base, {"budget_saver": 5})
        comfort = adjust_weights(base, {"comfort": 5})
        assert saver.cost > base.cost > comfort.cost
        assert comfort.quality > base.quality > saver.quality
        for w in (saver, comfort):
            assert w.cost + w.time + w.quality == pytest.approx(1)

    def test_weights_never_collapse_to_zero(self):
        w = adjust_weights(Weights(0.06, 0.5, 0.44), {"comfort": 1000})
        assert w.cost > 0


# --------------------------------------------------------------------------- budget
class TestBudget:
    def test_lines_sum_to_total_and_use_plan_currency(self):
        trip = make_trip(budget=Money.of(8000, "USD"))
        b = build_budget(trip, flight("f", 500, 600), hotel("h", 100), FX)
        assert b.currency == "USD"
        assert sum(ln.amount.amount for ln in b.lines) == b.total.amount
        assert {ln.category for ln in b.lines} == set(BudgetCategory)

    def test_known_figures(self):
        # 2 travellers, 7 days / 6 nights, Japan (index 1.15), mid-range.
        trip = make_trip(budget=Money.of(9000, "USD"))
        b = build_budget(trip, flight("f", 500, 600), hotel("h", 100), FX)
        assert b.line(BudgetCategory.FLIGHTS) == Money.of(1000, "USD")  # 2 x 500
        assert b.line(BudgetCategory.LODGING) == Money.of(600, "USD")  # 1 room x 6 nights x 100
        person_days = 2 * 7
        daily = Decimal(85) * Decimal("1.15")
        assert b.line(BudgetCategory.FOOD) == Money.of(daily * Decimal("0.40") * person_days, "USD")
        subtotal = sum(
            ln.amount.amount for ln in b.lines if ln.category is not BudgetCategory.BUFFER
        )
        assert b.line(BudgetCategory.BUFFER).amount == (subtotal * Decimal("0.10")).quantize(
            Decimal("0.01")
        )

    def test_verdicts(self):
        f, h = flight("f", 500, 600), hotel("h", 100)
        over = build_budget(make_trip(budget=Money.of(500, "USD")), f, h, FX)
        assert over.verdict is BudgetVerdict.OVER_BUDGET and over.remaining < 0
        comfortable = build_budget(make_trip(budget=Money.of(50000, "USD")), f, h, FX)
        assert (
            comfortable.verdict is BudgetVerdict.WITHIN_BUDGET and comfortable.utilization_pct < 20
        )
        none = build_budget(make_trip(budget=None), f, h, FX)
        assert none.verdict is BudgetVerdict.NO_LIMIT and none.remaining is None

    def test_tight_when_under_ten_percent_headroom(self):
        f, h = flight("f", 500, 600), hotel("h", 100)
        total = build_budget(make_trip(budget=Money.of(50000, "USD")), f, h, FX).total.amount
        tight = build_budget(make_trip(budget=Money.of(total * Decimal("1.05"), "USD")), f, h, FX)
        assert tight.verdict is BudgetVerdict.TIGHT

    def test_non_usd_plan_currency_is_converted(self):
        trip = make_trip(budget=Money.of(2_000_000, "PKR"))
        b = build_budget(trip, flight("f", 500, 600), hotel("h", 100), FX)
        assert b.currency == "PKR"
        assert b.line(BudgetCategory.FLIGHTS) == Money.of(1000 * 280, "PKR")

    def test_without_flight_or_hotel_estimates_and_says_so(self):
        b = build_budget(make_trip(), None, None, FX)
        assert b.line(BudgetCategory.FLIGHTS) is None
        assert any("Flights are not included" in a for a in b.assumptions)
        lodging = next(ln for ln in b.lines if ln.category is BudgetCategory.LODGING)
        assert lodging.is_estimate and "typical" in lodging.basis

    def test_cheap_destinations_cost_less(self):
        pk = make_trip(destination=make_place("Lahore", country_code="PK"))
        ch = make_trip(destination=make_place("Zurich", country_code="CH"))
        assert build_budget(pk, None, None, FX).total < build_budget(ch, None, None, FX).total

    def test_missing_fx_rate_raises(self):
        from tripmate.domain.money import MissingRateError

        with pytest.raises(MissingRateError):
            build_budget(make_trip(budget=Money.of(1000, "THB")), None, None, FX)

    def test_party_size_drives_rooms(self):
        three = build_budget(make_trip(travelers=3), None, None, FX).line(BudgetCategory.LODGING)
        two = build_budget(make_trip(travelers=2), None, None, FX).line(BudgetCategory.LODGING)
        assert three == two.scaled(2)  # 3 people need 2 rooms, 2 people need 1
