from __future__ import annotations

import json
from datetime import date

import pytest

from tests.helpers import make_trip
from tripmate.evals.cases import EvalCase, Expect, load_cases
from tripmate.evals.offline import reference_itinerary
from tripmate.evals.run import BASELINE, main
from tripmate.evals.runner import check_expectations, compare_to_baseline, run_all, summarize
from tripmate.evals.scorers import WEIGHTS, score_itinerary
from tripmate.schemas.itinerary import Itinerary, ItineraryDraft, draft_to_itinerary

pytestmark = pytest.mark.anyio


class TestDataset:
    def test_dataset_loads_with_unique_ids_and_covers_each_area(self):
        cases = load_cases()
        tags = {t for c in cases for t in c.tags}
        assert (
            len(cases) >= 15
            and {"core", "safety", "clarify", "currency", "budget", "honesty", "limits"} <= tags
        )

    def test_unknown_fields_in_a_case_are_rejected(self, tmp_path):
        p = tmp_path / "d.jsonl"
        p.write_text('{"id":"x","description":"d","message":"m","bogus":1}\n')
        with pytest.raises(ValueError, match="line 1"):
            load_cases(p)

    def test_duplicate_ids_rejected(self, tmp_path):
        p = tmp_path / "d.jsonl"
        line = '{"id":"x","description":"d","message":"m"}\n'
        p.write_text(line + line)
        with pytest.raises(ValueError, match="duplicate"):
            load_cases(p)


class TestGoldenRun:
    async def test_every_golden_case_passes_offline(self):
        results = await run_all(load_cases())
        failed = {r.id: r.failures for r in results if not r.passed}
        assert not failed, failed

    async def test_results_are_deterministic(self):
        a = summarize(await run_all(load_cases(tags={"core"})))
        b = summarize(await run_all(load_cases(tags={"core"})))
        assert a == b

    async def test_current_results_do_not_regress_against_committed_baseline(self):
        baseline = json.loads(BASELINE.read_text())
        summary = summarize(await run_all(load_cases()))
        assert compare_to_baseline(summary, baseline) == []

    def test_cli_exit_code_is_zero_on_green(self):
        assert main([]) == 0


class TestRegressionGate:
    BASE = {"pass_rate": 1.0, "mean_score": 90.0, "scores": {"a": 90.0, "b": 90.0}}

    def test_no_change_ok_and_improvement_ok(self):
        assert compare_to_baseline({**self.BASE}, self.BASE) == []
        assert (
            compare_to_baseline(
                {"pass_rate": 1.0, "mean_score": 95.0, "scores": {"a": 99.0, "b": 91.0}}, self.BASE
            )
            == []
        )

    def test_small_noise_is_tolerated(self):
        assert (
            compare_to_baseline(
                {"pass_rate": 1.0, "mean_score": 89.0, "scores": {"a": 88.5, "b": 90.0}}, self.BASE
            )
            == []
        )

    def test_per_case_drop_pass_rate_and_missing_case_are_flagged(self):
        s = {"pass_rate": 0.9, "mean_score": 80.0, "scores": {"a": 70.0}}
        msgs = " | ".join(compare_to_baseline(s, self.BASE))
        assert (
            "pass rate" in msgs
            and "mean score" in msgs
            and "a: score fell" in msgs
            and "b: no longer" in msgs
        )


def _it(trip, mutate=None):
    data = reference_itinerary(trip.model_dump(mode="json"))
    if mutate:
        mutate(data)
    return draft_to_itinerary(ItineraryDraft.model_validate(data), trip.currency)


class TestScorers:
    def trip(self, **kw):
        return make_trip(
            start_date=date(2027, 3, 1), end_date=date(2027, 3, 4), interests=["food"], **kw
        )

    def test_weights_sum_to_one(self):
        assert abs(sum(WEIGHTS.values()) - 1) < 1e-9

    def test_good_itinerary_scores_high(self):
        t = self.trip()
        assert score_itinerary(_it(t), t, None, None).total > 70

    def test_overlap_is_a_hard_failure_that_zeroes_the_constraint_component(self):
        t = self.trip()

        def overlap(d):
            d["days"][0]["activities"][1]["start"] = "09:30"

        r = score_itinerary(_it(t, overlap), t, None, None)
        assert next(c for c in r.components if c.name == "constraints").score == 0.0
        assert r.total < score_itinerary(_it(t), t, None, None).total - 25

    def test_missing_meals_lower_the_meal_component(self):
        t = self.trip()

        def no_food(d):
            for day in d["days"]:
                for a in day["activities"]:
                    a["category"] = "sightseeing"

        comp = {c.name: c for c in score_itinerary(_it(t, no_food), t, None, None).components}
        assert comp["meals"].score == 0.0

    def test_uncovered_interests_lower_interest_score(self):
        t = make_trip(
            start_date=date(2027, 3, 1), end_date=date(2027, 3, 4), interests=["scuba diving"]
        )
        comp = {c.name: c for c in score_itinerary(_it(t), t, None, None).components}
        assert comp["interests"].score == 0.0

    def test_repetitive_itinerary_scores_lower_on_variety(self):
        t = self.trip()

        def same(d):
            for day in d["days"]:
                for a in day["activities"]:
                    a["title"], a["category"] = "Walk", "leisure"

        comp = {c.name: c for c in score_itinerary(_it(t, same), t, None, None).components}
        assert comp["variety"].score < 0.5


class TestExpectations:
    def test_wrong_status_destination_and_missing_disclosure_reported(self):
        state = {"status": "blocked", "notes": []}
        fails = check_expectations(
            Expect(destination="Tokyo", note_contains=["estimate"]), state, None
        )
        assert (
            any("status" in f for f in fails)
            and any("no trip" in f for f in fails)
            and any("disclosure" in f for f in fails)
        )

    def test_score_floor_enforced(self):
        assert any(
            "below minimum" in f
            for f in check_expectations(Expect(min_score=90), {"status": "awaiting_approval"}, 80.0)
        )

    def test_case_model_forbids_unknown_expect_fields(self):
        with pytest.raises(ValueError):
            EvalCase.model_validate(
                {"id": "x", "description": "d", "message": "m", "expect": {"nope": 1}}
            )


_ = Itinerary
