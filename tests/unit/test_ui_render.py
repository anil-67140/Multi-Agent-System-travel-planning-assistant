from __future__ import annotations

from tripmate.ui import render

BUDGET = {
    "currency": "USD",
    "total": {"amount": "2800.00", "currency": "USD"},
    "limit": {"amount": "3000", "currency": "USD"},
    "remaining": "200.00",
    "utilization_pct": 93.3,
    "verdict": "tight",
    "lines": [
        {
            "category": "local_transport",
            "amount": {"amount": "100", "currency": "USD"},
            "basis": "b",
            "is_estimate": True,
        },
        {
            "category": "flights",
            "amount": {"amount": "900", "currency": "USD"},
            "basis": "q",
            "is_estimate": False,
        },
    ],
}


def test_budget_rows_label_estimates_and_total():
    rows = render.budget_rows(BUDGET)
    assert rows[0][0] == "Local Transport" and rows[0][2] == "estimate" and rows[1][2] == "quoted"
    assert rows[-1][:2] == ["TOTAL", "2,800.00 USD"]
    assert render.budget_rows(None) == []


def test_budget_summary_over_and_under():
    assert "200 USD left" in render.budget_summary(BUDGET)
    over = {**BUDGET, "remaining": "-150"}
    assert "150 USD over" in render.budget_summary(over)
    assert "no budget set" in render.budget_summary({**BUDGET, "limit": None})


def test_itinerary_markdown_contains_days_times_and_costs():
    it = {
        "title": "Tokyo",
        "summary": "s",
        "tips": ["Carry cash"],
        "days": [
            {
                "day_number": 1,
                "date": "2026-12-01",
                "theme": "Culture",
                "weather_note": "Dry",
                "activities": [
                    {
                        "start": "09:00:00",
                        "end": "11:00:00",
                        "title": "Temple",
                        "description": "d",
                        "estimated_cost": {"amount": "20", "currency": "USD"},
                    }
                ],
            }
        ],
    }
    md = render.itinerary_markdown(it)
    assert (
        "Day 1 - 2026-12-01: Culture" in md
        and "09:00-11:00" in md
        and "20.00 USD" in md
        and "Carry cash" in md
    )
    assert "No itinerary" in render.itinerary_markdown(None)


def test_sources_are_linked_deduplicated_and_labelled_with_origin():
    s = {
        "provider": "tavily",
        "kind": "web_source",
        "origin": "live",
        "url": "https://x.example/a",
        "title": "Hotels",
        "retrieved_at": "2026-10-22T04:00:00Z",
    }
    md = render.sources_markdown(
        [s, s, {**s, "url": None, "origin": "estimate", "provider": "cost-model"}]
    )
    assert md.count("https://x.example/a") == 1 and "estimate" in md and "live" in md
    assert "No external sources" in render.sources_markdown([])


def test_problems_surface_notes_tool_errors_and_assumptions():
    md = render.problems_markdown(
        {
            "notes": ["n1"],
            "tool_errors": [{"provider": "weather", "message": "down", "retryable": True}],
            "issues": [{"message": "stale"}],
            "trip": {"assumptions": ["5 days"]},
        }
    )
    assert all(
        x in md for x in ("n1", "weather: down", "(can be retried)", "stale", "Assumed: 5 days")
    )
    assert "Nothing to flag" in render.problems_markdown({})


def test_bundle_choices_flag_over_budget():
    b = [
        {
            "label": "budget_saver",
            "total_cost": {"amount": "3500", "currency": "USD"},
            "travel_minutes": 600,
            "quality": 0.5,
            "within_budget": False,
        }
    ]
    ((text, value),) = render.bundle_choices(b)
    assert value == "budget_saver" and "over budget" in text and "10.0h" in text


def test_progress_marks_degraded_steps():
    md = render.progress_markdown(
        [
            {
                "type": "node_done",
                "node": "weather",
                "message": "Weather unavailable",
                "data": {"degraded": True},
            }
        ],
        "running",
    )
    assert "[warning]" in md and "Weather" in md


def test_problem_message_includes_field_errors_and_rate_limit_hint():
    msg = render.problem_message(
        422,
        {
            "detail": "The request is invalid.",
            "errors": [{"field": "travelers", "message": "too big"}],
        },
    )
    assert "travelers: too big" in msg
    assert "wait" in render.problem_message(429, {"detail": "Too many requests."})
    assert render.problem_message(500, None) == "Request failed (500)."
