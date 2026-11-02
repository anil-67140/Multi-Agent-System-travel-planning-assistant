"""Constraint checks for a generated itinerary.

The LLM proposes; this module disposes. Any ERROR is fed back to the model as a
repair instruction (bounded retries); WARNINGs are shown to the traveller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from itertools import pairwise

from tripmate.schemas.budget import BudgetBreakdown, BudgetCategory
from tripmate.schemas.itinerary import ActivityCategory, DayPlan, Itinerary
from tripmate.schemas.search import DailyWeather, WeatherReport
from tripmate.schemas.trip import Pace, TripRequest
from tripmate.validation.issues import Issue, Severity


@dataclass(frozen=True)
class ItineraryRules:
    max_active_minutes: dict[Pace, int] = field(
        default_factory=lambda: {Pace.RELAXED: 8 * 60, Pace.BALANCED: 10 * 60, Pace.PACKED: 13 * 60}
    )
    hard_max_active_minutes: int = 14 * 60
    earliest_start: time = time(6, 0)
    latest_end: time = time(23, 30)
    arrival_buffer_minutes: int = 150  # immigration + transfer + check-in
    activity_cost_overrun_factor: float = 1.5


def _minutes(t: time) -> int:
    return t.hour * 60 + t.minute


def validate_itinerary(
    itinerary: Itinerary,
    trip: TripRequest,
    *,
    budget: BudgetBreakdown | None = None,
    weather: WeatherReport | None = None,
    arrival_local: datetime | None = None,
    rules: ItineraryRules | None = None,
) -> list[Issue]:
    rules = rules or ItineraryRules()
    issues: list[Issue] = []

    # ---- structure vs the trip --------------------------------------------
    if len(itinerary.days) != trip.duration_days:
        issues.append(
            Issue(
                code="ITIN_DAY_COUNT",
                message=f"Itinerary has {len(itinerary.days)} days but the trip is {trip.duration_days} days.",
            )
        )
    for index, day in enumerate(itinerary.days):
        expected = trip.start_date + timedelta(days=index)
        if day.date != expected:
            issues.append(
                Issue(
                    code="ITIN_DATE_MISMATCH",
                    message=f"Day {day.day_number} is dated {day.date}, expected {expected}.",
                    path=f"days[{index}]",
                )
            )

    if itinerary.currency != trip.currency:
        issues.append(
            Issue(
                code="ITIN_CURRENCY",
                message=f"Itinerary currency {itinerary.currency} differs from plan currency {trip.currency}.",
            )
        )

    weather_by_date = {d.date: d for d in weather.days} if weather else {}
    total_activity_cost = Decimal(0)

    for d_index, day in enumerate(itinerary.days):
        issues.extend(_check_day(day, d_index, trip, rules, weather_by_date, arrival_local))
        for activity in day.activities:
            if activity.estimated_cost is not None:
                if activity.estimated_cost.currency != trip.currency:
                    issues.append(
                        Issue(
                            code="ITIN_CURRENCY",
                            message=f"'{activity.title}' cost is in {activity.estimated_cost.currency}, expected {trip.currency}.",
                            path=f"days[{d_index}]",
                        )
                    )
                else:
                    total_activity_cost += activity.estimated_cost.amount

    # ---- cost vs the budget allowance ------------------------------------
    if budget is not None:
        allowance = budget.line(BudgetCategory.ACTIVITIES)
        if allowance is not None and allowance.amount > 0:
            limit = allowance.amount * Decimal(str(rules.activity_cost_overrun_factor))
            if total_activity_cost > limit:
                issues.append(
                    Issue(
                        code="ITIN_ACTIVITY_COST",
                        message=(
                            f"Activities total {total_activity_cost} {trip.currency}, well above the "
                            f"{allowance.amount} {trip.currency} allowance in the budget."
                        ),
                        severity=Severity.WARNING,
                    )
                )
    return issues


def _check_day(
    day: DayPlan,
    d_index: int,
    trip: TripRequest,
    rules: ItineraryRules,
    weather_by_date: dict[date, DailyWeather],
    arrival_local: datetime | None,
) -> list[Issue]:
    issues: list[Issue] = []
    path = f"days[{d_index}]"
    ordered = sorted(day.activities, key=lambda a: _minutes(a.start))

    # overlaps
    for earlier, later in pairwise(ordered):
        if _minutes(later.start) < _minutes(earlier.end):
            issues.append(
                Issue(
                    code="ITIN_OVERLAP",
                    message=(
                        f"Day {day.day_number}: '{later.title}' ({later.start:%H:%M}) starts before "
                        f"'{earlier.title}' ends ({earlier.end:%H:%M})."
                    ),
                    path=path,
                )
            )

    # pacing
    active = sum(a.minutes for a in ordered if a.category not in (ActivityCategory.REST,))
    soft = rules.max_active_minutes[trip.pace]
    if active > rules.hard_max_active_minutes:
        issues.append(
            Issue(
                code="ITIN_TOO_BUSY",
                message=f"Day {day.day_number} schedules {active / 60:.1f} h of activity, over the hard limit.",
                path=path,
            )
        )
    elif active > soft:
        issues.append(
            Issue(
                code="ITIN_PACE",
                message=f"Day {day.day_number} schedules {active / 60:.1f} h, more than a '{trip.pace.value}' pace suggests.",
                severity=Severity.WARNING,
                path=path,
            )
        )

    if ordered and ordered[0].start < rules.earliest_start:
        issues.append(
            Issue(
                code="ITIN_EARLY_START",
                message=f"Day {day.day_number} starts at {ordered[0].start:%H:%M}.",
                severity=Severity.WARNING,
                path=path,
            )
        )
    if ordered and max(a.end for a in ordered) > rules.latest_end:
        issues.append(
            Issue(
                code="ITIN_LATE_END",
                message=f"Day {day.day_number} runs past {rules.latest_end:%H:%M}.",
                severity=Severity.WARNING,
                path=path,
            )
        )

    if not any(a.category is ActivityCategory.FOOD for a in ordered):
        issues.append(
            Issue(
                code="ITIN_NO_MEAL",
                message=f"Day {day.day_number} has no meal planned.",
                severity=Severity.WARNING,
                path=path,
            )
        )

    # arrival: nothing (except transport) may happen before the traveller lands
    if arrival_local is not None:
        if day.date < arrival_local.date():
            for a in ordered:
                if a.category is not ActivityCategory.TRANSPORT:
                    issues.append(
                        Issue(
                            code="ITIN_BEFORE_ARRIVAL",
                            message=f"Day {day.day_number}: '{a.title}' is planned before the traveller arrives on {arrival_local.date()}.",
                            path=path,
                        )
                    )
        elif day.date == arrival_local.date():
            ready = arrival_local + timedelta(minutes=rules.arrival_buffer_minutes)
            if ready.date() == day.date:
                ready_min = ready.hour * 60 + ready.minute
                for a in ordered:
                    if (
                        a.category is not ActivityCategory.TRANSPORT
                        and _minutes(a.start) < ready_min
                    ):
                        issues.append(
                            Issue(
                                code="ITIN_BEFORE_ARRIVAL",
                                message=(
                                    f"Day {day.day_number}: '{a.title}' starts {a.start:%H:%M}, before the "
                                    f"traveller is ready (arrival {arrival_local:%H:%M} + transfer/check-in)."
                                ),
                                path=path,
                            )
                        )

    # weather
    daily = weather_by_date.get(day.date)
    if daily is not None and daily.is_wet:
        for a in ordered:
            if a.outdoor:
                issues.append(
                    Issue(
                        code="ITIN_OUTDOOR_RAIN",
                        message=f"Day {day.day_number}: outdoor activity '{a.title}' on a wet-weather day.",
                        severity=Severity.WARNING,
                        path=path,
                    )
                )
    return issues
