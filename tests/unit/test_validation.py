from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pytest

from tests.helpers import NOW, TODAY, make_trip
from tripmate.domain.money import FxRates, Money
from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef
from tripmate.schemas import (
    Activity,
    ActivityCategory,
    BudgetBreakdown,
    DayPlan,
    FlightOption,
    HotelOption,
    Itinerary,
    Pace,
)
from tripmate.schemas.budget import BudgetCategory, BudgetLine
from tripmate.schemas.search import Cabin, DailyWeather, WeatherReport
from tripmate.validation import (
    Freshness,
    Severity,
    assess_freshness,
    check_flight_duration,
    check_flight_price,
    check_flight_times,
    check_fx_covers,
    check_hotel_price,
    check_single_currency,
    check_sources,
    check_trip_dates,
    errors,
    estimate_arrival,
    great_circle_km,
    has_errors,
    local_to_utc,
    validate_itinerary,
)

SRC = SourceRef(provider="t", kind=DataKind.FLIGHT_PRICE, origin=DataOrigin.LIVE, retrieved_at=NOW)
FX = FxRates(base="USD", rates={"EUR": "0.92", "PKR": 280, "JPY": 150}, as_of=NOW, source="t")


def codes(issues):
    return {i.code for i in issues}


# --------------------------------------------------------------------------- dates
class TestDates:
    def test_valid(self):
        assert check_trip_dates(date(2026, 12, 1), date(2026, 12, 7), today=TODAY) == []

    def test_end_before_start(self):
        assert codes(check_trip_dates(date(2026, 12, 7), date(2026, 12, 1), today=TODAY)) == {
            "DATE_ORDER"
        }

    def test_past(self):
        assert "DATE_IN_PAST" in codes(
            check_trip_dates(date(2026, 10, 1), date(2026, 10, 5), today=TODAY)
        )

    def test_too_far_and_too_long(self):
        assert "DATE_TOO_FAR" in codes(
            check_trip_dates(date(2028, 1, 1), date(2028, 1, 5), today=TODAY)
        )
        assert "TRIP_TOO_LONG" in codes(
            check_trip_dates(date(2026, 12, 1), date(2027, 1, 20), today=TODAY)
        )

    def test_short_notice_is_a_warning_not_an_error(self):
        issues = check_trip_dates(TODAY + timedelta(days=2), TODAY + timedelta(days=5), today=TODAY)
        assert codes(issues) == {"SHORT_NOTICE"} and not has_errors(issues)

    def test_today_is_not_in_the_past(self):
        assert "DATE_IN_PAST" not in codes(
            check_trip_dates(TODAY, TODAY + timedelta(days=2), today=TODAY)
        )


# --------------------------------------------------------------------------- currency
class TestCurrency:
    def test_single_currency(self):
        issues = check_single_currency(
            [("a", Money.of(1, "USD")), ("b", Money.of(1, "EUR")), ("c", None)], "USD"
        )
        assert codes(issues) == {"CURRENCY_MISMATCH"} and issues[0].path == "b"

    def test_fx_coverage(self):
        assert check_fx_covers(FX, ["USD", "EUR"]) == []
        assert codes(check_fx_covers(FX, ["THB", "THB"])) == {"FX_RATE_MISSING"}
        assert len(check_fx_covers(FX, ["THB", "THB"])) == 1


# --------------------------------------------------------------------------- distance / duration
class TestDurations:
    def test_great_circle_known_distances(self):
        # JFK -> LHR is about 5,540 km; Karachi -> Tokyo about 6,900 km.
        assert great_circle_km(40.6413, -73.7781, 51.4700, -0.4543) == pytest.approx(5540, rel=0.01)
        assert great_circle_km(24.9065, 67.1608, 35.5494, 139.7798) == pytest.approx(6950, rel=0.02)
        assert great_circle_km(10, 10, 10, 10) == 0

    def test_antipodal_does_not_blow_up(self):
        assert great_circle_km(0, 0, 0, 180) == pytest.approx(20015, rel=0.001)

    def test_impossibly_fast_flight_is_an_error(self):
        assert codes(check_flight_duration(120, 5540)) == {"FLIGHT_TOO_FAST"}

    def test_realistic_flights_pass(self):
        assert check_flight_duration(420, 5540) == []
        assert check_flight_duration(600, 6900) == []
        assert check_flight_duration(900, 6900, stops=1) == []

    def test_absurdly_long_flight_is_a_warning(self):
        issues = check_flight_duration(3000, 5540, stops=0)
        assert codes(issues) == {"FLIGHT_TOO_SLOW"} and issues[0].severity is Severity.WARNING


# --------------------------------------------------------------------------- time zones
class TestTimezones:
    def test_local_to_utc(self):
        assert local_to_utc(datetime(2026, 12, 1, 9, 0), "Asia/Tokyo") == datetime(
            2026, 12, 1, 0, 0, tzinfo=UTC
        )
        with pytest.raises(ValueError):
            local_to_utc(datetime(2026, 12, 1, 9, tzinfo=UTC), "Asia/Tokyo")

    def test_arrival_crosses_midnight_and_zones(self):
        # 23:00 Karachi (UTC+5) = 18:00 UTC; +9h = 03:00 UTC next day = 12:00 in Tokyo (UTC+9).
        arr = estimate_arrival(datetime(2026, 12, 1, 23, 0), "Asia/Karachi", "Asia/Tokyo", 9 * 60)
        assert arr.date() == date(2026, 12, 2) and (arr.hour, arr.minute) == (12, 0)

    def test_arrival_respects_dst(self):
        # London leaves UTC on 2026-10-25; a flight on the 24th (BST, UTC+1) vs 26th (GMT).
        before = estimate_arrival(datetime(2026, 10, 24, 10, 0), "Europe/London", "UTC", 60)
        after = estimate_arrival(datetime(2026, 10, 26, 10, 0), "Europe/London", "UTC", 60)
        assert (
            before.hour == 10 and after.hour == 11
        )  # 10:00 BST = 09:00 UTC; 10:00 GMT = 10:00 UTC

    def test_flight_time_consistency(self):
        def flight(arr_delta_min: int) -> FlightOption:
            dep = datetime(2026, 12, 1, 10, tzinfo=UTC)
            return FlightOption(
                id="f",
                origin_iata="KHI",
                destination_iata="HND",
                depart_date=date(2026, 12, 1),
                duration_minutes=600,
                price_per_traveler=Money.of(500, "USD"),
                source=SRC,
                departure=dep,
                arrival=dep + timedelta(minutes=arr_delta_min),
            )

        assert check_flight_times(flight(600)) == []
        assert codes(check_flight_times(flight(420))) == {"FLIGHT_DURATION_MISMATCH"}


# --------------------------------------------------------------------------- prices
def _flight(price, cabin=Cabin.ECONOMY, currency="USD") -> FlightOption:
    return FlightOption(
        id="f",
        origin_iata="KHI",
        destination_iata="HND",
        depart_date=date(2026, 12, 1),
        duration_minutes=600,
        price_per_traveler=Money.of(price, currency),
        source=SRC,
        cabin=cabin,
    )


def _hotel(price, currency="USD") -> HotelOption:
    return HotelOption(id="h", name="H", nightly_price=Money.of(price, currency), source=SRC)


class TestPrices:
    def test_flight_price_bounds(self):
        assert check_flight_price(_flight(450), 6900, FX) == []
        assert codes(check_flight_price(_flight(5), 6900, FX)) == {"PRICE_SUSPICIOUSLY_LOW"}
        assert codes(check_flight_price(_flight(90000), 6900, FX)) == {"PRICE_SUSPICIOUSLY_HIGH"}

    def test_business_class_allowed_to_cost_more(self):
        assert check_flight_price(_flight(9000, Cabin.BUSINESS), 6900, FX) == []

    def test_non_usd_prices_are_converted_before_checking(self):
        # 126,000 PKR ~= 450 USD -> fine; 1,400 PKR ~= 5 USD -> too low.
        assert check_flight_price(_flight(126000, currency="PKR"), 6900, FX) == []
        assert codes(check_flight_price(_flight(1400, currency="PKR"), 6900, FX)) == {
            "PRICE_SUSPICIOUSLY_LOW"
        }

    def test_unknown_currency_is_flagged_not_crashed(self):
        issues = check_flight_price(_flight(100, currency="THB"), 6900, FX)
        assert codes(issues) == {"PRICE_UNCHECKED"} and issues[0].severity is Severity.WARNING

    def test_hotel_price_bounds(self):
        assert check_hotel_price(_hotel(120), FX) == []
        assert codes(check_hotel_price(_hotel(1), FX)) == {"PRICE_SUSPICIOUSLY_LOW"}
        assert codes(check_hotel_price(_hotel(100000), FX)) == {"PRICE_SUSPICIOUSLY_HIGH"}


# --------------------------------------------------------------------------- freshness
class TestFreshness:
    @pytest.mark.parametrize(
        ("kind", "age", "expected"),
        [
            (DataKind.WEATHER_FORECAST, timedelta(hours=1), Freshness.FRESH),
            (DataKind.WEATHER_FORECAST, timedelta(hours=9), Freshness.STALE),
            (DataKind.WEATHER_FORECAST, timedelta(days=2), Freshness.EXPIRED),
            (DataKind.FX_RATE, timedelta(hours=30), Freshness.FRESH),
            (DataKind.FX_RATE, timedelta(days=5), Freshness.EXPIRED),
            (DataKind.FLIGHT_PRICE, timedelta(minutes=59), Freshness.FRESH),
            (DataKind.FLIGHT_PRICE, timedelta(hours=3), Freshness.EXPIRED),
        ],
    )
    def test_assess(self, kind, age, expected):
        assert assess_freshness(NOW - age, kind, NOW) is expected

    def test_boundary_is_inclusive(self):
        assert (
            assess_freshness(NOW - timedelta(hours=6), DataKind.WEATHER_FORECAST, NOW)
            is Freshness.FRESH
        )

    def test_check_sources(self):
        def src(kind, age, origin=DataOrigin.LIVE):
            return SourceRef(provider="p", kind=kind, origin=origin, retrieved_at=NOW - age)

        issues = check_sources(
            [
                src(DataKind.FX_RATE, timedelta(hours=1)),
                src(DataKind.FX_RATE, timedelta(hours=40)),
                src(DataKind.HOTEL_PRICE, timedelta(days=3)),
                src(DataKind.FLIGHT_PRICE, timedelta(days=400), DataOrigin.ESTIMATE),
            ],
            NOW,
        )
        assert [i.code for i in issues] == ["SOURCE_STALE", "SOURCE_EXPIRED", "SOURCE_ESTIMATE"]
        assert not has_errors(issues)


# --------------------------------------------------------------------------- itinerary
def _act(
    start, end, title="x", category=ActivityCategory.SIGHTSEEING, cost=None, outdoor=False
) -> Activity:
    return Activity(
        start=start,
        end=end,
        title=title,
        category=category,
        outdoor=outdoor,
        estimated_cost=Money.of(cost, "USD") if cost is not None else None,
    )


def _day(n, activities, d0=date(2026, 12, 1)) -> DayPlan:
    return DayPlan(day_number=n, date=d0 + timedelta(days=n - 1), activities=activities)


def _good_day(n) -> DayPlan:
    return _day(
        n,
        [
            _act(time(9), time(12), "Museum"),
            _act(time(12, 30), time(13, 30), "Lunch", ActivityCategory.FOOD),
            _act(time(14), time(17), "Park", outdoor=True),
        ],
    )


def _itinerary(days=None, currency="USD") -> Itinerary:
    return Itinerary(title="T", currency=currency, days=days or [_good_day(i) for i in range(1, 8)])


class TestItineraryValidation:
    def test_good_itinerary_has_no_errors(self):
        issues = validate_itinerary(_itinerary(), make_trip())
        assert errors(issues) == []

    def test_wrong_day_count(self):
        issues = validate_itinerary(_itinerary([_good_day(i) for i in range(1, 4)]), make_trip())
        assert "ITIN_DAY_COUNT" in codes(errors(issues))

    def test_date_mismatch(self):
        days = [_good_day(i) for i in range(1, 8)]
        issues = validate_itinerary(
            _itinerary(days), make_trip(start_date=date(2026, 12, 2), end_date=date(2026, 12, 8))
        )
        assert "ITIN_DATE_MISMATCH" in codes(errors(issues))

    def test_overlap_detected_but_adjacent_is_fine(self):
        overlap = _day(
            1,
            [
                _act(time(9), time(11), "A"),
                _act(time(10, 30), time(12), "B"),
                _act(time(12), time(13), "Lunch", ActivityCategory.FOOD),
            ],
        )
        adjacent = _day(
            1,
            [
                _act(time(9), time(11), "A"),
                _act(time(11), time(12), "B"),
                _act(time(12), time(13), "Lunch", ActivityCategory.FOOD),
            ],
        )
        trip = make_trip(start_date=date(2026, 12, 1), end_date=date(2026, 12, 1))
        assert "ITIN_OVERLAP" in codes(validate_itinerary(_itinerary([overlap]), trip))
        assert "ITIN_OVERLAP" not in codes(validate_itinerary(_itinerary([adjacent]), trip))

    def test_pacing_soft_and_hard(self):
        trip1 = make_trip(
            start_date=date(2026, 12, 1), end_date=date(2026, 12, 1), pace=Pace.RELAXED
        )
        nine_hours = _day(
            1,
            [
                _act(time(8), time(17), "Marathon"),
                _act(time(17, 30), time(18), "Dinner", ActivityCategory.FOOD),
            ],
        )
        issue = [
            i for i in validate_itinerary(_itinerary([nine_hours]), trip1) if i.code == "ITIN_PACE"
        ]
        assert issue and issue[0].severity is Severity.WARNING
        sixteen = _day(
            1,
            [
                _act(time(6), time(14), "A"),
                _act(time(14), time(22), "B"),
                _act(time(22), time(23), "C", ActivityCategory.FOOD),
            ],
        )
        assert "ITIN_TOO_BUSY" in codes(errors(validate_itinerary(_itinerary([sixteen]), trip1)))

    def test_rest_does_not_count_as_activity_time(self):
        trip = make_trip(
            start_date=date(2026, 12, 1), end_date=date(2026, 12, 1), pace=Pace.RELAXED
        )
        day = _day(
            1,
            [
                _act(time(8), time(11), "Walk"),
                _act(time(11), time(19), "Hotel pool", ActivityCategory.REST),
                _act(time(19), time(20), "Dinner", ActivityCategory.FOOD),
            ],
        )
        assert "ITIN_PACE" not in codes(validate_itinerary(_itinerary([day]), trip))

    def test_no_meal_warning(self):
        trip = make_trip(start_date=date(2026, 12, 1), end_date=date(2026, 12, 1))
        issues = validate_itinerary(_itinerary([_day(1, [_act(time(9), time(10))])]), trip)
        assert "ITIN_NO_MEAL" in codes(issues)

    def test_currency_mismatch(self):
        assert "ITIN_CURRENCY" in codes(
            errors(validate_itinerary(_itinerary(currency="EUR"), make_trip()))
        )

    def test_activity_in_other_currency(self):
        bad = Activity(start=time(9), end=time(10), title="x", estimated_cost=Money.of(5, "EUR"))
        trip = make_trip(start_date=date(2026, 12, 1), end_date=date(2026, 12, 1))
        day = _day(1, [bad, _act(time(12), time(13), "L", ActivityCategory.FOOD)])
        assert "ITIN_CURRENCY" in codes(errors(validate_itinerary(_itinerary([day]), trip)))

    def test_arrival_blocks_early_activities(self):
        trip = make_trip(start_date=date(2026, 12, 1), end_date=date(2026, 12, 1))
        arrival = datetime(2026, 12, 1, 13, 0, tzinfo=UTC)  # lands 13:00 -> ready 15:30
        early = _day(
            1,
            [
                _act(time(9), time(11), "Museum"),
                _act(time(16), time(17), "Dinner", ActivityCategory.FOOD),
            ],
        )
        late = _day(1, [_act(time(16), time(17), "Dinner", ActivityCategory.FOOD)])
        transport = _day(
            1,
            [
                _act(time(9), time(11), "Transfer", ActivityCategory.TRANSPORT),
                _act(time(16), time(17), "D", ActivityCategory.FOOD),
            ],
        )
        assert "ITIN_BEFORE_ARRIVAL" in codes(
            errors(validate_itinerary(_itinerary([early]), trip, arrival_local=arrival))
        )
        assert "ITIN_BEFORE_ARRIVAL" not in codes(
            validate_itinerary(_itinerary([late]), trip, arrival_local=arrival)
        )
        assert "ITIN_BEFORE_ARRIVAL" not in codes(
            validate_itinerary(_itinerary([transport]), trip, arrival_local=arrival)
        )

    def test_outdoor_on_rainy_day_warns(self):
        trip = make_trip(start_date=date(2026, 12, 1), end_date=date(2026, 12, 1))
        wx = WeatherReport(
            place_name="Tokyo",
            source=SRC,
            days=[
                DailyWeather(
                    date=date(2026, 12, 1),
                    temp_min_c=5,
                    temp_max_c=9,
                    weather_code=65,
                    precipitation_mm=20,
                )
            ],
        )
        day = _day(
            1,
            [
                _act(time(9), time(12), "Park", outdoor=True),
                _act(time(12), time(13), "L", ActivityCategory.FOOD),
            ],
        )
        issues = validate_itinerary(_itinerary([day]), trip, weather=wx)
        assert "ITIN_OUTDOOR_RAIN" in codes(issues) and not errors(issues)

    def test_activity_costs_over_allowance_warn(self):
        trip = make_trip(start_date=date(2026, 12, 1), end_date=date(2026, 12, 1))
        budget = BudgetBreakdown(
            currency="USD",
            total=Money.of(100, "USD"),
            lines=[
                BudgetLine(
                    category=BudgetCategory.ACTIVITIES, amount=Money.of(100, "USD"), basis="b"
                )
            ],
        )
        day = _day(
            1,
            [
                _act(time(9), time(12), "Show", cost=400),
                _act(time(12), time(13), "L", ActivityCategory.FOOD),
            ],
        )
        assert "ITIN_ACTIVITY_COST" in codes(
            validate_itinerary(_itinerary([day]), trip, budget=budget)
        )
