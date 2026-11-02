from __future__ import annotations

from datetime import UTC, date, datetime, time
from decimal import Decimal

import pytest
from pydantic import ValidationError

from tests.helpers import make_place, make_trip
from tripmate.domain.money import (
    CurrencyMismatchError,
    FxRates,
    MissingRateError,
    Money,
    UnknownCurrencyError,
    normalize_currency,
    sum_money,
)
from tripmate.schemas import (
    Activity,
    ActivityCategory,
    BudgetBreakdown,
    DayPlan,
    FlightOption,
    Itinerary,
    ParsedTripIntent,
    Place,
)
from tripmate.schemas.itinerary import ActivityDraft, DayDraft, ItineraryDraft, draft_to_itinerary
from tripmate.schemas.search import DailyWeather


class TestMoney:
    def test_quantises_to_currency_exponent(self):
        assert Money.of("10.005", "USD").amount == Decimal("10.01")  # half-up, not banker's
        assert Money.of(1234.5, "JPY").amount == Decimal("1235")  # JPY has no minor unit
        assert str(Money.of(1, "KWD")) == "1.000 KWD"  # KWD has three

    def test_float_is_not_binary_noise(self):
        assert Money.of(0.1, "USD").amount == Decimal("0.10")
        assert Money.of(0.1, "USD").scaled(3).amount == Decimal("0.30")

    @pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), "abc", True, 10**20])
    def test_rejects_invalid_amounts(self, bad):
        with pytest.raises(ValidationError):
            Money.of(bad, "USD")

    def test_rejects_unknown_currency(self):
        with pytest.raises(ValidationError):
            Money.of(1, "ZZZ")
        with pytest.raises(UnknownCurrencyError):
            normalize_currency("dollars")

    def test_currency_normalised(self):
        assert Money.of(5, " usd ").currency == "USD"

    def test_arithmetic_requires_same_currency(self):
        assert (Money.of(1, "USD") + Money.of(2, "USD")).amount == Decimal("3.00")
        with pytest.raises(CurrencyMismatchError):
            Money.of(1, "USD") + Money.of(1, "EUR")
        with pytest.raises(CurrencyMismatchError):
            Money.of(1, "USD") < Money.of(1, "EUR")

    def test_sum_money(self):
        assert sum_money([Money.of(1, "EUR"), Money.of("2.5", "EUR")], "EUR").amount == Decimal(
            "3.50"
        )
        assert sum_money([], "EUR").amount == 0

    def test_hashable_and_immutable(self):
        m = Money.of(1, "USD")
        assert hash(m) == hash(Money.of(1, "USD"))
        with pytest.raises(ValidationError):
            m.amount = Decimal(2)  # type: ignore[misc]


class TestFx:
    def _fx(self) -> FxRates:
        return FxRates(
            base="USD",
            rates={"EUR": "0.92", "PKR": 280, "JPY": 150},
            as_of=datetime(2026, 10, 21, tzinfo=UTC),
            source="test",
        )

    def test_cross_rate_through_base(self):
        assert self._fx().convert(Money.of(100, "EUR"), "PKR").amount == Decimal("30434.78")

    def test_same_currency_is_identity(self):
        m = Money.of(5, "USD")
        assert self._fx().convert(m, "USD") is m

    def test_target_precision_respected(self):
        assert self._fx().convert(Money.of(10, "USD"), "JPY").amount == Decimal("1500")

    def test_missing_rate(self):
        with pytest.raises(MissingRateError):
            self._fx().convert(Money.of(1, "USD"), "THB")
        assert not self._fx().supports("THB")
        assert self._fx().supports("USD")

    def test_rejects_non_positive_rates(self):
        with pytest.raises(ValidationError):
            FxRates(
                base="USD", rates={"EUR": 0}, as_of=datetime(2026, 1, 1, tzinfo=UTC), source="x"
            )

    def test_requires_aware_timestamp(self):
        with pytest.raises(ValidationError):
            FxRates(base="USD", rates={"EUR": 1}, as_of=datetime(2026, 1, 1), source="x")


class TestPlaceAndTrip:
    def test_place_normalises_codes(self):
        p = make_place(iata="hnd", country_code="jp")
        assert (p.iata, p.country_code) == ("HND", "JP")

    def test_place_rejects_bad_coordinates_and_zone(self):
        with pytest.raises(ValidationError):
            make_place(latitude=91)
        with pytest.raises(ValidationError):
            make_place(timezone="Mars/Olympus")
        with pytest.raises(ValidationError):
            Place(name="x", latitude=0, longitude=0, iata="TOOLONG")

    def test_duration_and_nights(self):
        t = make_trip()
        assert (t.duration_days, t.nights) == (7, 6)

    def test_end_before_start_rejected(self):
        with pytest.raises(ValidationError):
            make_trip(end_date=date(2026, 11, 1))

    def test_too_long_rejected(self):
        with pytest.raises(ValidationError):
            make_trip(end_date=date(2027, 2, 1))

    def test_currency_follows_budget_then_display(self):
        assert make_trip().currency == "USD"
        assert make_trip(budget=Money.of(500000, "PKR")).currency == "PKR"
        assert make_trip(budget=None, display_currency="eur").currency == "EUR"

    def test_edited_revalidates(self):
        t = make_trip()
        assert t.edited(travelers=4).travelers == 4
        with pytest.raises(ValidationError):
            t.edited(travelers=99)

    def test_json_round_trip(self):
        t = make_trip()
        again = type(t).model_validate(t.model_dump(mode="json"))
        assert again == t


class TestSearchSchemas:
    def test_flight_total_price(
        self,
    ):
        from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef

        src = SourceRef(
            provider="est",
            kind=DataKind.FLIGHT_PRICE,
            origin=DataOrigin.ESTIMATE,
            retrieved_at=datetime(2026, 10, 22, tzinfo=UTC),
        )
        f = FlightOption(
            id="f1",
            origin_iata="KHI",
            destination_iata="HND",
            depart_date=date(2026, 12, 1),
            duration_minutes=600,
            travelers=3,
            price_per_traveler=Money.of(400, "USD"),
            source=src,
        )
        assert f.total_price == Money.of(1200, "USD")
        assert f.model_dump(mode="json")["total_price"]["amount"] == "1200.00"

    def test_flight_arrival_must_follow_departure(self):
        from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef

        src = SourceRef(
            provider="e",
            kind=DataKind.FLIGHT_PRICE,
            origin=DataOrigin.LIVE,
            retrieved_at=datetime(2026, 10, 22, tzinfo=UTC),
        )
        t = datetime(2026, 12, 1, 10, tzinfo=UTC)
        with pytest.raises(ValidationError):
            FlightOption(
                id="f",
                origin_iata="KHI",
                destination_iata="HND",
                depart_date=date(2026, 12, 1),
                duration_minutes=600,
                price_per_traveler=Money.of(1, "USD"),
                source=src,
                departure=t,
                arrival=t,
            )

    def test_source_url_must_be_http(self):
        from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef

        with pytest.raises(ValidationError):
            SourceRef(
                provider="x",
                kind=DataKind.WEB_SOURCE,
                origin=DataOrigin.LIVE,
                retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
                url="javascript:alert(1)",
            )

    def test_naive_timestamps_rejected_and_normalised_to_utc(self):
        from datetime import timedelta, timezone

        from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef

        with pytest.raises(ValidationError):
            SourceRef(
                provider="x",
                kind=DataKind.WEB_SOURCE,
                origin=DataOrigin.LIVE,
                retrieved_at=datetime(2026, 1, 1),
            )
        s = SourceRef(
            provider="x",
            kind=DataKind.WEB_SOURCE,
            origin=DataOrigin.LIVE,
            retrieved_at=datetime(2026, 1, 1, 9, tzinfo=timezone(timedelta(hours=9))),
        )
        assert s.retrieved_at == datetime(2026, 1, 1, 0, tzinfo=UTC)

    def test_weather_range_and_wetness(self):
        with pytest.raises(ValidationError):
            DailyWeather(date=date(2026, 12, 1), temp_min_c=20, temp_max_c=10)
        assert DailyWeather(
            date=date(2026, 12, 1), temp_min_c=5, temp_max_c=9, weather_code=63
        ).is_wet
        assert not DailyWeather(
            date=date(2026, 12, 1), temp_min_c=5, temp_max_c=9, weather_code=1
        ).is_wet


class TestBudgetAndItinerary:
    def test_budget_total_must_equal_sum(self):
        from tripmate.schemas.budget import BudgetCategory, BudgetLine

        lines = [BudgetLine(category=BudgetCategory.FOOD, amount=Money.of(10, "USD"), basis="b")]
        BudgetBreakdown(currency="USD", lines=lines, total=Money.of(10, "USD"))
        with pytest.raises(ValidationError):
            BudgetBreakdown(currency="USD", lines=lines, total=Money.of(11, "USD"))
        with pytest.raises(ValidationError):
            BudgetBreakdown(currency="EUR", lines=lines, total=Money.of(10, "USD"))

    def test_activity_must_end_after_start(self):
        with pytest.raises(ValidationError):
            Activity(start=time(10), end=time(9), title="x")

    def test_itinerary_day_sequence_and_dates(self):
        a = Activity(start=time(9), end=time(10), title="x")
        d1 = DayPlan(day_number=1, date=date(2026, 12, 1), activities=[a])
        d3 = DayPlan(day_number=3, date=date(2026, 12, 3), activities=[a])
        with pytest.raises(ValidationError):
            Itinerary(title="t", currency="USD", days=[d1, d3])
        d2_bad_date = DayPlan(day_number=2, date=date(2026, 12, 5), activities=[a])
        with pytest.raises(ValidationError):
            Itinerary(title="t", currency="USD", days=[d1, d2_bad_date])

    def test_draft_conversion_applies_plan_currency(self):
        draft = ItineraryDraft(
            title="Trip",
            days=[
                DayDraft(
                    day_number=1,
                    date=date(2026, 12, 1),
                    activities=[
                        ActivityDraft(
                            start="09:00",
                            end="10:30",
                            title="Temple",
                            estimated_cost=12.5,
                            outdoor=True,
                            category=ActivityCategory.CULTURE,
                        )
                    ],
                )
            ],
        )
        it = draft_to_itinerary(draft, "PKR")
        cost = it.days[0].activities[0].estimated_cost
        assert cost == Money.of("12.50", "PKR")
        assert it.days[0].activities[0].minutes == 90

    def test_draft_rejects_bad_time(self):
        with pytest.raises(ValidationError):
            ActivityDraft(start="25:99", end="10:00", title="x")


def test_parsed_intent_accepts_loose_llm_output():
    p = ParsedTripIntent.model_validate_json(
        '{"destination":"Japan","start_date":"2026-12-01","budget_amount":200000,"budget_currency":"PKR",'
        '"style":"budget","interests":["food"]}'
    )
    assert p.destination == "Japan" and p.budget_amount == Decimal("200000") and p.wants_flights
