"""Pydantic schemas shared across agents, services and the API."""

from tripmate.schemas.budget import BudgetBreakdown, BudgetCategory, BudgetLine, BudgetVerdict
from tripmate.schemas.intake import GuardrailVerdict, ParsedTripIntent
from tripmate.schemas.itinerary import (
    Activity,
    ActivityCategory,
    DayPlan,
    Itinerary,
    ItineraryDraft,
    draft_to_itinerary,
)
from tripmate.schemas.plan import BundleLabel, PlanBundle, PlanEvent, PlanStatus
from tripmate.schemas.search import (
    Cabin,
    DailyWeather,
    FlightOption,
    HotelOption,
    SearchResult,
    ToolError,
    WeatherReport,
)
from tripmate.schemas.trip import MAX_TRIP_DAYS, Pace, Place, TravelStyle, TripRequest

__all__ = [
    "MAX_TRIP_DAYS",
    "Activity",
    "ActivityCategory",
    "BudgetBreakdown",
    "BudgetCategory",
    "BudgetLine",
    "BudgetVerdict",
    "BundleLabel",
    "Cabin",
    "DailyWeather",
    "DayPlan",
    "FlightOption",
    "GuardrailVerdict",
    "HotelOption",
    "Itinerary",
    "ItineraryDraft",
    "Pace",
    "ParsedTripIntent",
    "Place",
    "PlanBundle",
    "PlanEvent",
    "PlanStatus",
    "SearchResult",
    "ToolError",
    "TravelStyle",
    "TripRequest",
    "WeatherReport",
    "draft_to_itinerary",
]
