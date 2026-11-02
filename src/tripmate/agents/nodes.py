"""Workflow nodes. Each is ``async (state, deps) -> partial state update``.

Rules the nodes follow:
* The model extracts and writes prose; code does arithmetic, validation and ranking.
* A failing data source degrades the plan (and says so); it never crashes it.
* Nothing here imports LangGraph, so every node is unit-testable with fakes.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field

from tripmate.agents.deps import Deps
from tripmate.agents.guardrail import run_guardrail
from tripmate.agents.resolver import Clarification, UserDefaults, resolve_intent
from tripmate.domain.cost_model import HOTEL_USD_PER_NIGHT, cost_index, rooms_needed
from tripmate.domain.money import FxRates, MissingRateError, Money
from tripmate.domain.provenance import DataKind, DataOrigin, SourceRef
from tripmate.errors import LLMError
from tripmate.graph.state import PlanState, dump, dump_all, load
from tripmate.llm import UNTRUSTED_NOTICE, wrap_untrusted
from tripmate.optimizer import STYLE_WEIGHTS, adjust_weights, build_budget, optimize
from tripmate.providers.flights import resolve_airport
from tripmate.schemas.budget import BudgetBreakdown
from tripmate.schemas.intake import ParsedTripIntent
from tripmate.schemas.itinerary import Itinerary, ItineraryDraft, draft_to_itinerary
from tripmate.schemas.plan import BundleLabel, PlanBundle, PlanStatus
from tripmate.schemas.search import FlightOption, HotelOption, WeatherReport
from tripmate.schemas.trip import Pace, TravelStyle, TripRequest
from tripmate.validation import (
    check_flight_price,
    check_hotel_price,
    check_sources,
    errors,
    format_issues,
    great_circle_km,
    has_errors,
    validate_itinerary,
)
from tripmate.validation.issues import Issue

log = logging.getLogger(__name__)


def _trip(state: PlanState) -> TripRequest:
    trip = load(TripRequest, state.get("trip"))
    if trip is None:
        raise RuntimeError("workflow reached a trip-dependent node without a trip")
    return trip


def _err(e: Any) -> list[dict[str, Any]]:
    return [x.model_dump(mode="json") for x in e]


def _meta(state: PlanState) -> dict[str, Any]:
    # Langfuse groups every model call of a plan under one session.
    return {"session_id": state.get("plan_id"), "trace_user_id": state.get("user_id")}


# --------------------------------------------------------------------------- guardrail
async def guardrail_node(state: PlanState, deps: Deps) -> dict[str, Any]:
    if state.get("trip"):  # an edit of an already-vetted trip
        return {"status": PlanStatus.RUNNING.value}
    await deps.emit("node_start", "guardrail", "Checking your request", {})
    verdict = await run_guardrail(state.get("message", ""), deps.llm, _meta(state))
    if not verdict.allowed:
        await deps.emit(
            "blocked", "guardrail", verdict.reason, {"category": verdict.category.value}
        )
        return {"guardrail": verdict.model_dump(mode="json"), "status": PlanStatus.BLOCKED.value}
    return {"guardrail": verdict.model_dump(mode="json"), "status": PlanStatus.RUNNING.value}


# --------------------------------------------------------------------------- intake
INTAKE_SYSTEM = (
    "You extract trip details from a traveller's message. Fill only what is stated or clearly implied; "
    "leave other fields null. Never invent dates. Convert budgets to plain numbers (lakh=100000, "
    "crore=10000000, k=1000). If the destination is missing, set clarifying_question. "
    + UNTRUSTED_NOTICE
)


def defaults_from_prefs(prefs: dict[str, Any] | None) -> UserDefaults:
    p = prefs or {}

    def enum(cls: Any, value: Any) -> Any:
        try:
            return cls(value) if value else None
        except ValueError:
            return None

    return UserDefaults(
        home_city=p.get("home_city"),
        currency=p.get("currency"),
        style=enum(TravelStyle, p.get("style")),
        pace=enum(Pace, p.get("pace")),
        dietary=tuple(p.get("dietary", ())),
        interests=tuple(p.get("interests", ())),
    )


async def intake_node(state: PlanState, deps: Deps) -> dict[str, Any]:
    if state.get("trip"):
        return {}
    await deps.emit("node_start", "intake", "Understanding your trip", {})
    today = deps.clock.now().date()
    system = f"{INTAKE_SYSTEM}\nToday's date is {today.isoformat()}."
    try:
        intent = await deps.llm.structured(
            task="intake",
            system=system,
            user=wrap_untrusted(state["message"]),
            schema=ParsedTripIntent,
            tier="fast",
            metadata=_meta(state),
        )
    except LLMError:
        return {
            "status": PlanStatus.FAILED.value,
            "clarification": None,
            "notes": ["Could not understand the request (language model unavailable)."],
        }
    result = await resolve_intent(
        intent, deps.tools, today=today, defaults=defaults_from_prefs(state.get("prefs"))
    )
    if isinstance(result, Clarification):
        await deps.emit("clarification", "intake", result.question, {})
        return {"status": PlanStatus.NEEDS_CLARIFICATION.value, "clarification": result.question}
    await deps.emit(
        "trip_resolved", "intake", f"{result.destination.name}, {result.duration_days} days", {}
    )
    return {"trip": dump(result), "status": PlanStatus.RUNNING.value, "clarification": None}


# --------------------------------------------------------------------------- data fan-out
async def flights_node(state: PlanState, deps: Deps) -> dict[str, Any]:
    trip = _trip(state)
    if trip.origin is None:
        return {"flights": [], "notes": ["Flights skipped: no departure city."]}
    origin_air = resolve_airport(trip.origin, deps.airports)
    dest_air = resolve_airport(trip.destination, deps.airports)
    if origin_air is None or dest_air is None:
        return {
            "flights": [],
            "notes": ["Flights skipped: no airport found near the origin or destination."],
        }
    if origin_air.iata == dest_air.iata:
        return {
            "flights": [],
            "notes": ["Flights skipped: origin and destination share an airport."],
        }
    await deps.emit("node_start", "flights", "Looking at flights", {})
    o = trip.origin.model_copy(update={"iata": origin_air.iata})
    d = trip.destination.model_copy(update={"iata": dest_air.iata})
    result = await deps.tools.flights(o, d, trip.start_date, trip.end_date, trip.travelers)
    await deps.emit(
        "node_done",
        "flights",
        f"{len(result.items)} flight option(s)",
        {"degraded": result.degraded},
    )
    out: dict[str, Any] = {
        "flights": dump_all(result.items),
        "tool_errors": _err(result.errors),
        "sources": dump_all(result.sources),
    }
    if result.note and result.degraded:
        out["notes"] = [result.note]
    return out


class HotelFact(BaseModel):
    name: str
    nightly_price: float | None = Field(
        default=None, ge=0, description="Per room per night, as printed on the page"
    )
    currency: str | None = None
    stars: float | None = Field(default=None, ge=0, le=5)
    rating: float | None = Field(default=None, ge=0, le=10)
    area: str | None = None
    source_url: str = Field(description="Must be one of the provided URLs")


class HotelFacts(BaseModel):
    hotels: list[HotelFact] = Field(default_factory=list, max_length=6)


def estimate_hotels(trip: TripRequest, fx: FxRates, now: datetime) -> list[HotelOption]:
    """Honest fallback: typical rates by style, flagged as estimates with no fake names."""
    index = cost_index(trip.destination.country_code)
    rooms = rooms_needed(trip.travelers)
    out: list[HotelOption] = []
    tiers = [
        (TravelStyle.BUDGET, "Typical budget stay", 2.5),
        (TravelStyle.MID_RANGE, "Typical mid-range stay", 3.5),
        (TravelStyle.LUXURY, "Typical upscale stay", 4.5),
    ]
    for style, name, stars in tiers:
        usd = Money.of(HOTEL_USD_PER_NIGHT[style] * index, "USD")
        try:
            price = fx.convert(usd, trip.currency)
        except (KeyError, MissingRateError):
            continue
        out.append(
            HotelOption(
                id=f"est-{style.value}",
                name=name,
                stars=stars,
                nightly_price=price,
                rooms=rooms,
                nights=max(trip.nights, 1),
                price_is_estimate=True,
                source=SourceRef(
                    provider="tripmate-cost-model",
                    kind=DataKind.HOTEL_PRICE,
                    origin=DataOrigin.ESTIMATE,
                    retrieved_at=now,
                    note="Regional average, not a live quote",
                ),
            )
        )
    return out


async def hotels_node(state: PlanState, deps: Deps) -> dict[str, Any]:
    trip = _trip(state)
    await deps.emit("node_start", "hotels", "Searching hotels", {})
    query = f"best {trip.style.value.replace('_', ' ')} hotels in {trip.destination.name} price per night"
    found = await deps.tools.web_search(query, 6)
    hotels: list[HotelOption] = []
    notes: list[str] = []
    now = deps.clock.now()
    if found.items:
        allowed = {h.url: h for h in found.items}
        context = "\n\n".join(
            f"URL: {h.url}\nTITLE: {h.title}\nTEXT: {h.content[:900]}" for h in found.items
        )
        try:
            facts = await deps.llm.structured(
                task="hotel_extract",
                system=(
                    "Extract hotels that are explicitly mentioned in the search results. Use ONLY prices printed "
                    "in the text; leave nightly_price null if none is stated. source_url must be one of the "
                    "given URLs. Do not invent hotels. " + UNTRUSTED_NOTICE
                ),
                user=wrap_untrusted(context, "search_results", limit=8000),
                schema=HotelFacts,
                tier="fast",
                metadata=_meta(state),
            )
        except LLMError:
            facts = HotelFacts()
        for i, fact in enumerate(facts.hotels):
            hit = allowed.get(fact.source_url)
            if hit is None or fact.nightly_price is None or not fact.currency:
                continue  # unverifiable claims are dropped, not displayed
            try:
                price = Money.of(Decimal(str(fact.nightly_price)), fact.currency)
                hotels.append(
                    HotelOption(
                        id=f"web-{i}",
                        name=fact.name[:160],
                        area=fact.area,
                        stars=fact.stars,
                        rating=fact.rating,
                        nightly_price=price,
                        rooms=rooms_needed(trip.travelers),
                        nights=max(trip.nights, 1),
                        price_is_estimate=False,
                        source=SourceRef(
                            provider="tavily",
                            kind=DataKind.HOTEL_PRICE,
                            origin=DataOrigin.LIVE,
                            retrieved_at=now,
                            url=hit.url,
                            title=hit.title[:300],
                        ),
                    )
                )
            except (ValueError, KeyError):
                continue
    if not hotels:
        notes.append(
            "No hotel prices could be verified from live search; showing typical regional rates."
        )
    await deps.emit(
        "node_done", "hotels", f"{len(hotels)} verified hotel(s)", {"degraded": not hotels}
    )
    return {
        "hotels": dump_all(hotels),
        "tool_errors": _err(found.errors),
        "sources": dump_all(found.sources),
        "notes": notes,
    }


async def weather_node(state: PlanState, deps: Deps) -> dict[str, Any]:
    trip = _trip(state)
    await deps.emit("node_start", "weather", "Checking the weather", {})
    result = await deps.tools.weather(trip.destination, trip.start_date, trip.end_date)
    report = result.items[0] if result.items else None
    await deps.emit(
        "node_done",
        "weather",
        "Weather ready" if report else "Weather unavailable",
        {"degraded": report is None},
    )
    out: dict[str, Any] = {
        "weather": dump(report),
        "tool_errors": _err(result.errors),
        "sources": dump_all(result.sources),
    }
    if report is None:
        out["notes"] = ["Weather is unavailable; outdoor activities were not adjusted for it."]
    return out


async def fx_node(state: PlanState, deps: Deps) -> dict[str, Any]:
    rates = await deps.tools.fx_rates("USD")
    out: dict[str, Any] = {"fx": rates.model_dump(mode="json")}
    if rates.origin is DataOrigin.ESTIMATE:
        out["notes"] = ["Live exchange rates were unavailable; approximate rates were used."]
    return out


# --------------------------------------------------------------------------- assemble
def _convert_flight(f: FlightOption, fx: FxRates, cur: str) -> FlightOption | None:
    if f.price_per_traveler.currency == cur:
        return f
    try:
        return f.model_copy(
            update={
                "price_per_traveler": fx.convert(f.price_per_traveler, cur),
                "original_price_per_traveler": f.price_per_traveler,
            }
        )
    except (KeyError, MissingRateError):
        return None


def _convert_hotel(h: HotelOption, fx: FxRates, cur: str) -> HotelOption | None:
    if h.nightly_price.currency == cur:
        return h
    try:
        return h.model_copy(
            update={
                "nightly_price": fx.convert(h.nightly_price, cur),
                "original_nightly_price": h.nightly_price,
            }
        )
    except (KeyError, MissingRateError):
        return None


async def assemble_node(state: PlanState, deps: Deps) -> dict[str, Any]:
    """Join point: normalise currency, drop implausible data, record source-freshness warnings."""
    trip = _trip(state)
    fx = FxRates.model_validate(state["fx"])
    notes: list[str] = []
    issues: list[Issue] = []
    flights, hotels = [], []

    km = (
        great_circle_km(
            trip.origin.latitude,
            trip.origin.longitude,
            trip.destination.latitude,
            trip.destination.longitude,
        )
        if trip.origin
        else 0.0
    )
    for raw in state.get("flights") or []:
        f = _convert_flight(FlightOption.model_validate(raw), fx, trip.currency)
        if f is None:
            notes.append(f"Dropped flight {raw.get('id')}: no exchange rate for its currency.")
            continue
        found = check_flight_price(f, km, fx)
        if has_errors(found):
            notes.append(f"Dropped flight {f.id}: " + "; ".join(i.message for i in errors(found)))
            continue
        issues += found
        flights.append(f)
    for raw in state.get("hotels") or []:
        h = _convert_hotel(HotelOption.model_validate(raw), fx, trip.currency)
        if h is None:
            notes.append(f"Dropped hotel {raw.get('name')}: no exchange rate for its currency.")
            continue
        found = check_hotel_price(h, fx)
        if has_errors(found):
            notes.append(f"Dropped hotel {h.name}: " + "; ".join(i.message for i in errors(found)))
            continue
        issues += found
        hotels.append(h)
    if not hotels:
        hotels = estimate_hotels(trip, fx, deps.clock.now())

    srcs = [SourceRef.model_validate(s) for s in state.get("sources") or []]
    issues += check_sources(srcs, deps.clock.now())
    await deps.emit(
        "node_done",
        "assemble",
        f"{len(flights)} flight(s), {len(hotels)} hotel(s) after checks",
        {},
    )
    return {
        "flights": dump_all(flights),
        "hotels": dump_all(hotels),
        "issues": dump_all(issues),
        "notes": notes,
    }


# --------------------------------------------------------------------------- optimize + budget
async def optimize_node(state: PlanState, deps: Deps) -> dict[str, Any]:
    trip = _trip(state)
    flights = [FlightOption.model_validate(f) for f in state.get("flights") or []]
    hotels = [HotelOption.model_validate(h) for h in state.get("hotels") or []]
    prefs = state.get("prefs") or {}
    weights = adjust_weights(STYLE_WEIGHTS[trip.style], prefs.get("bundle_affinity", {}))
    # Non-travel costs are identical across bundles, so they shift totals but not the ranking;
    # including them keeps the "fits budget" flag honest.
    base = build_budget(trip, None, None, FxRates.model_validate(state["fx"]))
    other = sum(
        (
            ln.amount.amount
            for ln in base.lines
            if ln.category.value in ("food", "local_transport", "activities")
        ),
        Decimal(0),
    ) * Decimal("1.1")
    result = optimize(
        flights,
        hotels,
        currency=trip.currency,
        other_costs=other,
        budget=trip.budget,
        weights=weights,
    )
    label = state.get("selected_label")
    chosen = result.get(BundleLabel(label)) if label else None
    chosen = chosen or result.recommended
    await deps.emit(
        "node_done",
        "optimize",
        f"{len(result.bundles)} alternative(s) ranked",
        {"considered": result.considered, "pareto": result.front_size},
    )
    return {
        "bundles": dump_all(result.bundles),
        "selected_label": chosen.label.value if chosen else None,
        "notes": result.notes,
    }


async def budget_node(state: PlanState, deps: Deps) -> dict[str, Any]:
    trip = _trip(state)
    fx = FxRates.model_validate(state["fx"])
    bundle = next(
        (
            PlanBundle.model_validate(b)
            for b in state.get("bundles") or []
            if b["label"] == state.get("selected_label")
        ),
        None,
    )
    flight = hotel = None
    if bundle:
        flight = next(
            (
                FlightOption.model_validate(f)
                for f in state.get("flights") or []
                if f["id"] == bundle.flight_id
            ),
            None,
        )
        hotel = next(
            (
                HotelOption.model_validate(h)
                for h in state.get("hotels") or []
                if h["id"] == bundle.hotel_id
            ),
            None,
        )
    budget = build_budget(trip, flight, hotel, fx)
    await deps.emit("node_done", "budget", f"Budget: {budget.verdict.value}", {})
    return {"budget": dump(budget)}


# --------------------------------------------------------------------------- itinerary
ITIN_SYSTEM = (
    "You are a meticulous travel planner. Produce a day-by-day itinerary as JSON. Rules: one entry per trip "
    "day with the exact dates given; 24h local times; activities in chronological order without overlap; "
    "include meals; respect the pace; keep outdoor activities for dry days; estimated_cost is for the whole "
    "party in the stated currency and must be realistic; leave it null if unknown. Do not claim live prices "
    "or opening hours. " + UNTRUSTED_NOTICE
)


def _itinerary_brief(
    state: PlanState, trip: TripRequest, issues_text: str = "", previous: str = ""
) -> str:
    weather = load(WeatherReport, state.get("weather"))
    wx = "unavailable"
    if weather:
        wx = "; ".join(
            f"{d.date}: {d.condition}, {d.temp_min_c:.0f}-{d.temp_max_c:.0f}C, rain {d.precipitation_mm:.0f}mm"
            for d in weather.days
        )
        if weather.is_climatology:
            wx += " (typical climate, not a forecast)"
    budget = load(BudgetBreakdown, state.get("budget"))
    lines = [
        f"Destination: {trip.destination.name}",
        f"Dates: {trip.start_date} to {trip.end_date} ({trip.duration_days} days)",
        f"Travelers: {trip.travelers}",
        f"Style: {trip.style.value}; pace: {trip.pace.value}",
        f"Currency for costs: {trip.currency}",
        f"Interests: {', '.join(trip.interests) or 'none stated'}",
        f"Dietary: {', '.join(trip.dietary) or 'none'}",
        f"Constraints: {'; '.join(trip.constraints) or 'none'}",
        f"Weather: {wx}",
    ]
    if budget and budget.limit:
        activities = next(
            (ln.amount for ln in budget.lines if ln.category.value == "activities"), None
        )
        if activities:
            lines.append(f"Total activities budget for the party: {activities}")
    if state.get("feedback"):
        lines.append("Traveller feedback to apply: " + state["feedback"][:500])
    if previous:
        lines.append(f"Previous attempt (JSON):\n{previous}\nProblems to fix:\n{issues_text}")
    return wrap_untrusted("\n".join(lines), "trip_brief", limit=6000)


async def itinerary_node(state: PlanState, deps: Deps) -> dict[str, Any]:
    trip = _trip(state)
    budget = load(BudgetBreakdown, state.get("budget"))
    weather = load(WeatherReport, state.get("weather"))
    await deps.emit("node_start", "itinerary", "Drafting your itinerary", {})
    previous, issues_text = "", ""
    itinerary: Itinerary | None = None
    issues: list[Issue] = []
    for attempt in range(deps.max_itinerary_repairs + 1):
        try:
            draft = await deps.llm.structured(
                task="itinerary",
                system=ITIN_SYSTEM,
                user=_itinerary_brief(state, trip, issues_text, previous),
                schema=ItineraryDraft,
                tier="smart",
                temperature=0.5,
                metadata=_meta(state),
            )
            itinerary = draft_to_itinerary(draft, trip.currency)
        except (LLMError, ValueError) as exc:
            log.warning(
                "itinerary attempt failed", extra={"attempt": attempt, "error": type(exc).__name__}
            )
            issues = [Issue(code="ITIN_UNUSABLE", message=str(exc)[:300])]
            issues_text, previous = format_issues(issues), ""
            itinerary = None
            continue
        issues = validate_itinerary(itinerary, trip, budget=budget, weather=weather)
        if not has_errors(issues):
            break
        issues_text = format_issues(errors(issues))
        previous = json.dumps(itinerary.model_dump(mode="json"))[:6000]
        await deps.emit(
            "repair",
            "itinerary",
            f"Fixing {len(errors(issues))} problem(s)",
            {"attempt": attempt + 1},
        )
    if itinerary is None or has_errors(issues):
        await deps.emit(
            "node_done", "itinerary", "Could not produce a valid itinerary", {"degraded": True}
        )
        return {
            "status": PlanStatus.FAILED.value,
            "itinerary": None,
            "notes": ["The itinerary could not be validated; try changing the constraints."],
            "issues": [i.model_dump(mode="json") for i in issues],
        }
    version = int(state.get("itinerary_version", 0)) + 1
    await deps.emit("node_done", "itinerary", f"Itinerary v{version} ready", {"version": version})
    return {
        "itinerary": dump(itinerary),
        "itinerary_version": version,
        "issues": [i.model_dump(mode="json") for i in issues],
    }


# --------------------------------------------------------------------------- finalize
async def finalize_node(state: PlanState, deps: Deps) -> dict[str, Any]:
    """Runs after the human decision. Approved -> done. Rejected -> loop back with feedback."""
    approval = state.get("approval") or {}
    if approval.get("approved"):
        await deps.emit(
            "completed", "finalize", "Plan approved", {"version": state.get("itinerary_version")}
        )
        return {"status": PlanStatus.COMPLETED.value, "feedback": None}
    revisions = int(state.get("revisions", 0)) + 1
    if revisions > deps.max_revisions:
        await deps.emit(
            "completed", "finalize", "Revision limit reached; keeping the latest version", {}
        )
        return {
            "status": PlanStatus.COMPLETED.value,
            "revisions": revisions,
            "approval": None,
            "notes": [
                f"Stopped after {deps.max_revisions} revisions; the latest itinerary was kept."
            ],
        }
    return {
        "status": PlanStatus.RUNNING.value,
        "feedback": approval.get("feedback") or None,
        "revisions": revisions,
        "approval": None,
    }


def route_after_finalize(state: PlanState) -> str:
    return "done" if state.get("status") == PlanStatus.COMPLETED.value else "revise"


def route_after_guardrail(state: PlanState) -> str:
    return "stop" if state.get("status") == PlanStatus.BLOCKED.value else "intake"


FAN_OUT = ("flights", "hotels", "weather", "fx")


def route_after_intake(state: PlanState) -> list[str]:
    """Fan out to every data node, or stop (clarification needed / failed). Returns node names."""
    if state.get("trip") and state.get("status") == PlanStatus.RUNNING.value:
        return list(FAN_OUT)
    return ["__end__"]


def route_after_itinerary(state: PlanState) -> str:
    return "stop" if state.get("status") == PlanStatus.FAILED.value else "approval"
