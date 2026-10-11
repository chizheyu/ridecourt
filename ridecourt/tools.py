"""Deterministic tools that compute facts from trip and chat data.

These functions never call an LLM. They produce the ``Facts`` object consumed
by ``ridecourt.policy.evaluate``.
"""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Sequence

from .data import ChatLog, Facts, Fare, GPSPoint, Trip

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

_EARTH_RADIUS_M = 6_371_000.0


def haversine_m(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance in metres between two coordinates."""
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return _EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def gps_trace_distance_km(trace: Sequence[GPSPoint]) -> float:
    """Total path length of a GPS trace in kilometres."""
    total = 0.0
    for i in range(1, len(trace)):
        total += haversine_m(
            trace[i - 1].lat, trace[i - 1].lon, trace[i].lat, trace[i].lon
        )
    return total / 1000.0


def min_distance_to_point_m(
    trace: Sequence[GPSPoint], lat: float, lon: float
) -> float:
    """Minimum distance from any GPS point to a fixed coordinate."""
    if not trace:
        return float("inf")
    return min(haversine_m(p.lat, p.lon, lat, lon) for p in trace)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

_FMT = "%Y-%m-%dT%H:%M:%S"


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def wait_minutes_past_eta(trip: Trip) -> float:
    """Minutes the rider waited past the driver's ETA.

    ETA is approximated as ``accepted_at + 5 min`` (a fixed 5-minute expected
    arrival window).  If the driver arrived (``driver_arrived_at``), the wait
    is (arrived − eta); if the driver never arrived, the wait is measured up
    to ``trip_ended_at`` or ``filed_at`` (we use the latest available).
    """
    if not trip.accepted_at:
        return 0.0
    eta = _parse(trip.accepted_at)
    if trip.driver_arrived_at:
        arrived = _parse(trip.driver_arrived_at)
        delta = (arrived - eta).total_seconds()
        return max(delta / 60.0, 0.0)
    # driver never arrived — measure to trip end or cancelled
    ref = trip.trip_ended_at or trip.trip_started_at or trip.accepted_at
    ref_dt = _parse(ref)
    delta = (ref_dt - eta).total_seconds()
    return max(delta / 60.0, 0.0)


# ---------------------------------------------------------------------------
# Contact attempts
# ---------------------------------------------------------------------------

def count_contact_attempts(chat: ChatLog | None) -> int:
    """Count call attempts + chat messages sent *to* the relevant party.

    For no-show cases, contact attempts are calls (answered or not) plus chat
    messages from driver to rider.  We count both calls and driver→rider chat
    lines.
    """
    if chat is None:
        return 0
    calls = len(chat.calls)
    chat_lines = sum(
        1
        for m in chat.messages
        if m.sender == "driver"
    )
    return calls + chat_lines


# ---------------------------------------------------------------------------
# Fare recompute
# ---------------------------------------------------------------------------

def recompute_fare(fare: Fare) -> float:
    """Deterministic fare calculation from fare parameters.

    total = (base + distance_km × per_km + duration_min × per_min)
            × surge_multiplier + booking_fee − discount
    """
    variable = fare.base + fare.distance_km * fare.per_km + fare.duration_min * fare.per_min
    subtotal = variable * fare.surge_multiplier
    total = subtotal + fare.booking_fee - fare.discount
    return round(total, 2)


# ---------------------------------------------------------------------------
# Keyword detection
# ---------------------------------------------------------------------------

# Rider requested stop / route change keywords (from rider chat lines).
_REQUESTED_CHANGE_PATTERNS = re.compile(
    r"\b(stop|drop (by|off|me)|detour|different (route|way)|"
    r"go via|pass by|pick up|pickup (someone|my friend)|"
    r"take (the|a) long way|scenic|extra stop)\b",
    re.IGNORECASE,
)

# Road closure keywords (from any chat line, usually driver or system).
_ROAD_CLOSURE_PATTERNS = re.compile(
    r"\b(road closure|road closed|roadblock|road block|"
    r"accident|diversion|diverted|traffic jam|heavy traffic|"
    r"road work|roadwork|construction|lane closed|blocked)\b",
    re.IGNORECASE,
)

# Safety / threat keywords.
_SAFETY_PATTERNS = re.compile(
    r"\b(threat|threaten|afraid|scared|unsafe|harass|"
    r"assault|weapon|knife|hit|scream|shout|danger|"
    r"police|call the cops)\b",
    re.IGNORECASE,
)


def rider_requested_change(chat: ChatLog | None) -> bool:
    if chat is None:
        return False
    return any(
        m.sender == "rider" and _REQUESTED_CHANGE_PATTERNS.search(m.text)
        for m in chat.messages
    )


def road_closure_detected(chat: ChatLog | None) -> bool:
    if chat is None:
        return False
    return any(_ROAD_CLOSURE_PATTERNS.search(m.text) for m in chat.messages)


def safety_keywords_detected(chat: ChatLog | None) -> bool:
    if chat is None:
        return False
    return any(_SAFETY_PATTERNS.search(m.text) for m in chat.messages)


# ---------------------------------------------------------------------------
# Promo validity (simplified)
# ---------------------------------------------------------------------------

# Promo codes that are "valid" in our synthetic world.
# In the full system this would check booking time against an expiry.
_VALID_PROMO_CODES = {"WELCOME10", "NEWUSER", "WEEKEND", "OFF5"}


def promo_valid_at_booking(fare: Fare) -> bool:
    """Return True if the promo code is recognised and applicable."""
    if not fare.promo_code:
        return True  # no promo — nothing to validate, so not invalid
    return fare.promo_code in _VALID_PROMO_CODES


# ---------------------------------------------------------------------------
# Main entry: compute_facts
# ---------------------------------------------------------------------------

def compute_facts(trip: Trip, chat: ChatLog | None) -> Facts:
    """Compute all facts the policy engine needs from trip and chat data."""
    actual_km = gps_trace_distance_km(trip.gps_trace)
    optimal_km = trip.optimal_route.distance_km
    duration_actual = trip.fare.duration_min
    duration_optimal = trip.optimal_route.duration_min
    min_pickup = min_distance_to_point_m(
        trip.gps_trace, trip.pickup.lat, trip.pickup.lon
    )
    wait = wait_minutes_past_eta(trip)
    contacts = count_contact_attempts(chat)
    recomputed = recompute_fare(trip.fare)
    charged = trip.fare.total_charged
    requested = rider_requested_change(chat)
    closure = road_closure_detected(chat)
    valid_promo = promo_valid_at_booking(trip.fare)

    return Facts(
        distance_actual_km=round(actual_km, 4),
        distance_optimal_km=round(optimal_km, 4),
        duration_actual_min=round(duration_actual, 2),
        duration_optimal_min=round(duration_optimal, 2),
        rider_requested_change=requested,
        road_closure=closure,
        min_distance_to_pickup_m=round(min_pickup, 2),
        wait_minutes_past_eta=round(wait, 2),
        contact_attempts=contacts,
        recomputed_fare=recomputed,
        charged_total=charged,
        promo_valid_at_booking=valid_promo,
    )
