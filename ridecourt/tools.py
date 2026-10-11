"""Deterministic tools that compute facts from trip and chat data.

These functions never call an LLM. They produce the ``Facts`` object consumed
by ``ridecourt.policy.evaluate``.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import Sequence

from .data import (
    ChatLog,
    Dispute,
    Dataset,
    EvidenceItem,
    EvidencePack,
    Facts,
    Fare,
    GeoPoint,
    GPSPoint,
    OptimalRoute,
    Profile,
    Trip,
)

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

# Threatening / abusive language — quoted in evidence.
_THREATENING_PATTERNS = re.compile(
    r"\b(threat|threaten|kill|hurt|beat|punch|slap|"
    r"weapon|knife|gun|i'll (find|hurt|kill)|"
    r"watch your back|come after|regret this)\b",
    re.IGNORECASE,
)

_ABUSIVE_PATTERNS = re.compile(
    r"\b(stupid|idiot|moron|asshole|bitch|bastard|"
    r"fuck|shit|damn|crazy|ugly|trash|garbage|"
    r"incompetent|useless|worthless)\b",
    re.IGNORECASE,
)

# Agreement / disagreement signals in chat.
_AGREEMENT_PATTERNS = re.compile(
    r"\b(ok|okay|sure|yes|alright|fine|agreed|no problem|"
    r"that works|sounds good|i understand|got it)\b",
    re.IGNORECASE,
)
_DISAGREEMENT_PATTERNS = re.compile(
    r"\b(no|nope|don't|stop|wrong|unfair|ridiculous|"
    r"disagree|not ok|not okay|unacceptable|refuse|"
    r"that's wrong|no way)\b",
    re.IGNORECASE,
)

# Requested stop / route change — captures the phrase for quoting.
_REQUESTED_CHANGE_QUOTE_PATTERNS = re.compile(
    r"(can you stop[^.!?]*|stop at[^.!?]*|drop (by|off)[^.!?]*|"
    r"detour[^.!?]*|different (route|way)[^.!?]*|"
    r"go via[^.!?]*|pass by[^.!?]*|"
    r"pick up (someone|my friend)[^.!?]*|"
    r"take (the|a) long way[^.!?]*|scenic[^.!?]*|extra stop[^.!?]*)",
    re.IGNORECASE,
)

# Road closure — captures the phrase for quoting.
_ROAD_CLOSURE_QUOTE_PATTERNS = re.compile(
    r"(road closure[^.!?]*|road closed[^.!?]*|roadblock[^.!?]*|"
    r"road block[^.!?]*|accident[^.!?]*|diversion[^.!?]*|"
    r"diverted[^.!?]*|traffic jam[^.!?]*|heavy traffic[^.!?]*|"
    r"road work[^.!?]*|roadwork[^.!?]*|construction[^.!?]*|"
    r"lane closed[^.!?]*|blocked[^.!?]*)",
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
# Evidence tools — each returns a list of EvidenceItem (without stable IDs;
# build_evidence_pack assigns E1..En)
# ---------------------------------------------------------------------------

_STOP_SPEED_KMH = 3.0
_STOP_DURATION_MIN = 2.0


def _segment_ts(trace: Sequence[GPSPoint], idx: int) -> str:
    """Safe timestamp accessor."""
    if 0 <= idx < len(trace):
        return trace[idx].ts
    return ""


def _find_detour_segment(
    trace: Sequence[GPSPoint],
    pickup: GeoPoint,
    dropoff: GeoPoint,
) -> dict | None:
    """Find the start/end timestamps of the largest off-route segment.

    Heuristic: for each GPS point, compute the distance to the straight line
    between pickup and dropoff (cross-track distance approximation). The
    longest consecutive run of points that are significantly off the direct
    path is the detour segment.
    """
    if len(trace) < 3:
        return None

    # Bearing of the direct pickup→dropoff line.
    direct_bearing = _bearing(
        pickup.lat, pickup.lon, dropoff.lat, dropoff.lon
    )
    off_threshold_m = 200.0  # 200 m off the direct path = "off route"

    off_flags: list[bool] = []
    for p in trace:
        xtd = _cross_track_distance_m(
            p.lat, p.lon,
            pickup.lat, pickup.lon,
            dropoff.lat, dropoff.lon,
            direct_bearing,
        )
        off_flags.append(xtd > off_threshold_m)

    # Find the longest run of True in off_flags.
    best_start, best_end = -1, -1
    cur_start = -1
    for i, flag in enumerate(off_flags):
        if flag:
            if cur_start < 0:
                cur_start = i
        else:
            if cur_start >= 0 and (i - cur_start) > (best_end - best_start):
                best_start, best_end = cur_start, i - 1
            cur_start = -1
    if cur_start >= 0 and (len(off_flags) - cur_start) > (best_end - best_start):
        best_start, best_end = cur_start, len(off_flags) - 1

    if best_start < 0:
        return None
    return {
        "start_ts": _segment_ts(trace, best_start),
        "end_ts": _segment_ts(trace, best_end),
        "start_idx": best_start,
        "end_idx": best_end,
    }


def _bearing(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing in radians from point 1 to point 2."""
    rlat1, rlat2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(rlat2)
    x = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon)
    return math.atan2(y, x)


def _cross_track_distance_m(
    plat, plon,  # point to test
    start_lat, start_lon,
    end_lat, end_lon,
    bearing_start_end,
) -> float:
    """Approximate cross-track distance (perpendicular offset) in metres.

    Uses the along-track / cross-track spherical formula.
    """
    # Great-circle distance from start to the point.
    d13 = haversine_m(start_lat, start_lon, plat, plon) / _EARTH_RADIUS_M
    if d13 == 0:
        return 0.0
    # Bearing from start to the point.
    rlat1 = math.radians(start_lat)
    rlat2 = math.radians(plat)
    dlon = math.radians(plon - start_lon)
    y = math.sin(dlon) * math.cos(rlat2)
    x = math.cos(rlat1) * math.sin(rlat2) - math.sin(rlat1) * math.cos(rlat2) * math.cos(dlon)
    bearing13 = math.atan2(y, x)

    d_xt = math.asin(
        max(-1.0, min(1.0, math.sin(d13) * math.sin(bearing13 - bearing_start_end)))
    ) * _EARTH_RADIUS_M
    return abs(d_xt)


def _detect_unexpected_stops(
    trace: Sequence[GPSPoint],
    pickup: GeoPoint,
    dropoff: GeoPoint,
) -> list[dict]:
    """Detect stops (speed < 3 km/h for > 2 min) away from pickup/dropoff."""
    stops: list[dict] = []
    if not trace:
        return stops

    pickup_radius_m = 150.0
    dropoff_radius_m = 150.0

    i = 0
    n = len(trace)
    while i < n:
        if trace[i].speed_kmh < _STOP_SPEED_KMH:
            # Check we're away from pickup and dropoff.
            dist_pickup = haversine_m(
                trace[i].lat, trace[i].lon, pickup.lat, pickup.lon
            )
            dist_dropoff = haversine_m(
                trace[i].lat, trace[i].lon, dropoff.lat, dropoff.lon
            )
            if dist_pickup <= pickup_radius_m or dist_dropoff <= dropoff_radius_m:
                i += 1
                continue

            # Extend the stop run.
            j = i
            while j < n and trace[j].speed_kmh < _STOP_SPEED_KMH:
                j += 1
            if j > i:
                start_ts = trace[i].ts
                end_ts = trace[j - 1].ts if j - 1 < n else trace[-1].ts
                try:
                    dur = (_parse(end_ts) - _parse(start_ts)).total_seconds() / 60.0
                except Exception:
                    dur = 0.0
                if dur > _STOP_DURATION_MIN:
                    stops.append({
                        "start_ts": start_ts,
                        "end_ts": end_ts,
                        "duration_min": round(dur, 2),
                        "lat": trace[i].lat,
                        "lon": trace[i].lon,
                    })
            i = j
        else:
            i += 1
    return stops


def gps_analysis(trip: Trip) -> list[dict]:
    """Analyze GPS trace vs optimal route.

    Returns evidence items (without final IDs) for:
    - actual vs optimal distance and duration
    - deviation percent
    - detour segment (start/end timestamps)
    - unexpected stops
    """
    actual_km = gps_trace_distance_km(trip.gps_trace)
    optimal_km = trip.optimal_route.distance_km
    actual_min = trip.fare.duration_min
    optimal_min = trip.optimal_route.duration_min
    extra_km = round(actual_km - optimal_km, 4)
    pct = (
        round(extra_km / optimal_km * 100, 2)
        if optimal_km > 0
        else 0.0
    )
    detour = _find_detour_segment(trip.gps_trace, trip.pickup, trip.dropoff)
    stops = _detect_unexpected_stops(trip.gps_trace, trip.pickup, trip.dropoff)

    items: list[dict] = []

    items.append({
        "kind": "gps_analysis",
        "summary": (
            f"Actual distance {actual_km:.2f} km vs optimal "
            f"{optimal_km:.2f} km; deviation {pct:.1f}% "
            f"({extra_km:.2f} km extra)."
        ),
        "data": {
            "distance_actual_km": round(actual_km, 4),
            "distance_optimal_km": round(optimal_km, 4),
            "duration_actual_min": round(actual_min, 2),
            "duration_optimal_min": round(optimal_min, 2),
            "extra_km": extra_km,
            "deviation_pct": pct,
        },
        "source": "gps_analysis",
    })

    if detour:
        items.append({
            "kind": "gps_analysis",
            "summary": (
                f"Detour segment from {detour['start_ts']} to "
                f"{detour['end_ts']}."
            ),
            "data": {"detour_segment": detour},
            "source": "gps_analysis",
        })

    if stops:
        items.append({
            "kind": "gps_analysis",
            "summary": (
                f"{len(stops)} unexpected stop(s) detected "
                f"(speed < {_STOP_SPEED_KMH} km/h for > "
                f"{_STOP_DURATION_MIN} min away from pickup/dropoff)."
            ),
            "data": {"unexpected_stops": stops},
            "source": "gps_analysis",
        })

    return items


def arrival_check(trip: Trip) -> list[dict]:
    """Check driver arrival: closest distance to pickup, arrival vs ETA,
    minutes waited within 150 m."""
    min_dist = min_distance_to_point_m(
        trip.gps_trace, trip.pickup.lat, trip.pickup.lon
    ) if trip.gps_trace else float("inf")

    # Arrival timestamp = the GPS point closest to pickup (within the trace).
    arrival_ts: str | None = None
    if trip.gps_trace:
        closest_pt = min(
            trip.gps_trace,
            key=lambda p: haversine_m(p.lat, p.lon, trip.pickup.lat, trip.pickup.lon),
        )
        arrival_ts = closest_pt.ts

    # ETA = accepted_at + 5 min (expected arrival window).
    arrival_vs_eta = 0.0
    if trip.accepted_at and arrival_ts:
        eta = _parse(trip.accepted_at) + timedelta(minutes=5)
        arrival_vs_eta = (_parse(arrival_ts) - eta).total_seconds() / 60.0

    # Minutes the driver waited within 150 m of pickup.
    wait_within_150 = 0.0
    if trip.gps_trace and trip.driver_arrived_at:
        # Count GPS points within 150 m of pickup and sum their time intervals.
        within = [
            p for p in trip.gps_trace
            if haversine_m(p.lat, p.lon, trip.pickup.lat, trip.pickup.lon) <= 150.0
        ]
        if len(within) >= 2:
            wait_within_150 = (
                _parse(within[-1].ts) - _parse(within[0].ts)
            ).total_seconds() / 60.0

    # Fallback: use the existing wait_minutes_past_eta helper for the
    # policy engine's no-show logic.
    wait_past = wait_minutes_past_eta(trip)

    summary = (
        f"Driver closest distance to pickup: {min_dist:.0f} m; "
        f"arrival {arrival_ts or 'unknown'} vs ETA "
        f"{'+' if arrival_vs_eta >= 0 else ''}{arrival_vs_eta:.1f} min; "
        f"waited {wait_within_150:.1f} min within 150 m."
    )

    return [{
        "kind": "arrival_check",
        "summary": summary,
        "data": {
            "min_distance_to_pickup_m": round(min_dist, 2),
            "arrival_ts": arrival_ts,
            "arrival_vs_eta_min": round(arrival_vs_eta, 2),
            "wait_minutes_within_150m": round(wait_within_150, 2),
            "wait_minutes_past_eta": round(wait_past, 2),
        },
        "source": "arrival_check",
    }]


def chat_scan(chats: ChatLog | None) -> list[dict]:
    """Scan chat for: requested route changes/stops, road closures,
    agreements, disagreements, threatening/abusive language, contact attempts.
    Quotes the relevant lines."""
    if chats is None:
        return [{
            "kind": "chat_scan",
            "summary": "No chat log available.",
            "data": {
                "rider_requested_change": False,
                "rider_requested_change_quotes": [],
                "road_closure": False,
                "road_closure_quotes": [],
                "agreements": [],
                "disagreements": [],
                "abusive_language": [],
                "threatening_language": [],
                "contact_attempts": 0,
            },
            "source": "chat_scan",
        }]

    requested_quotes: list[str] = []
    closure_quotes: list[str] = []
    agreements: list[str] = []
    disagreements: list[str] = []
    abusive: list[str] = []
    threatening: list[str] = []

    for m in chats.messages:
        text = m.text.strip()
        # Requested change (rider messages only).
        if m.sender == "rider":
            match = _REQUESTED_CHANGE_QUOTE_PATTERNS.search(text)
            if match:
                requested_quotes.append(f"[{m.ts}] {m.sender}: \"{text}\"")
            elif _REQUESTED_CHANGE_PATTERNS.search(text):
                requested_quotes.append(f"[{m.ts}] {m.sender}: \"{text}\"")

        # Road closure (any sender).
        closure_match = _ROAD_CLOSURE_QUOTE_PATTERNS.search(text)
        if closure_match:
            closure_quotes.append(f"[{m.ts}] {m.sender}: \"{text}\"")
        elif _ROAD_CLOSURE_PATTERNS.search(text):
            closure_quotes.append(f"[{m.ts}] {m.sender}: \"{text}\"")

        # Agreements.
        if _AGREEMENT_PATTERNS.search(text):
            agreements.append(f"[{m.ts}] {m.sender}: \"{text}\"")

        # Disagreements.
        if _DISAGREEMENT_PATTERNS.search(text):
            disagreements.append(f"[{m.ts}] {m.sender}: \"{text}\"")

        # Threatening language.
        if _THREATENING_PATTERNS.search(text):
            threatening.append(f"[{m.ts}] {m.sender}: \"{text}\"")

        # Abusive language.
        if _ABUSIVE_PATTERNS.search(text):
            abusive.append(f"[{m.ts}] {m.sender}: \"{text}\"")

    contacts = count_contact_attempts(chats)

    parts: list[str] = []
    if requested_quotes:
        parts.append(f"{len(requested_quotes)} requested change(s)")
    if closure_quotes:
        parts.append(f"{len(closure_quotes)} road closure mention(s)")
    if agreements:
        parts.append(f"{len(agreements)} agreement(s)")
    if disagreements:
        parts.append(f"{len(disagreements)} disagreement(s)")
    if threatening:
        parts.append(f"{len(threatening)} threatening message(s)")
    if abusive:
        parts.append(f"{len(abusive)} abusive message(s)")
    parts.append(f"{contacts} contact attempt(s)")
    summary = "Chat scan: " + "; ".join(parts) + "."

    return [{
        "kind": "chat_scan",
        "summary": summary,
        "data": {
            "rider_requested_change": len(requested_quotes) > 0,
            "rider_requested_change_quotes": requested_quotes,
            "road_closure": len(closure_quotes) > 0,
            "road_closure_quotes": closure_quotes,
            "agreements": agreements,
            "disagreements": disagreements,
            "abusive_language": abusive,
            "threatening_language": threatening,
            "contact_attempts": contacts,
        },
        "source": "chat_scan",
    }]


def fare_check(trip: Trip) -> list[dict]:
    """Recompute the fare from components and compare with total charged;
    check promo applied or not."""
    recomputed = recompute_fare(trip.fare)
    charged = trip.fare.total_charged
    diff = round(charged - recomputed, 2)
    promo_applied = bool(trip.fare.promo_code) and trip.fare.discount > 0
    valid_promo = promo_valid_at_booking(trip.fare)

    summary = (
        f"Recomputed fare S${recomputed:.2f} vs charged S${charged:.2f} "
        f"(diff S${diff:.2f}); promo "
        f"{'applied' if promo_applied else 'not applied'} "
        f"({'valid' if valid_promo else 'invalid'})."
    )

    return [{
        "kind": "fare_check",
        "summary": summary,
        "data": {
            "recomputed_fare": recomputed,
            "charged_total": charged,
            "fare_difference": diff,
            "promo_code": trip.fare.promo_code,
            "promo_applied": promo_applied,
            "promo_valid_at_booking": valid_promo,
            "components": {
                "base": trip.fare.base,
                "per_km": trip.fare.per_km,
                "per_min": trip.fare.per_min,
                "distance_km": trip.fare.distance_km,
                "duration_min": trip.fare.duration_min,
                "surge_multiplier": trip.fare.surge_multiplier,
                "discount": trip.fare.discount,
                "booking_fee": trip.fare.booking_fee,
            },
        },
        "source": "fare_check",
    }]


def history_profile(
    profiles: dict[str, Profile],
    rider_id: str,
    driver_id: str,
) -> list[dict]:
    """Build history/profile evidence for rider and driver."""
    items: list[dict] = []
    rider = profiles.get(rider_id)
    driver = profiles.get(driver_id)

    if rider:
        filed = rider.disputes_filed_90d
        upheld = rider.disputes_upheld_90d
        rate = round(upheld / filed * 100, 1) if filed > 0 else 0.0
        items.append({
            "kind": "history_profile",
            "summary": (
                f"Rider {rider_id}: {filed} disputes filed (90d), "
                f"{upheld} upheld ({rate:.1f}%), "
                f"account age {rider.account_age_days}d, "
                f"rating {rider.rating:.1f}, "
                f"flags: {rider.prior_flags or 'none'}."
            ),
            "data": {
                "party": "rider",
                "id": rider_id,
                "disputes_filed_90d": filed,
                "disputes_upheld_90d": upheld,
                "upheld_rate": rate,
                "account_age_days": rider.account_age_days,
                "rating": rider.rating,
                "trips_completed": rider.trips_completed,
                "prior_flags": rider.prior_flags,
            },
            "source": "history_profile",
        })

    if driver:
        filed = driver.disputes_filed_90d
        upheld = driver.disputes_upheld_90d
        rate = round(upheld / filed * 100, 1) if filed > 0 else 0.0
        items.append({
            "kind": "history_profile",
            "summary": (
                f"Driver {driver_id}: {filed} disputes filed (90d), "
                f"{upheld} upheld ({rate:.1f}%), "
                f"account age {driver.account_age_days}d, "
                f"rating {driver.rating:.1f}, "
                f"flags: {driver.prior_flags or 'none'}."
            ),
            "data": {
                "party": "driver",
                "id": driver_id,
                "disputes_filed_90d": filed,
                "disputes_upheld_90d": upheld,
                "upheld_rate": rate,
                "account_age_days": driver.account_age_days,
                "rating": driver.rating,
                "trips_completed": driver.trips_completed,
                "prior_flags": driver.prior_flags,
            },
            "source": "history_profile",
        })

    return items


# ---------------------------------------------------------------------------
# Evidence pack builder
# ---------------------------------------------------------------------------

# Which tools to run for each dispute category.
_CATEGORY_TOOLS: dict[str, set[str]] = {
    "route_deviation": {"gps_analysis", "chat_scan", "fare_check", "history_profile"},
    "no_show_driver": {"arrival_check", "chat_scan", "history_profile"},
    "no_show_rider": {"arrival_check", "chat_scan", "history_profile"},
    "fare_error": {"fare_check", "chat_scan", "history_profile"},
    "safety": {"chat_scan", "gps_analysis", "arrival_check", "history_profile"},
}


def build_evidence_pack(
    dispute: Dispute,
    dataset: Dataset,
) -> EvidencePack:
    """Run the relevant tools for the dispute category and number the items
    E1..En."""
    trip = dataset.trip(dispute.trip_id)
    if trip is None:
        raise ValueError(f"Trip {dispute.trip_id} not found in dataset")
    chat = dataset.chat(dispute.trip_id)

    tool_names = _CATEGORY_TOOLS.get(dispute.category, set())
    raw_items: list[dict] = []

    if "gps_analysis" in tool_names:
        raw_items.extend(gps_analysis(trip))
    if "arrival_check" in tool_names:
        raw_items.extend(arrival_check(trip))
    if "chat_scan" in tool_names:
        raw_items.extend(chat_scan(chat))
    if "fare_check" in tool_names:
        raw_items.extend(fare_check(trip))
    if "history_profile" in tool_names:
        raw_items.extend(
            history_profile(dataset.profiles, trip.rider_id, trip.driver_id)
        )

    # Assign stable IDs E1, E2, ...
    evidence_items: list[EvidenceItem] = []
    for i, raw in enumerate(raw_items, start=1):
        evidence_items.append(
            EvidenceItem(id=f"E{i}", **raw)
        )
    return EvidencePack(items=evidence_items)


# ---------------------------------------------------------------------------
# Facts extraction from the evidence pack (single source of truth)
# ---------------------------------------------------------------------------

def _pack_data(pack: EvidencePack, kind: str) -> dict:
    """Merge all data dicts of a given kind into one dict."""
    merged: dict = {}
    for item in pack.by_kind(kind):
        merged.update(item.data)
    return merged


def facts_from_evidence_pack(pack: EvidencePack) -> Facts:
    """Build a ``Facts`` object from the evidence pack so the policy engine
    and the agents use the same numbers."""
    gps = _pack_data(pack, "gps_analysis")
    arrival = _pack_data(pack, "arrival_check")
    chat = _pack_data(pack, "chat_scan")
    fare = _pack_data(pack, "fare_check")

    # Extract rider/driver history items properly.
    rider_hist: dict | None = None
    driver_hist: dict | None = None
    for item in pack.by_kind("history_profile"):
        if item.data.get("party") == "rider":
            rider_hist = item.data
        elif item.data.get("party") == "driver":
            driver_hist = item.data

    facts = Facts(
        distance_actual_km=gps.get("distance_actual_km", 0.0),
        distance_optimal_km=gps.get("distance_optimal_km", 0.0),
        duration_actual_min=gps.get("duration_actual_min", 0.0),
        duration_optimal_min=gps.get("duration_optimal_min", 0.0),
        deviation_pct=gps.get("deviation_pct", 0.0),
        detour_segment=gps.get("detour_segment"),
        unexpected_stops=gps.get("unexpected_stops", []),
        rider_requested_change=chat.get("rider_requested_change", False),
        rider_requested_change_quotes=chat.get("rider_requested_change_quotes", []),
        road_closure=chat.get("road_closure", False),
        road_closure_quotes=chat.get("road_closure_quotes", []),
        agreements=chat.get("agreements", []),
        disagreements=chat.get("disagreements", []),
        abusive_language=chat.get("abusive_language", []),
        threatening_language=chat.get("threatening_language", []),
        contact_attempts=chat.get("contact_attempts", 0),
        min_distance_to_pickup_m=arrival.get("min_distance_to_pickup_m", float("inf")),
        arrival_ts=arrival.get("arrival_ts"),
        arrival_vs_eta_min=arrival.get("arrival_vs_eta_min", 0.0),
        wait_minutes_within_150m=arrival.get("wait_minutes_within_150m", 0.0),
        wait_minutes_past_eta=arrival.get("wait_minutes_past_eta", 0.0),
        recomputed_fare=fare.get("recomputed_fare", 0.0),
        charged_total=fare.get("charged_total", 0.0),
        fare_difference=fare.get("fare_difference", 0.0),
        promo_applied=fare.get("promo_applied", False),
        promo_valid_at_booking=fare.get("promo_valid_at_booking", True),
    )

    if rider_hist:
        facts.rider_disputes_filed_90d = rider_hist.get("disputes_filed_90d", 0)
        facts.rider_disputes_upheld_90d = rider_hist.get("disputes_upheld_90d", 0)
        facts.rider_upheld_rate = rider_hist.get("upheld_rate", 0.0)
        facts.rider_account_age_days = rider_hist.get("account_age_days", 0)
        facts.rider_rating = rider_hist.get("rating", 0.0)
        facts.rider_prior_flags = rider_hist.get("prior_flags", [])

    if driver_hist:
        facts.driver_disputes_filed_90d = driver_hist.get("disputes_filed_90d", 0)
        facts.driver_disputes_upheld_90d = driver_hist.get("disputes_upheld_90d", 0)
        facts.driver_upheld_rate = driver_hist.get("upheld_rate", 0.0)
        facts.driver_account_age_days = driver_hist.get("account_age_days", 0)
        facts.driver_rating = driver_hist.get("rating", 0.0)
        facts.driver_prior_flags = driver_hist.get("prior_flags", [])

    return facts


# ---------------------------------------------------------------------------
# Main entry: compute_facts (now reads from the evidence pack)
# ---------------------------------------------------------------------------

def compute_facts(trip: Trip, chat: ChatLog | None) -> Facts:
    """Compute all facts the policy engine needs from trip and chat data.

    This now delegates to the evidence tools so the policy engine and the
    agents share the same numbers (single source of truth).  For cases where
    a full ``Dataset`` and ``Dispute`` are available, prefer
    ``build_evidence_pack`` + ``facts_from_evidence_pack``.
    """
    # Build a minimal evidence pack directly from trip + chat (no dataset
    # needed, no history profile).  This keeps backward compatibility with
    # existing callers (test_data.py, test_policy.py helpers).
    raw_items: list[dict] = []
    raw_items.extend(gps_analysis(trip))
    raw_items.extend(arrival_check(trip))
    raw_items.extend(chat_scan(chat))
    raw_items.extend(fare_check(trip))

    pack = EvidencePack(
        items=[
            EvidenceItem(id=f"E{i}", **raw)
            for i, raw in enumerate(raw_items, start=1)
        ]
    )
    return facts_from_evidence_pack(pack)
