"""Generate the synthetic RideCourt v2.1 dataset.

Usage::

    .venv/bin/python scripts/generate_data.py --seed 7

Writes ``data/synthetic/`` with trips.json, chats.json, profiles.json,
disputes.json, precedents.json, and labels.json.

Labels come from scenario intent (the known expected outcome for each
scenario), not from the policy engine, so the policy engine can be
evaluated against the labels without circularity.

Covers every v2.1 section row:
  §1.3: route deviation (justified+informed, justified+not informed,
        not justified, within tolerance)
  §2:   no-show (both contacts met → fee upheld, one missing → voided,
        driver no-show → credit, 3+ no-shows → flag)
  §3.3: surge (increased+notified, increased+not notified, unchanged,
        decreased)
  §4:   property damage (verifiable cleaning, not verifiable, repair,
        rider dispute)
  §5:   safety (always human)
  §6:   routing (under S$10 AI final, S$10-50 with confidence, >S$50 human,
        route >2km human, fraud flag)
"""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

# Make the package importable when running the script directly.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ridecourt.data import (
    AppEvent,
    CallAttempt,
    ChatLog,
    ChatMessage,
    DamageEvidence,
    Dispute,
    Fare,
    GPSPoint,
    GeoPoint,
    OptimalRoute,
    Precedent,
    Profile,
    Trip,
    TrafficIncident,
)

# ---------------------------------------------------------------------------
# Singapore landmarks (real coordinates)
# ---------------------------------------------------------------------------

SG_LOCATIONS = [
    ("Tampines Mall", 1.3521, 103.9450),
    ("Jurong East MRT", 1.3329, 103.7436),
    ("Changi Airport T3", 1.3586, 103.9889),
    ("Orchard MRT", 1.3041, 103.8318),
    ("Punggol MRT", 1.4053, 103.7621),
    ("Marina Bay Sands", 1.2834, 103.8607),
    ("Bedok MRT", 1.3245, 103.9322),
    ("Bishan MRT", 1.3509, 103.8478),
    ("Woodlands MRT", 1.4382, 103.7867),
    ("Sengkang MRT", 1.3915, 103.8912),
    ("Bukit Batok MRT", 1.3494, 103.7496),
    ("Serangoon MRT", 1.3496, 103.8745),
    ("Clementi MRT", 1.3151, 103.7647),
    ("Hougang MRT", 1.3713, 103.8924),
    ("Ang Mo Kio MRT", 1.3698, 103.8446),
    ("Toa Payoh MRT", 1.3325, 103.8475),
    ("Bishan", 1.3509, 103.8478),
    ("Geylang", 1.3116, 103.8863),
    ("Lavender MRT", 1.3128, 103.8539),
    ("Bugis MRT", 1.3009, 103.8557),
]

# ---------------------------------------------------------------------------
# v2.1 constants
# ---------------------------------------------------------------------------

BASE_FARE = 3.0
PER_KM = 0.80  # v2.1 §1.3: per-km distance rate
PER_MIN = 0.25
BOOKING_FEE = 1.0
CANCELLATION_FEE = 5.0  # v2.1 §2.2: S$5.00

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_TIME = datetime(2026, 1, 1, 8, 0, 0)


def ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def add_minutes(dt: datetime, m: float) -> datetime:
    return dt + timedelta(minutes=m)


def haversine_m(lat1, lon1, lat2, lon2):
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return 6_371_000 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def interpolate_route(
    pickup: tuple[float, float],
    dropoff: tuple[float, float],
    n_points: int = 8,
) -> list[tuple[float, float]]:
    pts = []
    for i in range(n_points):
        t = i / (n_points - 1)
        lat = pickup[0] + (dropoff[0] - pickup[0]) * t
        lon = pickup[1] + (dropoff[1] - pickup[1]) * t
        pts.append((lat, lon))
    return pts


def detour_route(
    pickup: tuple[float, float],
    dropoff: tuple[float, float],
    n_points: int = 12,
    detour_factor: float = 1.4,
) -> list[tuple[float, float]]:
    mid_lat = (pickup[0] + dropoff[0]) / 2
    mid_lon = (pickup[1] + dropoff[1]) / 2
    dx = dropoff[0] - pickup[0]
    dy = dropoff[1] - pickup[1]
    length = math.sqrt(dx * dx + dy * dy) or 1
    perp_lat = -dy / length * 0.02 * detour_factor
    perp_lon = dx / length * 0.02 * detour_factor

    pts = []
    for i in range(n_points):
        t = i / (n_points - 1)
        lat = pickup[0] + (dropoff[0] - pickup[0]) * t
        lon = pickup[1] + (dropoff[1] - pickup[1]) * t
        bump = math.sin(math.pi * t) * detour_factor
        lat += perp_lat * bump
        lon += perp_lon * bump
        pts.append((lat, lon))
    return pts


def make_gps_trace(
    route_pts: list[tuple[float, float]],
    start_dt: datetime,
    interval_s: int = 15,
    speed_kmh: float = 45.0,
) -> list[GPSPoint]:
    trace = []
    dt_cur = start_dt
    for i, (lat, lon) in enumerate(route_pts):
        s = speed_kmh + random.uniform(-8, 8)
        trace.append(GPSPoint(ts=ts(dt_cur), lat=lat, lon=lon, speed_kmh=max(s, 5)))
        if i < len(route_pts) - 1:
            dt_cur = add_minutes(dt_cur, interval_s / 60.0)
    return trace


def trace_distance_km(trace: list[GPSPoint]) -> float:
    total = 0.0
    for i in range(1, len(trace)):
        total += haversine_m(
            trace[i - 1].lat, trace[i - 1].lon, trace[i].lat, trace[i].lon
        )
    return total / 1000.0


def route_distance_km(route_pts: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(1, len(route_pts)):
        total += haversine_m(
            route_pts[i - 1][0], route_pts[i - 1][1],
            route_pts[i][0], route_pts[i][1],
        )
    return total / 1000.0


def make_fare(
    distance_km: float,
    duration_min: float,
    surge: float = 1.0,
    promo_code: str = "",
    discount: float = 0.0,
    total_charged: float | None = None,
    cancellation_fee: float = CANCELLATION_FEE,
    quote_surge: float = 1.0,
    trip_surge: float = 1.0,
    surgeable_subtotal: float = 0.0,
    upfront_quote: float = 0.0,
    rider_notified_surge: bool = False,
    excess_distance_km: float = 0.0,
    excess_distance_fare: float = 0.0,
) -> Fare:
    variable = BASE_FARE + distance_km * PER_KM + duration_min * PER_MIN
    subtotal = variable * surge
    total = subtotal + BOOKING_FEE - discount
    return Fare(
        base=BASE_FARE,
        per_km=PER_KM,
        per_min=PER_MIN,
        distance_km=round(distance_km, 2),
        duration_min=round(duration_min, 2),
        surge_multiplier=surge,
        promo_code=promo_code,
        discount=discount,
        booking_fee=BOOKING_FEE,
        cancellation_fee=cancellation_fee,
        total_charged=round(total if total_charged is None else total_charged, 2),
        quote_surge_multiplier=quote_surge,
        trip_start_surge_multiplier=trip_surge,
        surgeable_subtotal=surgeable_subtotal,
        upfront_quote=upfront_quote,
        rider_notified_surge=rider_notified_surge,
        excess_distance_km=excess_distance_km,
        excess_distance_fare=excess_distance_fare,
    )


def _make_trip(
    trip_id: str,
    rider_id: str,
    driver_id: str,
    pickup: GeoPoint,
    dropoff: GeoPoint,
    requested_dt: datetime,
    accepted_dt: datetime,
    arrived_dt: datetime | None,
    started_dt: datetime | None,
    ended_dt: datetime | None,
    status: str,
    gps_trace: list[GPSPoint],
    optimal_route: OptimalRoute,
    fare: Fare,
    traffic_conditions: list[TrafficIncident] | None = None,
    damage_evidence: DamageEvidence | None = None,
    scheduled_pickup_at: str | None = None,
    app_events: list[AppEvent] | None = None,
) -> Trip:
    return Trip(
        trip_id=trip_id,
        rider_id=rider_id,
        driver_id=driver_id,
        pickup=pickup,
        dropoff=dropoff,
        requested_at=ts(requested_dt),
        accepted_at=ts(accepted_dt),
        driver_arrived_at=ts(arrived_dt) if arrived_dt else None,
        trip_started_at=ts(started_dt) if started_dt else None,
        trip_ended_at=ts(ended_dt) if ended_dt else None,
        scheduled_pickup_at=scheduled_pickup_at,
        status=status,
        gps_trace=gps_trace,
        optimal_route=optimal_route,
        fare=fare,
        traffic_conditions=traffic_conditions or [],
        damage_evidence=damage_evidence,
        app_events=app_events or [],
    )


def _pick_locations(rng: random.Random) -> tuple[tuple, tuple, str, str]:
    src = rng.choice(SG_LOCATIONS)
    dst = rng.choice([l for l in SG_LOCATIONS if l[0] != src[0]])
    return (
        (src[1], src[2]),
        (dst[1], dst[2]),
        src[0],
        dst[0],
    )


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------

def make_profiles(rng: random.Random) -> dict[str, Profile]:
    profiles = {}

    # Fraud-flagged rider (§6.3: raises confidence threshold to 95%)
    profiles["rider_fraud1"] = Profile(
        account_age_days=120,
        rating=3.5,
        trips_completed=22,
        disputes_filed_90d=5,
        disputes_upheld_90d=1,
        cancellations_30d=6,
        prior_flags=["fraud_flag"],
        fraud_flags=1,
        no_show_cancellations_30d=4,
    )

    # Repeat no-show rider (§2.5: 3+ no-shows → flag)
    profiles["rider_repeat_ns"] = Profile(
        account_age_days=300,
        rating=3.8,
        trips_completed=45,
        disputes_filed_90d=3,
        disputes_upheld_90d=1,
        cancellations_30d=5,
        prior_flags=["frequent_no_show"],
        fraud_flags=0,
        no_show_cancellations_30d=3,
    )

    # Normal riders
    for i in range(1, 30):
        filed = rng.randint(0, 3)
        upheld = rng.randint(0, min(filed, 2))
        profiles[f"rider_{i:03d}"] = Profile(
            account_age_days=rng.randint(30, 800),
            rating=round(rng.uniform(4.0, 5.0), 1),
            trips_completed=rng.randint(5, 300),
            disputes_filed_90d=filed,
            disputes_upheld_90d=upheld,
            cancellations_30d=rng.randint(0, 3),
            fraud_flags=0,
            no_show_cancellations_30d=rng.randint(0, 2),
        )

    # Normal drivers
    for i in range(1, 30):
        filed = rng.randint(0, 2)
        upheld = rng.randint(0, min(filed, 1))
        profiles[f"driver_{i:03d}"] = Profile(
            account_age_days=rng.randint(60, 1200),
            rating=round(rng.uniform(4.2, 5.0), 1),
            trips_completed=rng.randint(50, 2000),
            disputes_filed_90d=filed,
            disputes_upheld_90d=upheld,
            cancellations_30d=rng.randint(0, 5),
            fraud_flags=0,
            no_show_cancellations_30d=0,
        )

    return profiles


# ---------------------------------------------------------------------------
# §1 Route Deviation scenarios (covering every §1.3 row)
# ---------------------------------------------------------------------------

def gen_route_deviation(
    rng: random.Random, idx: int, profiles: dict
) -> tuple[Trip, ChatLog, Dispute, dict]:
    """Generate a route deviation case.

    Scenarios (cycling through rows of §1.3):
      0: justified + informed → no refund
      1: justified + NOT informed → partial refund
      2: NOT justified → full refund
      3: within tolerance (≤1 km) → no refund
      4: justified + NOT informed → partial refund (large deviation >2km → human)
      5: NOT justified → full refund
      6: justified + informed → no refund
      7: within tolerance → no refund
    """
    scenario = idx % 8
    justified = scenario in (0, 1, 4, 6)
    informed = scenario in (0, 6)
    within_tolerance = scenario in (3, 7)

    (plat, plon), (dlat, dlon), paddr, daddr = _pick_locations(rng)
    pickup = GeoPoint(lat=plat, lon=plon, address=paddr)
    dropoff = GeoPoint(lat=dlat, lon=dlon, address=daddr)

    requested_dt = BASE_TIME + timedelta(hours=idx)
    accepted_dt = requested_dt + timedelta(minutes=2)

    if within_tolerance:
        # Use the optimal route with a tiny wiggle — excess must be ≤1 km
        route_pts = interpolate_route((plat, plon), (dlat, dlon), n_points=10)
        # Add a small perpendicular bump to create minor excess
        dx = dlat - plat
        dy = dlon - plon
        length = math.sqrt(dx * dx + dy * dy) or 1
        perp_lat = -dy / length * 0.001  # ~100m bump
        perp_lon = dx / length * 0.001
        route_pts = [
            (lat + perp_lat * math.sin(math.pi * i / (len(route_pts) - 1)),
             lon + perp_lon * math.sin(math.pi * i / (len(route_pts) - 1)))
            for i, (lat, lon) in enumerate(route_pts)
        ]
    else:
        route_pts = detour_route((plat, plon), (dlat, dlon), n_points=14, detour_factor=1.5)

    optimal_pts = interpolate_route((plat, plon), (dlat, dlon), n_points=8)
    opt_km = route_distance_km(optimal_pts)
    opt_min = opt_km / 40 * 60

    start_dt = accepted_dt + timedelta(minutes=5)
    trace = make_gps_trace(route_pts, start_dt, interval_s=15, speed_kmh=40)
    actual_km = trace_distance_km(trace)
    actual_min = actual_km / 40 * 60

    excess_km = max(0.0, round(actual_km - opt_km, 2))
    excess_fare = round(excess_km * PER_KM, 2)

    fare = make_fare(actual_km, actual_min, surge=1.0,
                     excess_distance_km=excess_km,
                     excess_distance_fare=excess_fare)

    # Traffic conditions
    traffic: list[TrafficIncident] = []
    if justified:
        traffic = [TrafficIncident(
            incident_id=f"TC-{idx:03d}",
            road="PIE",
            direction="eastbound",
            stretch="between exits 10-16",
            incident_type="heavy_congestion",
            severity="high" if scenario == 4 else "moderate",
            timestamp=ts(start_dt),
            clear_time=ts(start_dt + timedelta(minutes=30)),
            description="Heavy congestion on PIE eastbound.",
            on_recommended_route=True,
        )]

    trip = _make_trip(
        trip_id=f"trip_rd_{idx:03d}",
        rider_id=f"rider_{rng.randint(1, 29):03d}",
        driver_id=f"driver_{rng.randint(1, 29):03d}",
        pickup=pickup, dropoff=dropoff,
        requested_dt=requested_dt, accepted_dt=accepted_dt,
        arrived_dt=accepted_dt + timedelta(minutes=4),
        started_dt=start_dt,
        ended_dt=start_dt + timedelta(minutes=actual_min),
        status="completed",
        gps_trace=trace,
        optimal_route=OptimalRoute(
            distance_km=round(opt_km, 2), duration_min=round(opt_min, 2),
            points=[GeoPoint(lat=p[0], lon=p[1]) for p in optimal_pts],
        ),
        fare=fare,
        traffic_conditions=traffic,
    )

    # Chat — driver must inform BEFORE the deviation point (which is early
    # in the GPS trace), so the message timestamp is before start_dt.
    messages = []
    if informed:
        messages.append({"ts": ts(start_dt - timedelta(minutes=1)),
                          "sender": "driver",
                          "text": "There's heavy traffic on PIE, taking an alternate route via ECP."})
    else:
        messages.append({"ts": ts(start_dt + timedelta(minutes=5)),
                          "sender": "rider",
                          "text": "Are we going the right way? This doesn't look right."})
        messages.append({"ts": ts(start_dt + timedelta(minutes=6)),
                          "sender": "driver",
                          "text": "Don't worry, I know the way."})
    messages.append({"ts": ts(start_dt + timedelta(minutes=3)),
                     "sender": "system", "text": "Trip in progress."})

    chat = ChatLog(trip_id=trip.trip_id,
                   messages=[ChatMessage(**m) for m in messages], calls=[])

    dispute = Dispute(
        dispute_id=f"disp_rd_{idx:03d}",
        trip_id=trip.trip_id,
        filed_by="rider",
        category="route_deviation",
        claim_text="The driver took a longer route and I was overcharged.",
        filed_at=ts(start_dt + timedelta(minutes=actual_min + 10)),
    )

    # Label from scenario intent
    if within_tolerance:
        label = {"outcome": "no_refund_within_tolerance", "amount": 0.0,
                  "applicable_clauses": ["§1.3"]}
    elif justified and informed:
        label = {"outcome": "no_refund_justified_informed", "amount": 0.0,
                  "applicable_clauses": ["§1.2", "§1.3", "§1.4"]}
    elif justified and not informed:
        label = {"outcome": "refund_partial_excess_distance", "amount": excess_fare,
                  "applicable_clauses": ["§1.2", "§1.3", "§1.4"]}
    else:
        label = {"outcome": "refund_full_excess_distance", "amount": excess_fare,
                  "applicable_clauses": ["§1.3", "§1.4"]}

    return trip, chat, dispute, label


# ---------------------------------------------------------------------------
# §2 No-Show scenarios (covering every §2 row)
# ---------------------------------------------------------------------------

def gen_no_show(
    rng: random.Random, idx: int, profiles: dict
) -> tuple[Trip, ChatLog, Dispute, dict]:
    """Generate a no-show case.

    Scenarios:
      0: rider no-show, both contacts met → fee upheld
      1: rider no-show, missing call → fee voided
      2: rider no-show, missing chat → fee voided
      3: rider no-show, repeat no-show (3+) → fee upheld + flag
      4: driver no-show, no contact → S$5 credit
      5: driver no-show, attempted contact → no credit
      6: rider no-show, both contacts met → fee upheld
      7: rider no-show, no contact → fee voided
    """
    scenario = idx % 8
    is_driver_no_show = scenario in (4, 5)
    has_chat = scenario not in (2, 7)  # missing chat
    has_call = scenario not in (1, 7)  # missing call
    use_repeat_ns = scenario == 3
    driver_no_contact = scenario == 4

    (plat, plon), (dlat, dlon), paddr, daddr = _pick_locations(rng)
    pickup = GeoPoint(lat=plat, lon=plon, address=paddr)
    dropoff = GeoPoint(lat=dlat, lon=dlon, address=daddr)

    requested_dt = BASE_TIME + timedelta(hours=idx)
    accepted_dt = requested_dt + timedelta(minutes=2)
    scheduled_dt = accepted_dt + timedelta(minutes=5)

    if is_driver_no_show:
        # Driver never arrived at pickup
        gps_pts = [(plat + 0.005, plon + 0.003)] * 3  # ~600m away
        arrived_dt = None
        ended_dt = scheduled_dt + timedelta(minutes=6)
    else:
        # Driver arrived, waited
        gps_pts = [(plat + 0.0003, plon + 0.0003)] * 4
        arrived_dt = accepted_dt + timedelta(minutes=4)
        ended_dt = arrived_dt + timedelta(minutes=8)

    trace = [GPSPoint(
        ts=ts(scheduled_dt + timedelta(minutes=i * 2)),
        lat=p[0], lon=p[1], speed_kmh=0,
    ) for i, p in enumerate(gps_pts)]

    rider_id = "rider_repeat_ns" if use_repeat_ns else f"rider_{rng.randint(1, 29):03d}"

    fare = make_fare(0, 0, surge=1.0, total_charged=CANCELLATION_FEE)

    trip = _make_trip(
        trip_id=f"trip_ns_{idx:03d}",
        rider_id=rider_id,
        driver_id=f"driver_{rng.randint(1, 29):03d}",
        pickup=pickup, dropoff=dropoff,
        requested_dt=requested_dt, accepted_dt=accepted_dt,
        arrived_dt=arrived_dt,
        started_dt=None,
        ended_dt=ended_dt,
        status="no_show_driver" if is_driver_no_show else "no_show_rider",
        gps_trace=trace,
        optimal_route=OptimalRoute(distance_km=0, duration_min=0,
                                    points=[pickup, dropoff]),
        fare=fare,
        scheduled_pickup_at=ts(scheduled_dt),
    )

    # Chat
    messages = [{"ts": ts(scheduled_dt), "sender": "system",
                 "text": "Driver status update."}]
    if has_chat and not is_driver_no_show:
        messages.append({"ts": ts(scheduled_dt + timedelta(minutes=1)),
                         "sender": "driver", "text": "I'm at the pickup point."})
        messages.append({"ts": ts(scheduled_dt + timedelta(minutes=3)),
                         "sender": "driver", "text": "Please come out or I'll have to cancel."})
    if is_driver_no_show and not driver_no_contact:
        messages.append({"ts": ts(scheduled_dt + timedelta(minutes=1)),
                         "sender": "driver", "text": "I'm running late, traffic is bad."})

    calls = []
    if has_call and not is_driver_no_show:
        calls.append({"ts": ts(scheduled_dt + timedelta(minutes=2)),
                      "sender": "driver", "recipient": "rider",
                      "duration_s": 15, "answered": False})

    chat = ChatLog(trip_id=trip.trip_id,
                   messages=[ChatMessage(**m) for m in messages],
                   calls=[CallAttempt(**c) for c in calls])

    category = "no_show_driver" if is_driver_no_show else "no_show_rider"
    dispute = Dispute(
        dispute_id=f"disp_ns_{idx:03d}",
        trip_id=trip.trip_id,
        filed_by="rider",
        category=category,
        claim_text="I was charged a cancellation fee unfairly.",
        filed_at=ts(ended_dt + timedelta(minutes=10)),
    )

    # Label from scenario intent
    if is_driver_no_show:
        if driver_no_contact:
            label = {"outcome": "driver_no_show_credit", "amount": 5.0,
                      "applicable_clauses": ["§2.4"]}
        else:
            label = {"outcome": "driver_no_show_no_credit", "amount": 0.0,
                      "applicable_clauses": ["§2.4"]}
    elif has_chat and has_call:
        label = {"outcome": "cancellation_fee_upheld", "amount": 5.0,
                  "applicable_clauses": ["§2.1", "§2.2", "§2.3"]}
    else:
        label = {"outcome": "cancellation_fee_voided", "amount": 0.0,
                  "applicable_clauses": ["§2.3"]}

    return trip, chat, dispute, label


# ---------------------------------------------------------------------------
# §3 Surge scenarios (covering every §3.3 row)
# ---------------------------------------------------------------------------

def gen_surge(
    rng: random.Random, idx: int, profiles: dict
) -> tuple[Trip, ChatLog, Dispute, dict]:
    """Generate a surge pricing case.

    Scenarios:
      0: surge increased, NOT notified → full refund
      1: surge increased, notified → no refund
      2: surge unchanged → no refund
      3: surge decreased → lower fare applies
      4: surge increased, NOT notified → full refund
      5: surge increased, notified → no refund
    """
    scenario = idx % 6
    increased = scenario in (0, 1, 4, 5)
    notified = scenario in (1, 5)
    decreased = scenario == 3

    (plat, plon), (dlat, dlon), paddr, daddr = _pick_locations(rng)
    pickup = GeoPoint(lat=plat, lon=plon, address=paddr)
    dropoff = GeoPoint(lat=dlat, lon=dlon, address=daddr)

    requested_dt = BASE_TIME + timedelta(hours=idx)
    accepted_dt = requested_dt + timedelta(minutes=2)
    arrived_dt = accepted_dt + timedelta(minutes=4)
    start_dt = arrived_dt + timedelta(minutes=1)

    route_pts = interpolate_route((plat, plon), (dlat, dlon), n_points=10)
    opt_km = route_distance_km(route_pts)
    opt_min = opt_km / 40 * 60
    trace = make_gps_trace(route_pts, start_dt, interval_s=15, speed_kmh=40)
    actual_km = trace_distance_km(trace)
    actual_min = len(trace) * 15 / 60.0

    quote_surge = 1.0
    trip_surge = 1.0
    if increased:
        trip_surge = 1.5
    elif decreased:
        trip_surge = 0.8

    surgeable = round(BASE_FARE + actual_km * PER_KM + actual_min * PER_MIN, 2)
    differential = round(surgeable * (trip_surge - quote_surge), 2) if increased else 0.0
    total = round(surgeable * trip_surge + BOOKING_FEE, 2)
    upfront = round(surgeable * quote_surge + BOOKING_FEE, 2)

    fare = make_fare(
        actual_km, actual_min, surge=trip_surge,
        total_charged=total,
        quote_surge=quote_surge, trip_surge=trip_surge,
        surgeable_subtotal=surgeable,
        upfront_quote=upfront,
        rider_notified_surge=notified,
    )

    trip = _make_trip(
        trip_id=f"trip_sg_{idx:03d}",
        rider_id=f"rider_{rng.randint(1, 29):03d}",
        driver_id=f"driver_{rng.randint(1, 29):03d}",
        pickup=pickup, dropoff=dropoff,
        requested_dt=requested_dt, accepted_dt=accepted_dt,
        arrived_dt=arrived_dt, started_dt=start_dt,
        ended_dt=start_dt + timedelta(minutes=actual_min),
        status="completed",
        gps_trace=trace,
        optimal_route=OptimalRoute(
            distance_km=round(opt_km, 2), duration_min=round(opt_min, 2),
            points=[GeoPoint(lat=p[0], lon=p[1]) for p in route_pts],
        ),
        fare=fare,
    )

    chat = ChatLog(trip_id=trip.trip_id, messages=[
        ChatMessage(ts=ts(start_dt), sender="system", text="Trip started."),
        ChatMessage(ts=ts(start_dt + timedelta(minutes=actual_min)),
                     sender="system", text="Trip completed."),
    ], calls=[])

    dispute = Dispute(
        dispute_id=f"disp_sg_{idx:03d}",
        trip_id=trip.trip_id,
        filed_by="rider",
        category="fare_error",
        claim_text="I was charged more than the upfront quote.",
        filed_at=ts(start_dt + timedelta(minutes=actual_min + 10)),
    )

    # Label from scenario intent
    if increased and not notified:
        label = {"outcome": "refund_surge_differential", "amount": differential,
                  "applicable_clauses": ["§3.2", "§3.3", "§3.4", "§3.5"]}
    elif increased and notified:
        label = {"outcome": "no_surge_refund_notified", "amount": 0.0,
                  "applicable_clauses": ["§3.1", "§3.2", "§3.3"]}
    elif decreased:
        label = {"outcome": "no_surge_refund", "amount": 0.0,
                  "applicable_clauses": ["§3.1", "§3.3"]}
    else:
        label = {"outcome": "no_surge_refund", "amount": 0.0,
                  "applicable_clauses": ["§3.1", "§3.3"]}

    return trip, chat, dispute, label


# ---------------------------------------------------------------------------
# §4 Property Damage scenarios (covering every §4 row)
# ---------------------------------------------------------------------------

def gen_property_damage(
    rng: random.Random, idx: int, profiles: dict
) -> tuple[Trip, ChatLog, Dispute, dict]:
    """Generate a property damage case.

    Scenarios:
      0: verifiable cleaning ≤S$50 → approved
      1: not verifiable (external photo) → rejected
      2: repair claim → human review
      3: rider disputes damage charge → human review
      4: verifiable cleaning ≤S$50 → approved
      5: not verifiable (no timestamp) → rejected
    """
    scenario = idx % 6
    verifiable = scenario in (0, 4)
    repair = scenario == 2
    rider_dispute = scenario == 3
    no_timestamp = scenario == 5

    (plat, plon), (dlat, dlon), paddr, daddr = _pick_locations(rng)
    pickup = GeoPoint(lat=plat, lon=plon, address=paddr)
    dropoff = GeoPoint(lat=dlat, lon=dlon, address=daddr)

    requested_dt = BASE_TIME + timedelta(hours=idx)
    accepted_dt = requested_dt + timedelta(minutes=2)
    arrived_dt = accepted_dt + timedelta(minutes=4)
    start_dt = arrived_dt + timedelta(minutes=1)

    route_pts = interpolate_route((plat, plon), (dlat, dlon), n_points=8)
    opt_km = route_distance_km(route_pts)
    opt_min = opt_km / 40 * 60
    trace = make_gps_trace(route_pts, start_dt, interval_s=15, speed_kmh=40)
    actual_km = trace_distance_km(trace)
    actual_min = len(trace) * 15 / 60.0

    claim_amount = rng.uniform(20, 50) if not repair else rng.uniform(80, 200)

    damage = DamageEvidence(
        photo_source="in_app_camera" if verifiable else "external_upload",
        has_timestamp=not no_timestamp,
        submitted_within_24h=verifiable or (not no_timestamp and scenario != 1),
        linked_to_trip=verifiable or scenario != 1,
        damage_type="repair" if repair else "cleaning",
        claim_amount=round(claim_amount, 2),
        has_certified_quote=repair,
        quote_within_7_days=repair,
    )

    fare = make_fare(actual_km, actual_min, surge=1.0)

    trip = _make_trip(
        trip_id=f"trip_pd_{idx:03d}",
        rider_id=f"rider_{rng.randint(1, 29):03d}",
        driver_id=f"driver_{rng.randint(1, 29):03d}",
        pickup=pickup, dropoff=dropoff,
        requested_dt=requested_dt, accepted_dt=accepted_dt,
        arrived_dt=arrived_dt, started_dt=start_dt,
        ended_dt=start_dt + timedelta(minutes=actual_min),
        status="completed",
        gps_trace=trace,
        optimal_route=OptimalRoute(
            distance_km=round(opt_km, 2), duration_min=round(opt_min, 2),
            points=[GeoPoint(lat=p[0], lon=p[1]) for p in route_pts],
        ),
        fare=fare,
        damage_evidence=damage,
    )

    chat = ChatLog(trip_id=trip.trip_id, messages=[
        ChatMessage(ts=ts(start_dt), sender="system", text="Trip started."),
        ChatMessage(ts=ts(start_dt + timedelta(minutes=actual_min)),
                     sender="system", text="Trip completed."),
    ], calls=[])

    dispute = Dispute(
        dispute_id=f"disp_pd_{idx:03d}",
        trip_id=trip.trip_id,
        filed_by="driver" if not rider_dispute else "rider",
        category="property_damage",
        claim_text="The rider caused damage to the vehicle." if not rider_dispute
                   else "I dispute the damage charge filed by the driver.",
        filed_at=ts(start_dt + timedelta(minutes=actual_min + 10)),
    )

    # Label from scenario intent
    if repair:
        label = {"outcome": "human_review_repair", "amount": 0.0,
                  "applicable_clauses": ["§4.3"]}
    elif rider_dispute:
        label = {"outcome": "human_review_rider_dispute", "amount": 0.0,
                  "applicable_clauses": ["§4.4"]}
    elif verifiable:
        label = {"outcome": "damage_claim_approved", "amount": round(claim_amount, 2),
                  "applicable_clauses": ["§4.2", "§4.3"]}
    else:
        label = {"outcome": "claim_rejected_insufficient_evidence", "amount": 0.0,
                  "applicable_clauses": ["§4.2", "§4.4"]}

    return trip, chat, dispute, label


# ---------------------------------------------------------------------------
# §5 Safety scenarios
# ---------------------------------------------------------------------------

def gen_safety(
    rng: random.Random, idx: int, profiles: dict
) -> tuple[Trip, ChatLog, Dispute, dict]:
    (plat, plon), (dlat, dlon), paddr, daddr = _pick_locations(rng)
    pickup = GeoPoint(lat=plat, lon=plon, address=paddr)
    dropoff = GeoPoint(lat=dlat, lon=dlon, address=daddr)

    requested_dt = BASE_TIME + timedelta(hours=idx)
    accepted_dt = requested_dt + timedelta(minutes=2)
    arrived_dt = accepted_dt + timedelta(minutes=4)
    start_dt = arrived_dt + timedelta(minutes=1)

    route_pts = interpolate_route((plat, plon), (dlat, dlon), n_points=8)
    opt_km = route_distance_km(route_pts)
    opt_min = opt_km / 40 * 60
    trace = make_gps_trace(route_pts, start_dt, interval_s=15, speed_kmh=40)
    actual_km = trace_distance_km(trace)
    actual_min = len(trace) * 15 / 60.0

    fare = make_fare(actual_km, actual_min, surge=1.0)

    trip = _make_trip(
        trip_id=f"trip_sf_{idx:03d}",
        rider_id=f"rider_{rng.randint(1, 29):03d}",
        driver_id=f"driver_{rng.randint(1, 29):03d}",
        pickup=pickup, dropoff=dropoff,
        requested_dt=requested_dt, accepted_dt=accepted_dt,
        arrived_dt=arrived_dt, started_dt=start_dt,
        ended_dt=start_dt + timedelta(minutes=actual_min),
        status="completed",
        gps_trace=trace,
        optimal_route=OptimalRoute(
            distance_km=round(opt_km, 2), duration_min=round(opt_min, 2),
            points=[GeoPoint(lat=p[0], lon=p[1]) for p in route_pts],
        ),
        fare=fare,
    )

    chat = ChatLog(trip_id=trip.trip_id, messages=[
        ChatMessage(ts=ts(start_dt + timedelta(minutes=2)),
                     sender="rider", text="I feel unsafe, the driver is shouting at me."),
        ChatMessage(ts=ts(start_dt + timedelta(minutes=3)),
                     sender="driver", text="Get out of my car if you don't like it!"),
        ChatMessage(ts=ts(start_dt + timedelta(minutes=4)),
                     sender="rider", text="I'm afraid, please let me out safely."),
    ], calls=[])

    dispute = Dispute(
        dispute_id=f"disp_sf_{idx:03d}",
        trip_id=trip.trip_id,
        filed_by="rider",
        category="safety",
        claim_text="I felt threatened and unsafe during the trip.",
        filed_at=ts(start_dt + timedelta(minutes=actual_min + 5)),
    )

    label = {"outcome": "escalate_safety", "amount": 0.0,
              "applicable_clauses": ["§5.1", "§5.2"]}
    return trip, chat, dispute, label


# ---------------------------------------------------------------------------
# Fraud-flagged rider case (§6.3: raises confidence threshold to 95%)
# ---------------------------------------------------------------------------

def gen_fraud_case(
    rng: random.Random, idx: int, profiles: dict
) -> tuple[Trip, ChatLog, Dispute, dict]:
    """Generate a surge case filed by rider_fraud1 (fraud_flags=1).

    The fraud flag raises the confidence threshold to 95% (§6.3), so
    even a low-value case routes to human review.
    """
    (plat, plon), (dlat, dlon), paddr, daddr = _pick_locations(rng)
    pickup = GeoPoint(lat=plat, lon=plon, address=paddr)
    dropoff = GeoPoint(lat=dlat, lon=dlon, address=daddr)

    requested_dt = BASE_TIME + timedelta(hours=idx)
    accepted_dt = requested_dt + timedelta(minutes=2)
    arrived_dt = accepted_dt + timedelta(minutes=4)
    start_dt = arrived_dt + timedelta(minutes=1)

    route_pts = interpolate_route((plat, plon), (dlat, dlon), n_points=10)
    opt_km = route_distance_km(route_pts)
    opt_min = opt_km / 40 * 60
    trace = make_gps_trace(route_pts, start_dt, interval_s=15, speed_kmh=40)
    actual_km = trace_distance_km(trace)
    actual_min = len(trace) * 15 / 60.0

    # Surge increased, not notified → refund, but fraud flag → human routing
    quote_surge = 1.0
    trip_surge = 1.5
    surgeable = round(BASE_FARE + actual_km * PER_KM + actual_min * PER_MIN, 2)
    differential = round(surgeable * (trip_surge - quote_surge), 2)
    total = round(surgeable * trip_surge + BOOKING_FEE, 2)
    upfront = round(surgeable * quote_surge + BOOKING_FEE, 2)

    fare = make_fare(
        actual_km, actual_min, surge=trip_surge,
        total_charged=total,
        quote_surge=quote_surge, trip_surge=trip_surge,
        surgeable_subtotal=surgeable,
        upfront_quote=upfront,
        rider_notified_surge=False,
    )

    trip = _make_trip(
        trip_id=f"trip_fa_{idx:03d}",
        rider_id="rider_fraud1",
        driver_id=f"driver_{rng.randint(1, 29):03d}",
        pickup=pickup, dropoff=dropoff,
        requested_dt=requested_dt, accepted_dt=accepted_dt,
        arrived_dt=arrived_dt, started_dt=start_dt,
        ended_dt=start_dt + timedelta(minutes=actual_min),
        status="completed",
        gps_trace=trace,
        optimal_route=OptimalRoute(
            distance_km=round(opt_km, 2), duration_min=round(opt_min, 2),
            points=[GeoPoint(lat=p[0], lon=p[1]) for p in route_pts],
        ),
        fare=fare,
    )

    chat = ChatLog(trip_id=trip.trip_id, messages=[
        ChatMessage(ts=ts(start_dt), sender="system", text="Trip started."),
        ChatMessage(ts=ts(start_dt + timedelta(minutes=actual_min)),
                     sender="system", text="Trip completed."),
    ], calls=[])

    dispute = Dispute(
        dispute_id=f"disp_fa_{idx:03d}",
        trip_id=trip.trip_id,
        filed_by="rider",
        category="fare_error",
        claim_text="I was charged more than the upfront quote.",
        filed_at=ts(start_dt + timedelta(minutes=actual_min + 10)),
    )

    # Label: surge refund due, but routes to human because of fraud flag
    label = {"outcome": "refund_surge_differential", "amount": differential,
              "applicable_clauses": ["§3.2", "§3.3", "§3.4", "§3.5"]}
    return trip, chat, dispute, label


# ---------------------------------------------------------------------------
# Precedents (v2.1)
# ---------------------------------------------------------------------------

def gen_precedents() -> list[Precedent]:
    data = [
        ("DISP-2026-0089", "route_deviation",
         "Route deviation without notification: driver had traffic justification "
         "but failed to notify rider. Partial refund of excess distance fare. "
         "4.0 km excess at S$0.80/km = S$3.20.",
         "refund_partial_excess_distance", 3.20, "ryde"),
        ("DISP-2026-0142", "fare_error",
         "Surge pricing without rider notification: surge increased from 1.0x "
         "to 1.5x, rider not notified. Full refund of surge differential. "
         "S$6.40 × 0.5 = S$3.20.",
         "refund_surge_differential", 3.20, "ryde"),
        ("DISP-2026-0211", "no_show_rider",
         "No-show cancellation without contact: driver cancelled without chat "
         "or call. Cancellation fee voided. S$5.00.",
         "cancellation_fee_voided", 5.00, "ryde"),
        ("PC-2025-001", "route_deviation",
         "Driver took 3 km detour without justification. Full refund S$2.40.",
         "refund_full_excess_distance", 2.40, "seed"),
        ("PC-2025-002", "route_deviation",
         "Rider requested stop. No refund.",
         "no_refund_rider_requested", 0.0, "seed"),
        ("PC-2025-003", "route_deviation",
         "Traffic-justified, rider informed. No refund.",
         "no_refund_justified_informed", 0.0, "seed"),
        ("PC-2025-004", "no_show_driver",
         "Driver no-show, 0 contact. S$5 credit.",
         "driver_no_show_credit", 5.00, "seed"),
        ("PC-2025-006", "no_show_rider",
         "Both contacts met. Fee S$5 upheld.",
         "cancellation_fee_upheld", 5.00, "seed"),
        ("PC-2025-008", "fare_error",
         "Surge increased, rider notified. No refund.",
         "no_surge_refund_notified", 0.0, "seed"),
        ("PC-2025-010", "property_damage",
         "Verifiable cleaning claim S$35. Approved.",
         "damage_claim_approved", 35.00, "seed"),
        ("PC-2025-012", "safety",
         "Safety incident. Escalated to human.",
         "escalate_safety", 0.0, "seed"),
    ]
    return [Precedent(case_id=c, category=cat, summary=s, outcome=o,
                      amount=a, source=src)
            for c, cat, s, o, a, src in data]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    profiles = make_profiles(rng)

    trips: list[Trip] = []
    chats: list[ChatLog] = []
    disputes: list[Dispute] = []
    labels: list[dict] = []

    # §1 Route deviation: 8 scenarios
    for i in range(8):
        t, c, d, lbl = gen_route_deviation(rng, i, profiles)
        trips.append(t); chats.append(c); disputes.append(d)
        labels.append({"dispute_id": d.dispute_id, **lbl})

    # §2 No-show: 8 scenarios
    for i in range(8):
        t, c, d, lbl = gen_no_show(rng, i, profiles)
        trips.append(t); chats.append(c); disputes.append(d)
        labels.append({"dispute_id": d.dispute_id, **lbl})

    # §3 Surge: 6 scenarios
    for i in range(6):
        t, c, d, lbl = gen_surge(rng, i, profiles)
        trips.append(t); chats.append(c); disputes.append(d)
        labels.append({"dispute_id": d.dispute_id, **lbl})

    # §4 Property damage: 6 scenarios
    for i in range(6):
        t, c, d, lbl = gen_property_damage(rng, i, profiles)
        trips.append(t); chats.append(c); disputes.append(d)
        labels.append({"dispute_id": d.dispute_id, **lbl})

    # §5 Safety: 4 scenarios
    for i in range(4):
        t, c, d, lbl = gen_safety(rng, i, profiles)
        trips.append(t); chats.append(c); disputes.append(d)
        labels.append({"dispute_id": d.dispute_id, **lbl})

    # Extra route deviation for §6 routing (>2km → human)
    for i in range(8, 14):
        t, c, d, lbl = gen_route_deviation(rng, i, profiles)
        trips.append(t); chats.append(c); disputes.append(d)
        labels.append({"dispute_id": d.dispute_id, **lbl})

    # Extra no-show for variety
    for i in range(8, 12):
        t, c, d, lbl = gen_no_show(rng, i, profiles)
        trips.append(t); chats.append(c); disputes.append(d)
        labels.append({"dispute_id": d.dispute_id, **lbl})

    # Extra surge for variety
    for i in range(6, 10):
        t, c, d, lbl = gen_surge(rng, i, profiles)
        trips.append(t); chats.append(c); disputes.append(d)
        labels.append({"dispute_id": d.dispute_id, **lbl})

    # Extra property damage
    for i in range(6, 10):
        t, c, d, lbl = gen_property_damage(rng, i, profiles)
        trips.append(t); chats.append(c); disputes.append(d)
        labels.append({"dispute_id": d.dispute_id, **lbl})

    # Fraud-flagged rider case (§6.3)
    t, c, d, lbl = gen_fraud_case(rng, 0, profiles)
    trips.append(t); chats.append(c); disputes.append(d)
    labels.append({"dispute_id": d.dispute_id, **lbl})

    # Precedents
    precedents = gen_precedents()

    # Write output
    out_dir = Path(__file__).resolve().parent.parent / "data" / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, data):
        with open(out_dir / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    write("trips.json", [t.model_dump() for t in trips])
    write("chats.json", [c.model_dump() for c in chats])
    write("profiles.json", {k: v.model_dump() for k, v in profiles.items()})
    write("disputes.json", [d.model_dump() for d in disputes])
    write("precedents.json", [p.model_dump() for p in precedents])
    write("labels.json", labels)

    print(f"Generated {len(trips)} trips, {len(chats)} chats, "
          f"{len(disputes)} disputes, {len(precedents)} precedents, "
          f"{len(labels)} labels in {out_dir}")

    from collections import Counter
    cats = Counter(d.category for d in disputes)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
