"""Generate the synthetic RideCourt dataset.

Usage::

    .venv/bin/python scripts/generate_data.py --seed 7

Writes ``data/synthetic/`` with trips.json, chats.json, profiles.json,
disputes.json, precedents.json, and labels.json.

Labels are computed with the policy engine so agents never see them.
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
    ChatLog,
    Dispute,
    Fare,
    GPSPoint,
    GeoPoint,
    OptimalRoute,
    Precedent,
    Profile,
    Trip,
)
from ridecourt.policy import evaluate
from ridecourt.tools import compute_facts

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
    """Straight-ish polyline from pickup to dropoff."""
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
    """Route with a detour loop — longer than the straight path."""
    mid_lat = (pickup[0] + dropoff[0]) / 2
    mid_lon = (pickup[1] + dropoff[1]) / 2
    # Offset midpoint perpendicular to create a loop
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
        # Add a bump in the middle
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
    """Generate GPS points every 15 seconds along the route."""
    trace = []
    dt_cur = start_dt
    for i, (lat, lon) in enumerate(route_pts):
        # Vary speed slightly
        s = speed_kmh + random.uniform(-8, 8)
        trace.append(GPSPoint(ts=ts(dt_cur), lat=lat, lon=lon, speed_kmh=max(s, 5)))
        # Advance time by interval
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


# ---------------------------------------------------------------------------
# Fare parameters (Singapore-like)
# ---------------------------------------------------------------------------

BASE_FARE = 3.0
PER_KM = 0.75
PER_MIN = 0.25
BOOKING_FEE = 1.0
CANCELLATION_FEE = 4.0


def make_fare(
    distance_km: float,
    duration_min: float,
    surge: float = 1.0,
    promo_code: str = "",
    discount: float = 0.0,
    total_charged: float | None = None,
    cancellation_fee: float = CANCELLATION_FEE,
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
    )


# ---------------------------------------------------------------------------
# Profile generation
# ---------------------------------------------------------------------------

def make_profiles(rng: random.Random) -> dict[str, Profile]:
    profiles = {}

    # Bad-faith rider: 6 upheld fare disputes in 90 days
    profiles["rider_bf1"] = Profile(
        account_age_days=400,
        rating=3.8,
        trips_completed=52,
        disputes_filed_90d=8,
        disputes_upheld_90d=6,
        cancellations_30d=4,
        prior_flags=["excessive_disputes"],
    )

    # Bad-faith driver: claims no-show but GPS shows 600m away
    profiles["driver_bf1"] = Profile(
        account_age_days=200,
        rating=3.5,
        trips_completed=180,
        disputes_filed_90d=2,
        disputes_upheld_90d=1,
        cancellations_30d=12,
        prior_flags=["gps_anomaly", "frequent_cancellation"],
    )

    # Normal riders
    for i in range(1, 30):
        profiles[f"rider_{i:03d}"] = Profile(
            account_age_days=rng.randint(30, 800),
            rating=round(rng.uniform(4.0, 5.0), 1),
            trips_completed=rng.randint(5, 300),
            disputes_filed_90d=rng.randint(0, 3),
            disputes_upheld_90d=rng.randint(0, 2),
            cancellations_30d=rng.randint(0, 3),
        )

    # Normal drivers
    for i in range(1, 30):
        profiles[f"driver_{i:03d}"] = Profile(
            account_age_days=rng.randint(60, 1200),
            rating=round(rng.uniform(4.2, 5.0), 1),
            trips_completed=rng.randint(50, 2000),
            disputes_filed_90d=rng.randint(0, 2),
            disputes_upheld_90d=rng.randint(0, 1),
            cancellations_30d=rng.randint(0, 5),
        )

    return profiles


# ---------------------------------------------------------------------------
# Trip generation per dispute category
# ---------------------------------------------------------------------------

def _pick_locations(rng: random.Random) -> tuple[tuple, tuple, str, str]:
    src = rng.choice(SG_LOCATIONS)
    dst = rng.choice([l for l in SG_LOCATIONS if l[0] != src[0]])
    return (
        (src[1], src[2]),
        (dst[1], dst[2]),
        src[0],
        dst[0],
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
        status=status,
        gps_trace=gps_trace,
        optimal_route=optimal_route,
        fare=fare,
    )


# ---- Route deviation (16: 8 valid, 8 invalid) ----

def gen_route_deviation(
    rng: random.Random, idx: int, profiles: dict
) -> tuple[Trip, ChatLog, Dispute]:
    valid = idx < 8
    requested_stop = (not valid) and rng.random() < 0.5
    road_closure_note = (not valid) and not requested_stop

    (plat, plon), (dlat, dlon), paddr, daddr = _pick_locations(rng)
    pickup = GeoPoint(lat=plat, lon=plon, address=paddr)
    dropoff = GeoPoint(lat=dlat, lon=dlon, address=daddr)

    requested_dt = BASE_TIME + timedelta(hours=idx)
    accepted_dt = requested_dt + timedelta(minutes=2)

    if valid:
        # Deviated route
        route_pts = detour_route((plat, plon), (dlat, dlon), n_points=14, detour_factor=1.5)
    else:
        route_pts = interpolate_route((plat, plon), (dlat, dlon), n_points=10)

    optimal_pts = interpolate_route((plat, plon), (dlat, dlon), n_points=8)
    opt_km = route_distance_km(optimal_pts)
    opt_min = opt_km / 40 * 60  # ~40 km/h

    start_dt = accepted_dt + timedelta(minutes=5)
    trace = make_gps_trace(route_pts, start_dt, interval_s=15, speed_kmh=40)
    actual_km = trace_distance_km(trace)
    # Duration must be consistent with distance and speed so a detour (longer
    # distance) always takes longer than the optimal route.  Using the GPS
    # point count would under-count because the trace is sparse.
    actual_min = actual_km / 40 * 60  # same speed basis as optimal

    surge = rng.choice([1.0, 1.2, 1.5, 1.8])
    fare = make_fare(actual_km, actual_min, surge=surge)

    trip = _make_trip(
        trip_id=f"trip_rd_{idx:03d}",
        rider_id=f"rider_{rng.randint(1, 29):03d}",
        driver_id=f"driver_{rng.randint(1, 29):03d}",
        pickup=pickup,
        dropoff=dropoff,
        requested_dt=requested_dt,
        accepted_dt=accepted_dt,
        arrived_dt=accepted_dt + timedelta(minutes=4),
        started_dt=start_dt,
        ended_dt=start_dt + timedelta(minutes=actual_min),
        status="completed",
        gps_trace=trace,
        optimal_route=OptimalRoute(distance_km=round(opt_km, 2), duration_min=round(opt_min, 2), points=[GeoPoint(lat=p[0], lon=p[1]) for p in optimal_pts]),
        fare=fare,
    )

    # Chat
    messages = []
    if requested_stop:
        messages.append({
            "ts": ts(start_dt + timedelta(minutes=2)),
            "sender": "rider",
            "text": "Can we stop by a convenience store on the way?",
        })
    if road_closure_note:
        messages.append({
            "ts": ts(start_dt + timedelta(minutes=1)),
            "sender": "driver",
            "text": "There's a road closure on the usual route, taking a detour.",
        })
    messages.append({
        "ts": ts(start_dt + timedelta(minutes=5)),
        "sender": "system",
        "text": "Trip in progress.",
    })

    chat = ChatLog(
        trip_id=trip.trip_id,
        messages=[_ for _ in messages],
        calls=[],
    )

    dispute = Dispute(
        dispute_id=f"disp_rd_{idx:03d}",
        trip_id=trip.trip_id,
        filed_by="rider",
        category="route_deviation",
        claim_text="The driver took a longer route and I was overcharged.",
        filed_at=ts(start_dt + timedelta(minutes=actual_min + 10)),
    )
    return trip, chat, dispute


# ---- No-show by driver (10) ----

def gen_no_show_driver(
    rng: random.Random, idx: int, profiles: dict
) -> tuple[Trip, ChatLog, Dispute]:
    far_away = rng.random() < 0.6  # 60% driver never came within 150m
    use_bad_faith_driver = idx == 0  # one bad-faith driver case

    (plat, plon), (dlat, dlon), paddr, daddr = _pick_locations(rng)
    pickup = GeoPoint(lat=plat, lon=plon, address=paddr)
    dropoff = GeoPoint(lat=dlat, lon=dlon, address=daddr)

    requested_dt = BASE_TIME + timedelta(hours=idx)
    accepted_dt = requested_dt + timedelta(minutes=2)
    eta = accepted_dt + timedelta(minutes=5)

    if use_bad_faith_driver:
        # Driver GPS shows arrival 600m away
        far_away = True
        driver_id = "driver_bf1"
        # GPS trace near pickup but 600m off
        offset_lat = 0.005  # ~550m
        offset_lon = 0.003
        gps_pts = [(plat + offset_lat, plon + offset_lon)] * 3
    elif far_away:
        # Driver somewhere far
        offset = 0.003  # ~330m+
        gps_pts = [(plat + offset, plon + offset)] * 3
    else:
        # Driver close but waited > 5 min past ETA
        gps_pts = [(plat + 0.0005, plon + 0.0005)] * 3
        # Set wait time past ETA
        pass

    trace = [
        GPSPoint(ts=ts(eta + timedelta(minutes=i * 2)), lat=p[0], lon=p[1], speed_kmh=0)
        for i, p in enumerate(gps_pts)
    ]

    # For wait-based no-show: driver arrived but cancelled after rider waited
    if not far_away:
        ended_dt = eta + timedelta(minutes=7)  # > 5 min past ETA
        arrived_dt = None  # never marked arrived
    else:
        ended_dt = eta + timedelta(minutes=3)
        arrived_dt = None

    fare = make_fare(0, 0, surge=1.0, total_charged=CANCELLATION_FEE)

    trip = _make_trip(
        trip_id=f"trip_ns_d_{idx:03d}",
        rider_id=f"rider_{rng.randint(1, 29):03d}",
        driver_id=driver_id if use_bad_faith_driver else f"driver_{rng.randint(1, 29):03d}",
        pickup=pickup,
        dropoff=dropoff,
        requested_dt=requested_dt,
        accepted_dt=accepted_dt,
        arrived_dt=arrived_dt,
        started_dt=None,
        ended_dt=ended_dt,
        status="no_show_driver",
        gps_trace=trace,
        optimal_route=OptimalRoute(distance_km=0, duration_min=0, points=[pickup, dropoff]),
        fare=fare,
    )

    # Chat: rider tries to contact driver
    chat = ChatLog(
        trip_id=trip.trip_id,
        messages=[
            {"ts": ts(eta), "sender": "rider", "text": "Where are you? I've been waiting."},
            {"ts": ts(eta + timedelta(minutes=2)), "sender": "rider", "text": "Are you coming?"},
        ],
        calls=[
            {"ts": ts(eta), "sender": "rider", "recipient": "driver", "duration_s": 0, "answered": False},
        ],
    )

    dispute = Dispute(
        dispute_id=f"disp_ns_d_{idx:03d}",
        trip_id=trip.trip_id,
        filed_by="rider",
        category="no_show_driver",
        claim_text="The driver never showed up and I was charged a cancellation fee.",
        filed_at=ts(ended_dt + timedelta(minutes=10)),
    )
    return trip, chat, dispute


# ---- No-show by rider / cancellation fee (10) ----

def gen_no_show_rider(
    rng: random.Random, idx: int, profiles: dict
) -> tuple[Trip, ChatLog, Dispute]:
    valid = idx < 6  # 6 valid (fee stands), 4 invalid

    (plat, plon), (dlat, dlon), paddr, daddr = _pick_locations(rng)
    pickup = GeoPoint(lat=plat, lon=plon, address=paddr)
    dropoff = GeoPoint(lat=dlat, lon=dlon, address=daddr)

    requested_dt = BASE_TIME + timedelta(hours=idx)
    accepted_dt = requested_dt + timedelta(minutes=2)
    arrived_dt = accepted_dt + timedelta(minutes=4)
    eta = accepted_dt + timedelta(minutes=5)

    if valid:
        # Driver waited >= 5 min, within 150m, contacted rider
        ended_dt = eta + timedelta(minutes=3)
        gps_pts = [(plat + 0.0003, plon + 0.0003)] * 4
        contacts = 2
    else:
        # Invalid: driver didn't wait long enough or didn't contact
        ended_dt = arrived_dt + timedelta(minutes=2)  # only ~2 min
        gps_pts = [(plat + 0.0003, plon + 0.0003)] * 2
        contacts = 0  # no contact attempt

    trace = [
        GPSPoint(ts=ts(arrived_dt + timedelta(minutes=i * 2)), lat=p[0], lon=p[1], speed_kmh=0)
        for i, p in enumerate(gps_pts)
    ]

    fare = make_fare(0, 0, surge=1.0, total_charged=CANCELLATION_FEE)

    trip = _make_trip(
        trip_id=f"trip_ns_r_{idx:03d}",
        rider_id=f"rider_{rng.randint(1, 29):03d}",
        driver_id=f"driver_{rng.randint(1, 29):03d}",
        pickup=pickup,
        dropoff=dropoff,
        requested_dt=requested_dt,
        accepted_dt=accepted_dt,
        arrived_dt=arrived_dt,
        started_dt=None,
        ended_dt=ended_dt,
        status="no_show_rider",
        gps_trace=trace,
        optimal_route=OptimalRoute(distance_km=0, duration_min=0, points=[pickup, dropoff]),
        fare=fare,
    )

    messages = [
        {"ts": ts(arrived_dt), "sender": "system", "text": "Driver has arrived at the pickup point."},
    ]
    calls = []
    if valid:
        messages.append({"ts": ts(arrived_dt + timedelta(minutes=1)), "sender": "driver", "text": "I'm at the pickup point, where are you?"})
        messages.append({"ts": ts(arrived_dt + timedelta(minutes=3)), "sender": "driver", "text": "Please come out or I'll have to cancel."})
        calls.append({"ts": ts(arrived_dt + timedelta(minutes=2)), "sender": "driver", "recipient": "rider", "duration_s": 15, "answered": False})

    chat = ChatLog(trip_id=trip.trip_id, messages=messages, calls=calls)

    dispute = Dispute(
        dispute_id=f"disp_ns_r_{idx:03d}",
        trip_id=trip.trip_id,
        filed_by="rider",
        category="no_show_rider",
        claim_text="I was charged a cancellation fee but the driver didn't wait.",
        filed_at=ts(ended_dt + timedelta(minutes=10)),
    )
    return trip, chat, dispute


# ---- Fare error (6) ----

def gen_fare_error(
    rng: random.Random, idx: int, profiles: dict
) -> tuple[Trip, ChatLog, Dispute]:
    overcharged = idx < 3  # 3 overcharged, 3 correct/under
    bad_promo = idx == 4  # one with invalid promo
    use_bad_faith_rider = idx == 5  # bad-faith rider with 6 upheld disputes

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

    surge = rng.choice([1.0, 1.2, 1.5])

    if bad_promo:
        promo_code = "EXPIRED99"
        discount = 5.0  # discount was applied but promo is invalid
        recomputed_total = (BASE_FARE + actual_km * PER_KM + actual_min * PER_MIN) * surge + BOOKING_FEE
        # charged with the invalid discount
        charged_total = recomputed_total - discount
    elif overcharged:
        promo_code = ""
        discount = 0.0
        recomputed_total = (BASE_FARE + actual_km * PER_KM + actual_min * PER_MIN) * surge + BOOKING_FEE
        charged_total = recomputed_total + rng.uniform(1.0, 5.0)  # overcharge
    else:
        promo_code = "WELCOME10"
        discount = 2.0
        recomputed_total = (BASE_FARE + actual_km * PER_KM + actual_min * PER_MIN) * surge + BOOKING_FEE - discount
        charged_total = recomputed_total  # correct or within tolerance

    fare = make_fare(
        actual_km, actual_min, surge=surge,
        promo_code=promo_code, discount=discount,
        total_charged=round(charged_total, 2),
    )

    rider_id = "rider_bf1" if use_bad_faith_rider else f"rider_{rng.randint(1, 29):03d}"

    trip = _make_trip(
        trip_id=f"trip_fa_{idx:03d}",
        rider_id=rider_id,
        driver_id=f"driver_{rng.randint(1, 29):03d}",
        pickup=pickup,
        dropoff=dropoff,
        requested_dt=requested_dt,
        accepted_dt=accepted_dt,
        arrived_dt=arrived_dt,
        started_dt=start_dt,
        ended_dt=start_dt + timedelta(minutes=actual_min),
        status="completed",
        gps_trace=trace,
        optimal_route=OptimalRoute(distance_km=round(opt_km, 2), duration_min=round(opt_min, 2), points=[GeoPoint(lat=p[0], lon=p[1]) for p in route_pts]),
        fare=fare,
    )

    chat = ChatLog(
        trip_id=trip.trip_id,
        messages=[
            {"ts": ts(start_dt), "sender": "system", "text": "Trip started."},
            {"ts": ts(start_dt + timedelta(minutes=actual_min)), "sender": "system", "text": "Trip completed."},
        ],
        calls=[],
    )

    dispute = Dispute(
        dispute_id=f"disp_fa_{idx:03d}",
        trip_id=trip.trip_id,
        filed_by="rider",
        category="fare_error",
        claim_text="I think the fare I was charged is incorrect.",
        filed_at=ts(start_dt + timedelta(minutes=actual_min + 10)),
    )
    return trip, chat, dispute


# ---- Safety incident (6) ----

def gen_safety(
    rng: random.Random, idx: int, profiles: dict
) -> tuple[Trip, ChatLog, Dispute]:
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

    surge = 1.0
    fare = make_fare(actual_km, actual_min, surge=surge)

    trip = _make_trip(
        trip_id=f"trip_sf_{idx:03d}",
        rider_id=f"rider_{rng.randint(1, 29):03d}",
        driver_id=f"driver_{rng.randint(1, 29):03d}",
        pickup=pickup,
        dropoff=dropoff,
        requested_dt=requested_dt,
        accepted_dt=accepted_dt,
        arrived_dt=arrived_dt,
        started_dt=start_dt,
        ended_dt=start_dt + timedelta(minutes=actual_min),
        status="completed",
        gps_trace=trace,
        optimal_route=OptimalRoute(distance_km=round(opt_km, 2), duration_min=round(opt_min, 2), points=[GeoPoint(lat=p[0], lon=p[1]) for p in route_pts]),
        fare=fare,
    )

    # Safety chat: threatening language
    safety_messages = [
        {"ts": ts(start_dt + timedelta(minutes=2)), "sender": "rider", "text": "I feel unsafe, the driver is shouting at me."},
        {"ts": ts(start_dt + timedelta(minutes=3)), "sender": "driver", "text": "Get out of my car if you don't like it!"},
        {"ts": ts(start_dt + timedelta(minutes=4)), "sender": "rider", "text": "I'm afraid, please let me out safely."},
    ]

    chat = ChatLog(trip_id=trip.trip_id, messages=safety_messages, calls=[])

    dispute = Dispute(
        dispute_id=f"disp_sf_{idx:03d}",
        trip_id=trip.trip_id,
        filed_by="rider",
        category="safety",
        claim_text="I felt threatened and unsafe during the trip.",
        filed_at=ts(start_dt + timedelta(minutes=actual_min + 5)),
    )
    return trip, chat, dispute


# ---------------------------------------------------------------------------
# Precedents (12 past rulings)
# ---------------------------------------------------------------------------

def gen_precedents(rng: random.Random) -> list[Precedent]:
    data = [
        ("PC-2025-001", "route_deviation", "Driver took a 3 km detour without rider consent. Refund S$4.20.", "refund", 4.20),
        ("PC-2025-002", "route_deviation", "Rider asked for a stop en route. No refund — P-RD-2.", "no_refund", 0.0),
        ("PC-2025-003", "route_deviation", "Road closure on Bukit Timah, driver detoured. No refund.", "no_refund", 0.0),
        ("PC-2025-004", "no_show_driver", "Driver GPS 400m from pickup. Refund cancellation + S$3 credit.", "refund_and_credit", 7.0),
        ("PC-2025-005", "no_show_driver", "Driver arrived within 150m, waited 6 min. No refund.", "no_refund", 0.0),
        ("PC-2025-006", "no_show_rider", "Driver waited 7 min, called rider. Cancellation fee stands.", "cancellation_fee_stands", 4.0),
        ("PC-2025-007", "no_show_rider", "Driver waited only 2 min. Cancellation fee refunded.", "refund", 4.0),
        ("PC-2025-008", "fare_error", "Charged S$12.50 vs recomputed S$9.80. Refund S$2.70.", "refund_overcharge", 2.70),
        ("PC-2025-009", "fare_error", "Promo code expired at booking. Discount removed.", "refund_overcharge", 5.0),
        ("PC-2025-010", "fare_error", "Fare correct within S$0.50 tolerance. No refund.", "no_refund", 0.0),
        ("PC-2025-011", "safety", "Rider felt threatened. Escalated to human reviewer.", "escalate", 0.0),
        ("PC-2025-012", "route_deviation", "Deviation 12% under threshold. No refund.", "no_refund", 0.0),
    ]
    return [Precedent(case_id=c, category=cat, summary=s, outcome=o, amount=a) for c, cat, s, o, a in data]


# ---------------------------------------------------------------------------
# Labels computation
# ---------------------------------------------------------------------------

def compute_labels(
    trips: list[Trip],
    chats: list[ChatLog],
    disputes: list[Dispute],
    profiles: dict[str, Profile],
) -> list[dict]:
    """Compute labels using the policy engine — agents never see these."""
    trip_map = {t.trip_id: t for t in trips}
    chat_map = {c.trip_id: c for c in chats}
    labels = []
    for d in disputes:
        trip = trip_map[d.trip_id]
        chat = chat_map.get(d.trip_id)
        facts = compute_facts(trip, chat)
        result = evaluate(d, trip, profiles, facts, trip.fare)
        labels.append({
            "dispute_id": d.dispute_id,
            "outcome": result.outcome,
            "amount": result.amount,
            "applicable_clauses": result.applicable_clauses,
        })
    return labels


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

    # Route deviation: 16 (8 valid, 8 invalid)
    for i in range(16):
        t, c, d = gen_route_deviation(rng, i, profiles)
        trips.append(t)
        chats.append(c)
        disputes.append(d)

    # No-show by driver: 10
    for i in range(10):
        t, c, d = gen_no_show_driver(rng, i, profiles)
        trips.append(t)
        chats.append(c)
        disputes.append(d)

    # No-show by rider / cancellation fee: 10
    for i in range(10):
        t, c, d = gen_no_show_rider(rng, i, profiles)
        trips.append(t)
        chats.append(c)
        disputes.append(d)

    # Fare error: 6
    for i in range(6):
        t, c, d = gen_fare_error(rng, i, profiles)
        trips.append(t)
        chats.append(c)
        disputes.append(d)

    # Safety: 6
    for i in range(6):
        t, c, d = gen_safety(rng, i, profiles)
        trips.append(t)
        chats.append(c)
        disputes.append(d)

    # Precedents
    precedents = gen_precedents(rng)

    # Labels
    labels = compute_labels(trips, chats, disputes, profiles)

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

    # Category counts
    from collections import Counter
    cats = Counter(d.category for d in disputes)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
