"""Tests for the deterministic evidence tools.

Hand-built small trips so each scenario has a known expected outcome:
- a detour of known length
- a driver who stopped 600 m from pickup
- a chat with "can you stop at the 7-Eleven"
- a threatening message
- a promo not applied

Run with: .venv/bin/python -m pytest tests/test_tools.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

from ridecourt.data import (
    ChatLog,
    ChatMessage,
    Dispute,
    Fare,
    GeoPoint,
    GPSPoint,
    OptimalRoute,
    Profile,
    Trip,
)
from ridecourt.tools import (
    arrival_check,
    build_evidence_pack,
    chat_scan,
    compute_facts,
    fare_check,
    gps_analysis,
    history_profile,
    haversine_m,
    recompute_fare,
)
from ridecourt.policy import evaluate


# ---------------------------------------------------------------------------
# Helpers — build small, controlled fixtures
# ---------------------------------------------------------------------------

PICKUP = GeoPoint(lat=1.3509, lon=103.8478, address="Bishan MRT")
DROPOFF = GeoPoint(lat=1.3128, lon=103.8539, address="Lavender MRT")


def _fare(**kw) -> Fare:
    defaults = dict(
        base=3.0, per_km=0.75, per_min=0.25,
        distance_km=10.0, duration_min=20.0,
        surge_multiplier=1.0, promo_code="", discount=0.0,
        booking_fee=1.0, cancellation_fee=4.0, total_charged=14.5,
    )
    defaults.update(kw)
    return Fare(**defaults)


def _trip(gps_trace=None, fare=None, pickup=PICKUP, dropoff=DROPOFF, **kw) -> Trip:
    f = fare or _fare()
    defaults = dict(
        trip_id="trip_test",
        rider_id="rider_001",
        driver_id="driver_001",
        pickup=pickup,
        dropoff=dropoff,
        requested_at="2026-01-01T08:00:00",
        accepted_at="2026-01-01T08:02:00",
        driver_arrived_at="2026-01-01T08:06:00",
        trip_started_at="2026-01-01T08:07:00",
        trip_ended_at="2026-01-01T08:27:00",
        status="completed",
        gps_trace=gps_trace or [],
        optimal_route=OptimalRoute(distance_km=10.0, duration_min=20.0),
        fare=f,
    )
    defaults.update(kw)
    return Trip(**defaults)


def _straight_trace(
    start: GeoPoint,
    end: GeoPoint,
    n: int = 20,
    ts_start: str = "2026-01-01T08:07:00",
    speed_kmh: float = 40.0,
) -> list[GPSPoint]:
    """Build a straight-ish GPS trace from *start* to *end* with *n* points."""
    pts: list[GPSPoint] = []
    for i in range(n):
        frac = i / (n - 1)
        lat = start.lat + (end.lat - start.lat) * frac
        lon = start.lon + (end.lon - start.lon) * frac
        # timestamps 1 min apart
        from datetime import datetime, timedelta
        ts = (
            datetime.fromisoformat(ts_start) + timedelta(minutes=i)
        ).isoformat()
        pts.append(GPSPoint(ts=ts, lat=lat, lon=lon, speed_kmh=speed_kmh))
    return pts


def _detour_trace(
    start: GeoPoint,
    end: GeoPoint,
    detour_offset_lat: float = 0.05,  # ~5.5 km north detour
    n: int = 20,
) -> list[GPSPoint]:
    """Build a trace that goes straight, loops north (detour), then continues.
    The detour is large enough that actual distance > optimal + 15%."""
    from datetime import datetime, timedelta
    pts: list[GPSPoint] = []
    ts = datetime.fromisoformat("2026-01-01T08:07:00")
    # First quarter: straight to a midpoint
    mid1_lat = start.lat + (end.lat - start.lat) * 0.25
    mid1_lon = start.lon + (end.lon - start.lon) * 0.25
    # Detour point (offset north)
    detour_lat = mid1_lat + detour_offset_lat
    detour_lon = mid1_lon
    # Midpoint after detour
    mid2_lat = start.lat + (end.lat - start.lat) * 0.5
    mid2_lon = start.lon + (end.lon - start.lon) * 0.5

    waypoints = [start, GeoPoint(lat=mid1_lat, lon=mid1_lon),
                 GeoPoint(lat=detour_lat, lon=detour_lon),
                 GeoPoint(lat=mid2_lat, lon=mid2_lon), end]
    # interpolate 8 points between each waypoint pair (denser = more accurate distance)
    for wi in range(len(waypoints) - 1):
        a, b = waypoints[wi], waypoints[wi + 1]
        for j in range(8):
            frac = j / 8
            lat = a.lat + (b.lat - a.lat) * frac
            lon = a.lon + (b.lon - a.lon) * frac
            pts.append(GPSPoint(
                ts=ts.isoformat(), lat=lat, lon=lon, speed_kmh=40.0
            ))
            ts = ts + timedelta(minutes=1)
    return pts


def _chat(*messages: tuple[str, str, str]) -> ChatLog:
    """Build a ChatLog from (ts, sender, text) tuples."""
    return ChatLog(
        trip_id="trip_test",
        messages=[ChatMessage(ts=t, sender=s, text=x) for t, s, x in messages],
        calls=[],
    )


# ---------------------------------------------------------------------------
# gps_analysis
# ---------------------------------------------------------------------------

class TestGpsAnalysis:
    def test_no_deviation_straight_route(self):
        trace = _straight_trace(PICKUP, DROPOFF, n=10)
        trip = _trip(gps_trace=trace)
        items = gps_analysis(trip)
        # Should have at least the distance comparison item
        assert len(items) >= 1
        data = items[0]["data"]
        assert data["distance_actual_km"] > 0
        # No detour or stops expected on a straight route
        detour_items = [i for i in items if "detour_segment" in i["data"]]
        assert len(detour_items) == 0

    def test_detour_of_known_length(self):
        """A trace with a north detour should have actual > optimal and a
        detectable detour segment."""
        trace = _detour_trace(PICKUP, DROPOFF)  # default 0.05 offset → ~15 km
        trip = _trip(gps_trace=trace)
        items = gps_analysis(trip)
        data = items[0]["data"]
        # The detour adds significant distance
        assert data["distance_actual_km"] > data["distance_optimal_km"]
        assert data["deviation_pct"] > 15.0
        # A detour segment should be detected
        detour_items = [i for i in items if "detour_segment" in i["data"]]
        assert len(detour_items) == 1
        seg = detour_items[0]["data"]["detour_segment"]
        assert "start_ts" in seg and "end_ts" in seg

    def test_unexpected_stop_detected(self):
        """A GPS trace with a 3-minute stop (speed 0) away from pickup/dropoff."""
        trace = _straight_trace(PICKUP, DROPOFF, n=20, speed_kmh=40.0)
        # Insert 4 consecutive zero-speed points in the middle (away from ends)
        from datetime import timedelta
        base = trace[10]
        for i in range(4):
            trace.insert(
                11 + i,
                GPSPoint(
                    ts=(trace[10].ts_obj + timedelta(minutes=i + 1)).isoformat()
                    if hasattr(trace[10], "ts_obj")
                    else base.ts,
                    lat=base.lat,
                    lon=base.lon,
                    speed_kmh=0.0,
                ),
            )
        # Rebuild timestamps cleanly
        from datetime import datetime
        ts = datetime.fromisoformat("2026-01-01T08:07:00")
        for p in trace:
            p.ts = ts.isoformat()
            ts = ts + timedelta(minutes=1)
        trip = _trip(gps_trace=trace)
        items = gps_analysis(trip)
        stop_items = [i for i in items if "unexpected_stops" in i["data"]]
        assert len(stop_items) >= 1
        stops = stop_items[0]["data"]["unexpected_stops"]
        assert len(stops) >= 1
        assert stops[0]["duration_min"] > 2.0


# ---------------------------------------------------------------------------
# arrival_check
# ---------------------------------------------------------------------------

class TestArrivalCheck:
    def test_driver_600m_from_pickup(self):
        """Driver's closest GPS point is 600 m from pickup — too far."""
        # Place a GPS point ~600m away from pickup (move ~0.0054 deg lat)
        far_point = GeoPoint(lat=PICKUP.lat + 0.0054, lon=PICKUP.lon)
        trace = [
            GPSPoint(ts="2026-01-01T08:06:00", lat=far_point.lat, lon=far_point.lon, speed_kmh=10.0),
            GPSPoint(ts="2026-01-01T08:07:00", lat=far_point.lat + 0.001, lon=far_point.lon, speed_kmh=20.0),
        ]
        trip = _trip(gps_trace=trace)
        items = arrival_check(trip)
        data = items[0]["data"]
        # min distance should be around 600 m
        assert 500 < data["min_distance_to_pickup_m"] < 700
        assert data["min_distance_to_pickup_m"] > 150.0

    def test_driver_arrives_at_pickup(self):
        """Driver arrives right at the pickup — within 150 m."""
        trace = [
            GPSPoint(ts="2026-01-01T08:06:00", lat=PICKUP.lat, lon=PICKUP.lon, speed_kmh=5.0),
            GPSPoint(ts="2026-01-01T08:07:00", lat=PICKUP.lat + 0.0001, lon=PICKUP.lon, speed_kmh=30.0),
        ]
        trip = _trip(gps_trace=trace)
        items = arrival_check(trip)
        data = items[0]["data"]
        assert data["min_distance_to_pickup_m"] <= 150.0
        assert data["arrival_ts"] == "2026-01-01T08:06:00"

    def test_arrival_vs_eta(self):
        """ETA = accepted_at + 5 min; driver arrives 2 min late."""
        trace = [
            GPSPoint(ts="2026-01-01T08:09:00", lat=PICKUP.lat, lon=PICKUP.lon, speed_kmh=5.0),
        ]
        # accepted_at = 08:02, ETA = 08:07, arrival = 08:09 → +2 min
        trip = _trip(gps_trace=trace)
        items = arrival_check(trip)
        data = items[0]["data"]
        assert abs(data["arrival_vs_eta_min"] - 2.0) < 0.1


# ---------------------------------------------------------------------------
# chat_scan
# ---------------------------------------------------------------------------

class TestChatScan:
    def test_requested_stop_7eleven(self):
        """Rider asks 'can you stop at the 7-Eleven' — requested change detected."""
        chat = _chat(
            ("2026-01-01T08:08:00", "rider", "can you stop at the 7-Eleven on the way?"),
            ("2026-01-01T08:08:30", "driver", "Sure, no problem."),
        )
        items = chat_scan(chat)
        data = items[0]["data"]
        assert data["rider_requested_change"] is True
        assert len(data["rider_requested_change_quotes"]) == 1
        quote = data["rider_requested_change_quotes"][0]
        assert "7-Eleven" in quote
        assert quote.startswith("[")

    def test_threatening_message(self):
        """A threatening message is detected and quoted."""
        chat = _chat(
            ("2026-01-01T08:10:00", "driver", "If you don't pay I will hurt you."),
            ("2026-01-01T08:11:00", "rider", "That's a threat!"),
        )
        items = chat_scan(chat)
        data = items[0]["data"]
        assert len(data["threatening_language"]) >= 1
        assert any("hurt" in q for q in data["threatening_language"])

    def test_abusive_language(self):
        chat = _chat(
            ("2026-01-01T08:10:00", "driver", "You are an idiot."),
        )
        items = chat_scan(chat)
        data = items[0]["data"]
        assert len(data["abusive_language"]) >= 1

    def test_road_closure_mention(self):
        chat = _chat(
            ("2026-01-01T08:09:00", "driver", "There's a road closure on Orchard, I have to divert."),
        )
        items = chat_scan(chat)
        data = items[0]["data"]
        assert data["road_closure"] is True
        assert len(data["road_closure_quotes"]) == 1

    def test_agreement_and_disagreement(self):
        chat = _chat(
            ("2026-01-01T08:08:00", "rider", "Can we go via Orchard?"),
            ("2026-01-01T08:08:10", "driver", "No, that's too far."),
            ("2026-01-01T08:08:20", "rider", "Okay, fine."),
        )
        items = chat_scan(chat)
        data = items[0]["data"]
        assert len(data["agreements"]) >= 1
        assert len(data["disagreements"]) >= 1

    def test_contact_attempts_with_calls(self):
        from ridecourt.data import CallAttempt
        chat = ChatLog(
            trip_id="trip_test",
            messages=[ChatMessage(ts="2026-01-01T08:08:00", sender="driver", text="Where are you?")],
            calls=[CallAttempt(ts="2026-01-01T08:08:30", sender="driver", recipient="rider", duration_s=10, answered=True)],
        )
        items = chat_scan(chat)
        data = items[0]["data"]
        # 1 call + 1 driver chat message = 2 contact attempts
        assert data["contact_attempts"] == 2

    def test_none_chat(self):
        items = chat_scan(None)
        assert len(items) == 1
        assert items[0]["data"]["rider_requested_change"] is False
        assert items[0]["data"]["contact_attempts"] == 0


# ---------------------------------------------------------------------------
# fare_check
# ---------------------------------------------------------------------------

class TestFareCheck:
    def test_promo_not_applied(self):
        """A fare with a promo code but discount=0 → promo not applied."""
        fare = _fare(promo_code="WELCOME10", discount=0.0, total_charged=14.5)
        trip = _trip(fare=fare)
        items = fare_check(trip)
        data = items[0]["data"]
        assert data["promo_applied"] is False
        assert data["promo_valid_at_booking"] is True
        # The recomputed fare includes the discount (0) so it should match
        assert data["recomputed_fare"] == recompute_fare(fare)

    def test_promo_applied(self):
        fare = _fare(promo_code="WELCOME10", discount=2.0, total_charged=12.5)
        trip = _trip(fare=fare)
        items = fare_check(trip)
        data = items[0]["data"]
        assert data["promo_applied"] is True
        assert data["promo_valid_at_booking"] is True

    def test_overcharge_detected(self):
        # components: 3 + 10*0.75 + 20*0.25 = 3+7.5+5 = 15.5, *1.0 + 1 - 0 = 16.5
        fare = _fare(total_charged=18.0)
        trip = _trip(fare=fare)
        items = fare_check(trip)
        data = items[0]["data"]
        assert data["fare_difference"] > 0.5  # overcharged
        assert data["charged_total"] > data["recomputed_fare"]

    def test_fare_correct(self):
        fare = _fare(total_charged=16.5)  # matches recompute
        trip = _trip(fare=fare)
        items = fare_check(trip)
        data = items[0]["data"]
        assert abs(data["fare_difference"]) < 0.01

    def test_invalid_promo(self):
        fare = _fare(promo_code="EXPIRED", discount=2.0, total_charged=14.5)
        trip = _trip(fare=fare)
        items = fare_check(trip)
        data = items[0]["data"]
        assert data["promo_valid_at_booking"] is False


# ---------------------------------------------------------------------------
# history_profile
# ---------------------------------------------------------------------------

class TestHistoryProfile:
    def test_rider_and_driver_profiles(self):
        profiles = {
            "rider_001": Profile(
                account_age_days=365, rating=4.7, trips_completed=120,
                disputes_filed_90d=4, disputes_upheld_90d=2,
                prior_flags=["late_cancel"],
            ),
            "driver_001": Profile(
                account_age_days=730, rating=4.9, trips_completed=2000,
                disputes_filed_90d=2, disputes_upheld_90d=1,
                prior_flags=[],
            ),
        }
        items = history_profile(profiles, "rider_001", "driver_001")
        assert len(items) == 2
        rider_item = next(i for i in items if i["data"]["party"] == "rider")
        driver_item = next(i for i in items if i["data"]["party"] == "driver")
        assert rider_item["data"]["disputes_upheld_90d"] == 2
        assert rider_item["data"]["upheld_rate"] == 50.0
        assert driver_item["data"]["rating"] == 4.9

    def test_bad_faith_rider(self):
        profiles = {
            "rider_bf": Profile(
                account_age_days=400, rating=3.5, trips_completed=30,
                disputes_filed_90d=8, disputes_upheld_90d=6,
                prior_flags=["gps_anomaly"],
            ),
        }
        items = history_profile(profiles, "rider_bf", "driver_x")
        assert len(items) == 1
        assert items[0]["data"]["upheld_rate"] == 75.0


# ---------------------------------------------------------------------------
# build_evidence_pack
# ---------------------------------------------------------------------------

class TestBuildEvidencePack:
    def test_route_deviation_pack_ids(self):
        """A route deviation case should produce E1..En with stable IDs."""
        trace = _detour_trace(PICKUP, DROPOFF)  # default large detour
        fare = _fare()
        trip = _trip(gps_trace=trace, fare=fare)
        chat = _chat(
            ("2026-01-01T08:08:00", "rider", "Why did we go the long way?"),
        )
        profiles = {
            "rider_001": Profile(account_age_days=100, rating=4.5, trips_completed=50),
            "driver_001": Profile(account_age_days=200, rating=4.8, trips_completed=500),
        }
        from ridecourt.data import Dataset
        ds = Dataset(
            trips=[trip], chats=[chat], profiles=profiles,
            disputes=[], precedents=[], labels={},
        )
        dispute = Dispute(
            dispute_id="d1", trip_id="trip_test", filed_by="rider",
            category="route_deviation", claim_text="detour",
            filed_at="2026-01-01T09:00:00",
        )
        pack = build_evidence_pack(dispute, ds)
        ids = [i.id for i in pack.items]
        assert ids == [f"E{i}" for i in range(1, len(ids) + 1)]
        # Should include gps_analysis and chat_scan kinds
        kinds = {i.kind for i in pack.items}
        assert "gps_analysis" in kinds
        assert "chat_scan" in kinds
        assert "fare_check" in kinds
        assert "history_profile" in kinds

    def test_no_show_driver_pack_has_arrival_check(self):
        trace = [
            GPSPoint(ts="2026-01-01T08:06:00", lat=PICKUP.lat + 0.0054, lon=PICKUP.lon, speed_kmh=5.0),
        ]
        trip = _trip(gps_trace=trace, status="no_show_driver")
        chat = _chat(("2026-01-01T08:08:00", "driver", "I'm here, where are you?"))
        profiles = {
            "rider_001": Profile(account_age_days=100, rating=4.5, trips_completed=50),
            "driver_001": Profile(account_age_days=200, rating=4.8, trips_completed=500),
        }
        from ridecourt.data import Dataset
        ds = Dataset(
            trips=[trip], chats=[chat], profiles=profiles,
            disputes=[], precedents=[], labels={},
        )
        dispute = Dispute(
            dispute_id="d2", trip_id="trip_test", filed_by="rider",
            category="no_show_driver", claim_text="driver no show",
            filed_at="2026-01-01T09:00:00",
        )
        pack = build_evidence_pack(dispute, ds)
        kinds = {i.kind for i in pack.items}
        assert "arrival_check" in kinds
        assert "chat_scan" in kinds
        # fare_check provides the cancellation fee amount for no-show cases
        assert "fare_check" in kinds


# ---------------------------------------------------------------------------
# compute_facts reads from the evidence pack
# ---------------------------------------------------------------------------

class TestComputeFactsFromPack:
    def test_facts_match_evidence_data(self):
        """compute_facts should produce numbers identical to the evidence pack."""
        trace = _detour_trace(PICKUP, DROPOFF)  # default large detour
        fare = _fare(total_charged=18.0)  # overcharged
        trip = _trip(gps_trace=trace, fare=fare)
        chat = _chat(
            ("2026-01-01T08:08:00", "rider", "can you stop at the 7-Eleven?"),
        )
        facts = compute_facts(trip, chat)

        # Build the equivalent pack manually to compare
        raw: list[dict] = []
        from ridecourt.tools import (
            gps_analysis, arrival_check, chat_scan, fare_check,
        )
        raw.extend(gps_analysis(trip))
        raw.extend(arrival_check(trip))
        raw.extend(chat_scan(chat))
        raw.extend(fare_check(trip))

        # Facts should match the evidence data
        gps_data = raw[0]["data"]
        assert facts.distance_actual_km == gps_data["distance_actual_km"]
        assert facts.distance_optimal_km == gps_data["distance_optimal_km"]
        assert facts.deviation_pct == gps_data["deviation_pct"]

        arrival_data = next(r["data"] for r in raw if r["kind"] == "arrival_check")
        assert facts.min_distance_to_pickup_m == arrival_data["min_distance_to_pickup_m"]

        chat_data = next(r["data"] for r in raw if r["kind"] == "chat_scan")
        assert facts.rider_requested_change is True
        assert len(facts.rider_requested_change_quotes) == 1

        fare_data = next(r["data"] for r in raw if r["kind"] == "fare_check")
        assert facts.recomputed_fare == fare_data["recomputed_fare"]
        assert facts.charged_total == fare_data["charged_total"]
        assert facts.fare_difference == fare_data["fare_difference"]

    def test_compute_facts_feeds_policy_engine(self):
        """The facts from compute_facts should work with evaluate()."""
        trace = _detour_trace(PICKUP, DROPOFF)  # default large detour (~15 km)
        fare = _fare(total_charged=16.5)
        trip = _trip(gps_trace=trace, fare=fare)
        chat = _chat()  # no special chat
        facts = compute_facts(trip, chat)
        dispute = Dispute(
            dispute_id="d3", trip_id="trip_test", filed_by="rider",
            category="route_deviation", claim_text="detour",
            filed_at="2026-01-01T09:00:00",
        )
        result = evaluate(dispute, trip, {}, facts, trip.fare)
        # Significant deviation → refund_route_deviation
        assert result.outcome == "refund_route_deviation"
        assert result.amount > 0.0


# ---------------------------------------------------------------------------
# Summaries state numbers exactly as in data
# ---------------------------------------------------------------------------

class TestSummaryDataConsistency:
    def test_gps_summary_matches_data(self):
        trace = _straight_trace(PICKUP, DROPOFF, n=10)
        trip = _trip(gps_trace=trace)
        items = gps_analysis(trip)
        data = items[0]["data"]
        summary = items[0]["summary"]
        # The summary should contain the actual distance value
        actual_str = f"{data['distance_actual_km']:.2f}"
        assert actual_str in summary
        optimal_str = f"{data['distance_optimal_km']:.2f}"
        assert optimal_str in summary

    def test_fare_summary_matches_data(self):
        fare = _fare(total_charged=18.0)
        trip = _trip(fare=fare)
        items = fare_check(trip)
        data = items[0]["data"]
        summary = items[0]["summary"]
        recomp_str = f"S${data['recomputed_fare']:.2f}"
        charged_str = f"S${data['charged_total']:.2f}"
        assert recomp_str in summary
        assert charged_str in summary
