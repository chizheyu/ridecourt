"""Unit tests for the policy engine — every clause, plus edge cases.

Run with: .venv/bin/python -m pytest tests/test_policy.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

from ridecourt.data import Dispute, Facts, Fare, Profile, Trip, GeoPoint, OptimalRoute
from ridecourt.policy import (
    clause_bf_1,
    clause_fa_1,
    clause_ns_1,
    clause_ns_2,
    clause_rd_1,
    clause_rd_2,
    clause_sf_1,
    evaluate,
    NS_PICKUP_RADIUS_M,
    NS_WAIT_MIN,
    RD_DEVIATION_KM,
    RD_DEVIATION_PCT,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fare(**kw) -> Fare:
    defaults = dict(
        base=3.0, per_km=0.75, per_min=0.25,
        distance_km=10.0, duration_min=20.0,
        surge_multiplier=1.0, promo_code="", discount=0.0,
        booking_fee=1.0, cancellation_fee=4.0, total_charged=14.5,
    )
    defaults.update(kw)
    return Fare(**defaults)


def _dispute(category="route_deviation", filed_by="rider", **kw) -> Dispute:
    defaults = dict(
        dispute_id="disp_test",
        trip_id="trip_test",
        filed_by=filed_by,
        category=category,
        claim_text="test claim",
        filed_at="2026-01-01T08:00:00",
    )
    defaults.update(kw)
    return Dispute(**defaults)


def _trip(fare: Fare | None = None, **kw) -> Trip:
    f = fare or _fare()
    defaults = dict(
        trip_id="trip_test",
        rider_id="rider_001",
        driver_id="driver_001",
        pickup=GeoPoint(lat=1.3521, lon=103.9450, address="Tampines"),
        dropoff=GeoPoint(lat=1.3041, lon=103.8318, address="Orchard"),
        requested_at="2026-01-01T08:00:00",
        accepted_at="2026-01-01T08:02:00",
        driver_arrived_at="2026-01-01T08:06:00",
        trip_started_at="2026-01-01T08:07:00",
        trip_ended_at="2026-01-01T08:27:00",
        status="completed",
        gps_trace=[],
        optimal_route=OptimalRoute(distance_km=10.0, duration_min=20.0),
        fare=f,
    )
    defaults.update(kw)
    return Trip(**defaults)


def _facts(**kw) -> Facts:
    defaults = dict(
        distance_actual_km=10.0,
        distance_optimal_km=10.0,
        duration_actual_min=20.0,
        duration_optimal_min=20.0,
        rider_requested_change=False,
        road_closure=False,
        min_distance_to_pickup_m=50.0,
        wait_minutes_past_eta=0.0,
        contact_attempts=0,
        recomputed_fare=14.5,
        charged_total=14.5,
        promo_valid_at_booking=True,
    )
    defaults.update(kw)
    return Facts(**defaults)


# ---------------------------------------------------------------------------
# P-RD-1: route deviation
# ---------------------------------------------------------------------------

class TestRouteDeviation:
    def test_no_deviation(self):
        f = _facts(distance_actual_km=10.0, distance_optimal_km=10.0)
        r = clause_rd_1(f)
        assert r.outcome == "no_deviation"
        assert r.amount == 0.0

    def test_deviation_within_15_percent(self):
        # 14% over — not enough
        f = _facts(distance_actual_km=11.4, distance_optimal_km=10.0)
        r = clause_rd_1(f)
        assert r.outcome == "no_deviation"

    def test_deviation_exactly_15_percent(self):
        # Exactly 15% — not strictly > 15%, so no deviation
        f = _facts(distance_actual_km=11.5, distance_optimal_km=10.0)
        r = clause_rd_1(f)
        assert r.outcome == "no_deviation"

    def test_deviation_over_15_percent_but_under_1_5km(self):
        # 20% over but only 0.5 km extra — not enough
        f = _facts(distance_actual_km=3.0, distance_optimal_km=2.5)
        r = clause_rd_1(f)
        assert r.outcome == "no_deviation"

    def test_deviation_significant(self):
        f = _facts(
            distance_actual_km=13.0, distance_optimal_km=10.0,
            duration_actual_min=26.0, duration_optimal_min=20.0,
        )
        r = clause_rd_1(f)
        assert r.outcome != "no_deviation"  # significant deviation

    def test_refund_computation(self):
        f = _facts(
            distance_actual_km=13.0, distance_optimal_km=10.0,
            duration_actual_min=26.0, duration_optimal_min=20.0,
        )
        trip = _trip()
        d = _dispute(category="route_deviation")
        result = evaluate(d, trip, {}, f, trip.fare)
        # (3.0 * 0.75 + 6.0 * 0.25) * 1.0 = 2.25 + 1.50 = 3.75
        assert result.outcome == "refund_route_deviation"
        assert result.amount == 3.75

    def test_refund_with_surge(self):
        f = _facts(
            distance_actual_km=13.0, distance_optimal_km=10.0,
            duration_actual_min=26.0, duration_optimal_min=20.0,
        )
        fare = _fare(surge_multiplier=1.5)
        trip = _trip(fare=fare)
        d = _dispute(category="route_deviation")
        result = evaluate(d, trip, {}, f, trip.fare)
        # (2.25 + 1.50) * 1.5 = 5.625 → 5.62
        assert result.amount == 5.62

    def test_negative_extra_time_clamped(self):
        """A detour that is longer in distance but shorter in time must
        not produce a negative time contribution."""
        f = _facts(
            distance_actual_km=16.87, distance_optimal_km=10.0,
            duration_actual_min=17.1, duration_optimal_min=20.0,
            # extra_km=6.87, extra_min=-2.9 → clamped to 0
        )
        trip = _trip()
        d = _dispute(category="route_deviation")
        result = evaluate(d, trip, {}, f, trip.fare)
        assert result.outcome == "refund_route_deviation"
        # 6.87 * 0.75 + 0 * 0.25 = 5.1525 → 5.15
        assert result.amount == 5.15
        # Computation must not contain a negative S$ value
        assert "S$-0" not in result.computation
        assert "max(0," in result.computation

    def test_negative_extra_distance_clamped(self):
        """If actual distance is less than optimal (edge case), extra_km
        is clamped to 0."""
        f = _facts(
            distance_actual_km=10.0, distance_optimal_km=10.0,
            duration_actual_min=25.0, duration_optimal_min=20.0,
        )
        r = clause_rd_1(f)
        # extra_km=0 → 0% → no deviation
        assert r.outcome == "no_deviation"


# ---------------------------------------------------------------------------
# P-RD-2: requested stop / rider-requested change
# ---------------------------------------------------------------------------

class TestRequestedChange:
    def test_no_requested_change(self):
        f = _facts(rider_requested_change=False)
        r = clause_rd_2(f)
        assert r.outcome == "not_applicable"

    def test_requested_change(self):
        f = _facts(rider_requested_change=True)
        r = clause_rd_2(f)
        assert r.outcome == "requested_change"
        assert r.amount == 0.0

    def test_requested_change_overrides_deviation(self):
        f = _facts(
            distance_actual_km=13.0, distance_optimal_km=10.0,
            rider_requested_change=True,
        )
        trip = _trip()
        d = _dispute(category="route_deviation")
        result = evaluate(d, trip, {}, f, trip.fare)
        assert result.outcome == "no_refund_rider_requested"
        assert result.amount == 0.0


# ---------------------------------------------------------------------------
# Road closure overrides deviation
# ---------------------------------------------------------------------------

class TestRoadClosure:
    def test_road_closure_overrides_deviation(self):
        f = _facts(
            distance_actual_km=13.0, distance_optimal_km=10.0,
            road_closure=True,
        )
        trip = _trip()
        d = _dispute(category="route_deviation")
        result = evaluate(d, trip, {}, f, trip.fare)
        assert result.outcome == "no_refund_road_closure"
        assert result.amount == 0.0


# ---------------------------------------------------------------------------
# P-NS-1: driver no-show
# ---------------------------------------------------------------------------

class TestDriverNoShow:
    def test_driver_too_far(self):
        # 151m > 150m
        f = _facts(min_distance_to_pickup_m=151.0, charged_total=4.0)
        r = clause_ns_1(f)
        assert r.outcome == "refund_and_credit"
        assert r.amount == 7.0  # 4 + 3

    def test_driver_within_150m(self):
        # 149m < 150m — not too far
        f = _facts(min_distance_to_pickup_m=149.0, charged_total=4.0, wait_minutes_past_eta=0.0)
        r = clause_ns_1(f)
        assert r.outcome == "no_show_not_proven"
        assert r.amount == 0.0

    def test_exactly_150m(self):
        # Exactly 150m — not strictly > 150m, so not "too far"
        f = _facts(min_distance_to_pickup_m=150.0, charged_total=4.0)
        r = clause_ns_1(f)
        assert r.outcome == "no_show_not_proven"

    def test_rider_waited_over_5_min(self):
        f = _facts(min_distance_to_pickup_m=50.0, wait_minutes_past_eta=6.0, charged_total=4.0)
        r = clause_ns_1(f)
        assert r.outcome == "refund_and_credit"
        assert r.amount == 7.0

    def test_rider_waited_exactly_5_min(self):
        # Exactly 5 min — not strictly > 5 min
        f = _facts(min_distance_to_pickup_m=50.0, wait_minutes_past_eta=5.0, charged_total=4.0)
        r = clause_ns_1(f)
        assert r.outcome == "no_show_not_proven"

    def test_rider_waited_4_9_min(self):
        f = _facts(min_distance_to_pickup_m=50.0, wait_minutes_past_eta=4.9, charged_total=4.0)
        r = clause_ns_1(f)
        assert r.outcome == "no_show_not_proven"

    def test_no_cancellation_fee(self):
        f = _facts(min_distance_to_pickup_m=200.0, charged_total=0.0)
        r = clause_ns_1(f)
        assert r.outcome == "refund_and_credit"
        assert r.amount == 3.0  # just the credit


# ---------------------------------------------------------------------------
# P-NS-2: rider no-show / cancellation fee stands
# ---------------------------------------------------------------------------

class TestRiderNoShow:
    def test_fee_stands(self):
        f = _facts(
            min_distance_to_pickup_m=50.0,
            wait_minutes_within_150m=5.0,
            contact_attempts=2,
            charged_total=4.0,
        )
        r = clause_ns_2(f)
        assert r.outcome == "cancellation_fee_stands"
        assert r.amount == 4.0

    def test_exactly_5_min_and_150m(self):
        # Boundary: 5.0 min within 150m, 150m, 1 contact → stands
        f = _facts(
            min_distance_to_pickup_m=150.0,
            wait_minutes_within_150m=5.0,
            contact_attempts=1,
            charged_total=4.0,
        )
        r = clause_ns_2(f)
        assert r.outcome == "cancellation_fee_stands"

    def test_no_contact_attempt(self):
        f = _facts(
            min_distance_to_pickup_m=50.0,
            wait_minutes_within_150m=5.0,
            contact_attempts=0,
            charged_total=4.0,
        )
        r = clause_ns_2(f)
        assert r.outcome == "cancellation_fee_invalid"
        assert r.amount == 0.0

    def test_driver_did_not_wait_long_enough_within_150m(self):
        """P-NS-2 uses wait_minutes_within_150m, not wait_minutes_past_eta.
        If the driver only waited 4.9 min within 150m, the fee is invalid
        even if wait_minutes_past_eta is >= 5."""
        f = _facts(
            min_distance_to_pickup_m=50.0,
            wait_minutes_within_150m=4.9,
            wait_minutes_past_eta=6.0,  # past ETA is irrelevant for P-NS-2
            contact_attempts=2,
            charged_total=4.0,
        )
        r = clause_ns_2(f)
        assert r.outcome == "cancellation_fee_invalid"

    def test_fee_stands_uses_within_150m_not_past_eta(self):
        """The bug that caused disp_ns_r_001 to fail: the driver waited
        6 min within 150m but only 4 min past the ETA.  P-NS-2 must use
        the within-150m time (6.0 >= 5.0 → stands), not the past-ETA time."""
        f = _facts(
            min_distance_to_pickup_m=47.0,
            wait_minutes_within_150m=6.0,
            wait_minutes_past_eta=4.0,
            contact_attempts=3,
            charged_total=4.0,
        )
        r = clause_ns_2(f)
        assert r.outcome == "cancellation_fee_stands"
        assert r.amount == 4.0
        # The computation string must mention within 150 m
        assert "within 150 m" in r.computation


# ---------------------------------------------------------------------------
# P-FA-1: fare errors
# ---------------------------------------------------------------------------

class TestFareError:
    def test_overcharge(self):
        f = _facts(charged_total=15.0, recomputed_fare=14.0, promo_valid_at_booking=True)
        r = clause_fa_1(f)
        assert r.outcome == "refund_overcharge"
        assert r.amount == 1.0

    def test_within_tolerance(self):
        f = _facts(charged_total=14.5, recomputed_fare=14.2, promo_valid_at_booking=True)
        r = clause_fa_1(f)
        assert r.outcome == "fare_correct"
        assert r.amount == 0.0

    def test_exactly_50_cents(self):
        # Exactly S$0.50 — not strictly > 0.50
        f = _facts(charged_total=14.5, recomputed_fare=14.0, promo_valid_at_booking=True)
        r = clause_fa_1(f)
        assert r.outcome == "fare_correct"

    def test_51_cents_over(self):
        f = _facts(charged_total=14.51, recomputed_fare=14.0, promo_valid_at_booking=True)
        r = clause_fa_1(f)
        assert r.outcome == "refund_overcharge"
        assert r.amount == 0.51

    def test_undercharge(self):
        f = _facts(charged_total=14.0, recomputed_fare=15.0, promo_valid_at_booking=True)
        r = clause_fa_1(f)
        assert r.outcome == "undercharge_noted"
        assert r.amount == 0.0  # not collected

    def test_valid_promo(self):
        f = _facts(charged_total=12.0, recomputed_fare=14.0, promo_valid_at_booking=True)
        r = clause_fa_1(f)
        # charged < recomputed → undercharge
        assert r.outcome == "undercharge_noted"


# ---------------------------------------------------------------------------
# P-SF-1: safety always escalates
# ---------------------------------------------------------------------------

class TestSafety:
    def test_always_escalates(self):
        f = _facts()
        r = clause_sf_1(f)
        assert r.outcome == "escalate"
        assert r.amount == 0.0

    def test_safety_escalates_via_evaluate(self):
        trip = _trip()
        d = _dispute(category="safety")
        result = evaluate(d, trip, {}, _facts(), trip.fare)
        assert result.outcome == "escalate"
        assert result.amount == 0.0
        assert "P-SF-1" in result.applicable_clauses

    def test_safety_escalates_regardless_of_facts(self):
        trip = _trip()
        d = _dispute(category="safety")
        # Even with crazy facts
        f = _facts(distance_actual_km=100, charged_total=999)
        result = evaluate(d, trip, {}, f, trip.fare)
        assert result.outcome == "escalate"


# ---------------------------------------------------------------------------
# P-BF-1: bad faith
# ---------------------------------------------------------------------------

class TestBadFaith:
    def test_bad_faith_flagged(self):
        profiles = {
            "rider_bf1": Profile(
                account_age_days=400, rating=3.8, trips_completed=52,
                disputes_filed_90d=8, disputes_upheld_90d=6,
            )
        }
        r = clause_bf_1(profiles, "rider_bf1")
        assert r.outcome == "bad_faith_flagged"

    def test_no_bad_faith(self):
        profiles = {
            "rider_001": Profile(
                account_age_days=100, rating=4.5, trips_completed=50,
                disputes_filed_90d=2, disputes_upheld_90d=1,
            )
        }
        r = clause_bf_1(profiles, "rider_001")
        assert r.outcome == "no_bad_faith"

    def test_profile_not_found(self):
        r = clause_bf_1({}, "unknown")
        assert r.outcome == "profile_not_found"

    def test_exactly_three_upheld(self):
        profiles = {
            "r": Profile(
                account_age_days=200, rating=4.0, trips_completed=30,
                disputes_filed_90d=4, disputes_upheld_90d=3,
            )
        }
        r = clause_bf_1(profiles, "r")
        assert r.outcome == "bad_faith_flagged"
