"""§2.1 free wait and §2.4 late arrival.

The free wait runs from the later of the scheduled pickup time and the
driver's arrival to the cancellation; a rider no-show fee stands only after
five minutes of it. A driver who arrives more than five minutes after the
scheduled pickup time is a driver no-show (§2.4).
"""

import pytest

from ridecourt.data import Facts, GeoPoint, GPSPoint, OptimalRoute, Trip
from ridecourt.policy import compute_confidence, section_2_no_show
from ridecourt.tools import arrival_check
from ridecourt.ryde_adapter import RydeDatasetAdapter, RYDE_SAMPLES_DIR
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack


def _trip(arrived: str | None, scheduled: str | None, cancelled: str | None) -> Trip:
    pickup = GeoPoint(lat=1.30, lon=103.80, address="pickup")
    trace = [GPSPoint(ts="2026-01-01T08:00:00", lat=1.30, lon=103.80, speed_kmh=0),
             GPSPoint(ts="2026-01-01T08:10:00", lat=1.30, lon=103.80, speed_kmh=0)]
    return Trip(
        trip_id="t1", rider_id="r1", driver_id="d1", pickup=pickup, dropoff=pickup,
        requested_at="2026-01-01T07:50:00", accepted_at="2026-01-01T07:51:00",
        driver_arrived_at=arrived, scheduled_pickup_at=scheduled, cancelled_at=cancelled,
        status="no_show_rider", gps_trace=trace,
        optimal_route=OptimalRoute(distance_km=0, duration_min=0, points=[pickup, pickup]),
    )


def test_free_wait_counts_from_later_of_scheduled_and_arrival():
    data = arrival_check(_trip("2026-01-01T08:00:00", "2026-01-01T08:02:00", "2026-01-01T08:08:00"))[0]["data"]
    assert data["free_wait_minutes"] == pytest.approx(6.0)
    data = arrival_check(_trip("2026-01-01T08:03:00", "2026-01-01T08:00:00", "2026-01-01T08:08:00"))[0]["data"]
    assert data["free_wait_minutes"] == pytest.approx(5.0)


def test_cancelled_before_free_wait_ended_voids_fee():
    f = Facts(driver_made_contact=True, driver_chat_messages=1, driver_call_attempts=1,
              free_wait_minutes=3.0)
    r = section_2_no_show(f, "no_show_rider")
    assert r.outcome == "cancellation_fee_voided"
    assert r.applicable_clauses == ["§2.1"]


def test_fee_upheld_after_free_wait_with_both_contacts():
    f = Facts(driver_made_contact=True, driver_chat_messages=1, driver_call_attempts=1,
              free_wait_minutes=6.0)
    r = section_2_no_show(f, "no_show_rider")
    assert r.outcome == "cancellation_fee_upheld"


def test_late_arrival_is_driver_no_show():
    data = arrival_check(_trip("2026-01-01T08:08:00", "2026-01-01T08:00:00", None))[0]["data"]
    assert data["driver_no_show"] is True
    data = arrival_check(_trip("2026-01-01T08:04:00", "2026-01-01T08:00:00", None))[0]["data"]
    assert data["driver_no_show"] is False


def test_missing_cancellation_time_lowers_clarity():
    f = Facts(driver_made_contact=True, driver_chat_messages=1, driver_call_attempts=1)
    _, factors = compute_confidence(f, "no_show_rider", 5.0, outcome="cancellation_fee_upheld")
    assert factors["policy_clarity"] < 1.0


@pytest.mark.skipif(not RYDE_SAMPLES_DIR.exists(), reason="data/ryde_samples/ directory not found")
def test_disp_002_free_wait_six_minutes():
    ds = RydeDatasetAdapter().load()
    dispute = ds.dispute("DISP-002")
    facts = facts_from_evidence_pack(build_evidence_pack(dispute, ds))
    # scheduled 08:45, arrived 08:43, cancelled 08:51
    assert facts.free_wait_minutes == pytest.approx(6.0)
    assert facts.free_wait_expired is True
