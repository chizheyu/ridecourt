"""Tests for the dataset loader and generated data.

Run with: .venv/bin/python -m pytest tests/test_data.py -q
"""

import os
from pathlib import Path

os.environ.setdefault("LLM_PROVIDER", "mock")

from ridecourt.data import load_dataset, Dataset, DATA_DIR
from ridecourt.tools import compute_facts
from ridecourt.policy import evaluate


class TestDatasetLoads:
    def test_loads_without_error(self):
        ds = load_dataset()
        assert isinstance(ds, Dataset)

    def test_data_dir_exists(self):
        assert DATA_DIR.exists()
        assert (DATA_DIR / "trips.json").exists()
        assert (DATA_DIR / "chats.json").exists()
        assert (DATA_DIR / "profiles.json").exists()
        assert (DATA_DIR / "disputes.json").exists()
        assert (DATA_DIR / "precedents.json").exists()
        assert (DATA_DIR / "labels.json").exists()

    def test_trips_count(self):
        ds = load_dataset()
        assert len(ds.trips) == 51

    def test_chats_count(self):
        ds = load_dataset()
        assert len(ds.chats) == 51

    def test_disputes_count(self):
        ds = load_dataset()
        assert len(ds.disputes) == 51

    def test_precedents_count(self):
        ds = load_dataset()
        assert len(ds.precedents) == 11

    def test_labels_count(self):
        ds = load_dataset()
        assert len(ds.labels) == 51


class TestCategoryCounts:
    def test_route_deviation_count(self):
        ds = load_dataset()
        rd = [d for d in ds.disputes.values() if d.category == "route_deviation"]
        assert len(rd) == 14

    def test_no_show_driver_count(self):
        ds = load_dataset()
        nsd = [d for d in ds.disputes.values() if d.category == "no_show_driver"]
        assert len(nsd) == 2

    def test_no_show_rider_count(self):
        ds = load_dataset()
        nsr = [d for d in ds.disputes.values() if d.category == "no_show_rider"]
        assert len(nsr) == 10

    def test_fare_error_count(self):
        ds = load_dataset()
        fe = [d for d in ds.disputes.values() if d.category == "fare_error"]
        assert len(fe) == 11  # 10 surge + 1 fraud-flagged

    def test_safety_count(self):
        ds = load_dataset()
        sf = [d for d in ds.disputes.values() if d.category == "safety"]
        assert len(sf) == 4

    def test_property_damage_count(self):
        ds = load_dataset()
        pd = [d for d in ds.disputes.values() if d.category == "property_damage"]
        assert len(pd) == 10


class TestLabels:
    def test_every_dispute_has_label(self):
        ds = load_dataset()
        for did, dispute in ds.disputes.items():
            label = ds.label(did)
            assert label is not None, f"No label for {did}"
            assert hasattr(label, "outcome")
            assert hasattr(label, "amount")

    def test_safety_labels_escalate(self):
        ds = load_dataset()
        for did, dispute in ds.disputes.items():
            if dispute.category == "safety":
                label = ds.label(did)
                assert label.outcome == "escalate_safety", (
                    f"Safety case {did} should escalate, got {label.outcome}"
                )
                assert label.amount == 0.0

    def test_labels_match_policy_engine(self):
        """Labels should be consistent with the policy engine recomputation."""
        ds = load_dataset()
        for did, dispute in ds.disputes.items():
            trip = ds.trip(dispute.trip_id)
            chat = ds.chat(dispute.trip_id)
            assert trip is not None
            facts = compute_facts(trip, chat)
            result = evaluate(dispute, trip, ds.profiles, facts, trip.fare)
            label = ds.label(did)
            assert label.outcome == result.outcome, (
                f"{did}: label outcome {label.outcome} != "
                f"policy outcome {result.outcome}"
            )
            assert abs(label.amount - result.amount) < 0.01, (
                f"{did}: label amount {label.amount} != "
                f"policy amount {result.amount}"
            )

    def test_labels_cite_v2_sections(self):
        """Every label's applicable_clauses should use v2.1 §N.M format."""
        ds = load_dataset()
        for did, dispute in ds.disputes.items():
            label = ds.label(did)
            for clause in label.applicable_clauses:
                assert clause.startswith("§"), (
                    f"{did}: clause {clause} doesn't start with §"
                )


class TestProfiles:
    def test_fraud_flagged_rider_exists(self):
        ds = load_dataset()
        assert "rider_fraud1" in ds.profiles
        p = ds.profiles["rider_fraud1"]
        assert p.fraud_flags >= 1

    def test_repeat_no_show_rider_exists(self):
        ds = load_dataset()
        assert "rider_repeat_ns" in ds.profiles
        p = ds.profiles["rider_repeat_ns"]
        assert p.no_show_cancellations_30d >= 3

    def test_upheld_never_exceeds_filed(self):
        """Invariant: disputes_upheld_90d <= disputes_filed_90d for every
        profile.  Upheld is a subset of filed; 200% or 999% rates are bugs."""
        ds = load_dataset()
        for pid, p in ds.profiles.items():
            assert p.disputes_upheld_90d <= p.disputes_filed_90d, (
                f"Profile {pid}: upheld ({p.disputes_upheld_90d}) > "
                f"filed ({p.disputes_filed_90d})"
            )


class TestTrips:
    def test_every_trip_has_gps_trace(self):
        ds = load_dataset()
        for trip in ds.trips.values():
            assert len(trip.gps_trace) > 0, f"{trip.trip_id} has no GPS trace"

    def test_every_dispute_has_trip(self):
        ds = load_dataset()
        for dispute in ds.disputes.values():
            assert ds.trip(dispute.trip_id) is not None

    def test_every_dispute_has_chat(self):
        ds = load_dataset()
        for dispute in ds.disputes.values():
            assert ds.chat(dispute.trip_id) is not None

    def test_arrived_after_accepted(self):
        """Invariant: driver_arrived_at >= accepted_at when both are set."""
        ds = load_dataset()
        for trip in ds.trips.values():
            if trip.accepted_at and trip.driver_arrived_at:
                assert trip.driver_arrived_at >= trip.accepted_at, (
                    f"Trip {trip.trip_id}: arrived < accepted"
                )

    def test_gps_trace_within_trip_bounds(self):
        """Invariant: GPS trace timestamps must be within the trip's time
        window.  First point >= driver_arrived_at, last <= trip_ended_at."""
        ds = load_dataset()
        for trip in ds.trips.values():
            if not trip.gps_trace:
                continue
            first_ts = trip.gps_trace[0].ts
            last_ts = trip.gps_trace[-1].ts
            if trip.driver_arrived_at:
                assert first_ts >= trip.driver_arrived_at, (
                    f"Trip {trip.trip_id}: first GPS ts {first_ts} < "
                    f"arrived {trip.driver_arrived_at}"
                )
            if trip.trip_ended_at:
                assert last_ts <= trip.trip_ended_at, (
                    f"Trip {trip.trip_id}: last GPS ts {last_ts} > "
                    f"ended {trip.trip_ended_at}"
                )

    def test_no_show_driver_has_ended_at(self):
        """Invariant: no_show_driver cases without driver_arrived_at must
        have trip_ended_at so wait_minutes_past_eta can compute."""
        ds = load_dataset()
        for dispute in ds.disputes.values():
            if dispute.category != "no_show_driver":
                continue
            trip = ds.trip(dispute.trip_id)
            assert trip is not None
            if trip.driver_arrived_at is None:
                assert trip.trip_ended_at is not None, (
                    f"Trip {trip.trip_id}: no_show_driver without ended_at"
                )
