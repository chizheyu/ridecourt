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
        assert len(ds.trips) == 48

    def test_chats_count(self):
        ds = load_dataset()
        assert len(ds.chats) == 48

    def test_disputes_count(self):
        ds = load_dataset()
        assert len(ds.disputes) == 48

    def test_precedents_count(self):
        ds = load_dataset()
        assert len(ds.precedents) == 12

    def test_labels_count(self):
        ds = load_dataset()
        assert len(ds.labels) == 48


class TestCategoryCounts:
    def test_route_deviation_count(self):
        ds = load_dataset()
        rd = [d for d in ds.disputes.values() if d.category == "route_deviation"]
        assert len(rd) == 16

    def test_no_show_driver_count(self):
        ds = load_dataset()
        nsd = [d for d in ds.disputes.values() if d.category == "no_show_driver"]
        assert len(nsd) == 10

    def test_no_show_rider_count(self):
        ds = load_dataset()
        nsr = [d for d in ds.disputes.values() if d.category == "no_show_rider"]
        assert len(nsr) == 10

    def test_fare_error_count(self):
        ds = load_dataset()
        fe = [d for d in ds.disputes.values() if d.category == "fare_error"]
        assert len(fe) == 6

    def test_safety_count(self):
        ds = load_dataset()
        sf = [d for d in ds.disputes.values() if d.category == "safety"]
        assert len(sf) == 6


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
                assert label.outcome == "escalate", (
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


class TestProfiles:
    def test_bad_faith_rider_exists(self):
        ds = load_dataset()
        assert "rider_bf1" in ds.profiles
        p = ds.profiles["rider_bf1"]
        assert p.disputes_upheld_90d >= 3

    def test_bad_faith_driver_exists(self):
        ds = load_dataset()
        assert "driver_bf1" in ds.profiles
        p = ds.profiles["driver_bf1"]
        assert "gps_anomaly" in p.prior_flags


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
