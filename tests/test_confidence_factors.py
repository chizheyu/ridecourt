"""Tests for the §6.3 confidence computation.

Each of the four factors (evidence completeness, policy clarity, precedent
alignment, party credibility) must move correctly when its inputs change.
Also tests the three official Ryde cases for expected confidence and routing.
"""

import pytest

from ridecourt.data import Facts
from ridecourt.policy import (
    W_EVIDENCE,
    W_CLARITY,
    W_PRECEDENT,
    W_CREDIBILITY,
    RT_CONFIDENCE_95,
    compute_confidence,
    evaluate,
)
from ridecourt.ryde_adapter import RydeDatasetAdapter
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack
from ridecourt.precedent import retrieve_precedents


def _facts(**kw) -> Facts:
    """Build a Facts object with sensible defaults."""
    defaults = dict(
        distance_actual_km=20.0,
        distance_optimal_km=15.0,
        duration_actual_min=30.0,
        duration_optimal_min=25.0,
        extra_min=5.0,
        deviation_pct=33.3,
        rider_informed_deviation=False,
        deviation_point_ts="2026-01-01T08:10:00",
        traffic_justified=False,
        traffic_incidents_on_route=[],
        driver_chat_messages=2,
        driver_call_attempts=1,
        driver_made_contact=True,
        min_distance_to_pickup_m=10.0,
        arrival_ts="2026-01-01T08:06:00",
        arrival_vs_eta_min=1.0,
        wait_minutes_within_150m=5.0,
        wait_minutes_past_eta=0.0,
        charged_total=16.50,
        recomputed_fare=12.00,
        fare_difference=4.50,
        upfront_quote=12.00,
        quote_surge_multiplier=1.0,
        trip_start_surge_multiplier=1.5,
        surgeable_subtotal=9.00,
        rider_notified_surge=False,
        surge_increased=True,
        surge_decreased=False,
        surge_differential=4.50,
        rider_account_age_days=540,
        rider_rating=4.8,
        rider_fraud_flags=0,
        driver_account_age_days=720,
        driver_rating=4.7,
        driver_fraud_flags=0,
    )
    defaults.update(kw)
    return Facts(**defaults)


class TestEvidenceCompleteness:
    def test_all_fields_present(self):
        f = _facts()
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["evidence_completeness"] == 1.0

    def test_missing_deviation_point(self):
        f = _facts(deviation_point_ts=None)
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["evidence_completeness"] < 1.0
        assert "deviation point" in factors["evidence_reason"].lower()

    def test_missing_arrival_for_no_show(self):
        f = _facts(arrival_ts=None, min_distance_to_pickup_m=float("inf"))
        conf, factors = compute_confidence(f, "no_show_rider", 5.0, outcome="upheld")
        assert factors["evidence_completeness"] < 1.0


class TestPolicyClarity:
    def test_surge_not_notified_high_clarity(self):
        f = _facts(surge_increased=True, rider_notified_surge=False)
        conf, factors = compute_confidence(f, "fare_error", 4.50, outcome="refund")
        assert factors["policy_clarity"] >= 0.95

    def test_no_surge_no_difference_high_clarity(self):
        f = _facts(surge_increased=False, fare_difference=0, charged_total=12.0, upfront_quote=12.0)
        conf, factors = compute_confidence(f, "fare_error", 0.0, outcome="no_refund")
        assert factors["policy_clarity"] >= 0.90

    def test_no_show_zero_contact_high_clarity(self):
        f = _facts(driver_chat_messages=0, driver_call_attempts=0)
        conf, factors = compute_confidence(f, "no_show_rider", 0.0, outcome="voided")
        assert factors["policy_clarity"] >= 0.95


class TestPrecedentAlignment:
    def test_strong_match_high(self):
        f = _facts()
        conf, factors = compute_confidence(f, "route_deviation", 3.44,
                                            has_precedent_match=True,
                                            precedent_similarity=0.5,
                                            outcome="refund_partial")
        assert factors["precedent_alignment"] >= 0.95

    def test_weak_match_moderate(self):
        f = _facts()
        conf, factors = compute_confidence(f, "route_deviation", 3.44,
                                            has_precedent_match=True,
                                            precedent_similarity=0.2,
                                            outcome="refund_partial")
        assert factors["precedent_alignment"] == 0.80

    def test_no_match_low(self):
        f = _facts()
        conf, factors = compute_confidence(f, "route_deviation", 3.44,
                                            has_precedent_match=False,
                                            outcome="refund_partial")
        assert factors["precedent_alignment"] == 0.50


class TestPartyCredibility:
    def test_established_good_rating_high(self):
        f = _facts(rider_account_age_days=540, rider_rating=4.8, rider_fraud_flags=0)
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["party_credibility"] >= 0.95

    def test_fraud_flag_on_winning_party_lower(self):
        f = _facts(rider_fraud_flags=1)
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["party_credibility"] <= 0.55
        assert "fraud flag" in factors["credibility_reason"]

    def test_driver_wins_driver_credibility(self):
        """When the driver wins (fee upheld), use driver credibility."""
        f = _facts(driver_account_age_days=900, driver_rating=4.9, driver_fraud_flags=0)
        conf, factors = compute_confidence(f, "no_show_rider", 5.0, outcome="upheld")
        assert factors["party_credibility"] >= 0.95
        assert "driver" in factors["credibility_reason"]


class TestOfficialCasesConfidence:
    """Test the three official Ryde cases for expected confidence and routing."""

    @pytest.fixture(scope="class")
    def ds(self):
        return RydeDatasetAdapter().load()

    def _evaluate(self, ds, dispute_id: str):
        disp = ds.dispute(dispute_id)
        trip = ds.trip(disp.trip_id)
        pack = build_evidence_pack(disp, ds)
        facts = facts_from_evidence_pack(pack)
        precedents = retrieve_precedents(disp, ds.precedents, top_k=3)
        has_match = len(precedents) > 0
        best_sim = precedents[0].similarity if precedents else 0.0
        return evaluate(disp, trip, ds.profiles, facts, trip.fare,
                        has_precedent_match=has_match,
                        precedent_similarity=best_sim)

    def test_disp001_human_high_confidence(self, ds):
        """DISP-001 routes to human via §6.2 with high confidence."""
        pr = self._evaluate(ds, "DISP-001")
        assert pr.routing == "human"
        assert pr.confidence >= 0.90
        assert "§6.2" in pr.routing_reason

    def test_disp002_ai_final_high_confidence(self, ds):
        """DISP-002 ≥ 0.95 confidence, AI final despite fraud flag."""
        pr = self._evaluate(ds, "DISP-002")
        assert pr.confidence >= 0.95
        assert pr.routing == "ai_final"

    def test_disp003_ai_final(self, ds):
        """DISP-003 AI final."""
        pr = self._evaluate(ds, "DISP-003")
        assert pr.routing == "ai_final"

    def test_disp001_confidence_factors(self, ds):
        """DISP-001: list the four confidence factors with reasons."""
        pr = self._evaluate(ds, "DISP-001")
        f = pr.confidence_factors
        assert f["evidence_completeness"] == 1.0
        assert "all key evidence fields present" in f["evidence_reason"]
        assert f["policy_clarity"] >= 0.90
        assert "rider not informed" in f["policy_clarity_reason"]
        assert f["precedent_alignment"] >= 0.80
        assert "winning party" in f["credibility_reason"]

    def test_disp002_confidence_factors(self, ds):
        """DISP-002: list the four confidence factors with reasons."""
        pr = self._evaluate(ds, "DISP-002")
        f = pr.confidence_factors
        assert f["evidence_completeness"] == 1.0
        assert f["policy_clarity"] >= 0.90
        assert "contact" in f["policy_clarity_reason"].lower()
        assert f["precedent_alignment"] >= 0.80
        assert "winning party" in f["credibility_reason"]

    def test_disp003_confidence_factors(self, ds):
        """DISP-003: list the four confidence factors with reasons."""
        pr = self._evaluate(ds, "DISP-003")
        f = pr.confidence_factors
        assert f["evidence_completeness"] == 1.0
        assert f["policy_clarity"] >= 0.95
        assert "surge" in f["policy_clarity_reason"].lower()
        assert f["precedent_alignment"] >= 0.80
        assert "winning party" in f["credibility_reason"]


class TestFraudFlagRoutingToHuman:
    """A synthetic fraud-flag case where the ruling favours the flagged
    party must route to human (confidence < 0.95)."""

    def test_fraud_flag_winning_party_routes_human(self):
        """When the winning party has a fraud flag and confidence < 0.95,
        the case routes to human."""
        f = _facts(
            rider_fraud_flags=1,
            rider_account_age_days=10,
            rider_rating=3.5,
            outcome="refund_partial",
        )
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["party_credibility"] <= 0.55
        # With low credibility, confidence should be < 0.95
        assert conf < RT_CONFIDENCE_95
