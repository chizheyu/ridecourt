"""Tests for the §6.3 confidence computation.

Each of the four factors (evidence completeness, policy clarity, precedent
alignment, party credibility) is now computed from structured rules, not
constants.  Tests verify each rule on the three official cases (within 0.02)
and on targeted edge cases.

Expected outcomes:
  DISP-001 ~0.975 (clarity 0.9 because rider_informed inferred from chat) → human via §6.2
  DISP-002 1.0   → AI final
  DISP-003 1.0   → AI final
"""

import pytest

from ridecourt.data import Facts, Precedent
from ridecourt.policy import (
    W_EVIDENCE,
    W_CLARITY,
    W_PRECEDENT,
    W_CREDIBILITY,
    RT_CONFIDENCE_95,
    compute_confidence,
    evaluate,
    _numeric_condition_score,
)
from ridecourt.ryde_adapter import RydeDatasetAdapter
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack
from ridecourt.precedent import retrieve_precedents, PrecedentMatch


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
        rider_disputes_upheld_90d=0,
        driver_account_age_days=720,
        driver_rating=4.7,
        driver_fraud_flags=0,
        driver_disputes_upheld_90d=0,
    )
    defaults.update(kw)
    return Facts(**defaults)


# ---------------------------------------------------------------------------
# Factor 1: Evidence completeness
# ---------------------------------------------------------------------------

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

    def test_unknown_category_not_trivially_true(self):
        """Evidence completeness for unknown categories should not be
        trivially true — it checks that a fare or dispute value is recorded."""
        f = _facts(charged_total=0.0)
        conf, factors = compute_confidence(f, "promo_code", 0.0, outcome="no_refund")
        assert factors["evidence_completeness"] < 1.0


# ---------------------------------------------------------------------------
# Factor 2: Policy clarity
# ---------------------------------------------------------------------------

class TestPolicyClarity:
    def test_numeric_condition_score_formula(self):
        """min(1.0, 0.6 + 2×|v−t|/t).  At threshold → 0.6; far past → 1.0."""
        assert _numeric_condition_score(1.0, 1.0) == pytest.approx(0.6)
        assert _numeric_condition_score(2.0, 1.0) == pytest.approx(1.0)
        assert _numeric_condition_score(1.5, 1.0) == pytest.approx(1.0)  # capped
        # Below threshold: |v−t| is large, so score is also high (condition clearly not met)
        assert _numeric_condition_score(0.0, 1.0) == pytest.approx(1.0)  # capped

    def test_surge_not_notified_high_clarity(self):
        """Surge increased, rider not notified: system-record booleans → 1.0."""
        f = _facts(surge_increased=True, rider_notified_surge=False)
        conf, factors = compute_confidence(f, "fare_error", 4.50, outcome="refund")
        assert factors["policy_clarity"] == pytest.approx(1.0, abs=0.02)

    def test_no_surge_high_clarity(self):
        """No surge: only system-record boolean → clarity 1.0."""
        f = _facts(surge_increased=False, fare_difference=0, charged_total=12.0, upfront_quote=12.0)
        conf, factors = compute_confidence(f, "fare_error", 0.0, outcome="no_refund")
        assert factors["policy_clarity"] == pytest.approx(1.0, abs=0.02)

    def test_no_show_zero_contact_high_clarity(self):
        """No-show with zero contact: system-record boolean → 1.0."""
        f = _facts(driver_chat_messages=0, driver_call_attempts=0, driver_made_contact=False)
        conf, factors = compute_confidence(f, "no_show_rider", 0.0, outcome="voided")
        assert factors["policy_clarity"] == pytest.approx(1.0, abs=0.02)

    def test_route_deviation_clarity_min_includes_chat(self):
        """Route deviation: clarity is min over conditions.  rider_informed
        is inferred from chat (free text) → 0.9, so clarity should be 0.9."""
        f = _facts(
            distance_actual_km=20.0,
            distance_optimal_km=15.0,
            traffic_justified=True,
            traffic_incidents_on_route=[{"road": "PIE"}],
            rider_informed_deviation=False,
        )
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["policy_clarity"] == pytest.approx(0.9, abs=0.02)

    def test_shrinking_wait_margin_lowers_clarity(self):
        """Shrinking the wait margin (closer to threshold) lowers clarity."""
        # Wait far past threshold → numeric score closer to 1.0
        f_far = _facts(wait_minutes_past_eta=10.0, driver_made_contact=True)
        _, factors_far = compute_confidence(f_far, "no_show_rider", 5.0, outcome="upheld")

        # Wait barely past threshold → numeric score = 0.6
        f_close = _facts(wait_minutes_past_eta=5.1, driver_made_contact=True)
        _, factors_close = compute_confidence(f_close, "no_show_rider", 5.0, outcome="upheld")

        assert factors_far["policy_clarity"] > factors_close["policy_clarity"]


# ---------------------------------------------------------------------------
# Factor 3: Precedent alignment
# ---------------------------------------------------------------------------

class TestPrecedentAlignment:
    def test_agrees_matching_facts_same_outcome(self):
        """A precedent with matching deciding facts and same outcome → agrees → 1.0."""
        f = _facts()
        prec = PrecedentMatch(
            case_id="P-001",
            category="route_deviation",
            summary="route deviation refund",
            outcome="refund_partial",
            amount=3.0,
            similarity=0.5,
            reason="same category",
            deciding_facts={
                "rider_informed": False,
                "traffic_justified": False,
                "deviation_exceeds_tolerance": True,
            },
        )
        conf, factors = compute_confidence(
            f, "route_deviation", 3.44,
            outcome="refund_partial",
            retrieved_precedents=[prec],
        )
        assert factors["precedent_alignment"] == pytest.approx(1.0, abs=0.02)
        assert "agrees" in factors["precedent_reason"]

    def test_conflict_matching_facts_different_outcome(self):
        """A precedent with matching deciding facts but different outcome → conflict → 0.5."""
        f = _facts()
        prec = PrecedentMatch(
            case_id="P-002",
            category="route_deviation",
            summary="route deviation no refund",
            outcome="no_refund_within_tolerance",
            amount=0.0,
            similarity=0.5,
            reason="same category",
            deciding_facts={
                "rider_informed": False,
                "traffic_justified": False,
                "deviation_exceeds_tolerance": True,
            },
        )
        conf, factors = compute_confidence(
            f, "route_deviation", 3.44,
            outcome="refund_partial",
            retrieved_precedents=[prec],
        )
        assert factors["precedent_alignment"] == pytest.approx(0.5, abs=0.02)
        assert "conflicts" in factors["precedent_reason"]

    def test_distinguished_different_facts(self):
        """A precedent with different deciding facts → distinguished → 1.0
        (neutral: no agreement, no conflict)."""
        f = _facts()
        prec = PrecedentMatch(
            case_id="P-003",
            category="route_deviation",
            summary="route deviation justified",
            outcome="no_refund_justified_informed",
            amount=0.0,
            similarity=0.5,
            reason="same category",
            deciding_facts={
                "rider_informed": True,
                "traffic_justified": True,
                "deviation_exceeds_tolerance": True,
            },
        )
        conf, factors = compute_confidence(
            f, "route_deviation", 3.44,
            outcome="refund_partial",
            retrieved_precedents=[prec],
        )
        assert factors["precedent_alignment"] == pytest.approx(1.0, abs=0.02)
        assert "distinguished" in factors["precedent_reason"]

    def test_no_retrieved_precedents_fallback(self):
        """When no retrieved_precedents provided, fallback to similarity-based."""
        f = _facts()
        conf, factors = compute_confidence(f, "route_deviation", 3.44,
                                           has_precedent_match=True,
                                           precedent_similarity=0.5,
                                           outcome="refund_partial")
        assert factors["precedent_alignment"] == pytest.approx(1.0, abs=0.02)

    def test_no_match_no_retrieved_fallback(self):
        """No retrieved_precedents and no match → 0.85."""
        f = _facts()
        conf, factors = compute_confidence(f, "route_deviation", 3.44,
                                           has_precedent_match=False,
                                           outcome="refund_partial")
        assert factors["precedent_alignment"] == pytest.approx(0.85, abs=0.02)


# ---------------------------------------------------------------------------
# Factor 4: Party credibility
# ---------------------------------------------------------------------------

class TestPartyCredibility:
    def test_no_adverse_signal_high(self):
        """No fraud, no upheld disputes → credibility 1.0."""
        f = _facts(rider_account_age_days=540, rider_rating=4.8, rider_fraud_flags=0)
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["party_credibility"] == pytest.approx(1.0, abs=0.02)

    def test_fraud_flag_on_winning_party_lower(self):
        """Fraud flag on winning party → 0.6."""
        f = _facts(rider_fraud_flags=1)
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["party_credibility"] == pytest.approx(0.6, abs=0.02)
        assert "fraud flag" in factors["credibility_reason"]

    def test_fraud_flag_on_losing_party_no_effect(self):
        """Fraud flag on the losing party doesn't affect credibility."""
        f = _facts(driver_fraud_flags=1, rider_fraud_flags=0)
        # Rider wins (refund) → check rider credibility, driver's fraud doesn't matter
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["party_credibility"] == pytest.approx(1.0, abs=0.02)

    def test_three_upheld_disputes_lower(self):
        """3+ upheld disputes same type in 90d on winning party → 0.6."""
        f = _facts(rider_disputes_upheld_90d=3, rider_fraud_flags=0)
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["party_credibility"] == pytest.approx(0.6, abs=0.02)
        assert "upheld" in factors["credibility_reason"].lower()

    def test_new_account_alone_no_effect(self):
        """A new account with no other signal does NOT lower credibility."""
        f = _facts(rider_account_age_days=2, rider_rating=3.0, rider_fraud_flags=0,
                    rider_disputes_upheld_90d=0)
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["party_credibility"] == pytest.approx(1.0, abs=0.02)

    def test_driver_wins_driver_credibility(self):
        """When the driver wins (fee upheld), use driver credibility."""
        f = _facts(driver_account_age_days=900, driver_rating=4.9, driver_fraud_flags=0)
        conf, factors = compute_confidence(f, "no_show_rider", 5.0, outcome="upheld")
        assert factors["party_credibility"] == pytest.approx(1.0, abs=0.02)
        assert "driver" in factors["credibility_reason"]


# ---------------------------------------------------------------------------
# Official Ryde cases — confidence and routing
# ---------------------------------------------------------------------------

class TestOfficialCasesConfidence:
    """Test the three official Ryde cases for expected confidence and routing.

    Expected:
      DISP-001 ~0.975 (clarity 0.9 because rider_informed inferred from chat) → human via §6.2
      DISP-002 1.0   → AI final
      DISP-003 1.0   → AI final
    """

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
                        precedent_similarity=best_sim,
                        retrieved_precedents=precedents)

    def test_disp001_confidence_value(self, ds):
        """DISP-001 confidence ≈ 0.975 (within 0.02)."""
        pr = self._evaluate(ds, "DISP-001")
        assert pr.confidence == pytest.approx(0.975, abs=0.02)

    def test_disp001_routing_human(self, ds):
        """DISP-001 routes to human via §6.2 (excess > 2 km)."""
        pr = self._evaluate(ds, "DISP-001")
        assert pr.routing == "human"
        assert "§6.2" in pr.routing_reason

    def test_disp001_clarity_factor(self, ds):
        """DISP-001 clarity = 0.9 (rider_informed inferred from chat → 0.9)."""
        pr = self._evaluate(ds, "DISP-001")
        f = pr.confidence_factors
        assert f["policy_clarity"] == pytest.approx(0.9, abs=0.02)
        assert "rider_informed" in f["policy_clarity_reason"]
        assert "chat" in f["policy_clarity_reason"].lower()

    def test_disp001_all_factors(self, ds):
        """DISP-001: all four confidence factors with reasons."""
        pr = self._evaluate(ds, "DISP-001")
        f = pr.confidence_factors
        assert f["evidence_completeness"] == 1.0
        assert "all key evidence fields present" in f["evidence_reason"]
        assert f["policy_clarity"] == pytest.approx(0.9, abs=0.02)
        assert f["precedent_alignment"] >= 0.80
        assert "winning party" in f["credibility_reason"]

    def test_disp002_confidence_value(self, ds):
        """DISP-002 confidence = 1.0 (within 0.02)."""
        pr = self._evaluate(ds, "DISP-002")
        assert pr.confidence == pytest.approx(1.0, abs=0.02)

    def test_disp002_routing_ai_final(self, ds):
        """DISP-002 AI final despite fraud flag."""
        pr = self._evaluate(ds, "DISP-002")
        assert pr.routing == "ai_final"

    def test_disp002_all_factors(self, ds):
        """DISP-002: all four confidence factors with reasons."""
        pr = self._evaluate(ds, "DISP-002")
        f = pr.confidence_factors
        assert f["evidence_completeness"] == 1.0
        assert f["policy_clarity"] == pytest.approx(1.0, abs=0.02)
        assert "contact" in f["policy_clarity_reason"].lower()
        assert f["precedent_alignment"] >= 0.80
        assert "winning party" in f["credibility_reason"]

    def test_disp003_confidence_value(self, ds):
        """DISP-003 confidence = 1.0 (within 0.02)."""
        pr = self._evaluate(ds, "DISP-003")
        assert pr.confidence == pytest.approx(1.0, abs=0.02)

    def test_disp003_routing_ai_final(self, ds):
        """DISP-003 AI final."""
        pr = self._evaluate(ds, "DISP-003")
        assert pr.routing == "ai_final"

    def test_disp003_all_factors(self, ds):
        """DISP-003: all four confidence factors with reasons."""
        pr = self._evaluate(ds, "DISP-003")
        f = pr.confidence_factors
        assert f["evidence_completeness"] == 1.0
        assert f["policy_clarity"] == pytest.approx(1.0, abs=0.02)
        assert "surge" in f["policy_clarity_reason"].lower()
        assert f["precedent_alignment"] >= 0.80
        assert "winning party" in f["credibility_reason"]


# ---------------------------------------------------------------------------
# Fraud flag routing
# ---------------------------------------------------------------------------

class TestFraudFlagRoutingToHuman:
    """A case where the ruling favours a fraud-flagged party must route
    to human (credibility 0.6 → confidence < 0.95)."""

    def test_fraud_flag_winning_party_routes_human(self):
        """When the winning party has a fraud flag, credibility drops to 0.6
        and confidence falls below 0.95."""
        f = _facts(
            rider_fraud_flags=1,
            rider_account_age_days=10,
            rider_rating=3.5,
        )
        conf, factors = compute_confidence(f, "route_deviation", 3.44, outcome="refund_partial")
        assert factors["party_credibility"] == pytest.approx(0.6, abs=0.02)
        assert conf < RT_CONFIDENCE_95
