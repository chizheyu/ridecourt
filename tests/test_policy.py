"""Unit tests for the v2.1 policy engine — every section, table-driven.

Run with: .venv/bin/python -m pytest tests/test_policy.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest

from ridecourt.data import (
    Dispute,
    Facts,
    Fare,
    Profile,
    Trip,
    GeoPoint,
    OptimalRoute,
)
from ridecourt.policy import (
    VALID_SECTIONS,
    compute_confidence,
    compute_dispute_value,
    evaluate,
    section_1_route_deviation,
    section_2_no_show,
    section_3_surge,
    section_4_property_damage,
    section_5_safety,
    section_6_routing,
    NS_CANCELLATION_FEE,
    NS_FREE_WAIT_MIN,
    NS_REPEAT_FLAG_THRESHOLD,
    PD_CLEANING_MAX,
    RD_DEVIATION_THRESHOLD_KM,
    RD_TOLERANCE_KM,
    RT_CONFIDENCE_80,
    RT_CONFIDENCE_95,
    RT_10_TO_50,
    RT_UNDER_10,
    W_CLARITY,
    W_CREDIBILITY,
    W_EVIDENCE,
    W_PRECEDENT,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PER_KM = 0.80  # v2.1 per-km rate


def _fare(**kw) -> Fare:
    defaults = dict(
        base=3.0,
        per_km=PER_KM,
        per_min=0.25,
        distance_km=10.0,
        duration_min=20.0,
        surge_multiplier=1.0,
        promo_code="",
        discount=0.0,
        booking_fee=1.0,
        cancellation_fee=5.0,
        total_charged=14.5,
        quote_surge_multiplier=1.0,
        trip_start_surge_multiplier=1.0,
        surgeable_subtotal=0.0,
        upfront_quote=0.0,
        rider_notified_surge=False,
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
    )
    defaults.update(kw)
    return Facts(**defaults)


def _profile(**kw) -> Profile:
    defaults = dict(
        account_age_days=100,
        rating=4.5,
        trips_completed=50,
        disputes_filed_90d=2,
        disputes_upheld_90d=1,
        fraud_flags=0,
        no_show_cancellations_30d=0,
    )
    defaults.update(kw)
    return Profile(**defaults)


# ---------------------------------------------------------------------------
# §1 Route Deviation — table-driven for every §1.3 row
# ---------------------------------------------------------------------------

class TestSection1RouteDeviation:
    """Table-driven tests for §1.3 refund matrix.

    Rows:
    1. Within 1 km tolerance → no refund
    2. Traffic-justified + rider informed → no refund
    3. Traffic-justified + rider NOT informed → partial refund
    4. NOT traffic-justified → full refund
    """

    @pytest.mark.parametrize("excess_km", [0.0, 0.5, 1.0])
    def test_within_tolerance_no_refund(self, excess_km):
        """§1.3: deviation ≤ 1 km → within tolerance, no refund."""
        f = _facts(distance_actual_km=10.0 + excess_km, distance_optimal_km=10.0)
        r = section_1_route_deviation(f, per_km_rate=PER_KM)
        assert r.outcome == "no_refund_within_tolerance"
        assert r.amount == 0.0
        assert "§1.3" in r.applicable_clauses

    def test_justified_informed_no_refund(self):
        """§1.3 + §1.4: traffic-justified + rider informed → no refund."""
        f = _facts(
            distance_actual_km=14.0,
            distance_optimal_km=10.0,
            traffic_justified=True,
            rider_informed_deviation=True,
        )
        r = section_1_route_deviation(f, per_km_rate=PER_KM)
        assert r.outcome == "no_refund_justified_informed"
        assert r.amount == 0.0
        assert "§1.2" in r.applicable_clauses
        assert "§1.4" in r.applicable_clauses

    def test_justified_not_informed_partial_refund(self):
        """§1.3 + §1.4: traffic-justified + rider NOT informed → partial refund."""
        f = _facts(
            distance_actual_km=14.0,
            distance_optimal_km=10.0,
            traffic_justified=True,
            rider_informed_deviation=False,
        )
        r = section_1_route_deviation(f, per_km_rate=PER_KM)
        assert r.outcome == "refund_partial_excess_distance"
        # 4.0 km excess × S$0.80/km = S$3.20
        assert r.amount == 3.20
        assert "§1.2" in r.applicable_clauses

    def test_not_justified_full_refund(self):
        """§1.3 + §1.4: NOT traffic-justified → full excess distance refund."""
        f = _facts(
            distance_actual_km=14.0,
            distance_optimal_km=10.0,
            traffic_justified=False,
            rider_informed_deviation=False,
        )
        r = section_1_route_deviation(f, per_km_rate=PER_KM)
        assert r.outcome == "refund_full_excess_distance"
        # 4.0 km × S$0.80 = S$3.20
        assert r.amount == 3.20
        assert "§1.3" in r.applicable_clauses
        assert "§1.4" in r.applicable_clauses

    def test_tolls_time_surge_not_refunded(self):
        """§1.3: only excess distance fare is refunded — tolls, time, surge excluded."""
        f = _facts(
            distance_actual_km=14.0,
            distance_optimal_km=10.0,
            duration_actual_min=30.0,
            duration_optimal_min=20.0,
            traffic_justified=False,
        )
        r = section_1_route_deviation(f, per_km_rate=PER_KM)
        # Only 4 km × 0.80 = 3.20, no time component
        assert r.amount == 3.20
        assert "time" in r.computation.lower() or "toll" in r.computation.lower()

    def test_exactly_1km_tolerance(self):
        """Exactly 1.0 km excess → within tolerance (≤, not <)."""
        f = _facts(distance_actual_km=11.0, distance_optimal_km=10.0)
        r = section_1_route_deviation(f, per_km_rate=PER_KM)
        assert r.outcome == "no_refund_within_tolerance"

    def test_just_over_tolerance(self):
        """1.01 km excess → beyond tolerance."""
        f = _facts(
            distance_actual_km=11.01,
            distance_optimal_km=10.0,
            traffic_justified=False,
        )
        r = section_1_route_deviation(f, per_km_rate=PER_KM)
        assert r.outcome != "no_refund_within_tolerance"


# ---------------------------------------------------------------------------
# §2 No-Show & Cancellation — table-driven for every §2 row
# ---------------------------------------------------------------------------

class TestSection2NoShow:
    """Table-driven tests for §2 no-show matrix.

    Rows:
    1. §2.3: both chat AND call met → fee upheld (§2.2)
    2. §2.3: missing call → fee voided
    3. §2.3: missing chat → fee voided
    4. §2.4: driver no-show + no contact → S$5 credit
    5. §2.4: driver no-show + contact attempted → no credit
    6. §2.5: 3+ no-shows → flag account
    """

    def test_both_contacts_met_fee_upheld(self):
        """§2.3 met (≥1 chat AND ≥1 call) → §2.2 fee upheld."""
        f = _facts(
            driver_chat_messages=2,
            driver_call_attempts=1,
            driver_made_contact=True,
        )
        r = section_2_no_show(f, "no_show_rider")
        assert r.outcome == "cancellation_fee_upheld"
        assert r.amount == NS_CANCELLATION_FEE
        assert "§2.2" in r.applicable_clauses
        assert "§2.3" in r.applicable_clauses

    def test_missing_call_fee_voided(self):
        """§2.3: driver has chat but no call → fee voided."""
        f = _facts(
            driver_chat_messages=2,
            driver_call_attempts=0,
            driver_made_contact=False,
        )
        r = section_2_no_show(f, "no_show_rider")
        assert r.outcome == "cancellation_fee_voided"
        assert r.amount == 0.0
        assert "§2.3" in r.applicable_clauses

    def test_missing_chat_fee_voided(self):
        """§2.3: driver has call but no chat → fee voided."""
        f = _facts(
            driver_chat_messages=0,
            driver_call_attempts=1,
            driver_made_contact=False,
        )
        r = section_2_no_show(f, "no_show_rider")
        assert r.outcome == "cancellation_fee_voided"
        assert r.amount == 0.0

    def test_driver_no_show_no_contact_credit(self):
        """§2.4: driver no-show + 0 contact → S$5 credit."""
        f = _facts(
            driver_no_show=True,
            driver_chat_messages=0,
            driver_call_attempts=0,
        )
        r = section_2_no_show(f, "no_show_driver")
        assert r.outcome == "driver_no_show_credit"
        assert r.amount == 5.0
        assert "§2.4" in r.applicable_clauses

    def test_driver_no_show_with_contact_no_credit(self):
        """§2.4: driver no-show but contact attempted → no credit."""
        f = _facts(
            driver_no_show=True,
            driver_chat_messages=1,
            driver_call_attempts=1,
        )
        r = section_2_no_show(f, "no_show_driver")
        assert r.outcome == "driver_no_show_no_credit"
        assert r.amount == 0.0

    def test_repeat_no_show_flag(self):
        """§2.5: 3+ no-show cancellations in 30d → flag."""
        f = _facts(
            driver_chat_messages=1,
            driver_call_attempts=1,
            driver_made_contact=True,
            rider_no_show_cancellations_30d=NS_REPEAT_FLAG_THRESHOLD,
        )
        r = section_2_no_show(f, "no_show_rider")
        actions = [a for a in r.actions if a.get("action") == "flag_account_review"]
        assert len(actions) == 1

    def test_below_repeat_threshold_no_flag(self):
        """§2.5: 2 no-shows → no flag."""
        f = _facts(
            driver_chat_messages=1,
            driver_call_attempts=1,
            driver_made_contact=True,
            rider_no_show_cancellations_30d=2,
        )
        r = section_2_no_show(f, "no_show_rider")
        actions = [a for a in r.actions if a.get("action") == "flag_account_review"]
        assert len(actions) == 0


# ---------------------------------------------------------------------------
# §3 Surge — table-driven for every §3.3 row
# ---------------------------------------------------------------------------

class TestSection3Surge:
    """Table-driven tests for §3.3 surge matrix.

    Rows:
    1. Surge increased + rider NOT notified → refund differential
    2. Surge increased + rider notified → no refund
    3. Surge unchanged → no refund
    4. Surge decreased → no refund (lower fare applies)
    """

    def test_surge_increased_not_notified_refund(self):
        """§3.3: surge increased, rider not notified → full refund of differential."""
        f = _facts(
            surge_increased=True,
            surge_decreased=False,
            quote_surge_multiplier=1.0,
            trip_start_surge_multiplier=1.5,
            surgeable_subtotal=9.0,
            rider_notified_surge=False,
        )
        r = section_3_surge(f)
        assert r.outcome == "refund_surge_differential"
        # 9.0 × (1.5 − 1.0) = 4.50
        assert r.amount == 4.50
        assert "§3.2" in r.applicable_clauses
        assert "§3.3" in r.applicable_clauses
        assert "§3.4" in r.applicable_clauses
        assert "§3.5" in r.applicable_clauses

    def test_surge_increased_notified_no_refund(self):
        """§3.3: surge increased, rider notified → no refund."""
        f = _facts(
            surge_increased=True,
            surge_decreased=False,
            quote_surge_multiplier=1.0,
            trip_start_surge_multiplier=1.5,
            surgeable_subtotal=9.0,
            rider_notified_surge=True,
        )
        r = section_3_surge(f)
        assert r.outcome == "no_surge_refund_notified"
        assert r.amount == 0.0

    def test_surge_unchanged_no_refund(self):
        """§3.3: surge unchanged → quoted fare applies."""
        f = _facts(
            surge_increased=False,
            surge_decreased=False,
            quote_surge_multiplier=1.5,
            trip_start_surge_multiplier=1.5,
        )
        r = section_3_surge(f)
        assert r.outcome == "no_surge_refund"
        assert r.amount == 0.0

    def test_surge_decreased_no_refund(self):
        """§3.3: surge decreased → lower fare applies automatically."""
        f = _facts(
            surge_increased=False,
            surge_decreased=True,
            quote_surge_multiplier=1.5,
            trip_start_surge_multiplier=1.2,
        )
        r = section_3_surge(f)
        assert r.outcome == "no_surge_refund"
        assert r.amount == 0.0

    def test_booking_fee_exempt(self):
        """§3.5: booking fee is exempt from surge differential."""
        f = _facts(
            surge_increased=True,
            surge_decreased=False,
            quote_surge_multiplier=1.0,
            trip_start_surge_multiplier=2.0,
            surgeable_subtotal=10.0,
            rider_notified_surge=False,
        )
        r = section_3_surge(f)
        # 10.0 × (2.0 − 1.0) = 10.00 — booking fee not included
        assert r.amount == 10.00
        assert "§3.5" in r.applicable_clauses


# ---------------------------------------------------------------------------
# §4 Property Damage — table-driven for every §4 row
# ---------------------------------------------------------------------------

class TestSection4PropertyDamage:
    """Table-driven tests for §4 matrix.

    Rows:
    1. Verifiable cleaning claim ≤ S$50 → approved
    2. Not verifiable (external photo) → rejected
    3. Repair claim → human review
    4. Rider dispute → human review (implied by §4.4)
    """

    def test_verifiable_cleaning_approved(self):
        """§4.2 + §4.3: verifiable photo, cleaning ≤ S$50 → approved."""
        f = _facts(
            damage_type="cleaning",
            damage_claim_amount=35.0,
            photo_source="in_app_camera",
            has_timestamp=True,
            submitted_within_24h=True,
            linked_to_trip=True,
        )
        r = section_4_property_damage(f)
        assert r.outcome == "damage_claim_approved"
        assert r.amount == 35.0
        assert "§4.2" in r.applicable_clauses
        assert "§4.3" in r.applicable_clauses

    def test_not_verifiable_rejected(self):
        """§4.2: external upload → not verifiable → rejected."""
        f = _facts(
            damage_type="cleaning",
            damage_claim_amount=35.0,
            photo_source="external_upload",
            has_timestamp=True,
            submitted_within_24h=True,
            linked_to_trip=True,
        )
        r = section_4_property_damage(f)
        assert r.outcome == "claim_rejected_insufficient_evidence"
        assert r.amount == 0.0

    def test_repair_claim_human_review(self):
        """§4.3: repair claim → human review."""
        f = _facts(
            damage_type="repair",
            damage_claim_amount=200.0,
            has_certified_quote=False,
        )
        r = section_4_property_damage(f)
        assert r.outcome == "human_review_repair"
        assert r.routing == "human"

    def test_cleaning_over_50_human_review(self):
        """§4.3: cleaning claim > S$50 → routed to human (§6.2)."""
        f = _facts(
            damage_type="cleaning",
            damage_claim_amount=75.0,
            photo_source="in_app_camera",
            has_timestamp=True,
            submitted_within_24h=True,
            linked_to_trip=True,
        )
        r = section_4_property_damage(f)
        # §4.3 caps cleaning at S$50
        assert r.amount == 50.0

    def test_missing_timestamp_rejected(self):
        """§4.2: no timestamp → not verifiable."""
        f = _facts(
            damage_type="cleaning",
            damage_claim_amount=30.0,
            photo_source="in_app_camera",
            has_timestamp=False,
            submitted_within_24h=True,
            linked_to_trip=True,
        )
        r = section_4_property_damage(f)
        assert r.outcome == "claim_rejected_insufficient_evidence"

    def test_not_linked_to_trip_rejected(self):
        """§4.2: not linked to trip → not verifiable."""
        f = _facts(
            damage_type="cleaning",
            damage_claim_amount=30.0,
            photo_source="in_app_camera",
            has_timestamp=True,
            submitted_within_24h=True,
            linked_to_trip=False,
        )
        r = section_4_property_damage(f)
        assert r.outcome == "claim_rejected_insufficient_evidence"


# ---------------------------------------------------------------------------
# §5 Safety
# ---------------------------------------------------------------------------

class TestSection5Safety:
    def test_safety_always_human(self):
        r = section_5_safety()
        assert r.outcome == "escalate_safety"
        assert r.routing == "human"
        assert r.amount == 0.0
        assert "§5.1" in r.applicable_clauses

    def test_safety_via_evaluate(self):
        trip = _trip()
        d = _dispute(category="safety")
        result = evaluate(d, trip, {}, _facts(), trip.fare)
        assert result.outcome == "escalate_safety"
        assert result.routing == "human"

    def test_safety_indicators_in_claim_escalate(self):
        """§5.2: safety indicators in claim escalate any category."""
        trip = _trip()
        d = _dispute(category="route_deviation")
        f = _facts(safety_indicators_in_claim=True)
        result = evaluate(d, trip, {}, f, trip.fare)
        assert result.routing == "human"
        assert "§5.2" in result.applicable_clauses


# ---------------------------------------------------------------------------
# §6 Routing — table-driven for every §6 row
# ---------------------------------------------------------------------------

class TestSection6Routing:
    """Table-driven tests for §6 routing matrix.

    Rows:
    1. §6.1: dispute value < S$10, high confidence → AI final
    2. §6.1: dispute value S$10–S$50, confidence ≥ 80% → AI final
    3. §6.1: dispute value S$10–S$50, confidence < 80% → human
    4. §6.1: dispute value > S$50 → human
    5. §6.2: safety → human
    6. §6.2: route deviation > 2 km → human
    7. §6.3: fraud flag, confidence < 95% → human
    8. §6.3: fraud flag, confidence ≥ 95% → AI final (if no other reason)
    """

    def test_under_10_high_confidence_ai_final(self):
        """§6.1: dispute value < S$10, high confidence → AI final."""
        f = _facts()
        routing, reason = section_6_routing(
            "fare_error", f, policy_amount=5.0,
            confidence=0.90, confidence_factors={},
        )
        assert routing == "ai_final"

    def test_10_to_50_high_confidence_ai_final(self):
        """§6.1: dispute value S$10–S$50, confidence ≥ 80% → AI final."""
        f = _facts()
        routing, reason = section_6_routing(
            "fare_error", f, policy_amount=25.0,
            confidence=0.85, confidence_factors={},
        )
        assert routing == "ai_final"

    def test_10_to_50_low_confidence_human(self):
        """§6.1: dispute value S$10–S$50, confidence < 80% → human."""
        f = _facts()
        routing, reason = section_6_routing(
            "fare_error", f, policy_amount=25.0,
            confidence=0.70, confidence_factors={},
        )
        assert routing == "human"
        assert "§6.1" in reason

    def test_over_50_human(self):
        """§6.1: dispute value > S$50 → human."""
        f = _facts()
        routing, reason = section_6_routing(
            "fare_error", f, policy_amount=75.0,
            confidence=0.95, confidence_factors={},
        )
        assert routing == "human"
        assert "§6.1" in reason

    def test_safety_always_human(self):
        """§6.2: safety → human."""
        f = _facts()
        routing, reason = section_6_routing(
            "safety", f, policy_amount=0.0,
            confidence=0.99, confidence_factors={},
        )
        assert routing == "human"
        assert "§6.2" in reason

    def test_route_deviation_over_2km_human(self):
        """§6.2: route deviation > 2 km → human."""
        f = _facts(distance_actual_km=13.0, distance_optimal_km=10.0)
        routing, reason = section_6_routing(
            "route_deviation", f, policy_amount=2.40,
            confidence=0.90, confidence_factors={},
        )
        assert routing == "human"
        assert "§6.2" in reason

    def test_fraud_flag_low_confidence_human(self):
        """§6.3: fraud flag, confidence < 95% → human."""
        f = _facts()
        routing, reason = section_6_routing(
            "fare_error", f, policy_amount=5.0,
            confidence=0.90, confidence_factors={},
            has_fraud_flag=True,
        )
        assert routing == "human"
        assert "§6.3" in reason

    def test_strictest_wins(self):
        """Multiple reasons → human (strictest wins)."""
        f = _facts(distance_actual_km=15.0, distance_optimal_km=10.0)
        routing, reason = section_6_routing(
            "route_deviation", f, policy_amount=60.0,
            confidence=0.70, confidence_factors={},
            has_fraud_flag=True,
        )
        assert routing == "human"
        # Should mention multiple §6 sections
        assert "§6.1" in reason
        assert "§6.2" in reason
        assert "§6.3" in reason


# ---------------------------------------------------------------------------
# Confidence computation
# ---------------------------------------------------------------------------

class TestComputeConfidence:
    def test_weights_sum_to_one(self):
        """The four confidence weights must sum to 1.0."""
        total = W_EVIDENCE + W_CLARITY + W_PRECEDENT + W_CREDIBILITY
        assert abs(total - 1.0) < 0.001

    def test_similarity_alone_does_not_raise_confidence(self):
        """A similar precedent whose outcome was not compared does not raise
        confidence; only an agreeing precedent does (test_confidence_factors)."""
        f = _facts()
        conf_with, factors_with = compute_confidence(
            f, "route_deviation", 5.0, has_precedent_match=True,
            precedent_similarity=0.5,
        )
        conf_without, factors_without = compute_confidence(
            f, "route_deviation", 5.0, has_precedent_match=False,
        )
        assert conf_with == conf_without

    def test_fraud_flags_lower_credibility(self):
        """Fraud flags on the winning party lower credibility."""
        f = _facts(rider_fraud_flags=1)
        # outcome="refund..." → rider wins → rider has fraud flag
        conf, factors = compute_confidence(f, "route_deviation", 5.0,
                                            outcome="refund_partial")
        assert factors["party_credibility"] == 0.6
        assert "fraud flag" in factors["credibility_reason"]

    def test_confidence_returns_factors(self):
        """The confidence function returns both confidence and factor dict."""
        f = _facts()
        conf, factors = compute_confidence(f, "fare_error", 5.0)
        assert "evidence_completeness" in factors
        assert "policy_clarity" in factors
        assert "precedent_alignment" in factors
        assert "party_credibility" in factors


# ---------------------------------------------------------------------------
# Dispute value computation
# ---------------------------------------------------------------------------

class TestComputeDisputeValue:
    def test_property_damage_uses_claim_amount(self):
        f = _facts(damage_claim_amount=75.0)
        val = compute_dispute_value("property_damage", f, 0.0)
        # Capped at S$50
        assert val == 50.0

    def test_no_show_uses_cancellation_fee(self):
        f = _facts()
        val = compute_dispute_value("no_show_rider", f, 0.0)
        assert val == NS_CANCELLATION_FEE

    def test_fare_error_uses_policy_amount(self):
        f = _facts()
        val = compute_dispute_value("fare_error", f, 4.50)
        assert val == 4.50

    def test_route_deviation_uses_policy_amount(self):
        f = _facts()
        val = compute_dispute_value("route_deviation", f, 3.20)
        assert val == 3.20


# ---------------------------------------------------------------------------
# VALID_SECTIONS
# ---------------------------------------------------------------------------

class TestValidSections:
    def test_all_major_sections_present(self):
        for s in ["§1", "§2", "§3", "§4", "§5", "§6", "§7"]:
            assert s in VALID_SECTIONS

    def test_subsections_present(self):
        for s in ["§1.3", "§2.3", "§3.3", "§4.2", "§5.1", "§6.1", "§6.2", "§6.3"]:
            assert s in VALID_SECTIONS


# ---------------------------------------------------------------------------
# Top-level evaluate integration
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_safety_escalates(self):
        trip = _trip()
        d = _dispute(category="safety")
        result = evaluate(d, trip, {}, _facts(), trip.fare)
        assert result.outcome == "escalate_safety"
        assert result.routing == "human"

    def test_route_deviation_no_tolerance(self):
        trip = _trip()
        d = _dispute(category="route_deviation")
        f = _facts(
            distance_actual_km=14.0,
            distance_optimal_km=10.0,
            traffic_justified=False,
        )
        result = evaluate(d, trip, {}, f, trip.fare)
        assert result.outcome == "refund_full_excess_distance"
        assert result.amount == 3.20

    def test_fraud_flag_in_routing(self):
        """When the claimant has fraud flags, §6.3 applies."""
        trip = _trip()
        d = _dispute(category="fare_error")
        profiles = {"rider_001": _profile(fraud_flags=1)}
        f = _facts(
            surge_increased=True,
            quote_surge_multiplier=1.0,
            trip_start_surge_multiplier=1.5,
            surgeable_subtotal=9.0,
            rider_notified_surge=False,
        )
        result = evaluate(d, trip, profiles, f, trip.fare)
        assert result.routing == "human"
        # §6.3 fraud flag notice should be in actions
        fraud_actions = [
            a for a in result.actions if a.get("action") == "fraud_flag_notice"
        ]
        assert len(fraud_actions) >= 1

    def test_unknown_category_human(self):
        trip = _trip()
        d = _dispute(category="promo_code")
        result = evaluate(d, trip, {}, _facts(), trip.fare)
        assert result.routing == "human"
