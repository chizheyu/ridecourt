"""Table-driven tests for v2.1 §6 routing rules.

Tests every row of §6.1, §6.2, and §6.3, plus the strictest-wins principle.

Run with: .venv/bin/python -m pytest tests/test_v2_routing.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest

from ridecourt.data import Facts
from ridecourt.policy import (
    compute_confidence,
    compute_dispute_value,
    section_6_routing,
    NS_CANCELLATION_FEE,
    PD_CLEANING_MAX,
    RD_DEVIATION_THRESHOLD_KM,
    RT_CONFIDENCE_80,
    RT_CONFIDENCE_95,
    RT_10_TO_50,
    RT_UNDER_10,
)


def _facts(**kw) -> Facts:
    defaults = dict(
        distance_actual_km=10.0,
        distance_optimal_km=10.0,
    )
    defaults.update(kw)
    return Facts(**defaults)


# ---------------------------------------------------------------------------
# §6.1 Monetary thresholds
# ---------------------------------------------------------------------------


class TestSection61Monetary:
    """§6.1: Under S$10 → AI final; S$10–S$50 needs 80%+; >S$50 → human."""

    def test_under_10_ai_final(self):
        f = _facts()
        routing, _ = section_6_routing(
            "fare_error", f, policy_amount=5.0,
            confidence=0.90, confidence_factors={},
        )
        assert routing == "ai_final"

    def test_exactly_10_ai_final_with_high_confidence(self):
        """S$10.00 is in the S$10–S$50 range; needs 80%+ confidence."""
        f = _facts()
        routing, _ = section_6_routing(
            "fare_error", f, policy_amount=10.0,
            confidence=0.85, confidence_factors={},
        )
        assert routing == "ai_final"

    def test_10_to_50_high_confidence_ai_final(self):
        f = _facts()
        routing, _ = section_6_routing(
            "fare_error", f, policy_amount=25.0,
            confidence=0.80, confidence_factors={},
        )
        assert routing == "ai_final"

    def test_10_to_50_low_confidence_human(self):
        f = _facts()
        routing, reason = section_6_routing(
            "fare_error", f, policy_amount=25.0,
            confidence=0.79, confidence_factors={},
        )
        assert routing == "human"
        assert "§6.1" in reason

    def test_over_50_human(self):
        f = _facts()
        routing, reason = section_6_routing(
            "fare_error", f, policy_amount=51.0,
            confidence=0.99, confidence_factors={},
        )
        assert routing == "human"
        assert "§6.1" in reason

    def test_exactly_50_human_if_low_confidence(self):
        """S$50.00 is within the S$10–S$50 range, needs 80%+ confidence."""
        f = _facts()
        routing, _ = section_6_routing(
            "fare_error", f, policy_amount=50.0,
            confidence=0.80, confidence_factors={},
        )
        assert routing == "ai_final"

    def test_exactly_51_human(self):
        """S$51.00 is above the S$50 threshold."""
        f = _facts()
        routing, _ = section_6_routing(
            "fare_error", f, policy_amount=51.0,
            confidence=0.99, confidence_factors={},
        )
        assert routing == "human"


# ---------------------------------------------------------------------------
# §6.2 Category-specific thresholds
# ---------------------------------------------------------------------------


class TestSection62Category:
    """§6.2: safety → human; route deviation ≤2 km + 80% confidence; repair → human."""

    def test_safety_always_human(self):
        f = _facts()
        routing, reason = section_6_routing(
            "safety", f, policy_amount=0.0,
            confidence=0.99, confidence_factors={},
        )
        assert routing == "human"
        assert "§6.2" in reason

    def test_route_deviation_under_2km_ai_final(self):
        f = _facts(distance_actual_km=11.0, distance_optimal_km=10.0)
        routing, _ = section_6_routing(
            "route_deviation", f, policy_amount=0.80,
            confidence=0.85, confidence_factors={},
        )
        assert routing == "ai_final"

    def test_route_deviation_over_2km_human(self):
        f = _facts(distance_actual_km=13.0, distance_optimal_km=10.0)
        routing, reason = section_6_routing(
            "route_deviation", f, policy_amount=2.40,
            confidence=0.90, confidence_factors={},
        )
        assert routing == "human"
        assert "§6.2" in reason

    def test_route_deviation_exactly_2km_ai_final(self):
        """Exactly 2.0 km excess → not strictly > 2 km → AI final."""
        f = _facts(distance_actual_km=12.0, distance_optimal_km=10.0)
        routing, _ = section_6_routing(
            "route_deviation", f, policy_amount=1.60,
            confidence=0.85, confidence_factors={},
        )
        assert routing == "ai_final"

    def test_route_deviation_low_confidence_human(self):
        """Route deviation with confidence < 80% → human (§6.2)."""
        f = _facts(distance_actual_km=11.0, distance_optimal_km=10.0)
        routing, reason = section_6_routing(
            "route_deviation", f, policy_amount=0.80,
            confidence=0.70, confidence_factors={},
        )
        assert routing == "human"
        assert "§6.2" in reason

    def test_repair_damage_always_human(self):
        f = _facts(damage_type="repair", damage_claim_amount=200.0)
        routing, reason = section_6_routing(
            "property_damage", f, policy_amount=0.0,
            confidence=0.90, confidence_factors={},
        )
        assert routing == "human"
        assert "§6.2" in reason

    def test_damage_over_50_human(self):
        f = _facts(damage_type="cleaning", damage_claim_amount=75.0)
        routing, reason = section_6_routing(
            "property_damage", f, policy_amount=50.0,
            confidence=0.90, confidence_factors={},
        )
        assert routing == "human"
        assert "§6.2" in reason


# ---------------------------------------------------------------------------
# §6.3 Fraud flag
# ---------------------------------------------------------------------------


class TestSection63Fraud:
    """§6.3: fraud flag ≥1 → confidence threshold raised to 95%."""

    def test_fraud_flag_low_confidence_human(self):
        f = _facts()
        routing, reason = section_6_routing(
            "fare_error", f, policy_amount=5.0,
            confidence=0.90, confidence_factors={},
            has_fraud_flag=True,
        )
        assert routing == "human"
        assert "§6.3" in reason

    def test_fraud_flag_high_confidence_ai_final(self):
        """Fraud flag but confidence ≥ 95% and value < S$10 → AI final."""
        f = _facts()
        routing, _ = section_6_routing(
            "fare_error", f, policy_amount=5.0,
            confidence=0.96, confidence_factors={},
            has_fraud_flag=True,
        )
        assert routing == "ai_final"

    def test_fraud_flag_exactly_95_ai_final(self):
        """Exactly 95% confidence with fraud flag → AI final (≥, not >)."""
        f = _facts()
        routing, _ = section_6_routing(
            "fare_error", f, policy_amount=5.0,
            confidence=0.95, confidence_factors={},
            has_fraud_flag=True,
        )
        assert routing == "ai_final"

    def test_no_fraud_flag_normal_threshold(self):
        """Without fraud flag, 80% confidence is sufficient for S$10–S$50."""
        f = _facts()
        routing, _ = section_6_routing(
            "fare_error", f, policy_amount=15.0,
            confidence=0.80, confidence_factors={},
            has_fraud_flag=False,
        )
        assert routing == "ai_final"


# ---------------------------------------------------------------------------
# Strictest rule wins
# ---------------------------------------------------------------------------


class TestStrictestWins:
    """When multiple §6 rules trigger, the strictest (human) wins."""

    def test_multiple_reasons_human(self):
        f = _facts(distance_actual_km=15.0, distance_optimal_km=10.0)
        routing, reason = section_6_routing(
            "route_deviation", f, policy_amount=60.0,
            confidence=0.70, confidence_factors={},
            has_fraud_flag=True,
        )
        assert routing == "human"
        # Should mention §6.1, §6.2, and §6.3
        assert "§6.1" in reason
        assert "§6.2" in reason
        assert "§6.3" in reason

    def test_one_reason_is_enough(self):
        """Even a single §6 reason escalates to human."""
        f = _facts()
        routing, _ = section_6_routing(
            "fare_error", f, policy_amount=100.0,
            confidence=0.99, confidence_factors={},
        )
        assert routing == "human"

    def test_no_reasons_ai_final(self):
        """When no §6 rule triggers → AI final."""
        f = _facts()
        routing, _ = section_6_routing(
            "fare_error", f, policy_amount=5.0,
            confidence=0.90, confidence_factors={},
        )
        assert routing == "ai_final"


# ---------------------------------------------------------------------------
# Dispute value computation
# ---------------------------------------------------------------------------


class TestDisputeValue:
    def test_property_damage_capped_at_50(self):
        f = _facts(damage_claim_amount=200.0)
        val = compute_dispute_value("property_damage", f, 0.0)
        assert val == PD_CLEANING_MAX

    def test_no_show_uses_cancellation_fee(self):
        f = _facts()
        val = compute_dispute_value("no_show_rider", f, 0.0)
        assert val == NS_CANCELLATION_FEE

    def test_no_show_driver_uses_cancellation_fee(self):
        f = _facts()
        val = compute_dispute_value("no_show_driver", f, 0.0)
        assert val == NS_CANCELLATION_FEE

    def test_fare_error_uses_policy_amount(self):
        f = _facts()
        val = compute_dispute_value("fare_error", f, 4.50)
        assert val == 4.50

    def test_route_deviation_uses_policy_amount(self):
        f = _facts()
        val = compute_dispute_value("route_deviation", f, 3.20)
        assert val == 3.20
