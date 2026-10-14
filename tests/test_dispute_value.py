"""Test §6.1 dispute value computation.

The dispute value is the amount at stake — actual fare minus upfront quote
for route/fare disputes, NOT the engine's computed refund. A S$0 refund
with S$60 at stake must route to human.
"""

import pytest

from ridecourt.data import Facts
from ridecourt.policy import (
    compute_dispute_value,
    RT_UNDER_10,
    RT_10_TO_50,
    section_6_routing,
)


def _facts(**kw) -> Facts:
    defaults = dict(
        charged_total=16.50,
        upfront_quote=12.00,
        recomputed_fare=12.00,
        distance_actual_km=20.0,
        distance_optimal_km=20.0,
        rider_account_age_days=500,
        rider_rating=4.5,
        rider_fraud_flags=0,
        driver_account_age_days=500,
        driver_rating=4.5,
        driver_fraud_flags=0,
    )
    defaults.update(kw)
    return Facts(**defaults)


class TestDisputeValueRouteFare:
    """For route_deviation and fare_error, dispute value = charged - quote."""

    def test_fare_error_uses_charged_minus_quote(self):
        f = _facts(charged_total=16.50, upfront_quote=12.00)
        val = compute_dispute_value("fare_error", f, 4.50)
        assert val == 4.50

    def test_route_deviation_uses_charged_minus_quote(self):
        f = _facts(charged_total=35.00, upfront_quote=28.50)
        val = compute_dispute_value("route_deviation", f, 3.44)
        assert val == 6.50

    def test_zero_refund_high_stake_routes_to_human(self):
        """S$0 refund with S$60 at stake must route to human."""
        f = _facts(
            charged_total=72.00,
            upfront_quote=12.00,
            distance_actual_km=15.0,
            distance_optimal_km=15.0,  # within tolerance → S$0 refund
        )
        val = compute_dispute_value("route_deviation", f, 0.0)
        assert val == 60.0
        assert val > RT_10_TO_50, "S$60 dispute value must trigger human review"

        # Check routing: > S$50 → human
        routing, reason = section_6_routing(
            "route_deviation", f, 0.0,
            confidence=0.90,
            confidence_factors={},
            has_fraud_flag=False,
            has_precedent_match=False,
        )
        assert routing == "human"
        assert "§6.1" in reason

    def test_no_show_uses_cancellation_fee(self):
        f = _facts(charged_total=0.0, upfront_quote=0.0)
        val = compute_dispute_value("no_show_rider", f, 5.0)
        assert val == 5.0

    def test_property_damage_uses_claim_amount(self):
        f = _facts(damage_claim_amount=35.0)
        val = compute_dispute_value("property_damage", f, 35.0)
        assert val == 35.0

    def test_fallback_when_no_quote(self):
        """When no upfront quote, falls back to charged - recomputed."""
        f = _facts(charged_total=16.50, upfront_quote=0.0, recomputed_fare=14.00)
        val = compute_dispute_value("fare_error", f, 2.50)
        assert val == 2.50
