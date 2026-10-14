"""Tests for the v2.1 policy engine on the 3 official Ryde cases.

Policy engine tests (direct):
DISP-001 → human (partial refund, S$3.44, deviation > 2 km triggers §6.2)
DISP-002 → AI final (cancellation fee upheld, S$5.00)
DISP-003 → AI final (surge differential refund, S$4.50)

End-to-end orchestrator tests (LLM_PROVIDER=mock):
Each case is run through the full pipeline: evidence pack → advocates →
fraud agent → judge → guard → ruling/escalation.

Run with: .venv/bin/python -m pytest tests/test_v2_official_cases.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest

from ridecourt.data import Dataset
from ridecourt.orchestrator import Orchestrator
from ridecourt.policy import evaluate
from ridecourt.ryde_adapter import RydeDatasetAdapter, RYDE_SAMPLES_DIR
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack

pytestmark = pytest.mark.skipif(
    not RYDE_SAMPLES_DIR.exists(),
    reason="data/ryde_samples/ directory not found",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adapter() -> RydeDatasetAdapter:
    return RydeDatasetAdapter()


@pytest.fixture(scope="module")
def ds(adapter) -> Dataset:
    return adapter.load()


def _resolve(ds: Dataset, dispute_id: str):
    """Build evidence, facts, and evaluate for a case."""
    dispute = ds.dispute(dispute_id)
    trip = ds.trip(dispute.trip_id)
    pack = build_evidence_pack(dispute, ds)
    facts = facts_from_evidence_pack(pack)
    result = evaluate(dispute, trip, ds.profiles, facts, trip.fare)
    return dispute, trip, facts, result


# ---------------------------------------------------------------------------
# DISP-001: route deviation → human
# ---------------------------------------------------------------------------


class TestDisp001:
    """DISP-001: route deviation, partial refund S$3.44, routes to human."""

    def test_routes_to_human(self, ds):
        """DISP-001 must route to human (deviation > 2 km per §6.2)."""
        _, _, _, result = _resolve(ds, "DISP-001")
        assert result.routing == "human"

    def test_outcome_is_refund(self, ds):
        """The policy outcome should be a refund (partial or full excess)."""
        _, _, _, result = _resolve(ds, "DISP-001")
        assert "refund" in result.outcome

    def test_amount_matches_label(self, ds):
        """The refund amount should match the label (S$3.44)."""
        _, _, _, result = _resolve(ds, "DISP-001")
        lbl = ds.label("DISP-001")
        assert result.amount == lbl.amount

    def test_applicable_clauses_are_v21(self, ds):
        """Clauses should be §N.M format, not legacy P-XX-N."""
        _, _, _, result = _resolve(ds, "DISP-001")
        for c in result.applicable_clauses:
            assert c.startswith("§"), f"Expected §N.M citation, got {c}"

    def test_routing_reason_mentions_6_2(self, ds):
        """The routing reason should mention §6.2 (deviation > 2 km)."""
        _, _, _, result = _resolve(ds, "DISP-001")
        assert "§6.2" in result.routing_reason


# ---------------------------------------------------------------------------
# DISP-002: no-show → AI final (fee upheld)
# ---------------------------------------------------------------------------


class TestDisp002:
    """DISP-002: no-show charge, cancellation fee upheld, S$5.00."""

    def test_outcome_fee_upheld(self, ds):
        _, _, _, result = _resolve(ds, "DISP-002")
        assert result.outcome == "cancellation_fee_upheld"

    def test_amount_is_5(self, ds):
        _, _, _, result = _resolve(ds, "DISP-002")
        assert result.amount == 5.0

    def test_clauses_include_2_2_and_2_3(self, ds):
        _, _, _, result = _resolve(ds, "DISP-002")
        assert "§2.2" in result.applicable_clauses
        assert "§2.3" in result.applicable_clauses


# ---------------------------------------------------------------------------
# DISP-003: surge → AI final (refund S$4.50)
# ---------------------------------------------------------------------------


class TestDisp003:
    """DISP-003: surge differential refund, S$4.50."""

    def test_outcome_surge_refund(self, ds):
        _, _, _, result = _resolve(ds, "DISP-003")
        assert result.outcome == "refund_surge_differential"

    def test_amount_is_4_50(self, ds):
        _, _, _, result = _resolve(ds, "DISP-003")
        assert result.amount == 4.50

    def test_clauses_include_3_3(self, ds):
        _, _, _, result = _resolve(ds, "DISP-003")
        assert "§3.3" in result.applicable_clauses
        assert "§3.4" in result.applicable_clauses

    def test_routes_to_ai_final(self, ds):
        """DISP-003 should route to AI final (dispute value < S$10)."""
        _, _, _, result = _resolve(ds, "DISP-003")
        assert result.routing == "ai_final"


# ---------------------------------------------------------------------------
# End-to-end through the full orchestrator pipeline (mock LLM)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def orch(ds) -> Orchestrator:
    """Orchestrator wired to the Ryde sample dataset."""
    return Orchestrator(ds)


class TestE2EDisp001:
    """DISP-001 through the full pipeline: evidence → advocates → judge.

    Expected: routed to human (§6.2: deviation > 2 km), but the Judge still
    issues a ruling as a recommendation with amount S$3.44.
    """

    def test_no_guard_failures(self, orch):
        cf = orch.resolve("DISP-001")
        assert not cf.guard_problems, (
            f"Guard problems on DISP-001: {cf.guard_problems}"
        )

    def test_ruling_present_as_recommendation(self, orch):
        """DISP-001 routes to human, but the ruling is kept as a recommendation."""
        cf = orch.resolve("DISP-001")
        assert cf.ruling is not None, "Ruling should exist as a recommendation"
        assert cf.ruling.amount == 3.44

    def test_escalated_for_human_review(self, orch):
        cf = orch.resolve("DISP-001")
        assert cf.escalated is True
        assert "human" in cf.escalation_reason or "§6" in cf.escalation_reason

    def test_advocates_ran(self, orch):
        cf = orch.resolve("DISP-001")
        assert cf.rider_case is not None
        assert cf.driver_case is not None

    def test_evidence_pack_built(self, orch):
        cf = orch.resolve("DISP-001")
        assert len(cf.evidence_pack.items) > 0

    def test_policy_result_matches(self, orch):
        cf = orch.resolve("DISP-001")
        assert cf.policy_result is not None
        assert "refund" in cf.policy_result.outcome
        assert cf.policy_result.amount == 3.44


class TestE2EDisp002:
    """DISP-002 through the full pipeline.

    Expected: cancellation fee upheld, S$5.00, AI final. The rider has a
    fraud flag (§6.3 raises threshold to 95%), but confidence ≥ 0.95
    (driver is the winning party, highly credible) so the ruling is AI final.
    """

    def test_no_guard_failures(self, orch):
        cf = orch.resolve("DISP-002")
        assert not cf.guard_problems, (
            f"Guard problems on DISP-002: {cf.guard_problems}"
        )

    def test_ruling_present(self, orch):
        cf = orch.resolve("DISP-002")
        assert cf.ruling is not None
        assert cf.ruling.amount == 5.00

    def test_outcome_fee_upheld(self, orch):
        cf = orch.resolve("DISP-002")
        assert cf.policy_result.outcome == "cancellation_fee_upheld"

    def test_advocates_ran(self, orch):
        cf = orch.resolve("DISP-002")
        assert cf.rider_case is not None
        assert cf.driver_case is not None

    def test_fraud_agent_ran(self, orch):
        """DISP-002 rider has fraud_flags=1, so the fraud agent should detect risk."""
        cf = orch.resolve("DISP-002")
        assert cf.risk_output is not None
        assert cf.risk_output.risk in ("medium", "high")

    def test_routing_ai_final(self, orch):
        """DISP-002 routes to AI final (confidence ≥ 0.95 despite fraud flag)."""
        cf = orch.resolve("DISP-002")
        assert not cf.escalated
        assert cf.policy_result.routing == "ai_final"


class TestE2EDisp003:
    """DISP-003 through the full pipeline.

    Expected: surge differential refund, S$4.50, AI final (§6.1: <S$10).
    """

    def test_no_guard_failures(self, orch):
        cf = orch.resolve("DISP-003")
        assert not cf.guard_problems, (
            f"Guard problems on DISP-003: {cf.guard_problems}"
        )

    def test_ruling_present(self, orch):
        cf = orch.resolve("DISP-003")
        assert cf.ruling is not None
        assert cf.ruling.amount == 4.50

    def test_not_escalated(self, orch):
        """DISP-003 should be AI-final (§6.1: dispute value < S$10, no fraud)."""
        cf = orch.resolve("DISP-003")
        assert not cf.escalated, (
            f"DISP-003 should not escalate, reason: {cf.escalation_reason}"
        )

    def test_outcome_surge_refund(self, orch):
        cf = orch.resolve("DISP-003")
        assert cf.policy_result.outcome == "refund_surge_differential"

    def test_advocates_ran(self, orch):
        cf = orch.resolve("DISP-003")
        assert cf.rider_case is not None
        assert cf.driver_case is not None

    def test_ruling_cites_v21_sections(self, orch):
        """The Judge's reasoning should cite §N.M sections, not legacy P-XX."""
        cf = orch.resolve("DISP-003")
        if cf.ruling and cf.ruling.cites:
            has_section = any(
                c.startswith("§") or c.startswith("E") for c in cf.ruling.cites
            )
            assert has_section, (
                f"Ruling cites should include § or E# citations: {cf.ruling.cites}"
            )
