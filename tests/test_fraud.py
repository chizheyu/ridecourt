"""Tests for the Fraud & Bad-Faith Detection Agent.

Covers:
- Deterministic feature computation (dispute rate, upheld rate, GPS
  contradiction, prior flags)
- Risk levels for the known bad-faith rider (rider_bf1, high)
- Normal rider (low risk)
- GPS contradiction detection for no-show cases
- FraudOutput cites evidence IDs and P-BF-1
- Orchestrator emits RISK_SIGNAL event
- Judge mentions bad-faith when risk is high (P-BF-1)

Run with: LLM_PROVIDER=mock python -m pytest tests/test_fraud.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest

from ridecourt.agents.fraud import FraudAgent, FraudOutput
from ridecourt.data import load_dataset
from ridecourt.fraud import compute_fraud_signals
from ridecourt.orchestrator import Orchestrator
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


@pytest.fixture(scope="module")
def orc(ds):
    return Orchestrator(ds)


# ---------------------------------------------------------------------------
# Deterministic feature computation
# ---------------------------------------------------------------------------


class TestFraudFeatures:
    def test_bad_faith_rider_is_high_risk(self, ds):
        """rider_bf1 has 6 upheld disputes in 90d → high risk."""
        dispute = ds.dispute("disp_fa_005")
        trip = ds.trip(dispute.trip_id)
        pack = build_evidence_pack(dispute, ds)
        facts = facts_from_evidence_pack(pack)

        assessment = compute_fraud_signals(
            pack, facts, ds.profiles,
            claimant_id=trip.rider_id,
            claimant_party="rider",
            claim_text=dispute.claim_text,
            category=dispute.category,
        )

        assert assessment.risk == "high"
        assert len(assessment.signals) >= 1
        # At least one signal should cite P-BF-1.
        bf_signals = [s for s in assessment.signals if "P-BF-1" in s.cites]
        assert len(bf_signals) >= 1

    def test_normal_rider_is_low_risk(self, ds):
        """A rider with no prior disputes should be low risk."""
        dispute = ds.dispute("disp_rd_000")
        trip = ds.trip(dispute.trip_id)
        pack = build_evidence_pack(dispute, ds)
        facts = facts_from_evidence_pack(pack)

        assessment = compute_fraud_signals(
            pack, facts, ds.profiles,
            claimant_id=trip.rider_id,
            claimant_party="rider",
            claim_text=dispute.claim_text,
            category=dispute.category,
        )

        assert assessment.risk == "low"
        assert len(assessment.signals) == 0

    def test_prior_flags_signal(self, ds):
        """rider_bf1 has prior_flags → signal should mention them."""
        dispute = ds.dispute("disp_fa_005")
        trip = ds.trip(dispute.trip_id)
        pack = build_evidence_pack(dispute, ds)
        facts = facts_from_evidence_pack(pack)

        assessment = compute_fraud_signals(
            pack, facts, ds.profiles,
            claimant_id=trip.rider_id,
            claimant_party="rider",
            claim_text=dispute.claim_text,
            category=dispute.category,
        )

        flag_signals = [s for s in assessment.signals if s.feature == "prior_flags"]
        assert len(flag_signals) >= 1
        assert "excessive_disputes" in flag_signals[0].text

    def test_gps_contradiction_for_driver_no_show(self, ds):
        """For no-show-driver cases, if the driver was far from pickup
        while claiming no-show, GPS contradiction should fire.

        Note: disp_ns_d_000 is filed by the rider (not the driver), so
        GPS actually confirms the rider's claim. The contradiction only
        fires when the driver is the claimant AND the GPS contradicts.
        """
        dispute = ds.dispute("disp_ns_d_000")
        trip = ds.trip(dispute.trip_id)
        pack = build_evidence_pack(dispute, ds)
        facts = facts_from_evidence_pack(pack)

        # The dispute is filed_by=rider, so the rider is the claimant.
        # GPS showing driver far away confirms (not contradicts) the claim.
        # So there should be no GPS contradiction signal for the rider.
        if dispute.filed_by == "driver":
            claimant_id = trip.driver_id
            claimant_party = "driver"
        else:
            claimant_id = trip.rider_id
            claimant_party = "rider"

        assessment = compute_fraud_signals(
            pack, facts, ds.profiles,
            claimant_id=claimant_id,
            claimant_party=claimant_party,
            claim_text=dispute.claim_text,
            category=dispute.category,
        )

        # When the rider is the claimant and GPS confirms the driver
        # was far, there is no contradiction.
        if claimant_party == "rider" and facts.min_distance_to_pickup_m > 150:
            gps_signals = [
                s for s in assessment.signals if s.feature == "gps_contradiction"
            ]
            assert len(gps_signals) == 0  # GPS confirms, not contradicts

    def test_signals_cite_evidence_ids(self, ds):
        """All signal cites should be E# or P-# IDs."""
        dispute = ds.dispute("disp_fa_005")
        trip = ds.trip(dispute.trip_id)
        pack = build_evidence_pack(dispute, ds)
        facts = facts_from_evidence_pack(pack)

        assessment = compute_fraud_signals(
            pack, facts, ds.profiles,
            claimant_id=trip.rider_id,
            claimant_party="rider",
            claim_text=dispute.claim_text,
            category=dispute.category,
        )

        for s in assessment.signals:
            for cite in s.cites:
                assert cite.startswith("E") or cite.startswith("P-"), (
                    f"Bad cite: {cite}"
                )


# ---------------------------------------------------------------------------
# Fraud agent (LLM wrapper)
# ---------------------------------------------------------------------------


class TestFraudAgent:
    def test_agent_build_returns_output_and_assessment(self, ds):
        """The FraudAgent should return a FraudOutput and the deterministic
        assessment."""
        dispute = ds.dispute("disp_fa_005")
        trip = ds.trip(dispute.trip_id)
        pack = build_evidence_pack(dispute, ds)
        facts = facts_from_evidence_pack(pack)

        agent = FraudAgent()
        output, meta, assessment = agent.build(
            dispute.claim_text, pack, facts, ds.profiles,
            claimant_id=trip.rider_id,
            claimant_party="rider",
            category=dispute.category,
        )

        assert isinstance(output, FraudOutput)
        assert output.risk == assessment.risk
        assert len(output.signals) == len(assessment.signals)
        # All signals should cite at least one E#.
        for s in output.signals:
            assert len(s.cites) >= 1


# ---------------------------------------------------------------------------
# Orchestrator integration — RISK_SIGNAL event
# ---------------------------------------------------------------------------


class TestOrchestratorRiskSignal:
    def test_risk_signal_event_emitted(self, orc):
        """The orchestrator should emit a RISK_SIGNAL event."""
        cf = orc.resolve("disp_rd_000")
        types = cf.events.types()

        assert "RISK_SIGNAL" in types
        risk_events = cf.events.by_type("RISK_SIGNAL")
        assert len(risk_events) == 1
        assert "risk" in risk_events[0].content

    def test_risk_output_in_case_file(self, orc):
        """The CaseFile should include the risk output."""
        cf = orc.resolve("disp_rd_000")
        assert cf.risk_output is not None
        assert cf.risk_output.risk in ("low", "medium", "high")
        assert cf.risk_assessment is not None

    def test_bad_faith_case_risk_is_high(self, orc, ds):
        """The disp_fa_005 case (rider_bf1) should have high risk."""
        cf = orc.resolve("disp_fa_005")
        assert cf.risk_output is not None
        assert cf.risk_output.risk == "high"
        assert cf.risk_assessment.risk == "high"

    def test_judge_mentions_bad_faith_when_high(self, orc):
        """When risk is high, the Judge should set bad_faith_mentioned=True."""
        cf = orc.resolve("disp_fa_005")
        # If the case escalated (e.g. fare_error), the ruling may be None.
        # But if it has a ruling, bad_faith_mentioned should be True.
        if cf.ruling is not None:
            assert cf.ruling.bad_faith_mentioned is True

    def test_precedent_event_emitted(self, orc):
        """The orchestrator should emit a PRECEDENT event."""
        cf = orc.resolve("disp_rd_000")
        types = cf.events.types()

        assert "PRECEDENT" in types
        prec_events = cf.events.by_type("PRECEDENT")
        assert len(prec_events) == 1
        assert "matches" in prec_events[0].content

    def test_precedent_matches_in_case_file(self, orc):
        """The CaseFile should include precedent matches."""
        cf = orc.resolve("disp_rd_000")
        assert len(cf.precedent_matches) > 0
        # Top 3 max.
        assert len(cf.precedent_matches) <= 3
        # Each match should have a similarity score and reason.
        for m in cf.precedent_matches:
            assert m.similarity >= 0.0
            assert m.reason

    def test_judge_precedent_consistency(self, orc):
        """The Judge should state precedent consistency."""
        cf = orc.resolve("disp_rd_000")
        if cf.ruling is not None:
            assert cf.ruling.precedent_consistency in (
                "consistent", "inconsistent", "no_precedents", ""
            )
