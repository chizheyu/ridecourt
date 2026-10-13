"""Tests for the orchestrator — end-to-end pipeline with ``LLM_PROVIDER=mock``.

Covers:
- Route deviation case ends with the label's outcome and amount
- Requested-stop case ends with no refund
- No-show driver case ends with the label's outcome and amount
- Safety case escalates
- Guard failure leads to a re-ask, then escalation on second failure
- Event log has the expected event sequence

Run with: .venv/bin/python -m pytest tests/test_orchestrator.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest

from ridecourt.agents.advocate import AdvocateOutput, Argument
from ridecourt.agents.judge import JudgeOutput
from ridecourt.data import load_dataset
from ridecourt.events import EventLog
from ridecourt.guard import Problem
from ridecourt.orchestrator import Orchestrator, CaseFile


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
# Route deviation case — ruling matches the label
# ---------------------------------------------------------------------------


class TestRouteDeviationCase:
    def test_ruling_matches_label(self, orc, ds):
        cf = orc.resolve("disp_rd_000")
        label = ds.label("disp_rd_000")

        assert not cf.escalated
        assert cf.ruling is not None
        assert cf.ruling.outcome == label.outcome
        assert abs(cf.ruling.amount - label.amount) < 0.01
        assert cf.ruling.confidence >= 0.7
        assert cf.policy_result.outcome == label.outcome
        assert abs(cf.policy_result.amount - label.amount) < 0.01

    def test_ruling_has_cites(self, orc):
        cf = orc.resolve("disp_rd_000")
        assert cf.ruling is not None
        assert len(cf.ruling.cites) >= 1
        # All cites should be valid E# or P-# IDs.
        for cite in cf.ruling.cites:
            assert cite.startswith("E") or cite.startswith("P-")

    def test_ruling_has_messages(self, orc):
        cf = orc.resolve("disp_rd_000")
        assert cf.ruling is not None
        assert cf.ruling.message_to_rider
        assert cf.ruling.message_to_driver

    def test_advocate_cases_built(self, orc):
        cf = orc.resolve("disp_rd_000")
        assert cf.rider_case is not None
        assert cf.driver_case is not None
        assert len(cf.rider_case.arguments) >= 1
        assert len(cf.driver_case.arguments) >= 1
        # Each argument should cite at least one E#.
        for arg in cf.rider_case.arguments:
            assert len(arg.cites) >= 1
        for arg in cf.driver_case.arguments:
            assert len(arg.cites) >= 1

    def test_rebuttals_built(self, orc):
        cf = orc.resolve("disp_rd_000")
        assert cf.rider_rebuttal is not None
        assert cf.driver_rebuttal is not None


# ---------------------------------------------------------------------------
# Requested-stop case — ends with no refund
# ---------------------------------------------------------------------------


class TestRequestedStopCase:
    def test_no_refund(self, orc, ds):
        cf = orc.resolve("disp_rd_008")
        label = ds.label("disp_rd_008")

        assert label.outcome == "no_refund_rider_requested"
        assert not cf.escalated
        assert cf.ruling is not None
        assert cf.ruling.outcome == "no_refund_rider_requested"
        assert cf.ruling.amount == 0.0

    def test_policy_says_rider_requested(self, orc):
        cf = orc.resolve("disp_rd_008")
        assert cf.policy_result.outcome == "no_refund_rider_requested"
        assert "P-RD-2" in cf.policy_result.applicable_clauses


# ---------------------------------------------------------------------------
# No-show driver case
# ---------------------------------------------------------------------------


class TestNoShowDriverCase:
    def test_ruling_matches_label(self, orc, ds):
        cf = orc.resolve("disp_ns_d_000")
        label = ds.label("disp_ns_d_000")

        assert label.outcome == "refund_and_credit"
        assert not cf.escalated
        assert cf.ruling is not None
        assert cf.ruling.outcome == label.outcome
        assert abs(cf.ruling.amount - label.amount) < 0.01
        assert cf.ruling.amount > 0  # refund + credit


# ---------------------------------------------------------------------------
# No-show rider case
# ---------------------------------------------------------------------------


class TestNoShowRiderCase:
    def test_cancellation_fee_stands(self, orc, ds):
        """disp_ns_r_000: driver waited 6 min within 150m, 3 contact
        attempts → fee stands (P-NS-2 uses wait_minutes_within_150m)."""
        cf = orc.resolve("disp_ns_r_000")
        label = ds.label("disp_ns_r_000")

        assert label.outcome == "cancellation_fee_stands"
        assert not cf.escalated
        assert cf.ruling is not None
        assert cf.ruling.outcome == label.outcome
        assert cf.ruling.amount == label.amount

    def test_cancellation_fee_invalid(self, orc, ds):
        """disp_ns_r_006: driver didn't wait long enough or no contact
        → fee invalid."""
        cf = orc.resolve("disp_ns_r_006")
        label = ds.label("disp_ns_r_006")

        assert label.outcome == "cancellation_fee_invalid"
        assert not cf.escalated
        assert cf.ruling is not None
        assert cf.ruling.outcome == label.outcome
        assert cf.ruling.amount == 0.0


# ---------------------------------------------------------------------------
# Safety case — always escalates
# ---------------------------------------------------------------------------


class TestSafetyEscalates:
    def test_safety_escalates(self, orc):
        cf = orc.resolve("disp_sf_000")

        assert cf.escalated
        assert cf.escalation_reason == "safety_category"
        assert cf.ruling is None  # no ruling for safety

    def test_safety_skips_advocates(self, orc):
        cf = orc.resolve("disp_sf_000")
        # Safety short-circuits before advocates run.
        assert cf.rider_case is None
        assert cf.driver_case is None

    def test_safety_has_policy_result(self, orc):
        cf = orc.resolve("disp_sf_000")
        assert cf.policy_result is not None
        assert cf.policy_result.outcome == "escalate"
        assert "P-SF-1" in cf.policy_result.applicable_clauses


# ---------------------------------------------------------------------------
# Guard failure — re-ask once, then escalate
# ---------------------------------------------------------------------------


class TestGuardFailure:
    def test_guard_failure_reask_then_escalation(self, ds, monkeypatch):
        """When the mock LLM always produces a forbidden word, the guard
        should re-ask once, fail again, and the case escalates.
        """
        import ridecourt.orchestrator as orc_mod
        from ridecourt.llm import LLMMeta

        # Build an AdvocateOutput with a forbidden word so the guard catches it.
        bad_output = AdvocateOutput(
            position="This is guaranteed to be a refund based on E1.",
            arguments=[
                Argument(text="Evidence E1 shows deviation.", cites=["E1"]),
            ],
            requested_outcome="Refund S$5.52.",
        )

        # Patch the guard to always flag the forbidden word.
        original_check = orc_mod.guard_check

        def always_fail(output, pack, policy_ids, policy_result=None, **kwargs):
            return [Problem(
                kind="forbidden_word",
                detail="guaranteed found",
                field="position",
            )]

        monkeypatch.setattr(orc_mod, "guard_check", always_fail)

        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_000")

        assert cf.escalated
        assert "guard" in cf.escalation_reason
        # Guard problems recorded.
        assert len(cf.guard_problems) > 0

    def test_guard_reask_succeeds_on_second_try(self, ds, monkeypatch):
        """If the guard fails the first time but passes on re-ask, the case
        should proceed normally (not escalate)."""
        import ridecourt.orchestrator as orc_mod

        call_count = {"n": 0}

        original_check = orc_mod.guard_check

        def fail_once(output, pack, policy_ids, policy_result=None, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [Problem(
                    kind="unknown_cite",
                    detail="first call fails",
                    cite="E99",
                    field="position",
                )]
            return original_check(output, pack, policy_ids, policy_result=policy_result, **kwargs)

        monkeypatch.setattr(orc_mod, "guard_check", fail_once)

        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_000")

        # Second call succeeded → no escalation.
        assert not cf.escalated
        assert cf.ruling is not None


# ---------------------------------------------------------------------------
# Rebuttal guard rejection — drop rebuttal, continue to Judge
# ---------------------------------------------------------------------------


class TestRebuttalGuardRejection:
    """When a rebuttal fails the guard after one re-ask, it is dropped with a
    GUARD_REJECTED event and the case continues to the Judge with the original
    claims.  Escalation happens only for initial-claim or Judge-ruling guard
    failures, not for rebuttals."""

    def test_rebuttal_guard_failure_drops_rebuttal(self, ds, monkeypatch):
        """A rebuttal that always fails the guard is dropped (set to None),
        a GUARD_REJECTED event is emitted, and the case is NOT escalated."""
        import ridecourt.orchestrator as orc_mod

        original_check = orc_mod.guard_check

        call_state = {"n": 0}

        def fail_rebuttals(output, pack, policy_ids, policy_result=None, **kwargs):
            call_state["n"] += 1
            # Calls 1-2: rider + driver initial claims → pass.
            # Calls 3-4: rider + driver rebuttals (first try) → fail.
            # Calls 5-6: rider + driver rebuttals (re-ask) → fail again.
            # Call 7+: judge output → pass.
            if call_state["n"] >= 3 and call_state["n"] <= 6:
                return [Problem(
                    kind="forbidden_word",
                    detail="guaranteed found in rebuttal",
                    field="position",
                )]
            return original_check(output, pack, policy_ids, policy_result=policy_result, **kwargs)

        monkeypatch.setattr(orc_mod, "guard_check", fail_rebuttals)

        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_000")

        # Case is NOT escalated — rebuttal guard failure does not escalate.
        assert not cf.escalated
        assert cf.ruling is not None
        # Rebuttals are dropped (None).
        assert cf.rider_rebuttal is None
        assert cf.driver_rebuttal is None
        # Original claims are still present.
        assert cf.rider_case is not None
        assert cf.driver_case is not None
        # GUARD_REJECTED events were emitted.
        types = cf.events.types()
        rejected = cf.events.by_type("GUARD_REJECTED")
        assert len(rejected) == 2  # one per advocate
        assert "GUARD_REJECTED" in types
        # Each GUARD_REJECTED shows the problem.
        for ev in rejected:
            assert ev.content["stage"] == "rebuttal"
            assert len(ev.content["problems"]) >= 1
        # Case ends with a RULING, not an ESCALATION.
        assert types[-1] == "RULING"

    def test_rebuttal_guard_reask_then_pass_continues(self, ds, monkeypatch):
        """If the rebuttal guard fails the first time but passes on re-ask,
        the rebuttal is kept and the case proceeds normally."""
        import ridecourt.orchestrator as orc_mod

        original_check = orc_mod.guard_check

        call_state = {"n": 0}

        def fail_rebuttal_once(output, pack, policy_ids, policy_result=None, **kwargs):
            call_state["n"] += 1
            # Calls 3 and 5: first rebuttal try → fail.
            # Calls 4 and 6: re-ask → pass.
            if call_state["n"] in (3, 5):
                return [Problem(
                    kind="unknown_cite",
                    detail="first rebuttal try fails",
                    cite="E99",
                    field="position",
                )]
            return original_check(output, pack, policy_ids, policy_result=policy_result, **kwargs)

        monkeypatch.setattr(orc_mod, "guard_check", fail_rebuttal_once)

        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_000")

        assert not cf.escalated
        assert cf.ruling is not None
        # Rebuttals were kept (re-ask passed).
        assert cf.rider_rebuttal is not None
        assert cf.driver_rebuttal is not None
        # No GUARD_REJECTED events — re-ask succeeded.
        rejected = cf.events.by_type("GUARD_REJECTED")
        assert len(rejected) == 0

    def test_one_rebuttal_fails_other_passes(self, ds, monkeypatch):
        """If only one rebuttal fails (both tries) but the other passes,
        the failing rebuttal is dropped and the case continues to the Judge."""
        # Directly patch _rebuttal_guarded to fail for the rider side only.
        original_rebuttal = Orchestrator._rebuttal_guarded

        def fail_one_rebuttal(self, advocate, claim_text, pack, opponent,
                              policy_result, policy_ids):
            if advocate.side == "rider":
                # Always fail — both tries.
                from ridecourt.llm import LLMMeta
                meta = LLMMeta(
                    provider="mock", model="mock",
                    agent_name="rider_advocate", tokens=100, latency_ms=0,
                )
                return None, meta, [Problem(
                    kind="forbidden_word",
                    detail="guaranteed found",
                    field="position",
                )], None
            return original_rebuttal(
                self, advocate, claim_text, pack, opponent,
                policy_result, policy_ids,
            )

        monkeypatch.setattr(Orchestrator, "_rebuttal_guarded", fail_one_rebuttal)

        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_000")

        # Case continues to Judge — not escalated.
        assert not cf.escalated
        assert cf.ruling is not None
        # Rider rebuttal dropped, driver rebuttal kept.
        assert cf.rider_rebuttal is None
        assert cf.driver_rebuttal is not None
        # One GUARD_REJECTED event for the rider.
        rejected = cf.events.by_type("GUARD_REJECTED")
        assert len(rejected) == 1
        assert rejected[0].content["agent"] == "rider_advocate"
        assert rejected[0].content["stage"] == "rebuttal"

    def test_guard_escalation_only_for_initial_claims(self, ds, monkeypatch):
        """Guard failure on initial claims escalates (guard_failure_advocate),
        even if rebuttals would have been fine."""
        import ridecourt.orchestrator as orc_mod

        original_check = orc_mod.guard_check

        call_state = {"n": 0}

        def fail_claims_only(output, pack, policy_ids, policy_result=None, **kwargs):
            call_state["n"] += 1
            # Calls 1-2: initial claims → always fail (both tries).
            # Since both claims fail, the case escalates before rebuttals run.
            if call_state["n"] <= 2:
                return [Problem(
                    kind="forbidden_word",
                    detail="guaranteed found in claim",
                    field="position",
                )]
            return original_check(output, pack, policy_ids, policy_result=policy_result, **kwargs)

        monkeypatch.setattr(orc_mod, "guard_check", fail_claims_only)

        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_000")

        assert cf.escalated
        assert cf.escalation_reason == "guard_failure_advocate"
        # No ruling — case was escalated.
        assert cf.ruling is None

    def test_guard_escalation_on_judge_ruling(self, ds, monkeypatch):
        """Guard failure on the Judge's ruling escalates
        (guard_failure_judge), not dropped."""
        import ridecourt.orchestrator as orc_mod

        original_check = orc_mod.guard_check

        # The mock LLM produces clean advocate output, so claims and
        # rebuttals pass on the first try: 2 claim calls + 2 rebuttal calls
        # = 4 guard calls before the Judge.  The Judge's first guard call
        # is #5; its re-ask is #6.
        call_state = {"n": 0}

        def fail_judge_only(output, pack, policy_ids, policy_result=None, **kwargs):
            call_state["n"] += 1
            # Calls 1-4: advocates + rebuttals → pass.
            # Call 5+: judge output → always fail.
            if call_state["n"] >= 5:
                return [Problem(
                    kind="forbidden_word",
                    detail="guaranteed found in judge ruling",
                    field="reasoning",
                )]
            return original_check(output, pack, policy_ids, policy_result=policy_result, **kwargs)

        monkeypatch.setattr(orc_mod, "guard_check", fail_judge_only)

        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_000")

        assert cf.escalated
        assert cf.escalation_reason == "guard_failure_judge"
        assert len(cf.guard_problems) > 0

        # The ESCALATION event content must include the guard problems so
        # a reviewer can see why the case was escalated.
        esc_events = cf.events.by_type("ESCALATION")
        assert len(esc_events) >= 1
        content = esc_events[-1].content
        assert content["reason"] == "guard_failure_judge"
        assert "problems" in content
        assert len(content["problems"]) > 0
        assert "first_round_problems" in content
        assert "judge_output" in content


# ---------------------------------------------------------------------------
# Event log sequence
# ---------------------------------------------------------------------------


class TestEventLogSequence:
    def test_normal_case_event_sequence(self, orc):
        cf = orc.resolve("disp_rd_000")
        types = cf.events.types()

        # Normal flow: DISPUTE_FILED → EVIDENCE_REQUEST → EVIDENCE → POLICY_EVAL
        # → CLAIM × 2 → REBUTTAL × 2 → RULING
        assert types[0] == "DISPUTE_FILED"
        assert "EVIDENCE_REQUEST" in types
        assert "EVIDENCE" in types
        assert "POLICY_EVAL" in types
        # Two claims (rider + driver).
        claim_count = types.count("CLAIM")
        assert claim_count == 2
        # Two rebuttals.
        rebuttal_count = types.count("REBUTTAL")
        assert rebuttal_count == 2
        # Ends with a ruling (not escalation).
        assert types[-1] == "RULING"

    def test_safety_case_event_sequence(self, orc):
        cf = orc.resolve("disp_sf_001")
        types = cf.events.types()

        # Safety: DISPUTE_FILED → EVIDENCE_REQUEST → EVIDENCE → POLICY_EVAL
        # → ESCALATION (no advocates)
        assert types[0] == "DISPUTE_FILED"
        assert "EVIDENCE_REQUEST" in types
        assert "EVIDENCE" in types
        assert "POLICY_EVAL" in types
        assert "CLAIM" not in types  # no advocates run
        assert types[-1] == "ESCALATION"

    def test_events_have_tokens_and_latency(self, orc):
        cf = orc.resolve("disp_rd_002")
        # At least the CLAIM and RULING events should have tokens.
        claim_events = cf.events.by_type("CLAIM")
        for ev in claim_events:
            assert ev.tokens >= 0
            assert ev.latency_ms >= 0
        ruling_events = cf.events.by_type("RULING")
        assert len(ruling_events) == 1
        assert ruling_events[0].tokens >= 0

    def test_total_tokens_tracked(self, orc):
        cf = orc.resolve("disp_rd_003")
        # Mock provider charges 100 tokens per call.
        # At minimum: 2 advocates + 2 rebuttals + 1 judge = 5 calls.
        assert cf.total_tokens >= 500

    def test_jsonl_persisted(self, orc):
        cf = orc.resolve("disp_rd_004")
        # The JSONL file should exist under runs/.
        from pathlib import Path

        jsonl_path = Path("runs") / f"{cf.dispute_id}.jsonl"
        assert jsonl_path.exists()
        # Should be replayable.
        events = EventLog.replay(jsonl_path)
        assert len(events) == len(cf.events.events)
        assert events[0].type == "DISPUTE_FILED"

    def test_event_cites_populated(self, orc):
        cf = orc.resolve("disp_rd_005")
        claim_events = cf.events.by_type("CLAIM")
        for ev in claim_events:
            # Each claim should cite at least one E# or P-#.
            assert len(ev.cites) >= 1


# ---------------------------------------------------------------------------
# Judge rules — amount must match policy engine
# ---------------------------------------------------------------------------


class TestJudgeRules:
    def test_judge_amount_matches_policy(self, orc):
        cf = orc.resolve("disp_rd_006")
        assert cf.ruling is not None
        assert abs(cf.ruling.amount - cf.policy_result.amount) < 0.01

    def test_judge_outcome_matches_policy(self, orc):
        cf = orc.resolve("disp_rd_007")
        assert cf.ruling is not None
        assert cf.ruling.outcome == cf.policy_result.outcome

    def test_judge_confidence_above_threshold(self, orc):
        cf = orc.resolve("disp_rd_000")
        assert cf.ruling is not None
        assert cf.ruling.confidence >= 0.7
