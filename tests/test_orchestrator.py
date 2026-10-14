"""Tests for the orchestrator — end-to-end pipeline with ``LLM_PROVIDER=mock``.

Covers:
- Route deviation case (AI-final) ends with the label's outcome and amount
- Large-deviation route case escalates to human (§6.2)
- No-show rider case ends with the label's outcome and amount
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
# Route deviation case (within tolerance, AI-final) — ruling matches label
# ---------------------------------------------------------------------------


class TestRouteDeviationCase:
    def test_ruling_matches_label(self, orc, ds):
        """disp_rd_003: within tolerance, routes to AI final."""
        cf = orc.resolve("disp_rd_003")
        label = ds.label("disp_rd_003")

        assert not cf.escalated
        assert cf.ruling is not None
        assert cf.ruling.outcome == label.outcome
        assert abs(cf.ruling.amount - label.amount) < 0.01
        assert cf.policy_result.outcome == label.outcome
        assert abs(cf.policy_result.amount - label.amount) < 0.01

    def test_ruling_has_cites(self, orc):
        cf = orc.resolve("disp_rd_003")
        assert cf.ruling is not None
        assert len(cf.ruling.cites) >= 1
        # All cites should be valid E# or §N.M IDs.
        for cite in cf.ruling.cites:
            assert cite.startswith("E") or cite.startswith("§")

    def test_ruling_has_messages(self, orc):
        cf = orc.resolve("disp_rd_003")
        assert cf.ruling is not None
        assert cf.ruling.message_to_rider
        assert cf.ruling.message_to_driver

    def test_advocate_cases_built(self, orc):
        cf = orc.resolve("disp_rd_003")
        assert cf.rider_case is not None
        assert cf.driver_case is not None
        assert len(cf.rider_case.arguments) >= 1
        assert len(cf.driver_case.arguments) >= 1
        for arg in cf.rider_case.arguments:
            assert len(arg.cites) >= 1
        for arg in cf.driver_case.arguments:
            assert len(arg.cites) >= 1

    def test_rebuttals_built(self, orc):
        cf = orc.resolve("disp_rd_003")
        assert cf.rider_rebuttal is not None
        assert cf.driver_rebuttal is not None


# ---------------------------------------------------------------------------
# Large-deviation route case — escalates to human (§6.2)
# ---------------------------------------------------------------------------


class TestLargeDeviationEscalation:
    def test_large_deviation_routes_to_human(self, orc):
        """disp_rd_000 has >2 km deviation → §6.2 routes to human."""
        cf = orc.resolve("disp_rd_000")

        assert cf.escalated
        # The ruling is kept as a recommendation for the human reviewer.
        assert cf.ruling is not None
        assert cf.policy_result.routing == "human"
        assert "§6.2" in cf.policy_result.routing_reason


# ---------------------------------------------------------------------------
# No-show rider case
# ---------------------------------------------------------------------------


class TestNoShowRiderCase:
    def test_cancellation_fee_stands(self, orc, ds):
        """disp_ns_000: both contacts met → fee upheld."""
        cf = orc.resolve("disp_ns_000")
        label = ds.label("disp_ns_000")

        assert label.outcome == "cancellation_fee_upheld"
        assert not cf.escalated
        assert cf.ruling is not None
        assert cf.ruling.outcome == label.outcome
        assert cf.ruling.amount == label.amount

    def test_cancellation_fee_voided(self, orc, ds):
        """disp_ns_001: missing contact → fee voided."""
        cf = orc.resolve("disp_ns_001")
        label = ds.label("disp_ns_001")

        assert label.outcome == "cancellation_fee_voided"
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
        assert cf.rider_case is None
        assert cf.driver_case is None

    def test_safety_has_policy_result(self, orc):
        cf = orc.resolve("disp_sf_000")
        assert cf.policy_result is not None
        assert cf.policy_result.outcome == "escalate_safety"
        assert "§5.1" in cf.policy_result.applicable_clauses


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

        bad_output = AdvocateOutput(
            position="This is guaranteed to be a refund based on E1.",
            arguments=[
                Argument(text="Evidence E1 shows deviation.", cites=["E1"]),
            ],
            requested_outcome="Refund S$5.52.",
        )

        original_check = orc_mod.guard_check

        def always_fail(output, pack, policy_ids, policy_result=None, **kwargs):
            return [Problem(
                kind="forbidden_word",
                detail="guaranteed found",
                field="position",
            )]

        monkeypatch.setattr(orc_mod, "guard_check", always_fail)

        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_003")

        assert cf.escalated
        assert "guard" in cf.escalation_reason
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
        cf = orc.resolve("disp_rd_003")

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
            if call_state["n"] >= 3 and call_state["n"] <= 6:
                return [Problem(
                    kind="forbidden_word",
                    detail="guaranteed found in rebuttal",
                    field="position",
                )]
            return original_check(output, pack, policy_ids, policy_result=policy_result, **kwargs)

        monkeypatch.setattr(orc_mod, "guard_check", fail_rebuttals)

        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_003")

        assert not cf.escalated
        assert cf.ruling is not None
        assert cf.rider_rebuttal is None
        assert cf.driver_rebuttal is None
        assert cf.rider_case is not None
        assert cf.driver_case is not None
        types = cf.events.types()
        rejected = cf.events.by_type("GUARD_REJECTED")
        assert len(rejected) == 2
        assert "GUARD_REJECTED" in types
        for ev in rejected:
            assert ev.content["stage"] == "rebuttal"
            assert len(ev.content["problems"]) >= 1
        assert types[-1] == "RULING"

    def test_rebuttal_guard_reask_then_pass_continues(self, ds, monkeypatch):
        """If the rebuttal guard fails the first time but passes on re-ask,
        the rebuttal is kept and the case proceeds normally."""
        import ridecourt.orchestrator as orc_mod

        original_check = orc_mod.guard_check

        call_state = {"n": 0}

        def fail_rebuttal_once(output, pack, policy_ids, policy_result=None, **kwargs):
            call_state["n"] += 1
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
        cf = orc.resolve("disp_rd_003")

        assert not cf.escalated
        assert cf.ruling is not None
        assert cf.rider_rebuttal is not None
        assert cf.driver_rebuttal is not None
        rejected = cf.events.by_type("GUARD_REJECTED")
        assert len(rejected) == 0

    def test_one_rebuttal_fails_other_passes(self, ds, monkeypatch):
        """If only one rebuttal fails (both tries) but the other passes,
        the failing rebuttal is dropped and the case continues to the Judge."""
        original_rebuttal = Orchestrator._rebuttal_guarded

        def fail_one_rebuttal(self, advocate, claim_text, pack, opponent,
                              policy_result, policy_ids):
            if advocate.side == "rider":
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
        cf = orc.resolve("disp_rd_003")

        assert not cf.escalated
        assert cf.ruling is not None
        assert cf.rider_rebuttal is None
        assert cf.driver_rebuttal is not None
        rejected = cf.events.by_type("GUARD_REJECTED")
        assert len(rejected) == 1
        assert rejected[0].content["agent"] == "rider_advocate"
        assert rejected[0].content["stage"] == "rebuttal"

    def test_guard_escalation_only_for_initial_claims(self, ds, monkeypatch):
        """Guard failure on initial claims escalates (guard_failure_advocate)."""
        import ridecourt.orchestrator as orc_mod

        original_check = orc_mod.guard_check

        call_state = {"n": 0}

        def fail_claims_only(output, pack, policy_ids, policy_result=None, **kwargs):
            call_state["n"] += 1
            if call_state["n"] <= 2:
                return [Problem(
                    kind="forbidden_word",
                    detail="guaranteed found in claim",
                    field="position",
                )]
            return original_check(output, pack, policy_ids, policy_result=policy_result, **kwargs)

        monkeypatch.setattr(orc_mod, "guard_check", fail_claims_only)

        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_003")

        assert cf.escalated
        assert cf.escalation_reason == "guard_failure_advocate"
        assert cf.ruling is None

    def test_guard_escalation_on_judge_ruling(self, ds, monkeypatch):
        """Guard failure on the Judge's ruling escalates
        (guard_failure_judge), not dropped."""
        import ridecourt.orchestrator as orc_mod

        original_check = orc_mod.guard_check

        call_state = {"n": 0}

        def fail_judge_only(output, pack, policy_ids, policy_result=None, **kwargs):
            call_state["n"] += 1
            if call_state["n"] >= 5:
                return [Problem(
                    kind="forbidden_word",
                    detail="guaranteed found in judge ruling",
                    field="reasoning",
                )]
            return original_check(output, pack, policy_ids, policy_result=policy_result, **kwargs)

        monkeypatch.setattr(orc_mod, "guard_check", fail_judge_only)

        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_003")

        assert cf.escalated
        assert cf.escalation_reason == "guard_failure_judge"
        assert len(cf.guard_problems) > 0

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
        cf = orc.resolve("disp_rd_003")
        types = cf.events.types()

        assert types[0] == "DISPUTE_FILED"
        assert "EVIDENCE_REQUEST" in types
        assert "EVIDENCE" in types
        assert "POLICY_EVAL" in types
        claim_count = types.count("CLAIM")
        assert claim_count == 2
        rebuttal_count = types.count("REBUTTAL")
        assert rebuttal_count == 2
        assert types[-1] == "RULING"

    def test_safety_case_event_sequence(self, orc):
        cf = orc.resolve("disp_sf_001")
        types = cf.events.types()

        assert types[0] == "DISPUTE_FILED"
        assert "EVIDENCE_REQUEST" in types
        assert "EVIDENCE" in types
        assert "POLICY_EVAL" in types
        assert "CLAIM" not in types
        assert types[-1] == "ESCALATION"

    def test_events_have_tokens_and_latency(self, orc):
        cf = orc.resolve("disp_rd_003")
        claim_events = cf.events.by_type("CLAIM")
        for ev in claim_events:
            assert ev.tokens >= 0
            assert ev.latency_ms >= 0
        ruling_events = cf.events.by_type("RULING")
        assert len(ruling_events) == 1
        assert ruling_events[0].tokens >= 0

    def test_total_tokens_tracked(self, orc):
        cf = orc.resolve("disp_rd_006")
        assert cf.total_tokens >= 500

    def test_jsonl_persisted(self, orc):
        cf = orc.resolve("disp_rd_009")
        from pathlib import Path

        jsonl_path = Path("runs") / f"{cf.dispute_id}.jsonl"
        assert jsonl_path.exists()
        events = EventLog.replay(jsonl_path)
        assert len(events) == len(cf.events.events)
        assert events[0].type == "DISPUTE_FILED"

    def test_event_cites_populated(self, orc):
        cf = orc.resolve("disp_rd_010")
        claim_events = cf.events.by_type("CLAIM")
        for ev in claim_events:
            assert len(ev.cites) >= 1


# ---------------------------------------------------------------------------
# Judge rules — amount must match policy engine
# ---------------------------------------------------------------------------


class TestJudgeRules:
    def test_judge_amount_matches_policy(self, orc):
        cf = orc.resolve("disp_rd_011")
        assert cf.ruling is not None
        assert abs(cf.ruling.amount - cf.policy_result.amount) < 0.01

    def test_judge_outcome_matches_policy(self, orc):
        cf = orc.resolve("disp_sg_000")
        assert cf.ruling is not None
        assert cf.ruling.outcome == cf.policy_result.outcome

    def test_judge_confidence_above_threshold(self, orc):
        cf = orc.resolve("disp_rd_003")
        assert cf.ruling is not None
        assert cf.ruling.confidence >= 0.7


# ---------------------------------------------------------------------------
# Full-suite mock sweep — no guard failures across all cases
# ---------------------------------------------------------------------------


class TestMockFullSuiteNoGuardFailures:
    """Resolve every case in the dataset with ``LLM_PROVIDER=mock`` and fail
    if any case escalates due to a guard problem.

    Safety cases escalate by design (§5.1) and are excluded.  Cases that
    route to human via §6 (e.g. large route deviation) also escalate by
    design.  A guard failure means the mock provider wrote a number or
    citation that is not backed by the evidence — a bug in the mock.
    """

    def test_no_guard_failures_all_cases(self, ds):
        orc = Orchestrator(ds)
        guard_failures: list[tuple[str, str, str]] = []
        for did in sorted(ds.disputes.keys()):
            cf = orc.resolve(did)
            d = ds.dispute(did)
            if cf.escalated and "guard_failure" in (cf.escalation_reason or ""):
                guard_failures.append(
                    (did, d.category, cf.escalation_reason)
                )
        assert guard_failures == [], (
            f"{len(guard_failures)} case(s) have guard failures with the mock "
            "provider — the mock wrote numbers or citations not in the evidence:\n  "
            + "\n  ".join(f"{did} [{cat}]: {reason}" for did, cat, reason in guard_failures)
        )
