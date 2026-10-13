"""Tests for orchestrator resilience — optional vs required step failures.

When the LLM provider fails after retries:

* **Optional steps** (rebuttal, Fraud agent explanation, Precedent) are
  dropped and the case continues.  A ``STEP_SKIPPED`` event is emitted
  recording which step was skipped and why.  The Fraud agent's
  deterministic risk level still reaches the Judge.

* **Required steps** (an advocate's initial claim, the Judge) escalate
  the case with reason ``model_unavailable`` and an ``ESCALATION`` event
  that carries the error, instead of raising.

Run with: .venv/bin/python -m pytest tests/test_orchestrator_resilience.py -q
"""

from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

from unittest.mock import patch

import pytest

from ridecourt.data import load_dataset
from ridecourt.llm import LLMError
from ridecourt.orchestrator import Orchestrator


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm_error(agent_name: str) -> LLMError:
    """A transient LLMError that survives retries."""
    return LLMError(
        f"gemini transport error: Server disconnected without sending a response."
    )


# ---------------------------------------------------------------------------
# Optional step: rebuttal failure is dropped, case continues
# ---------------------------------------------------------------------------


class TestRebuttalModelFailure:
    """When the rebuttal LLM call fails after retries, the rebuttal is
    dropped (not the guard path) and the case proceeds to the Judge."""

    def test_rebuttal_model_error_drops_rebuttal(self, ds):
        """A rebuttal whose LLM call fails is dropped with a STEP_SKIPPED
        event, and the case is NOT escalated."""
        orc = Orchestrator(ds)

        original_rebuttal = Orchestrator._rebuttal_guarded

        def fail_rebuttal(self, advocate, claim_text, pack, opponent,
                          policy_result, policy_ids):
            # Simulate the LLM failing after retries for the rider rebuttal.
            if advocate.side == "rider":
                return None, None, [], _make_llm_error("rider_advocate")
            return original_rebuttal(
                self, advocate, claim_text, pack, opponent,
                policy_result, policy_ids,
            )

        with patch.object(Orchestrator, "_rebuttal_guarded", fail_rebuttal):
            cf = orc.resolve("disp_rd_000")

        # Case continues — not escalated.
        assert not cf.escalated
        assert cf.ruling is not None
        # Rider rebuttal is dropped.
        assert cf.rider_rebuttal is None
        # Driver rebuttal is present (only rider failed).
        assert cf.driver_rebuttal is not None
        # A STEP_SKIPPED event was emitted for the rider rebuttal.
        skipped = cf.events.by_type("STEP_SKIPPED")
        assert len(skipped) == 1
        assert skipped[0].content["agent"] == "rider_advocate"
        assert skipped[0].content["stage"] == "rebuttal"
        assert skipped[0].content["reason"] == "model_unavailable"
        # Case ends with a RULING.
        assert cf.events.types()[-1] == "RULING"


# ---------------------------------------------------------------------------
# Optional step: Fraud agent LLM failure — assessment still reaches Judge
# ---------------------------------------------------------------------------


class TestFraudModelFailure:
    """When the Fraud agent's LLM explanation fails after retries, the
    deterministic risk level still reaches the Judge and the case
    continues."""

    def test_fraud_model_error_drops_explanation(self, ds):
        """The Fraud agent's LLM explanation fails; the deterministic
        assessment is retained and the case proceeds."""
        orc = Orchestrator(ds)

        original_fraud = Orchestrator._fraud_guarded

        def fail_fraud(self, claim_text, pack, facts, claimant_id,
                       claimant_party, category):
            # Simulate the LLM failing after retries.
            from ridecourt.fraud import compute_fraud_signals
            assessment = compute_fraud_signals(
                pack, facts, self.dataset.profiles,
                claimant_id, claimant_party,
                claim_text=claim_text, category=category,
            )
            return None, None, assessment, _make_llm_error("fraud_agent")

        with patch.object(Orchestrator, "_fraud_guarded", fail_fraud):
            cf = orc.resolve("disp_rd_000")

        # Case continues — not escalated.
        assert not cf.escalated
        assert cf.ruling is not None
        # risk_output is None (LLM explanation failed), but the
        # deterministic assessment is retained.
        assert cf.risk_output is None
        assert cf.risk_assessment is not None
        # A STEP_SKIPPED event was emitted for the fraud agent.
        skipped = cf.events.by_type("STEP_SKIPPED")
        fraud_skips = [
            e for e in skipped if e.content.get("step") == "fraud_agent"
        ]
        assert len(fraud_skips) == 1
        assert fraud_skips[0].content["reason"] == "model_unavailable"
        assert fraud_skips[0].content["deterministic_assessment"] == "retained"


# ---------------------------------------------------------------------------
# Required step: advocate claim failure escalates with model_unavailable
# ---------------------------------------------------------------------------


class TestAdvocateModelFailure:
    """When an advocate's initial claim LLM call fails after retries, the
    case escalates with reason ``model_unavailable`` instead of raising."""

    def test_advocate_model_error_escalates(self, ds):
        """The rider advocate's LLM call fails; the case escalates with
        reason model_unavailable and an ESCALATION event carries the error."""
        orc = Orchestrator(ds)

        original_advocate = Orchestrator._advocate_guarded

        def fail_rider_advocate(self, advocate, claim_text, pack,
                                policy_result, policy_ids):
            if advocate.side == "rider":
                return None, None, [], _make_llm_error("rider_advocate")
            return original_advocate(
                self, advocate, claim_text, pack,
                policy_result, policy_ids,
            )

        with patch.object(Orchestrator, "_advocate_guarded", fail_rider_advocate):
            cf = orc.resolve("disp_rd_000")

        # Case is escalated with model_unavailable.
        assert cf.escalated
        assert cf.escalation_reason == "model_unavailable"
        assert cf.ruling is None  # Judge never ran
        # An ESCALATION event was emitted with the error.
        escalations = cf.events.by_type("ESCALATION")
        assert len(escalations) == 1
        assert escalations[0].content["reason"] == "model_unavailable"
        assert "rider_advocate" in escalations[0].content["errors"]


# ---------------------------------------------------------------------------
# Required step: Judge failure escalates with model_unavailable
# ---------------------------------------------------------------------------


class TestJudgeModelFailure:
    """When the Judge's LLM call fails after retries, the case escalates
    with reason ``model_unavailable`` instead of raising."""

    def test_judge_model_error_escalates(self, ds):
        """The Judge's LLM call fails; the case escalates with reason
        model_unavailable and an ESCALATION event carries the error."""
        orc = Orchestrator(ds)

        from ridecourt.agents.judge import Judge
        original_build = Judge.build

        def fail_judge(self, claim_text, pack, rider_case, driver_case,
                       policy_result, *, risk=None, precedents=None):
            raise _make_llm_error("judge")

        with patch.object(Judge, "build", fail_judge):
            cf = orc.resolve("disp_rd_000")

        # Case is escalated with model_unavailable.
        assert cf.escalated
        assert cf.escalation_reason == "model_unavailable"
        assert cf.ruling is None
        # An ESCALATION event was emitted with the error and step=judge.
        escalations = cf.events.by_type("ESCALATION")
        assert len(escalations) == 1
        assert escalations[0].content["reason"] == "model_unavailable"
        assert escalations[0].content["step"] == "judge"
        assert "error" in escalations[0].content
