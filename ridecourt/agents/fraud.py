"""Fraud & Bad-Faith Detection Agent.

Runs in parallel with the advocates using the history_profile evidence
and the full evidence pack.  Outputs risk level and signals with
citations.  Deterministic features are computed first
(``ridecourt.fraud.compute_fraud_signals``); the LLM only explains them.

Output schema (per SPEC section 6)::

    {risk: low/medium/high, signals: [{text, cites}]}
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..data import EvidencePack, Facts, Profile
from ..fraud import FraudAssessment, compute_fraud_signals
from ..llm import LLMMeta, complete_json


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class FraudSignalOut(BaseModel):
    """A single fraud signal in the agent output."""

    text: str
    cites: list[str] = Field(default_factory=list)


class FraudOutput(BaseModel):
    """Output of the Fraud & Bad-Faith Agent."""

    risk: str  # low | medium | high
    signals: list[FraudSignalOut] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _system_prompt() -> str:
    return (
        "You are the Fraud & Bad-Faith Detection Agent in RideCourt, a "
        "dispute resolution system for ride-hailing.\n\n"
        "You receive the claimant's profile history, the evidence pack, "
        "and deterministic fraud signals computed by the system.\n\n"
        "Your job: assess the claimant's credibility and output a risk "
        "level (low, medium, high) with supporting signals.\n\n"
        "Rules:\n"
        "1. Every signal must cite at least one evidence ID (E#) and/or "
        "policy section ID (§N.M) from the provided context.\n"
        "2. Use the deterministic signals as your basis — do not invent "
        "new signals or numbers. You may rephrase the signal text.\n"
        "3. If no signals are present, risk is 'low'.\n"
        "4. Per §6.3, three or more upheld disputes of the same type "
        "in 90 days lowers credibility — you must flag this as 'high' "
        "risk.\n\n"
        "Return JSON matching the provided schema."
    )


def _user_prompt(
    claim_text: str,
    pack: EvidencePack,
    assessment: FraudAssessment,
) -> str:
    lines: list[str] = []
    lines.append(f"Claim [CLAIM]: {claim_text}\n")
    lines.append(f"Claimant: {assessment.claimant_id}")
    lines.append(f"Computed risk level: {assessment.risk}\n")

    lines.append("Evidence pack (history items):")
    for item in pack.by_kind("history_profile"):
        lines.append(f"  {item.id}: {item.summary}")
    for item in pack.by_kind("arrival_check"):
        lines.append(f"  {item.id}: {item.summary}")
    lines.append("")

    lines.append(f"Deterministic signals ({len(assessment.signals)}):")
    for s in assessment.signals:
        cites_str = ", ".join(s.cites)
        lines.append(f"  [{s.feature}] {s.text} [{cites_str}]")
    lines.append("")

    lines.append(
        "Output the risk level and signals. Cite E# and P-.. IDs. "
        "Do not invent new numbers or signals."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fraud agent
# ---------------------------------------------------------------------------


class FraudAgent:
    """Fraud & Bad-Faith Detection Agent.

    Computes deterministic features first, then calls the LLM to explain
    them.  The LLM output is validated by the guard.
    """

    name = "fraud_agent"

    def build(
        self,
        claim_text: str,
        pack: EvidencePack,
        facts: Facts,
        profiles: dict[str, Profile],
        claimant_id: str,
        claimant_party: str,
        category: str = "",
    ) -> tuple[FraudOutput, LLMMeta, FraudAssessment]:
        """Assess fraud risk for the claimant.

        Returns ``(output, meta, assessment)`` where *assessment* is the
        deterministic computation (used by the orchestrator and tests).
        """
        assessment = compute_fraud_signals(
            pack, facts, profiles, claimant_id, claimant_party,
            claim_text=claim_text, category=category,
        )

        system = _system_prompt()
        user = _user_prompt(claim_text, pack, assessment)
        result, meta = complete_json(
            system,
            user,
            FraudOutput,
            self.name,
            evidence_pack=pack,
            context={"assessment": assessment},
        )
        return result, meta, assessment
