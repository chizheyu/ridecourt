"""Judge agent — receives both advocate cases, the policy engine result,
risk signals and precedents (when present), and issues a ruling.

Judge rules (per SPEC section 6 + v2.1 §6):

1. The **amount** must equal the policy engine's amount; the Judge never
   invents numbers.
2. The **outcome** comes from the policy engine. If the Judge disagrees
   with the policy engine, it escalates instead of overriding.
3. **§6 routing**: if the policy engine says routing=human, the case
   escalates regardless of the Judge's confidence.
4. **Safety category** always escalates with a neutral case summary.
5. If risk signals are provided and the risk is 'high', the Judge must
   mention bad-faith signals (§6.3).

Output schema::

    {outcome, amount, confidence 0-1, reasoning, cites,
     message_to_rider, message_to_driver}
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..data import EvidencePack, PolicyResult
from ..llm import LLMMeta, complete_json
from .advocate import AdvocateOutput


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class JudgeOutput(BaseModel):
    """Structured ruling from the Judge agent."""

    outcome: str
    amount: float = 0.0
    confidence: float = 0.0
    reasoning: str
    cites: list[str] = Field(default_factory=list)
    message_to_rider: str
    message_to_driver: str
    # Stretch: precedent consistency — "consistent" | "inconsistent" | "no_precedents"
    precedent_consistency: str = ""
    # Stretch: bad-faith mentioned — the Judge must mention high-risk signals (§6.3)
    bad_faith_mentioned: bool = False


# ---------------------------------------------------------------------------
# Confidence threshold (kept for backward compat — v2.1 routing is by
# policy_result.routing, but low Judge confidence still escalates)
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# System + user prompt builders
# ---------------------------------------------------------------------------


def _system_prompt() -> str:
    return (
        "You are the Judge in RideCourt, a dispute resolution system for "
        "ride-hailing.\n\n"
        "You receive both advocates' cases, the policy engine result, and "
        "optional risk/precedent information. Your job is to issue a ruling.\n\n"
        "Judge rules:\n"
        "1. The amount must equal the policy engine's amount. Never invent "
        "numbers. Copy numbers exactly as shown in the evidence or policy "
        "computation — never compute, round, or estimate new numbers. If "
        "you need a value, cite the evidence item that already states it.\n"
        "2. The outcome comes from the policy engine. If you disagree with "
        "the policy engine, you must escalate — never override.\n"
        "3. If the policy engine says routing=human (§6), the case escalates "
        "regardless of your confidence.\n"
        "4. Safety incidents always escalate with a neutral case summary.\n"
        "5. If risk signals are provided and the risk is 'high', you MUST "
        "mention the bad-faith signals explicitly in your reasoning (§6.3) "
        "and set bad_faith_mentioned=true.\n"
        "6. If precedents are provided, you MUST say whether your ruling is "
        "consistent with them (precedent_consistency: 'consistent' or "
        "'inconsistent'; 'no_precedents' if none provided).\n"
        "7. Cite v2.1 policy sections exactly as §1.3, §2.3, §6.2 (no prefix). Cite "
        "precedents as DISP-YYYY-NNNN or PC-YYYY-NNN. Cite evidence as E#.\n"
        "8. The dispute's claim text has citation ID \"CLAIM\". You may "
        "quote a number from the claim only in a sentence that cites CLAIM "
        "and attributes it to the claimant. Do not state a claim number as "
        "a fact of the case without citing CLAIM.\n\n"
        "Your messages to the rider and driver must be in plain English, "
        "polite, and specific (cite the section and the key fact).\n\n"
        "Return JSON matching the provided schema."
    )


def _user_prompt(
    claim_text: str,
    pack: EvidencePack,
    rider_case: AdvocateOutput,
    driver_case: AdvocateOutput,
    policy_result: PolicyResult,
    risk: Any | None = None,
    precedents: list[Any] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"Claim [CLAIM]: {claim_text}\n")

    lines.append("Evidence pack:")
    for item in pack.items:
        lines.append(f"  {item.id} [{item.kind}]: {item.summary}")
        data = getattr(item, "data", {})
        if data:
            num_vals = {
                k: v for k, v in data.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if num_vals:
                pairs = ", ".join(f"{k}={v}" for k, v in num_vals.items())
                lines.append(f"    values: {pairs}")
    lines.append("")

    lines.append("Rider Advocate case:")
    lines.append(f"  position: {rider_case.position}")
    for i, arg in enumerate(rider_case.arguments):
        lines.append(f"  arg {i + 1}: {arg.text} [{', '.join(arg.cites)}]")
    lines.append(f"  requested_outcome: {rider_case.requested_outcome}")
    lines.append("")

    lines.append("Driver Advocate case:")
    lines.append(f"  position: {driver_case.position}")
    for i, arg in enumerate(driver_case.arguments):
        lines.append(f"  arg {i + 1}: {arg.text} [{', '.join(arg.cites)}]")
    lines.append(f"  requested_outcome: {driver_case.requested_outcome}")
    lines.append("")

    lines.append("Policy engine result:")
    lines.append(f"  clauses: {', '.join(policy_result.applicable_clauses)}")
    lines.append(f"  outcome: {policy_result.outcome}")
    lines.append(f"  amount: S${policy_result.amount:.2f}")
    lines.append(f"  computation: {policy_result.computation}")
    lines.append("")

    if risk is not None:
        risk_level = getattr(risk, "risk", "low")
        lines.append(f"Risk signals: {risk_level}")
        signals = getattr(risk, "signals", [])
        for s in signals:
            cites_str = ", ".join(getattr(s, "cites", []))
            lines.append(f"  signal: {getattr(s, 'text', '')} [{cites_str}]")
        if risk_level == "high":
            lines.append(
                "  NOTE: risk is HIGH — you must mention bad-faith signals "
                "in reasoning and set bad_faith_mentioned=true (§6.3)."
            )
        lines.append("")

    if precedents:
        lines.append(f"Precedents ({len(precedents)}):")
        for p in precedents[:3]:
            pid = getattr(p, "case_id", str(p))
            summary = getattr(p, "summary", str(p))
            outcome = getattr(p, "outcome", "")
            amount = getattr(p, "amount", 0.0)
            source = getattr(p, "source", "seed")
            lines.append(
                f"  {pid}: {summary} → {outcome} S${amount:.2f} "
                f"(source={source})"
            )
        lines.append(
            "  You must state whether your ruling is consistent with these "
            "precedents (precedent_consistency). You may cite a precedent's "
            f"ID (e.g. PC-2025-004) and refer to its facts. Numbers that come "
            "from a precedent are accepted only when you cite that "
            "precedent's ID."
        )
        lines.append("")

    lines.append(
        "Issue a ruling: outcome, amount (must equal policy engine amount), "
        "confidence (0-1), reasoning, cites, message_to_rider, "
        "message_to_driver, precedent_consistency, bad_faith_mentioned."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge agent
# ---------------------------------------------------------------------------


class Judge:
    """Judge agent — applies the ruling rules after both advocates have spoken."""

    name = "judge"

    def build(
        self,
        claim_text: str,
        pack: EvidencePack,
        rider_case: AdvocateOutput,
        driver_case: AdvocateOutput,
        policy_result: PolicyResult,
        *,
        risk: Any | None = None,
        precedents: list[Any] | None = None,
    ) -> tuple[JudgeOutput, LLMMeta]:
        """Call the LLM to produce a JudgeOutput.

        The Judge's amount must match the policy engine's amount. If the
        LLM returns a different amount, the output is **corrected** to
        the policy engine's amount (the orchestrator then decides whether
        to escalate based on the discrepancy).
        """
        system = _system_prompt()
        user = _user_prompt(
            claim_text, pack, rider_case, driver_case,
            policy_result, risk=risk, precedents=precedents,
        )

        # Build context for the mock provider so it can set
        # bad_faith_mentioned and precedent_consistency deterministically.
        context: dict[str, Any] = {}
        if risk is not None:
            # The FraudOutput carries risk + signals; pass the risk level.
            risk_level = getattr(risk, "risk", "low")
            # Reconstruct a minimal assessment-like object for the mock.
            from ..fraud import FraudAssessment, FraudSignal
            signals = getattr(risk, "signals", [])
            context["assessment"] = FraudAssessment(
                risk=risk_level,
                signals=[
                    FraudSignal(
                        text=getattr(s, "text", ""),
                        cites=list(getattr(s, "cites", [])),
                    )
                    for s in signals
                ],
            )
        if precedents:
            context["matches"] = precedents

        result, meta = complete_json(
            system,
            user,
            JudgeOutput,
            self.name,
            evidence_pack=pack,
            policy_result=policy_result,
            context=context,
        )
        # Enforce: the Judge's amount must equal the policy engine's amount.
        # The Judge never invents numbers.
        if abs(result.amount - policy_result.amount) > 0.01:
            result.amount = round(policy_result.amount, 2)
        return result, meta


# ---------------------------------------------------------------------------
# Ruling rules — decide RULING vs ESCALATION
# ---------------------------------------------------------------------------


def should_escalate(
    judge_output: JudgeOutput,
    policy_result: PolicyResult,
    *,
    category: str = "",
    rider_case: AdvocateOutput | None = None,
    driver_case: AdvocateOutput | None = None,
) -> bool:
    """Determine whether the Judge's output should escalate.

    v2.1 §6 routing takes precedence: if the policy engine says
    routing=human, the case escalates regardless of the Judge's
    confidence.

    Also escalates when:
    - Safety category (always, §5).
    - Confidence < 0.7 (Judge's own confidence, secondary signal).
    - Judge's outcome differs from the policy engine's outcome.
    """
    # v2.1 §6: policy engine routing takes precedence
    if getattr(policy_result, "routing", "ai_final") == "human":
        return True
    if category == "safety":
        return True
    if judge_output.confidence < CONFIDENCE_THRESHOLD:
        return True
    if judge_output.outcome != policy_result.outcome:
        return True
    return False


def neutral_case_summary(
    claim_text: str,
    pack: EvidencePack,
    policy_result: PolicyResult,
    *,
    reason: str = "safety_category",
    judge_output: JudgeOutput | None = None,
) -> str:
    """Build a neutral case summary for escalation.

    The summary is factual — it describes what the evidence shows and what
    the policy engine computed, without taking a side. Works for any
    escalation reason.
    """
    parts: list[str] = []
    parts.append(f"Escalation reason: {reason}")
    parts.append(f"Dispute: {claim_text}")
    # Find category from evidence if possible.
    parts.append("Evidence:")
    for item in pack.items:
        parts.append(f"  {item.id}: {item.summary}")
    parts.append(f"Policy engine: {policy_result.computation}")
    if judge_output is not None:
        parts.append(
            f"Judge outcome: {judge_output.outcome}, "
            f"confidence: {judge_output.confidence:.2f}"
        )
    return "\n".join(parts)
