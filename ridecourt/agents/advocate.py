"""Advocate agent — one class for both Rider Advocate and Driver Advocate.

Both advocates share the same output schema and reasoning strategy; they
differ only in which side they argue for.  The agent's job is to build
the strongest case for its side using **only** the evidence pack and the
policy, cite evidence IDs (``E#``) and policy sections (``§1.3``) for
every argument, and briefly acknowledge evidence against its side
(judges reward candour).

Output schema (per SPEC section 6)::

    {position, arguments: [{text, cites}], requested_outcome}
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..data import EvidencePack, PolicyResult
from ..llm import LLMMeta, complete_json


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class Argument(BaseModel):
    """A single argument with evidence/policy citations."""

    text: str
    cites: list[str] = Field(default_factory=list)


class AdvocateOutput(BaseModel):
    """Output shape for both Rider Advocate and Driver Advocate."""

    position: str
    arguments: list[Argument] = Field(default_factory=list)
    requested_outcome: str


Side = Literal["rider", "driver"]


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------


def _system_prompt(side: Side) -> str:
    who = "Rider Advocate" if side == "rider" else "Driver Advocate"
    target = "the rider" if side == "rider" else "the driver"
    return (
        f"You are the {who} in RideCourt, a dispute resolution system for "
        "ride-hailing.\n\n"
        "Your job: argue for your side using ONLY the evidence pack and the "
        "policy provided. You must not invent facts or numbers.\n\n"
        "Rules:\n"
        "1. Every argument must cite at least one evidence ID (E#) and/or "
        "policy section of Ryde Policy v2.1 written exactly as §1.3 (no "
        "prefix) that exists in the provided context.\n"
        "2. Copy numbers exactly as shown in the evidence. Never compute, "
        "round, or estimate new numbers. If you need a difference (e.g. "
        "extra minutes, fare difference), cite the evidence item that "
        "already states it — the evidence pack includes derived quantities "
        "and a policy computation item for this purpose.\n"
        "3. The dispute's claim text is shown with the citation ID "
        "\"CLAIM\". You may quote a number from the claim only in a "
        "sentence that cites CLAIM and attributes it to the claimant "
        "(e.g. \"The rider says the driver overcharged a certain amount "
        "(CLAIM)\"). Do not state a claim number as a fact of the case "
        "without citing CLAIM — the guard will reject it.\n"
        "4. Briefly acknowledge evidence that cuts against your side. "
        "Judges reward candour.\n"
        f"5. Your position and requested_outcome should favour {target}.\n"
        "6. Be concise and specific.\n\n"
        "Return JSON matching the provided schema."
    )


def _user_prompt(
    side: Side,
    claim_text: str,
    pack: EvidencePack,
    policy_result: PolicyResult | None,
) -> str:
    """Build the user prompt with the evidence pack and policy context."""
    # Evidence pack summary
    lines: list[str] = []
    lines.append(f"Claim [CLAIM]: {claim_text}\n")
    lines.append("Evidence pack:")
    for item in pack.items:
        lines.append(f"  {item.id} [{item.kind}]: {item.summary}")
        # Show key numeric data values so the agent can copy them exactly.
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

    # Policy result
    if policy_result is not None:
        lines.append(f"Policy engine result:")
        lines.append(f"  clauses: {', '.join(policy_result.applicable_clauses)}")
        lines.append(f"  outcome: {policy_result.outcome}")
        lines.append(f"  amount: S${policy_result.amount:.2f}")
        lines.append(f"  computation: {policy_result.computation}")
        lines.append("")

    lines.append(
        f"Argue for {'the rider' if side == 'rider' else 'the driver'}. "
        "Cite E# IDs and v2.1 sections written as §1.3. Acknowledge counter-evidence briefly."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Advocate agent
# ---------------------------------------------------------------------------


class Advocate:
    """Advocate agent — argues for one side using evidence and policy.

    Parameters
    ----------
    side : ``"rider"`` or ``"driver"``
        Which side to argue for.
    """

    def __init__(self, side: Side):
        self.side: Side = side
        self.name: str = (
            "rider_advocate" if side == "rider" else "driver_advocate"
        )

    def build(
        self,
        claim_text: str,
        pack: EvidencePack,
        policy_result: PolicyResult | None = None,
    ) -> tuple[AdvocateOutput, LLMMeta]:
        """Call the LLM to produce an AdvocateOutput for this side."""
        system = _system_prompt(self.side)
        user = _user_prompt(self.side, claim_text, pack, policy_result)
        result, meta = complete_json(
            system,
            user,
            AdvocateOutput,
            self.name,
            evidence_pack=pack,
            policy_result=policy_result,
        )
        return result, meta

    def rebut(
        self,
        claim_text: str,
        pack: EvidencePack,
        opponent_output: AdvocateOutput,
        policy_result: PolicyResult | None = None,
    ) -> tuple[AdvocateOutput, LLMMeta]:
        """Produce a short rebuttal after seeing the opponent's claim."""
        who = "Rider Advocate" if self.side == "rider" else "Driver Advocate"
        opp = "Driver Advocate" if self.side == "rider" else "Rider Advocate"
        system = (
            f"You are the {who}. The {opp} has presented their case. "
            "Provide a short rebuttal: identify the weakest point in the "
            "opponent's argument and reinforce your side's position. "
            "Cite E# IDs and v2.1 sections written as §1.3. Be brief (2-3 arguments max). "
            "Copy numbers exactly as shown in the evidence — never compute, "
            "round, or estimate new numbers. The claim text has citation "
            "ID \"CLAIM\"; you may quote a number from the claim only in a "
            "sentence that cites CLAIM and attributes it to the claimant. "
            "Return JSON matching the provided schema."
        )
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
        lines.append(f"Opponent ({opp}) case:")
        lines.append(f"  position: {opponent_output.position}")
        for i, arg in enumerate(opponent_output.arguments):
            cites_str = ", ".join(arg.cites)
            lines.append(f"  argument {i + 1}: {arg.text} [{cites_str}]")
        lines.append(f"  requested_outcome: {opponent_output.requested_outcome}")
        lines.append("")
        if policy_result is not None:
            lines.append(f"Policy engine: {policy_result.outcome}, S${policy_result.amount:.2f}")
            lines.append("")
        lines.append(
            f"Argue for {'the rider' if self.side == 'rider' else 'the driver'}. "
            "Cite E# IDs and v2.1 sections written as §1.3."
        )
        user = "\n".join(lines)
        result, meta = complete_json(
            system,
            user,
            AdvocateOutput,
            self.name,
            evidence_pack=pack,
            policy_result=policy_result,
        )
        return result, meta


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def rider_advocate() -> Advocate:
    return Advocate(side="rider")


def driver_advocate() -> Advocate:
    return Advocate(side="driver")
