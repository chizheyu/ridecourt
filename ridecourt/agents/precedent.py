"""Policy & Precedent Agent — retrieves similar past rulings.

Uses ``ridecourt.precedent.retrieve_precedents`` (TF-IDF keyword
similarity) to find the 3 most similar past rulings from
``precedents.json`` and gives them to the Judge.

The Judge must say whether its ruling is consistent with the precedents.

Output schema (per SPEC section 6)::

    top 3 similar rulings with similarity reason
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..data import Dispute, Precedent
from ..llm import LLMMeta, complete_json
from ..precedent import PrecedentMatch, retrieve_precedents


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class PrecedentResult(BaseModel):
    """A single precedent match in the agent output."""

    case_id: str
    category: str
    summary: str
    outcome: str
    amount: float = 0.0
    similarity: float = 0.0
    reason: str = ""
    source: str = "seed"


class PrecedentOutput(BaseModel):
    """Output of the Precedent Agent."""

    matches: list[PrecedentResult] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _system_prompt() -> str:
    return (
        "You are the Precedent Agent in RideCourt, a dispute resolution "
        "system for ride-hailing.\n\n"
        "You receive the dispute and a set of past rulings (precedents). "
        "Your job: select the 3 most relevant precedents and explain why "
        "they are similar.\n\n"
        "Rules:\n"
        "1. Use the deterministic TF-IDF retrieval results provided.\n"
        "2. For each match, explain the similarity reason briefly.\n"
        "3. If a precedent comes from a human override (source='override'), "
        "note that it was 'learned from override'.\n\n"
        "Return JSON matching the provided schema."
    )


def _user_prompt(
    claim_text: str,
    matches: list[PrecedentMatch],
) -> str:
    lines: list[str] = []
    lines.append(f"Dispute claim: {claim_text}\n")
    lines.append(f"Retrieved precedents ({len(matches)}):")
    for m in matches:
        lines.append(
            f"  {m.case_id} [{m.category}] sim={m.similarity:.2f}: "
            f"{m.summary} → {m.outcome} S${m.amount:.2f} "
            f"({m.reason}, source={m.source})"
        )
    lines.append("")
    lines.append(
        "Output the matches with their similarity reasons. "
        "Note any override-sourced precedents."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Precedent agent
# ---------------------------------------------------------------------------


class PrecedentAgent:
    """Precedent Agent — retrieves and explains similar past rulings."""

    name = "precedent_agent"

    def build(
        self,
        dispute: Dispute,
        precedents: list[Precedent],
    ) -> tuple[PrecedentOutput, LLMMeta, list[PrecedentMatch]]:
        """Retrieve the top-3 similar precedents and build the agent output.

        Returns ``(output, meta, matches)`` where *matches* is the raw
        deterministic retrieval (used by the orchestrator and tests).
        """
        matches = retrieve_precedents(dispute, precedents, top_k=3)

        system = _system_prompt()
        user = _user_prompt(dispute.claim_text, matches)
        result, meta = complete_json(
            system,
            user,
            PrecedentOutput,
            self.name,
            context={"matches": matches},
        )
        return result, meta, matches
