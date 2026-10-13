"""Serialization helpers for CaseFile <-> plain dict.

The store persists case files as JSON-serializable dicts. Events are
serialized separately (stored as their own document/collection) to stay
under the 1 MB Firestore document limit.
"""

from __future__ import annotations

from typing import Any

from .orchestrator import CaseFile


def serialize_case_file(cf: CaseFile) -> dict[str, Any]:
    """Convert a CaseFile dataclass to a JSON-serializable dict.

    Excludes events — they are stored separately via ``save_events``.
    """
    return {
        "dispute_id": cf.dispute_id,
        "dispute": cf.dispute.model_dump(),
        "evidence_pack": cf.evidence_pack.model_dump(),
        "rider_case": cf.rider_case.model_dump() if cf.rider_case else None,
        "driver_case": cf.driver_case.model_dump() if cf.driver_case else None,
        "rider_rebuttal": (
            cf.rider_rebuttal.model_dump() if cf.rider_rebuttal else None
        ),
        "driver_rebuttal": (
            cf.driver_rebuttal.model_dump() if cf.driver_rebuttal else None
        ),
        "policy_result": (
            cf.policy_result.model_dump() if cf.policy_result else None
        ),
        "ruling": cf.ruling.model_dump() if cf.ruling else None,
        "escalated": cf.escalated,
        "escalation_reason": cf.escalation_reason,
        "guard_problems": cf.guard_problems,
        "total_tokens": cf.total_tokens,
        "total_latency_ms": cf.total_latency_ms,
        "risk_output": cf.risk_output.model_dump() if cf.risk_output else None,
        "risk_assessment": _serialize_risk_assessment(cf.risk_assessment),
        "precedent_matches": [
            {
                "case_id": m.case_id,
                "category": m.category,
                "summary": m.summary,
                "outcome": m.outcome,
                "amount": m.amount,
                "similarity": m.similarity,
                "reason": m.reason,
                "source": m.source,
            }
            for m in cf.precedent_matches
        ] if cf.precedent_matches else [],
        "escalation": cf.escalation.to_dict() if cf.escalation else None,
    }


def _serialize_risk_assessment(assessment: Any) -> dict[str, Any] | None:
    if assessment is None:
        return None
    return {
        "risk": assessment.risk,
        "claimant_id": assessment.claimant_id,
        "signals": [
            {"text": s.text, "cites": s.cites, "feature": s.feature}
            for s in assessment.signals
        ],
    }


def serialize_events(cf: CaseFile) -> list[dict[str, Any]]:
    """Extract events from a CaseFile as a list of dicts."""
    if cf.events is None:
        return []
    return [e.model_dump() for e in cf.events.events]
