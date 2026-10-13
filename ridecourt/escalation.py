"""Escalation protocol — formal escalation records and an in-memory queue.

Per SPEC section 3 + stretch goals:

Triggers:
- Confidence below ``CONFIDENCE_THRESHOLD`` (0.7)
- Guard failure after one retry
- Agent disagreement with the policy engine (outcome mismatch)
- Safety category (always)

Each escalation creates a record with a neutral case summary for a human
reviewer. The queue is visible via ``GET /api/escalations``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


EscalationReason = Literal[
    "low_confidence",
    "guard_failure_advocate",
    "guard_failure_judge",
    "outcome_mismatch",
    "safety_category",
    "agent_policy_disagreement",
    "model_unavailable",
    "system_error",
    "unknown",
]


@dataclass
class Escalation:
    """A single escalation record for a human reviewer."""

    case_id: str
    reason: EscalationReason
    summary: str
    severity: str = "normal"  # "normal" | "high" (safety = high)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    status: str = "open"  # "open" | "resolved"
    reviewer: str = ""
    resolution: str = ""
    resolved_at: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "reason": self.reason,
            "summary": self.summary,
            "severity": self.severity,
            "created_at": self.created_at,
            "status": self.status,
            "reviewer": self.reviewer,
            "resolution": self.resolution,
            "resolved_at": self.resolved_at,
        }


class EscalationQueue:
    """In-memory escalation queue for human reviewers."""

    def __init__(self) -> None:
        self._items: list[Escalation] = []

    def add(self, escalation: Escalation) -> Escalation:
        """Add an escalation to the queue."""
        self._items.append(escalation)
        return escalation

    def all(self) -> list[Escalation]:
        """Return all escalations, sorted by severity (high first) then age."""
        severity_order = {"high": 0, "normal": 1}
        return sorted(
            self._items,
            key=lambda e: (severity_order.get(e.severity, 1), e.created_at),
        )

    def open(self) -> list[Escalation]:
        """Return only open escalations."""
        return [e for e in self.all() if e.status == "open"]

    def resolve(self, case_id: str, reviewer: str, resolution: str) -> bool:
        """Mark all open escalations for *case_id* as resolved.

        A human reviewer's override resolves every open escalation for
        that case. Returns True if at least one was resolved.
        """
        found = False
        for e in self._items:
            if e.case_id == case_id and e.status == "open":
                e.status = "resolved"
                e.reviewer = reviewer
                e.resolution = resolution
                e.resolved_at = datetime.now(timezone.utc).isoformat()
                found = True
        return found

    def by_case(self, case_id: str) -> list[Escalation]:
        """Return all escalations for a case."""
        return [e for e in self._items if e.case_id == case_id]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
