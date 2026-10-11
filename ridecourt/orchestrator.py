"""Orchestrator — runs the full dispute resolution pipeline.

``resolve(dispute_id) -> CaseFile``:

1. Build the evidence pack (EVIDENCE_REQUEST / EVIDENCE events).
2. Call the policy engine (POLICY_EVAL event).
3. Retrieve precedents (PRECEDENT event).
4. Run both advocates + the Fraud agent in parallel (CLAIM + RISK_SIGNAL events).
5. Give each advocate the other's claim for one rebuttal round
   (REBUTTAL events).
6. Run the guard on every agent output (re-ask once, then escalate).
7. Call the Judge with risk + precedents (RULING or ESCALATION event).
8. On escalation, create an Escalation record in the queue.

Tokens and latency are recorded per step via the event log.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .agents.advocate import Advocate, AdvocateOutput
from .agents.fraud import FraudAgent, FraudOutput
from .agents.judge import (
    Judge,
    JudgeOutput,
    should_escalate,
    neutral_case_summary,
    CONFIDENCE_THRESHOLD,
)
from .agents.precedent import PrecedentAgent, PrecedentOutput
from .data import Dataset, Dispute, EvidencePack, PolicyResult
from .escalation import Escalation, EscalationQueue
from .events import EventLog, monotonic_ms
from .fraud import FraudAssessment
from .guard import Problem, check as guard_check
from .llm import CaseBudget, LLMError, set_case_budget
from .policy import evaluate
from .precedent import PrecedentMatch, retrieve_precedents
from .tools import build_evidence_pack, facts_from_evidence_pack, policy_evidence_item


# ---------------------------------------------------------------------------
# CaseFile — the full output of resolve()
# ---------------------------------------------------------------------------


@dataclass
class CaseFile:
    """The complete result of resolving one dispute."""

    dispute_id: str
    dispute: Dispute
    evidence_pack: EvidencePack
    rider_case: AdvocateOutput | None = None
    driver_case: AdvocateOutput | None = None
    rider_rebuttal: AdvocateOutput | None = None
    driver_rebuttal: AdvocateOutput | None = None
    policy_result: PolicyResult | None = None
    ruling: JudgeOutput | None = None
    escalated: bool = False
    escalation_reason: str = ""
    guard_problems: list[dict[str, Any]] = field(default_factory=list)
    events: EventLog | None = None
    total_tokens: int = 0
    total_latency_ms: int = 0
    # Stretch: fraud + precedent
    risk_output: FraudOutput | None = None
    risk_assessment: FraudAssessment | None = None
    precedent_matches: list[PrecedentMatch] = field(default_factory=list)
    escalation: Escalation | None = None


# ---------------------------------------------------------------------------
# Guard-wrapped agent call
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    output: BaseModel | None
    problems: list[Problem]
    reasked: bool
    failed: bool


def _guard_check(
    output: BaseModel,
    pack: EvidencePack,
    policy_ids: list[str],
    policy_result: PolicyResult | None = None,
    agent_role: str = "advocate",
) -> list[Problem]:
    """Run the guard on an agent output."""
    return guard_check(
        output, pack, policy_ids,
        policy_result=policy_result,
        agent_role=agent_role,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Runs the full pipeline for one or more disputes."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.judge = Judge()
        self.fraud_agent = FraudAgent()
        self.precedent_agent = PrecedentAgent()
        self.escalation_queue = EscalationQueue()

    def resolve(self, dispute_id: str) -> CaseFile:
        """Resolve a dispute end-to-end. Returns a CaseFile."""
        dispute = self.dataset.dispute(dispute_id)
        if dispute is None:
            raise ValueError(f"Dispute {dispute_id} not found")

        # Set up event log + token budget for this case.
        log = EventLog(case_id=dispute_id)
        budget = CaseBudget()
        set_case_budget(budget)

        try:
            return self._run(dispute, log, budget)
        finally:
            set_case_budget(None)
            log.close()

    def _run(
        self,
        dispute: Dispute,
        log: EventLog,
        budget: CaseBudget,
    ) -> CaseFile:
        trip = self.dataset.trip(dispute.trip_id)
        if trip is None:
            raise ValueError(f"Trip {dispute.trip_id} not found")

        # --- 1. DISPUTE_FILED --------------------------------------------
        log.emit(
            "DISPUTE_FILED",
            from_agent="system",
            content={
                "dispute_id": dispute.dispute_id,
                "category": dispute.category,
                "claim_text": dispute.claim_text,
                "filed_by": dispute.filed_by,
            },
        )

        # --- 2. Build evidence pack --------------------------------------
        log.emit(
            "EVIDENCE_REQUEST",
            from_agent="orchestrator",
            content={"category": dispute.category, "trip_id": dispute.trip_id},
        )

        t0 = time.monotonic()
        pack = build_evidence_pack(dispute, self.dataset)
        latency = monotonic_ms(t0)

        log.emit(
            "EVIDENCE",
            from_agent="evidence_tools",
            content={
                "items": [
                    {"id": i.id, "kind": i.kind, "summary": i.summary}
                    for i in pack.items
                ],
            },
            latency_ms=latency,
        )

        # --- 3. Policy engine (facts + evaluate) -------------------------
        facts = facts_from_evidence_pack(pack)
        t0 = time.monotonic()
        policy_result = evaluate(
            dispute, trip, self.dataset.profiles, facts, trip.fare,
        )
        latency = monotonic_ms(t0)

        # Append the policy computation as a citable evidence item so agents
        # can cite E# instead of doing arithmetic on derived quantities.
        next_id = len(pack.items) + 1
        pack.items.append(
            policy_evidence_item(policy_result, facts, next_id)
        )

        policy_ids = list(policy_result.applicable_clauses)

        log.emit(
            "POLICY_EVAL",
            from_agent="policy_engine",
            content={
                "outcome": policy_result.outcome,
                "amount": policy_result.amount,
                "computation": policy_result.computation,
                "clauses": policy_result.applicable_clauses,
            },
            latency_ms=latency,
        )

        # --- Safety short-circuit ----------------------------------------
        # Safety cases skip the advocates and go straight to escalation.
        if dispute.category == "safety":
            summary = neutral_case_summary(
                dispute.claim_text, pack, policy_result,
                reason="safety_category",
            )
            esc = Escalation(
                case_id=dispute.dispute_id,
                reason="safety_category",
                summary=summary,
                severity="high",
            )
            self.escalation_queue.add(esc)
            log.emit(
                "ESCALATION",
                from_agent="judge",
                content={
                    "reason": "safety_category",
                    "summary": summary,
                },
            )
            cf = CaseFile(
                dispute_id=dispute.dispute_id,
                dispute=dispute,
                evidence_pack=pack,
                policy_result=policy_result,
                escalated=True,
                escalation_reason="safety_category",
                escalation=esc,
                events=log,
                total_tokens=budget.spent,
                total_latency_ms=log.total_latency_ms(),
            )
            return cf

        # --- 4. Precedent retrieval (PRECEDENT event) -------------------
        t0 = time.monotonic()
        precedent_matches = retrieve_precedents(
            dispute, self.dataset.precedents, top_k=3,
        )
        prec_latency = monotonic_ms(t0)

        # Check if any precedent was learned from an override.
        has_override_precedent = any(
            m.source == "override" for m in precedent_matches
        )

        log.emit(
            "PRECEDENT",
            from_agent="precedent_agent",
            to_agent="judge",
            content={
                "matches": [
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
                    for m in precedent_matches
                ],
                "learned_from_override": has_override_precedent,
            },
            latency_ms=prec_latency,
        )

        # --- 5. Run both advocates + fraud agent in parallel -----------
        rider_adv = Advocate(side="rider")
        driver_adv = Advocate(side="driver")

        # Determine the claimant (filing party).
        trip = self.dataset.trip(dispute.trip_id)
        claimant_id = trip.rider_id if dispute.filed_by == "rider" else trip.driver_id
        claimant_party = dispute.filed_by

        with ThreadPoolExecutor(max_workers=3) as pool:
            rider_future = pool.submit(
                self._advocate_guarded,
                rider_adv, dispute.claim_text, pack, policy_result, policy_ids,
            )
            driver_future = pool.submit(
                self._advocate_guarded,
                driver_adv, dispute.claim_text, pack, policy_result, policy_ids,
            )
            fraud_future = pool.submit(
                self._fraud_guarded,
                dispute.claim_text, pack, facts,
                claimant_id, claimant_party, dispute.category,
            )
            rider_result = rider_future.result()
            driver_result = driver_future.result()
            fraud_result = fraud_future.result()

        rider_case, rider_meta, rider_problems = rider_result
        driver_case, driver_meta, driver_problems = driver_result
        risk_output, risk_meta, risk_assessment = fraud_result

        # Record CLAIM events.
        if rider_case is not None:
            self._emit_advocate_event(
                log, "CLAIM", "rider_advocate", "judge",
                rider_case, rider_meta, rider_problems,
            )
        if driver_case is not None:
            self._emit_advocate_event(
                log, "CLAIM", "driver_advocate", "judge",
                driver_case, driver_meta, driver_problems,
            )

        # Record RISK_SIGNAL event.
        if risk_output is not None:
            risk_cites: list[str] = []
            for s in risk_output.signals:
                risk_cites.extend(s.cites)
            log.emit(
                "RISK_SIGNAL",
                from_agent="fraud_agent",
                to_agent="judge",
                content={
                    "risk": risk_output.risk,
                    "signals": [
                        {"text": s.text, "cites": s.cites}
                        for s in risk_output.signals
                    ],
                    "claimant_id": claimant_id,
                },
                cites=risk_cites,
                tokens=getattr(risk_meta, "tokens", 0),
                latency_ms=getattr(risk_meta, "latency_ms", 0),
            )

        # If either advocate failed the guard after re-ask, escalate.
        if rider_case is None or driver_case is None:
            reason = "guard_failure_advocate"
            summary = neutral_case_summary(
                dispute.claim_text, pack, policy_result,
                reason=reason,
            )
            esc = Escalation(
                case_id=dispute.dispute_id,
                reason=reason,
                summary=summary,
            )
            self.escalation_queue.add(esc)
            log.emit(
                "ESCALATION",
                from_agent="guard",
                content={
                    "reason": reason,
                    "summary": summary,
                    "rider_problems": [p.__dict__ for p in rider_problems],
                    "driver_problems": [p.__dict__ for p in driver_problems],
                },
            )
            cf = CaseFile(
                dispute_id=dispute.dispute_id,
                dispute=dispute,
                evidence_pack=pack,
                rider_case=rider_case,
                driver_case=driver_case,
                policy_result=policy_result,
                escalated=True,
                escalation_reason=reason,
                guard_problems=self._problem_dicts(rider_problems, driver_problems),
                risk_output=risk_output,
                risk_assessment=risk_assessment,
                precedent_matches=precedent_matches,
                escalation=esc,
                events=log,
                total_tokens=budget.spent,
                total_latency_ms=log.total_latency_ms(),
            )
            return cf

        # --- 5. Rebuttal round -------------------------------------------
        with ThreadPoolExecutor(max_workers=2) as pool:
            rider_reb_future = pool.submit(
                self._rebuttal_guarded,
                rider_adv, dispute.claim_text, pack, driver_case, policy_result, policy_ids,
            )
            driver_reb_future = pool.submit(
                self._rebuttal_guarded,
                driver_adv, dispute.claim_text, pack, rider_case, policy_result, policy_ids,
            )
            rider_reb = rider_reb_future.result()
            driver_reb = driver_reb_future.result()

        rider_rebuttal, rider_reb_meta, rider_reb_problems = rider_reb
        driver_rebuttal, driver_reb_meta, driver_reb_problems = driver_reb

        if rider_rebuttal is not None:
            self._emit_advocate_event(
                log, "REBUTTAL", "rider_advocate", "judge",
                rider_rebuttal, rider_reb_meta, rider_reb_problems,
            )
        else:
            # Rebuttal guard failed after re-ask — drop it and continue.
            log.emit(
                "GUARD_REJECTED",
                from_agent="guard",
                content={
                    "agent": "rider_advocate",
                    "stage": "rebuttal",
                    "problems": [p.__dict__ for p in rider_reb_problems],
                },
            )
        if driver_rebuttal is not None:
            self._emit_advocate_event(
                log, "REBUTTAL", "driver_advocate", "judge",
                driver_rebuttal, driver_reb_meta, driver_reb_problems,
            )
        else:
            # Rebuttal guard failed after re-ask — drop it and continue.
            log.emit(
                "GUARD_REJECTED",
                from_agent="guard",
                content={
                    "agent": "driver_advocate",
                    "stage": "rebuttal",
                    "problems": [p.__dict__ for p in driver_reb_problems],
                },
            )

        # Rebuttals are optional: a failed rebuttal is dropped (above) and
        # the Judge proceeds with the original claims. Escalation happens
        # only on initial-claim guard failure or Judge ruling guard failure.

        # --- 6. Judge ----------------------------------------------------
        t0 = time.monotonic()
        judge_output, judge_meta = self.judge.build(
            dispute.claim_text, pack,
            rider_case, driver_case,
            policy_result,
            risk=risk_output,
            precedents=precedent_matches,
        )
        judge_latency = monotonic_ms(t0)

        # Guard check on judge output — the Judge sees the whole pack,
        # so numbers may come from any evidence item (agent_role="judge").
        judge_problems = _guard_check(
            judge_output, pack, policy_ids, policy_result=policy_result,
            agent_role="judge",
        )

        if judge_problems:
            # Re-ask the Judge once.
            t0 = time.monotonic()
            judge_output, judge_meta = self.judge.build(
                dispute.claim_text, pack,
                rider_case, driver_case,
                policy_result,
                risk=risk_output,
                precedents=precedent_matches,
            )
            judge_latency += monotonic_ms(t0)
            judge_problems_2 = _guard_check(
                judge_output, pack, policy_ids, policy_result=policy_result,
                agent_role="judge",
            )
            if judge_problems_2:
                # Guard failed twice — escalate.
                reason = "guard_failure_judge"
                summary = neutral_case_summary(
                    dispute.claim_text, pack, policy_result,
                    reason=reason, judge_output=judge_output,
                )
                esc = Escalation(
                    case_id=dispute.dispute_id,
                    reason=reason,
                    summary=summary,
                )
                self.escalation_queue.add(esc)
                log.emit(
                    "ESCALATION",
                    from_agent="guard",
                    content={
                        "reason": reason,
                        "summary": summary,
                        "judge_output": judge_output.model_dump(),
                        "policy_outcome": policy_result.outcome,
                        "policy_amount": policy_result.amount,
                        "problems": [p.__dict__ for p in judge_problems_2],
                        "first_round_problems": [p.__dict__ for p in judge_problems],
                    },
                )
                cf = CaseFile(
                    dispute_id=dispute.dispute_id,
                    dispute=dispute,
                    evidence_pack=pack,
                    rider_case=rider_case,
                    driver_case=driver_case,
                    rider_rebuttal=rider_rebuttal,
                    driver_rebuttal=driver_rebuttal,
                    policy_result=policy_result,
                    escalated=True,
                    escalation_reason=reason,
                    guard_problems=self._problem_dicts(judge_problems_2),
                    risk_output=risk_output,
                    risk_assessment=risk_assessment,
                    precedent_matches=precedent_matches,
                    escalation=esc,
                    events=log,
                    total_tokens=budget.spent,
                    total_latency_ms=log.total_latency_ms(),
                )
                return cf

        # --- 7. Ruling vs Escalation -------------------------------------
        escalate = should_escalate(
            judge_output, policy_result,
            category=dispute.category,
            rider_case=rider_case,
            driver_case=driver_case,
        )

        if escalate:
            reason = self._escalation_reason(
                judge_output, policy_result, dispute.category,
                rider_case=rider_case, driver_case=driver_case,
            )
            summary = neutral_case_summary(
                dispute.claim_text, pack, policy_result,
                reason=reason, judge_output=judge_output,
            )
            severity = "high" if reason == "safety_category" else "normal"
            esc = Escalation(
                case_id=dispute.dispute_id,
                reason=reason,
                summary=summary,
                severity=severity,
            )
            self.escalation_queue.add(esc)
            log.emit(
                "ESCALATION",
                from_agent="judge",
                content={
                    "reason": reason,
                    "summary": summary,
                    "judge_output": judge_output.model_dump(),
                    "policy_outcome": policy_result.outcome,
                    "confidence": judge_output.confidence,
                },
                tokens=judge_meta.tokens,
                latency_ms=judge_latency,
            )
            cf = CaseFile(
                dispute_id=dispute.dispute_id,
                dispute=dispute,
                evidence_pack=pack,
                rider_case=rider_case,
                driver_case=driver_case,
                rider_rebuttal=rider_rebuttal,
                driver_rebuttal=driver_rebuttal,
                policy_result=policy_result,
                ruling=judge_output,
                escalated=True,
                escalation_reason=reason,
                guard_problems=self._problem_dicts(judge_problems),
                risk_output=risk_output,
                risk_assessment=risk_assessment,
                precedent_matches=precedent_matches,
                escalation=esc,
                events=log,
                total_tokens=budget.spent,
                total_latency_ms=log.total_latency_ms(),
            )
            return cf

        # Normal ruling.
        log.emit(
            "RULING",
            from_agent="judge",
            to_agent="system",
            content={
                "outcome": judge_output.outcome,
                "amount": judge_output.amount,
                "confidence": judge_output.confidence,
                "message_to_rider": judge_output.message_to_rider,
                "message_to_driver": judge_output.message_to_driver,
                "precedent_consistency": judge_output.precedent_consistency,
                "bad_faith_mentioned": judge_output.bad_faith_mentioned,
            },
            cites=judge_output.cites,
            tokens=judge_meta.tokens,
            latency_ms=judge_latency,
        )

        cf = CaseFile(
            dispute_id=dispute.dispute_id,
            dispute=dispute,
            evidence_pack=pack,
            rider_case=rider_case,
            driver_case=driver_case,
            rider_rebuttal=rider_rebuttal,
            driver_rebuttal=driver_rebuttal,
            policy_result=policy_result,
            ruling=judge_output,
            escalated=False,
            guard_problems=self._problem_dicts(judge_problems),
            risk_output=risk_output,
            risk_assessment=risk_assessment,
            precedent_matches=precedent_matches,
            events=log,
            total_tokens=budget.spent,
            total_latency_ms=log.total_latency_ms(),
        )
        return cf

    # -- Guarded advocate helpers ----------------------------------------

    def _advocate_guarded(
        self,
        advocate: Advocate,
        claim_text: str,
        pack: EvidencePack,
        policy_result: PolicyResult,
        policy_ids: list[str],
    ) -> tuple[AdvocateOutput | None, Any, list[Problem]]:
        """Call an advocate, run the guard, re-ask once, return result.

        Returns ``(output_or_None, meta, problems)``.
        If the output is None, the guard failed twice.
        """
        t0 = time.monotonic()
        output, meta = advocate.build(claim_text, pack, policy_result)
        problems = _guard_check(output, pack, policy_ids, policy_result=policy_result)
        # Record latency on the first call.
        meta.latency_ms += monotonic_ms(t0)  # accumulate for re-ask if any

        if problems:
            # Re-ask once.
            t0 = time.monotonic()
            output2, meta2 = advocate.build(claim_text, pack, policy_result)
            problems2 = _guard_check(
                output2, pack, policy_ids, policy_result=policy_result,
            )
            meta2.latency_ms += monotonic_ms(t0)
            if problems2:
                return None, meta2, problems2
            return output2, meta2, problems2

        return output, meta, problems

    def _rebuttal_guarded(
        self,
        advocate: Advocate,
        claim_text: str,
        pack: EvidencePack,
        opponent: AdvocateOutput,
        policy_result: PolicyResult,
        policy_ids: list[str],
    ) -> tuple[AdvocateOutput | None, Any, list[Problem]]:
        """Call rebuttal, run the guard, re-ask once, return result."""
        t0 = time.monotonic()
        output, meta = advocate.rebut(
            claim_text, pack, opponent, policy_result,
        )
        problems = _guard_check(output, pack, policy_ids, policy_result=policy_result)
        meta.latency_ms += monotonic_ms(t0)

        if problems:
            t0 = time.monotonic()
            output2, meta2 = advocate.rebut(
                claim_text, pack, opponent, policy_result,
            )
            problems2 = _guard_check(
                output2, pack, policy_ids, policy_result=policy_result,
            )
            meta2.latency_ms += monotonic_ms(t0)
            if problems2:
                return None, meta2, problems2
            return output2, meta2, problems2

        return output, meta, problems

    # -- Fraud agent helper -----------------------------------------------

    def _fraud_guarded(
        self,
        claim_text: str,
        pack: EvidencePack,
        facts: Any,
        claimant_id: str,
        claimant_party: str,
        category: str,
    ) -> tuple[FraudOutput | None, Any, FraudAssessment]:
        """Run the fraud agent. The assessment is deterministic so the
        guard is not needed — the signals cite real evidence IDs."""
        output, meta, assessment = self.fraud_agent.build(
            claim_text, pack, facts, self.dataset.profiles,
            claimant_id, claimant_party, category,
        )
        return output, meta, assessment

    # -- Event helpers ----------------------------------------------------

    def _emit_advocate_event(
        self,
        log: EventLog,
        event_type: str,
        from_agent: str,
        to_agent: str,
        case: AdvocateOutput,
        meta: Any,
        problems: list[Problem],
    ) -> None:
        """Emit a CLAIM or REBUTTAL event for an advocate output."""
        cites: list[str] = []
        for arg in case.arguments:
            cites.extend(arg.cites)
        log.emit(
            event_type,  # type: ignore[arg-type]
            from_agent=from_agent,
            to_agent=to_agent,
            content={
                "position": case.position,
                "arguments": [
                    {"text": a.text, "cites": a.cites}
                    for a in case.arguments
                ],
                "requested_outcome": case.requested_outcome,
                "guard_problems": [p.__dict__ for p in problems] if problems else [],
            },
            cites=cites,
            tokens=getattr(meta, "tokens", 0),
            latency_ms=getattr(meta, "latency_ms", 0),
        )

    def _problem_dicts(self, *problem_lists: list[Problem]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for pl in problem_lists:
            for p in pl:
                result.append(p.__dict__)
        return result

    def _escalation_reason(
        self,
        judge_output: JudgeOutput,
        policy_result: PolicyResult,
        category: str,
        *,
        rider_case: AdvocateOutput | None = None,
        driver_case: AdvocateOutput | None = None,
    ) -> str:
        if category == "safety":
            return "safety_category"
        if judge_output.confidence < CONFIDENCE_THRESHOLD:
            return "low_confidence"
        if judge_output.outcome != policy_result.outcome:
            return "outcome_mismatch"
        # Agent-policy disagreement
        policy_outcome = policy_result.outcome.lower()
        for case in (rider_case, driver_case):
            if case is not None:
                requested = case.requested_outcome.lower()
                if "refund" in requested and "no_refund" in policy_outcome:
                    return "agent_policy_disagreement"
                if "no_refund" in requested and "refund" in policy_outcome:
                    return "agent_policy_disagreement"
        return "unknown"
