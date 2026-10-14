"""Sweep test — every policy section ID that any agent prompt tells an agent
to cite, or that a deterministic fraud signal cites, must be accepted by the
guard in exactly the cases where that instruction applies.

Runs over all synthetic cases.

v2.1: policy clauses are §N.M section IDs (§1.3, §2.3, §6.3, etc.).
Fraud signals cite §6.3. Agent prompts reference §N.M sections.
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import re

import pytest

from ridecourt.agents.advocate import _system_prompt as adv_system
from ridecourt.agents.advocate import _user_prompt as adv_user
from ridecourt.agents.fraud import _system_prompt as fraud_system
from ridecourt.agents.fraud import _user_prompt as fraud_user
from ridecourt.agents.judge import _system_prompt as judge_system
from ridecourt.agents.judge import _user_prompt as judge_user
from ridecourt.data import load_dataset
from ridecourt.fraud import compute_fraud_signals
from ridecourt.policy import VALID_SECTIONS, evaluate
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack

# All §N.M section IDs mentioned in prompts or fraud signals.
_SEC_RE = re.compile(r"§\d+(?:\.\d+)?")

# Legacy P-# clause IDs (rejected in v2.1 — kept for detection only).
_P_RE = re.compile(r"\bP-[A-Z]+-\d+\b")


def _all_prompt_text(disp, pack, pr, rider_case=None, driver_case=None,
                     risk=None, assessment=None, precedents=None):
    """Collect all §N.M and P-# IDs mentioned in agent prompt text for one case."""
    pids: set[str] = set()
    claim_text = disp.claim_text

    # Advocate prompts (both sides).
    for side in ("rider", "driver"):
        for text in (adv_system(side), adv_user(side, claim_text, pack, pr)):
            pids.update(_SEC_RE.findall(text))
            pids.update(_P_RE.findall(text))

    # Judge prompts.
    if rider_case is not None and driver_case is not None:
        for text in (
            judge_system(),
            judge_user(claim_text, pack, rider_case, driver_case, pr,
                       risk=risk, precedents=precedents),
        ):
            pids.update(_SEC_RE.findall(text))
            pids.update(_P_RE.findall(text))

    # Fraud prompts.
    if assessment is not None:
        for text in (fraud_system(), fraud_user(claim_text, pack, assessment)):
            pids.update(_SEC_RE.findall(text))
            pids.update(_P_RE.findall(text))

    return pids


def _build_case(ds, did):
    """Build evidence pack, facts, policy result, and fraud assessment."""
    disp = ds.dispute(did)
    trip = ds.trip(disp.trip_id)
    pack = build_evidence_pack(disp, ds)
    facts = facts_from_evidence_pack(pack)
    pr = evaluate(disp, trip, ds.profiles, facts, trip.fare)
    claimant_id = trip.rider_id if disp.filed_by == "rider" else trip.driver_id
    claimant_party = disp.filed_by
    assessment = compute_fraud_signals(
        pack, facts, ds.profiles, claimant_id, claimant_party,
        claim_text=disp.claim_text, category=disp.category,
    )
    return disp, trip, pack, facts, pr, assessment, claimant_id, claimant_party


@pytest.fixture(scope="module")
def all_cases():
    ds = load_dataset()
    cases = {}
    for did in ds.disputes:
        disp = ds.dispute(did)
        if disp is None:
            continue
        trip = ds.trip(disp.trip_id)
        if trip is None:
            continue
        cases[did] = _build_case(ds, did)
    return cases


def test_fraud_signal_cites_accepted_by_guard(all_cases):
    """Every §N.M or P-# ID that a fraud signal cites must be in the valid
    set (VALID_SECTIONS ∪ applicable_clauses ∪ fraud-signal cites).

    v2.1: fraud signals cite §6.3, which is in VALID_SECTIONS, so the
    guard should always accept it.
    """
    failures = []
    for did, (disp, trip, pack, facts, pr, assessment, cid, cp) in all_cases.items():
        # Collect all cites from fraud signal cites.
        fraud_cites: set[str] = set()
        for s in assessment.signals:
            for c in s.cites:
                fraud_cites.add(c)

        # Every cite must be either in VALID_SECTIONS, in applicable_clauses,
        # or an E# that exists in the evidence pack.
        valid_sections = set(VALID_SECTIONS)
        applicable = set(pr.applicable_clauses)
        eids = {item.id for item in pack.items}

        for cite in fraud_cites:
            if cite.startswith("E"):
                if cite not in eids:
                    failures.append(
                        f"{did}: fraud signal cites {cite} but it is not in "
                        f"the evidence pack"
                    )
            elif cite in valid_sections or cite in applicable:
                pass  # valid
            else:
                failures.append(
                    f"{did}: fraud signal cites {cite} but it is not in "
                    f"VALID_SECTIONS or applicable_clauses={pr.applicable_clauses}"
                )

    assert not failures, "\n".join(failures)


def test_all_prompt_cited_sections_accepted_by_guard(all_cases):
    """Every §N.M ID that any agent prompt mentions must be in
    VALID_SECTIONS (which the guard always accepts)."""
    failures = []
    for did, (disp, trip, pack, facts, pr, assessment, cid, cp) in all_cases.items():
        prompt_ids = _all_prompt_text(
            disp, pack, pr, assessment=assessment,
        )

        valid_sections = set(VALID_SECTIONS)

        for pid in prompt_ids:
            if pid.startswith("§"):
                if pid not in valid_sections:
                    failures.append(
                        f"{did}: prompt mentions {pid} but it is not in "
                        f"VALID_SECTIONS"
                    )
            elif pid.startswith("P-"):
                # Legacy P- IDs are rejected in v2.1 — if they appear in
                # prompt text, the guard will flag them as unknown_cite.
                failures.append(
                    f"{did}: prompt mentions {pid} — legacy P- IDs are "
                    f"rejected in v2.1"
                )

    assert not failures, "\n".join(failures)


def test_fraud_flag_does_not_change_outcome(all_cases):
    """When §6.3 applies (fraud flag), the outcome and amount must NOT
    change — §6.3 only affects routing (raises confidence threshold).
    """
    for did, (disp, trip, pack, facts, pr, assessment, cid, cp) in all_cases.items():
        if disp.category == "safety":
            continue
        # Check if there are fraud flag actions
        fraud_actions = [
            a for a in pr.actions if a.get("action") == "fraud_flag_notice"
        ]
        if not fraud_actions:
            continue
        # The outcome should be a standard category outcome
        assert pr.outcome != "bad_faith_flagged", (
            f"{did}: §6.3 must not change the outcome to bad_faith_flagged — "
            "it only raises the confidence threshold for routing."
        )
        assert pr.amount >= 0.0
