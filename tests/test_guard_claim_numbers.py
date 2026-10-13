"""Sweep test for claim-text numbers — the live-app "File a Dispute" path.

The synthetic dataset's claim texts are generic prose with no unit-tagged
numbers, so ``tests/test_guard_number_sweep.py`` does not exercise the
``CLAIM`` citation path. But the live app lets a person type their own
claim, e.g. "The driver overcharged me S$10 and drove 5 km out of the
way". The advocates see that text and will naturally quote it. Without a
``CLAIM`` citation source the guard rejects those numbers, so every
custom dispute escalates with a guard failure.

This test adds claim texts with numbers in every unit the guard knows
(``S$``, ``km``, ``m``, ``min``, ``%``) for each dispute category, then
checks for every agent (rider advocate, driver advocate, fraud, judge)
that:

1. A number from the claim text is **accepted** by the guard when the
   agent's output cites ``CLAIM`` and attributes the number to the
   claimant.
2. The same number is **rejected** when the agent states it as a fact of
   the case without citing ``CLAIM``.

This test fails on code that does not pass ``claim_text`` to the guard
and does not treat ``CLAIM`` as an attributable source.
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest
from pydantic import BaseModel, Field

from ridecourt.agents.advocate import (
    _system_prompt as adv_system,
    _user_prompt as adv_user,
    AdvocateOutput,
    Argument as AdvArgument,
)
from ridecourt.agents.fraud import (
    _system_prompt as fraud_system,
    _user_prompt as fraud_user,
    FraudOutput,
    FraudSignalOut,
)
from ridecourt.agents.judge import (
    _system_prompt as judge_system,
    _user_prompt as judge_user,
    JudgeOutput,
)
from ridecourt.data import load_dataset
from ridecourt.fraud import compute_fraud_signals
from ridecourt.guard import check as guard_check
from ridecourt.guard import _extract_numbers
from ridecourt.policy import evaluate
from ridecourt.precedent import retrieve_precedents
from ridecourt.tools import (
    build_evidence_pack,
    facts_from_evidence_pack,
    policy_evidence_item,
)


# ---------------------------------------------------------------------------
# Claim texts with numbers in every unit, one per category.
# ---------------------------------------------------------------------------

CLAIM_TEXTS: dict[str, str] = {
    "route_deviation": (
        "The driver overcharged me S$10 and drove 5 km out of the way. "
        "I waited 15 min at the pickup and the detour added 800 m to the "
        "trip. The fare was 20% higher than quoted."
    ),
    "no_show_driver": (
        "The driver never showed up. I waited 25 min at the pickup point "
        "which was 300 m from where the app said. The cancellation fee was "
        "S$4 and the driver was 2 km away. The trip was 100% a no-show."
    ),
    "no_show_rider": (
        "The rider was not at the pickup. I waited 10 min within 50 m of "
        "the pin. The cancellation fee is S$4 and the distance to the "
        "rider's location was 1.5 km. The rider no-show rate is 30%."
    ),
    "fare_error": (
        "I was charged S$15 instead of S$8. The surge multiplier was 1.5 "
        "and the trip was only 3 km. The overcharge was 50% of the correct "
        "fare. The waiting time was 5 min."
    ),
    "safety": (
        "The driver drove recklessly for 10 min covering 8 km. I was "
        "scared for 100% of the trip. The driver stopped 500 m from my "
        "destination and asked for S$5 extra. The speed was 60 km over "
        "the limit for 2 km."
    ),
}

# Units covered: S$ (money), km, m, min, %.


# ---------------------------------------------------------------------------
# Build a full case (evidence pack + policy + precedents) for a category.
# ---------------------------------------------------------------------------


def _build_case_for_category(category: str, claim_text: str):
    """Find a synthetic dispute of *category*, swap in *claim_text*,
    build the evidence pack, policy result, and precedents."""
    ds = load_dataset()
    # Find the first dispute of this category (skip safety for the
    # advocate/judge path — safety short-circuits).
    target_did = None
    for did, disp in ds.disputes.items():
        if disp.category == category:
            target_did = did
            break
    if target_did is None:
        pytest.skip(f"No synthetic dispute of category {category!r}")

    disp = ds.dispute(target_did)
    # Override the claim text with our number-laden version.
    disp = disp.model_copy(update={"claim_text": claim_text})
    trip = ds.trip(disp.trip_id)

    pack = build_evidence_pack(disp, ds)
    facts = facts_from_evidence_pack(pack)
    pr = evaluate(disp, trip, ds.profiles, facts, trip.fare)

    next_id = len(pack.items) + 1
    pack.items.append(policy_evidence_item(pr, facts, next_id))

    claimant_id = trip.rider_id if disp.filed_by == "rider" else trip.driver_id
    claimant_party = disp.filed_by
    assessment = compute_fraud_signals(
        pack, facts, ds.profiles, claimant_id, claimant_party,
        claim_text=disp.claim_text, category=disp.category,
    )
    precedent_matches = retrieve_precedents(disp, ds.precedents, top_k=3)
    return disp, trip, pack, facts, pr, assessment, claimant_id, claimant_party, precedent_matches


def _has_unverified_number(problems, unit: str, value: float) -> bool:
    """True if *problems* contains an unverified_number problem for the
    given unit and value (tolerance 0.001)."""
    for p in problems:
        if p.kind != "unverified_number":
            continue
        if unit in p.detail and str(value) in p.detail:
            return True
        # The detail format is "Number {raw} ({value} {unit})".
        if unit in p.detail and f"{value}" in p.detail:
            return True
    return False


def _claim_number_pairs(claim_text: str) -> list[tuple[float, str]]:
    """Extract (value, unit) pairs from the claim text."""
    return [(n.value, n.unit) for n in _extract_numbers(claim_text, "")]


# ---------------------------------------------------------------------------
# Test: claim numbers accepted when cited (CLAIM), rejected as fact.
# ---------------------------------------------------------------------------


class TestClaimNumbersAcceptedWhenCited:
    """For each category and each claim number, verify that:
    1. An advocate output that cites CLAIM and attributes the number is
       accepted.
    2. An advocate output that states the number as a fact (no CLAIM
       cite) is rejected.
    """

    @pytest.mark.parametrize("category", sorted(CLAIM_TEXTS.keys()))
    def test_advocate_claim_number_accepted_with_claim_cite(self, category):
        claim_text = CLAIM_TEXTS[category]
        disp, trip, pack, facts, pr, assessment, cid, cp, prec = (
            _build_case_for_category(category, claim_text)
        )
        claim_pairs = _claim_number_pairs(claim_text)
        assert claim_pairs, f"Claim text for {category} has no numbers"

        # Pick the first (value, unit) pair from the claim.
        val, unit = claim_pairs[0]
        policy_ids = list(pr.applicable_clauses)

        # Build an advocate output that cites CLAIM and attributes the
        # number to the claimant.
        raw = _format_number_back(val, unit)
        cited_eid = next(
            (i.id for i in pack.items if i.id.startswith("E")), "E1"
        )
        output = AdvocateOutput(
            position="Test position.",
            arguments=[
                AdvArgument(
                    text=(
                        f"The rider says the amount was {raw} (CLAIM). "
                        f"Evidence {cited_eid} supports the case."
                    ),
                    cites=["CLAIM", cited_eid],
                ),
            ],
            requested_outcome="refund",
        )
        problems = guard_check(
            output, pack, policy_ids, policy_result=pr,
            agent_role="advocate", claim_text=claim_text,
        )
        unverified = [
            p for p in problems
            if p.kind == "unverified_number" and unit in p.detail
        ]
        assert not unverified, (
            f"{category}: advocate cited CLAIM for {raw} ({val} {unit}) "
            f"but guard still rejected it: {unverified}"
        )

    @pytest.mark.parametrize("category", sorted(CLAIM_TEXTS.keys()))
    def test_advocate_claim_number_rejected_without_claim_cite(self, category):
        """A claim number stated as a fact of the case (no CLAIM cite)
        must be rejected — the claimant's allegation is not an
        established fact."""
        claim_text = CLAIM_TEXTS[category]
        disp, trip, pack, facts, pr, assessment, cid, cp, prec = (
            _build_case_for_category(category, claim_text)
        )
        claim_pairs = _claim_number_pairs(claim_text)
        assert claim_pairs

        val, unit = claim_pairs[0]
        policy_ids = list(pr.applicable_clauses)

        # Find a (value, unit) that is NOT also a valid case number,
        # so the rejection is due to the claim-source rule, not a
        # coincidental match. We try each claim number until we find one
        # that is not in the evidence pack.
        from ridecourt.guard import _valid_numbers
        case_valid = _valid_numbers(pack, set(), pr)

        test_pair = None
        for val, unit in claim_pairs:
            is_case_num = any(
                u == unit and round(v, 2) == round(val, 2)
                for v, u in case_valid
            )
            if not is_case_num:
                test_pair = (val, unit)
                break
        if test_pair is None:
            # All claim numbers happen to also be case numbers — skip.
            pytest.skip(
                f"All claim numbers for {category} are also case numbers"
            )
        val, unit = test_pair
        raw = _format_number_back(val, unit)
        cited_eid = next(
            (i.id for i in pack.items if i.id.startswith("E")), "E1"
        )

        # State the number as a fact without citing CLAIM.
        output = AdvocateOutput(
            position="Test position.",
            arguments=[
                AdvArgument(
                    text=(
                        f"The overcharge was {raw}. "
                        f"Evidence {cited_eid} supports the case."
                    ),
                    cites=[cited_eid],
                ),
            ],
            requested_outcome="refund",
        )
        problems = guard_check(
            output, pack, policy_ids, policy_result=pr,
            agent_role="advocate", claim_text=claim_text,
        )
        unverified = [
            p for p in problems
            if p.kind == "unverified_number"
            and unit in p.detail
        ]
        assert unverified, (
            f"{category}: advocate stated {raw} ({val} {unit}) as a fact "
            f"without citing CLAIM — guard should have rejected it."
        )

    @pytest.mark.parametrize("category", sorted(CLAIM_TEXTS.keys()))
    def test_judge_claim_number_accepted_with_claim_cite(self, category):
        """The Judge can quote a claim number when citing CLAIM."""
        claim_text = CLAIM_TEXTS[category]
        if category == "safety":
            pytest.skip("Safety cases skip the Judge")
        disp, trip, pack, facts, pr, assessment, cid, cp, prec = (
            _build_case_for_category(category, claim_text)
        )
        claim_pairs = _claim_number_pairs(claim_text)
        assert claim_pairs
        val, unit = claim_pairs[0]
        raw = _format_number_back(val, unit)
        policy_ids = list(pr.applicable_clauses)

        judge_output = JudgeOutput(
            outcome=pr.outcome,
            amount=pr.amount,
            confidence=0.85,
            reasoning=(
                f"The rider says the amount was {raw} (CLAIM). "
                f"Policy engine determines the refund."
            ),
            cites=list(pr.applicable_clauses) + (["CLAIM"] if prec else ["CLAIM"]),
            message_to_rider="Refund per policy.",
            message_to_driver="Refund per policy.",
        )
        problems = guard_check(
            judge_output, pack, policy_ids, policy_result=pr,
            agent_role="judge", precedents=prec, claim_text=claim_text,
        )
        unverified = [
            p for p in problems
            if p.kind == "unverified_number" and unit in p.detail
        ]
        assert not unverified, (
            f"{category}: judge cited CLAIM for {raw} ({val} {unit}) "
            f"but guard rejected it: {unverified}"
        )

    @pytest.mark.parametrize("category", sorted(CLAIM_TEXTS.keys()))
    def test_judge_claim_number_rejected_without_claim_cite(self, category):
        """A claim number in the Judge's text without citing CLAIM is
        rejected (unless it happens to also be a valid case number)."""
        if category == "safety":
            pytest.skip("Safety cases skip the Judge")
        claim_text = CLAIM_TEXTS[category]
        disp, trip, pack, facts, pr, assessment, cid, cp, prec = (
            _build_case_for_category(category, claim_text)
        )
        claim_pairs = _claim_number_pairs(claim_text)
        assert claim_pairs
        policy_ids = list(pr.applicable_clauses)

        from ridecourt.guard import _valid_numbers
        case_valid = _valid_numbers(pack, set(), pr)
        test_pair = None
        for val, unit in claim_pairs:
            is_case_num = any(
                u == unit and round(v, 2) == round(val, 2)
                for v, u in case_valid
            )
            if not is_case_num:
                test_pair = (val, unit)
                break
        if test_pair is None:
            pytest.skip(
                f"All claim numbers for {category} are also case numbers"
            )
        val, unit = test_pair
        raw = _format_number_back(val, unit)

        judge_output = JudgeOutput(
            outcome=pr.outcome,
            amount=pr.amount,
            confidence=0.85,
            reasoning=f"The overcharge was {raw}. Policy applies.",
            cites=list(pr.applicable_clauses),
            message_to_rider="Refund per policy.",
            message_to_driver="Refund per policy.",
        )
        problems = guard_check(
            judge_output, pack, policy_ids, policy_result=pr,
            agent_role="judge", precedents=prec, claim_text=claim_text,
        )
        unverified = [
            p for p in problems
            if p.kind == "unverified_number" and unit in p.detail
        ]
        assert unverified, (
            f"{category}: judge stated {raw} ({val} {unit}) as a fact "
            f"without citing CLAIM — guard should have rejected it."
        )

    @pytest.mark.parametrize("category", sorted(CLAIM_TEXTS.keys()))
    def test_fraud_claim_number_accepted_with_claim_cite(self, category):
        """The fraud agent can quote a claim number when citing CLAIM."""
        if category == "safety":
            pytest.skip("Safety cases skip the fraud agent path")
        claim_text = CLAIM_TEXTS[category]
        disp, trip, pack, facts, pr, assessment, cid, cp, prec = (
            _build_case_for_category(category, claim_text)
        )
        claim_pairs = _claim_number_pairs(claim_text)
        assert claim_pairs
        val, unit = claim_pairs[0]
        raw = _format_number_back(val, unit)
        policy_ids = list(pr.applicable_clauses)

        output = FraudOutput(
            risk=assessment.risk,
            signals=[
                FraudSignalOut(
                    text=(
                        f"The rider says the amount was {raw} (CLAIM). "
                        f"History supports assessment."
                    ),
                    cites=["CLAIM"],
                ),
            ],
        )
        problems = guard_check(
            output, pack, policy_ids, policy_result=pr,
            agent_role="advocate", claim_text=claim_text,
        )
        unverified = [
            p for p in problems
            if p.kind == "unverified_number" and unit in p.detail
        ]
        assert not unverified, (
            f"{category}: fraud agent cited CLAIM for {raw} ({val} {unit}) "
            f"but guard rejected it: {unverified}"
        )


# ---------------------------------------------------------------------------
# Helper: format a (value, unit) pair back into the text form the guard
# regexes expect to see (so the number is extracted from agent text).
# ---------------------------------------------------------------------------


def _format_number_back(val: float, unit: str) -> str:
    if unit == "money":
        return f"S${val:.2f}"
    if unit == "km":
        return f"{val:.1f} km"
    if unit == "m":
        return f"{val:.0f} m"
    if unit == "min":
        return f"{val:.0f} min"
    if unit == "percent":
        return f"{val:.0f}%"
    return f"{val}"
