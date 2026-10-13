"""Sweep test — every number in every piece of text the prompt builder shows
an agent must be either accepted by the guard for that agent, or deliberately
excluded with a stated reason.

Runs over all 48 synthetic cases.

For each case, the test:
1. Builds the evidence pack, policy result, fraud assessment, and precedents.
2. For every agent (rider advocate, driver advocate, rebuttals, fraud, judge),
   collects every piece of text the prompt builder shows the agent:
   - system prompt
   - user prompt (claim text, evidence pack, policy computation, precedents,
     risk signals, opponent's case)
3. Extracts every number with a unit (money, km, m, min, percent) from that text.
4. Checks that each number is either:
   a. Accepted by the guard for that agent (appears in the valid number set
      for that agent's role), or
   b. Deliberately excluded with a stated reason (documented in EXCLUSIONS).

This test fails on code that does not pass precedents to the guard, because
precedent summaries contain numbers (e.g. 400 m from PC-2025-004) that the
Judge sees in its prompt and may mention — but the guard would reject them
as unverified.
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest

from ridecourt.agents.advocate import (
    _system_prompt as adv_system,
    _user_prompt as adv_user,
    AdvocateOutput,
)
from ridecourt.agents.fraud import (
    _system_prompt as fraud_system,
    _user_prompt as fraud_user,
)
from ridecourt.agents.judge import (
    _system_prompt as judge_system,
    _user_prompt as judge_user,
)
from ridecourt.data import load_dataset
from ridecourt.fraud import compute_fraud_signals
from ridecourt.guard import _extract_numbers, _valid_numbers
from ridecourt.policy import evaluate
from ridecourt.precedent import retrieve_precedents
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack


# ---------------------------------------------------------------------------
# Exclusions: numbers shown to an agent but deliberately NOT accepted by
# the guard, with the reason.  Each entry is (agent_role, unit, value,
# reason).
# ---------------------------------------------------------------------------

EXCLUSIONS: set[tuple[str, str, float, str]] = set()

# Claim text numbers: synthetic claim texts are generic ("The driver took a
# longer route", "I was charged a cancellation fee") and contain no
# unit-tagged numbers, so no exclusions are needed for synthetic claim
# text.  A separate test suite (tests/test_guard_claim_numbers.py) adds
# claim texts with numbers in every unit to verify the CLAIM citation
# path works for live (user-filed) disputes.

# Policy clause IDs in system prompts contain numbers (e.g. "P-RD-1",
# "P-BF-1") but these are IDs, not unit-tagged numbers — the guard's
# number regex only matches numbers followed by a unit (km, m, min,
# money, %), so they are not extracted and not a concern.

# Precedent summaries contain numbers (400 m, 3 km, S$4.20, etc.).
# The Judge sees these in its prompt.  After the fix, these numbers are
# accepted by the guard only when the Judge cites the precedent's ID
# (PC-YYYY-NNN).  The sweep test checks that every precedent number IS
# accepted by the guard when the precedent is cited.  No exclusion needed.


# ---------------------------------------------------------------------------
# Helpers: build full prompt text for each agent
# ---------------------------------------------------------------------------


def _build_case(ds, did):
    """Build evidence pack, facts, policy result, fraud assessment, precedents."""
    disp = ds.dispute(did)
    trip = ds.trip(disp.trip_id)
    pack = build_evidence_pack(disp, ds)
    facts = facts_from_evidence_pack(pack)
    pr = evaluate(disp, trip, ds.profiles, facts, trip.fare)

    # Append the policy evidence item (as the orchestrator does).
    from ridecourt.tools import policy_evidence_item
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


def _all_prompt_pieces_for_advocate(disp, pack, pr):
    """Return (label, text) pairs for every piece of text shown to an advocate."""
    pieces = []
    for side in ("rider", "driver"):
        pieces.append((f"advocate_{side}_system", adv_system(side)))
        pieces.append((f"advocate_{side}_user", adv_user(side, disp.claim_text, pack, pr)))
    return pieces


def _all_prompt_pieces_for_rebuttal(disp, pack, pr, opponent_output, side):
    """Return (label, text) pairs for the rebuttal prompt.

    The rebuttal is built inline in Advocate.rebut(), so we reconstruct
    the key text pieces: system prompt (inline), user prompt (inline with
    evidence + opponent case).
    """
    who = "Rider Advocate" if side == "rider" else "Driver Advocate"
    opp = "Driver Advocate" if side == "rider" else "Rider Advocate"
    system = (
        f"You are the {who}. The {opp} has presented their case. "
        "Provide a short rebuttal: identify the weakest point in the "
        "opponent's argument and reinforce your side's position. "
        "Cite E# and P-.. IDs. Be brief (2-3 arguments max). "
        "Copy numbers exactly as shown in the evidence — never compute, "
        "round, or estimate new numbers. "
        "Return JSON matching the provided schema."
    )
    # User prompt is built inline; reconstruct the relevant text.
    lines = []
    lines.append(f"Claim [CLAIM]: {disp.claim_text}\n")
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
    if pr is not None:
        lines.append(f"Policy engine: {pr.outcome}, S${pr.amount:.2f}")
    user = "\n".join(lines)
    return [
        (f"rebuttal_{side}_system", system),
        (f"rebuttal_{side}_user", user),
    ]


def _all_prompt_pieces_for_fraud(disp, pack, assessment):
    """Return (label, text) pairs for the fraud agent prompt."""
    return [
        ("fraud_system", fraud_system()),
        ("fraud_user", fraud_user(disp.claim_text, pack, assessment)),
    ]


def _all_prompt_pieces_for_judge(disp, pack, rider_case, driver_case, pr, risk, precedents):
    """Return (label, text) pairs for the judge prompt."""
    return [
        ("judge_system", judge_system()),
        ("judge_user", judge_user(
            disp.claim_text, pack, rider_case, driver_case, pr,
            risk=risk, precedents=precedents,
        )),
    ]


# ---------------------------------------------------------------------------
# Valid number sets per agent role
# ---------------------------------------------------------------------------


def _valid_nums_for_advocate(pack, pr, cited_eids=None):
    """Numbers accepted by the guard for an advocate (cited-evidence-only)."""
    # If no cites specified, use all items (same as judge for sweep purposes).
    if cited_eids is None:
        # Advocates cite evidence, but for the sweep we check all numbers
        # the advocate *sees* — which are from all evidence items in the pack.
        # The guard accepts numbers from cited evidence only, so for the sweep
        # we use all evidence (the broadest set the advocate might cite).
        return _valid_numbers(pack, set(), pr)
    return _valid_numbers(pack, cited_eids, pr)


def _valid_nums_for_judge(pack, pr):
    """Numbers accepted by the guard for the Judge (all evidence + policy)."""
    return _valid_numbers(pack, set(), pr)


def _precedent_nums(precedents):
    """Return dict: precedent_id -> list of (value, unit) pairs from summary."""
    result = {}
    for p in precedents:
        pid = getattr(p, "case_id", "")
        summary = getattr(p, "summary", "")
        pairs = [(n.value, n.unit) for n in _extract_numbers(summary, "")]
        result[pid] = pairs
    return result


def _all_precedent_nums(precedents):
    """All numbers from all precedents (for checking if a number is a precedent number).
    Includes both summary text numbers and the amount field."""
    nums = []
    for p in precedents:
        summary = getattr(p, "summary", "")
        for n in _extract_numbers(summary, ""):
            nums.append((n.value, n.unit, getattr(p, "case_id", "")))
        amount = getattr(p, "amount", None)
        if amount is not None:
            nums.append((float(amount), "money", getattr(p, "case_id", "")))
    return nums


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        if disp.category == "safety":
            continue  # safety cases skip advocates and judge
        cases[did] = _build_case(ds, did)
    return cases


@pytest.fixture(scope="module")
def mock_rider_case(all_cases):
    """Build a mock rider_case output for each case (needed for judge prompt)."""
    from ridecourt.llm import complete_json
    from ridecourt.agents.advocate import Advocate
    ds = load_dataset()
    results = {}
    for did, (disp, trip, pack, facts, pr, assessment, cid, cp, prec) in all_cases.items():
        adv = Advocate(side="rider")
        output, _ = complete_json(
            adv_system("rider"),
            adv_user("rider", disp.claim_text, pack, pr),
            AdvocateOutput,
            "rider_advocate",
            evidence_pack=pack,
            policy_result=pr,
        )
        results[did] = output
    return results


@pytest.fixture(scope="module")
def mock_driver_case(all_cases):
    """Build a mock driver_case output for each case (needed for judge prompt)."""
    from ridecourt.llm import complete_json
    from ridecourt.agents.advocate import Advocate
    ds = load_dataset()
    results = {}
    for did, (disp, trip, pack, facts, pr, assessment, cid, cp, prec) in all_cases.items():
        adv = Advocate(side="driver")
        output, _ = complete_json(
            adv_system("driver"),
            adv_user("driver", disp.claim_text, pack, pr),
            AdvocateOutput,
            "driver_advocate",
            evidence_pack=pack,
            policy_result=pr,
        )
        results[did] = output
    return results


# ---------------------------------------------------------------------------
# Test: every number in advocate prompts is accepted or excluded
# ---------------------------------------------------------------------------


def test_advocate_prompt_numbers_accepted(all_cases):
    """Every number with a unit in advocate prompt text must be accepted by
    the guard for the advocate role (numbers from evidence/policy computation).
    """
    failures = []
    for did, (disp, trip, pack, facts, pr, assessment, cid, cp, prec) in all_cases.items():
        pieces = _all_prompt_pieces_for_advocate(disp, pack, pr)
        # For advocates, the guard accepts numbers from cited evidence + policy.
        # The advocate prompt shows ALL evidence items, so all evidence numbers
        # are valid (the advocate can cite any of them).
        valid = _valid_nums_for_advocate(pack, pr)
        for label, text in pieces:
            extracted = _extract_numbers(text, label)
            for num in extracted:
                matched = any(
                    unit == num.unit and round(val, _dec_places(num.raw)) == num.value
                    for val, unit in valid
                )
                if not matched:
                    # Check if it's in the exclusions list.
                    is_excluded = any(
                        ex_role == "advocate" and ex_unit == num.unit
                        and abs(ex_val - num.value) < 0.001
                        for ex_role, ex_unit, ex_val, _ in EXCLUSIONS
                    )
                    if not is_excluded:
                        failures.append(
                            f"{did} [{label}]: number {num.raw} ({num.value} "
                            f"{num.unit}) in prompt text is not accepted by "
                            f"the guard for advocates."
                        )
    assert not failures, "\n".join(failures)


def test_fraud_prompt_numbers_accepted(all_cases):
    """Every number with a unit in fraud agent prompt text must be accepted
    by the guard for the fraud agent (same as advocate: cited evidence + policy).
    """
    failures = []
    for did, (disp, trip, pack, facts, pr, assessment, cid, cp, prec) in all_cases.items():
        pieces = _all_prompt_pieces_for_fraud(disp, pack, assessment)
        valid = _valid_nums_for_advocate(pack, pr)
        for label, text in pieces:
            extracted = _extract_numbers(text, label)
            for num in extracted:
                matched = any(
                    unit == num.unit and round(val, _dec_places(num.raw)) == num.value
                    for val, unit in valid
                )
                if not matched:
                    is_excluded = any(
                        ex_role == "fraud" and ex_unit == num.unit
                        and abs(ex_val - num.value) < 0.001
                        for ex_role, ex_unit, ex_val, _ in EXCLUSIONS
                    )
                    if not is_excluded:
                        failures.append(
                            f"{did} [{label}]: number {num.raw} ({num.value} "
                            f"{num.unit}) in prompt text is not accepted by "
                            f"the guard for the fraud agent."
                        )
    assert not failures, "\n".join(failures)


def test_judge_prompt_numbers_accepted(all_cases, mock_rider_case, mock_driver_case):
    """Every number with a unit in judge prompt text must be accepted by the
    guard for the judge role (all evidence + policy + precedents when cited).

    This test fails on code that does not pass precedents to the guard,
    because precedent summaries contain numbers (e.g. 400 m from
    PC-2025-004) that the Judge sees in its prompt but the guard would
    reject as unverified.
    """
    from ridecourt.guard import check as guard_check
    from ridecourt.agents.judge import JudgeOutput
    from pydantic import BaseModel, Field

    class _DummyJudgeOutput(BaseModel):
        """Minimal model to test guard number acceptance with a precedent number."""
        outcome: str = "test"
        amount: float = 0.0
        confidence: float = 0.9
        reasoning: str = ""
        cites: list[str] = Field(default_factory=list)
        message_to_rider: str = ""
        message_to_driver: str = ""

    failures = []
    for did, (disp, trip, pack, facts, pr, assessment, cid, cp, prec) in all_cases.items():
        rider_case = mock_rider_case[did]
        driver_case = mock_driver_case[did]
        risk = None  # The fraud output; not needed for prompt number check.
        pieces = _all_prompt_pieces_for_judge(
            disp, pack, rider_case, driver_case, pr, risk, prec,
        )

        valid = _valid_nums_for_judge(pack, pr)
        prec_nums = _all_precedent_nums(prec)

        for label, text in pieces:
            extracted = _extract_numbers(text, label)
            for num in extracted:
                # Check if it's a regular evidence/policy number.
                matched = any(
                    unit == num.unit and round(val, _dec_places(num.raw)) == num.value
                    for val, unit in valid
                )
                if matched:
                    continue
                # Check if it's a precedent number.
                is_precedent_num = any(
                    unit == num.unit and abs(val - num.value) < 0.001
                    for val, unit, _ in prec_nums
                )
                if is_precedent_num:
                    # Verify the guard accepts it when the precedent is cited.
                    # Find which precedent has this number.
                    prec_id = next(
                        pid for val, unit, pid in prec_nums
                        if unit == num.unit and abs(val - num.value) < 0.001
                    )
                    # Build a dummy output that cites the precedent and
                    # mentions the number.
                    dummy = _DummyJudgeOutput(
                        reasoning=f"Per {prec_id}, the distance was {num.raw}.",
                        cites=[prec_id],
                    )
                    problems = guard_check(
                        dummy, pack, pr.applicable_clauses,
                        policy_result=pr,
                        agent_role="judge",
                        precedents=prec,
                    )
                    num_problems = [
                        p for p in problems
                        if p.kind == "unverified_number"
                        and num.unit in p.detail
                    ]
                    if num_problems:
                        failures.append(
                            f"{did} [{label}]: precedent number {num.raw} "
                            f"({num.value} {num.unit}) from {prec_id} is shown "
                            f"to the Judge but the guard rejects it even when "
                            f"the precedent is cited."
                        )
                else:
                    # Not a precedent number and not an evidence number.
                    is_excluded = any(
                        ex_role == "judge" and ex_unit == num.unit
                        and abs(ex_val - num.value) < 0.001
                        for ex_role, ex_unit, ex_val, _ in EXCLUSIONS
                    )
                    if not is_excluded:
                        failures.append(
                            f"{did} [{label}]: number {num.raw} ({num.value} "
                            f"{num.unit}) in prompt text is not accepted by "
                            f"the guard for the judge."
                        )
    assert not failures, "\n".join(failures)


def test_judge_precedent_number_rejected_without_citation(all_cases):
    """A precedent number in Judge text that does NOT cite the precedent
    must be rejected by the guard — precedent facts are not facts of the
    current case.

    We only test numbers that are NOT also valid case numbers (i.e. numbers
    that are in the precedent but not in the evidence pack or policy
    computation for this case).  If a number happens to be both a precedent
    number and a case number, the guard correctly accepts it regardless.
    """
    from ridecourt.guard import check as guard_check
    from ridecourt.agents.judge import JudgeOutput

    ds = load_dataset()
    failures = []
    for did, (disp, trip, pack, facts, pr, assessment, cid, cp, prec) in all_cases.items():
        if not prec:
            continue
        # Compute the valid numbers for this case (evidence + policy).
        case_valid = _valid_nums_for_judge(pack, pr)
        # Find a precedent with a number in its summary that is NOT also
        # a valid case number.
        prec_with_nums = None
        prec_num = None
        for p in prec:
            summary = getattr(p, "summary", "")
            nums = _extract_numbers(summary, "")
            for n in nums:
                # Skip if this number is also a valid case number.
                is_case_num = any(
                    unit == n.unit and round(val, _dec_places(n.raw)) == n.value
                    for val, unit in case_valid
                )
                if not is_case_num:
                    prec_with_nums = p
                    prec_num = n
                    break
            if prec_with_nums is not None:
                break
        if prec_with_nums is None:
            continue

        # Build Judge output that mentions the precedent number WITHOUT
        # citing the precedent ID.
        dummy = JudgeOutput(
            outcome=pr.outcome,
            amount=pr.amount,
            confidence=0.85,
            reasoning=f"The driver was {prec_num.raw} from pickup.",
            cites=list(pr.applicable_clauses),
            message_to_rider="Refund per policy.",
            message_to_driver="Refund per policy.",
        )
        problems = guard_check(
            dummy, pack, pr.applicable_clauses,
            policy_result=pr,
            agent_role="judge",
            precedents=prec,
        )
        num_problems = [
            p for p in problems
            if p.kind == "unverified_number"
            and prec_num.unit in p.detail
        ]
        if not num_problems:
            failures.append(
                f"{did}: precedent number {prec_num.raw} ({prec_num.value} "
                f"{prec_num.unit}) from {getattr(prec_with_nums, 'case_id', '')} "
                f"was accepted by the guard WITHOUT citing the precedent — "
                f"precedent facts must not pass as facts of the current case."
            )
    # Only fail if we found cases with precedent numbers.
    assert not failures, "\n".join(failures)


def test_rebuttal_prompt_numbers_accepted(all_cases, mock_rider_case, mock_driver_case):
    """Every number in rebuttal prompt text must be accepted by the guard."""
    failures = []
    for did, (disp, trip, pack, facts, pr, assessment, cid, cp, prec) in all_cases.items():
        rider_case = mock_rider_case[did]
        driver_case = mock_driver_case[did]
        valid = _valid_nums_for_advocate(pack, pr)
        # Rider rebuttal sees driver_case.
        pieces = _all_prompt_pieces_for_rebuttal(
            disp, pack, pr, driver_case, "rider",
        )
        # Driver rebuttal sees rider_case.
        pieces += _all_prompt_pieces_for_rebuttal(
            disp, pack, pr, rider_case, "driver",
        )
        for label, text in pieces:
            extracted = _extract_numbers(text, label)
            for num in extracted:
                matched = any(
                    unit == num.unit and round(val, _dec_places(num.raw)) == num.value
                    for val, unit in valid
                )
                if not matched:
                    # Numbers from the opponent's case text might be in the
                    # opponent's text. Check if it's from evidence.
                    is_excluded = any(
                        ex_role == "advocate" and ex_unit == num.unit
                        and abs(ex_val - num.value) < 0.001
                        for ex_role, ex_unit, ex_val, _ in EXCLUSIONS
                    )
                    if not is_excluded:
                        failures.append(
                            f"{did} [{label}]: number {num.raw} ({num.value} "
                            f"{num.unit}) in rebuttal prompt text is not "
                            f"accepted by the guard."
                        )
    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _dec_places(num_str: str) -> int:
    """Count decimal places in a number string."""
    if "." not in num_str:
        return 0
    raw = num_str.strip()
    dot_idx = raw.rfind(".")
    if dot_idx < 0:
        return 0
    count = 0
    for ch in raw[dot_idx + 1:]:
        if ch.isdigit():
            count += 1
        else:
            break
    return count
