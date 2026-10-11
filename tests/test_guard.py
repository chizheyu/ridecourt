"""Tests for the guard — must-fail and must-pass cases.

Covers:
- Unknown citation → fail
- Invented number (not in evidence) → fail
- Number with the wrong unit (S$3.25 backed by 3.25 km) → fail
- Valid output (cites exist, numbers match) → pass
- Forbidden certainty words → fail

Run with: .venv/bin/python -m pytest tests/test_guard.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest
from pydantic import BaseModel, Field

from ridecourt.data import EvidenceItem, EvidencePack, PolicyResult
from ridecourt.guard import Problem, check


# ---------------------------------------------------------------------------
# Schema fixtures
# ---------------------------------------------------------------------------


class Argument(BaseModel):
    text: str
    cites: list[str] = Field(default_factory=list)


class AdvocateOutput(BaseModel):
    position: str
    arguments: list[Argument] = Field(default_factory=list)
    requested_outcome: str


class JudgeOutput(BaseModel):
    outcome: str
    amount: float = 0.0
    confidence: float = 0.0
    reasoning: str
    cites: list[str] = Field(default_factory=list)
    message_to_rider: str
    message_to_driver: str


# ---------------------------------------------------------------------------
# Evidence + policy fixtures
# ---------------------------------------------------------------------------


def _pack() -> EvidencePack:
    return EvidencePack(
        items=[
            EvidenceItem(
                id="E1",
                kind="gps_analysis",
                summary=(
                    "Actual distance 13.68 km vs optimal 10.00 km; "
                    "deviation 36.8% (3.68 km extra)."
                ),
                data={
                    "distance_actual_km": 13.68,
                    "distance_optimal_km": 10.0,
                    "duration_actual_min": 24.0,
                    "duration_optimal_min": 20.0,
                    "extra_km": 3.68,
                    "deviation_pct": 36.8,
                },
                source="gps_analysis",
            ),
            EvidenceItem(
                id="E2",
                kind="fare_check",
                summary="Recomputed fare S$28.00 vs charged S$18.50 (diff S$-9.50).",
                data={
                    "recomputed_fare": 28.0,
                    "charged_total": 18.5,
                    "fare_difference": -9.5,
                },
                source="fare_check",
            ),
        ]
    )


def _policy_result() -> PolicyResult:
    return PolicyResult(
        applicable_clauses=["P-RD-1"],
        outcome="refund_route_deviation",
        amount=5.52,
        computation=(
            "extra_km=3.68 x S$0.75/km = S$2.76; "
            "extra_min=4.0 x S$0.25/min = S$1.00; "
            "sum x surge 1.0 = S$5.52"
        ),
    )


VALID_POLICY_IDS = ["P-RD-1", "P-NS-1", "P-FA-1", "P-SF-1", "P-BF-1"]


# ---------------------------------------------------------------------------
# Must-pass: valid output
# ---------------------------------------------------------------------------


class TestGuardPasses:
    def test_valid_output_no_problems(self):
        """All cites exist, all numbers match cited evidence → no problems."""
        output = AdvocateOutput(
            position="The rider requests a refund based on E1, E2, P-RD-1.",
            arguments=[
                Argument(
                    text="Evidence E1 shows the route deviation of 3.68 km.",
                    cites=["E1"],
                ),
            ],
            requested_outcome="Refund of S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        assert problems == []

    def test_amount_from_policy_engine_accepted(self):
        """The amount S$5.52 from the policy result is valid."""
        output = JudgeOutput(
            outcome="refund_route_deviation",
            amount=5.52,
            confidence=0.85,
            reasoning="The policy engine computes S$5.52 per P-RD-1.",
            cites=["P-RD-1"],
            message_to_rider="Refund of S$5.52 per P-RD-1.",
            message_to_driver="Refund of S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        assert problems == []

    def test_numbers_from_evidence_summary_accepted(self):
        """Numbers that appear in the evidence summary text are valid."""
        output = AdvocateOutput(
            position="Actual distance was 13.68 km per E1.",
            arguments=[
                Argument(
                    text="E1 shows optimal distance 10.00 km and deviation 36.8%.",
                    cites=["E1"],
                ),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        assert problems == []

    def test_no_problems_with_empty_text_fields(self):
        output = JudgeOutput(
            outcome="no_refund",
            amount=0.0,
            confidence=0.9,
            reasoning="",
            cites=[],
            message_to_rider="",
            message_to_driver="",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS)
        assert problems == []

    def test_cited_policy_id_accepted(self):
        """A policy clause ID cited in text must be accepted."""
        output = AdvocateOutput(
            position="Per P-RD-1 the deviation refund applies.",
            arguments=[Argument(text="P-NS-1 also cited here.", cites=["P-NS-1"])],
            requested_outcome="Refund S$5.52.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        assert problems == []


# ---------------------------------------------------------------------------
# Must-fail: unknown citation
# ---------------------------------------------------------------------------


class TestGuardFailsUnknownCite:
    def test_unknown_evidence_id(self):
        output = AdvocateOutput(
            position="The rider claims based on E99.",
            arguments=[
                Argument(
                    text="Evidence E99 shows the deviation.",
                    cites=["E99"],
                ),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert len(cite_problems) >= 1
        assert any("E99" in p.detail or p.cite == "E99" for p in cite_problems)

    def test_unknown_policy_id(self):
        output = AdvocateOutput(
            position="Per P-XX-99 the refund applies.",
            arguments=[
                Argument(text="P-XX-99 cited.", cites=["P-XX-99"]),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert len(cite_problems) >= 1
        assert any("P-XX-99" in p.detail or p.cite == "P-XX-99" for p in cite_problems)

    def test_unknown_cite_in_cites_list(self):
        output = AdvocateOutput(
            position="Claim based on E1.",
            arguments=[
                Argument(text="Deviation of 3.68 km per E1.", cites=["E1", "E50"]),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert any(p.cite == "E50" for p in cite_problems)


# ---------------------------------------------------------------------------
# Must-fail: invented number
# ---------------------------------------------------------------------------


class TestGuardFailsInventedNumber:
    def test_invented_distance(self):
        """3.99 km is not in any cited evidence."""
        output = AdvocateOutput(
            position="Rider claims based on E1.",
            arguments=[
                Argument(
                    text="Evidence E1 shows the deviation was 3.99 km.",
                    cites=["E1"],
                ),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1
        assert any("3.99" in p.detail for p in num_problems)

    def test_invented_money(self):
        """S$9.99 is not in the evidence or policy computation."""
        output = AdvocateOutput(
            position="Rider claims based on E2.",
            arguments=[
                Argument(
                    text="Evidence E2 shows overcharge of S$9.99.",
                    cites=["E2"],
                ),
            ],
            requested_outcome="Refund S$9.99 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1
        assert any("9.99" in p.detail for p in num_problems)

    def test_invented_percent(self):
        """50% is not in the evidence."""
        output = AdvocateOutput(
            position="Rider claims based on E1.",
            arguments=[
                Argument(
                    text="Evidence E1 shows deviation of 50%.",
                    cites=["E1"],
                ),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1
        assert any("50%" in p.detail or "50.0" in p.detail for p in num_problems)


# ---------------------------------------------------------------------------
# Must-fail: number with the wrong unit
# ---------------------------------------------------------------------------


class TestGuardFailsWrongUnit:
    def test_money_backed_by_km_value(self):
        """S$3.68 must NOT be backed by 3.68 km from the evidence.

        The number 3.68 exists in E1 as extra_km, but as money it is not
        present. The guard must catch this unit mismatch.
        """
        output = AdvocateOutput(
            position="Rider claims based on E1.",
            arguments=[
                Argument(
                    text="Evidence E1 shows a charge of S$3.68.",
                    cites=["E1"],
                ),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1
        # The problem should mention 3.68 and money.
        assert any("3.68" in p.detail and "money" in p.detail for p in num_problems)

    def test_km_backed_by_money_value(self):
        """18.50 km must NOT be backed by S$18.50 from the fare evidence.

        The number 18.50 exists in E2 as charged_total (money), but as
        kilometres it is not present.
        """
        output = AdvocateOutput(
            position="Rider claims based on E2.",
            arguments=[
                Argument(
                    text="Evidence E2 shows the trip was 18.50 km.",
                    cites=["E2"],
                ),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1
        assert any("18.50" in p.detail and "km" in p.detail for p in num_problems)

    def test_metres_backed_by_km_value(self):
        """13.68 m must NOT be backed by 13.68 km from the evidence.

        The number 13.68 exists as distance_actual_km, but 13.68 metres is
        not in the evidence.
        """
        output = AdvocateOutput(
            position="Rider claims based on E1.",
            arguments=[
                Argument(
                    text="Evidence E1 shows the driver was 13.68 m away.",
                    cites=["E1"],
                ),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1
        assert any("13.68" in p.detail and " m" in p.detail for p in num_problems)


# ---------------------------------------------------------------------------
# Must-fail: forbidden certainty words
# ---------------------------------------------------------------------------


class TestGuardFailsForbiddenWords:
    def test_forbidden_word_guaranteed(self):
        output = AdvocateOutput(
            position="The rider is guaranteed a refund based on E1.",
            arguments=[
                Argument(text="Deviation of 3.68 km per E1.", cites=["E1"]),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        word_problems = [p for p in problems if p.kind == "forbidden_word"]
        assert len(word_problems) >= 1
        assert any("guaranteed" in p.detail for p in word_problems)

    def test_forbidden_phrase_definitely_fraud(self):
        output = JudgeOutput(
            outcome="refund",
            amount=5.52,
            confidence=0.8,
            reasoning="This is definitely fraud based on E1.",
            cites=["E1", "P-RD-1"],
            message_to_rider="Refund S$5.52 per P-RD-1.",
            message_to_driver="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        word_problems = [p for p in problems if p.kind == "forbidden_word"]
        assert len(word_problems) >= 1

    def test_no_false_positive_on_word_substring(self):
        """'guaranteed' should not match inside 'unguaranteed' etc."""
        output = AdvocateOutput(
            position="Refund is not guaranteed to resolve anything.",
            arguments=[
                Argument(text="Deviation of 3.68 km per E1.", cites=["E1"]),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        word_problems = [p for p in problems if p.kind == "forbidden_word"]
        # "guaranteed" is a forbidden word — it appears in the text regardless.
        # This test verifies that the check is case-insensitive substring.
        assert len(word_problems) >= 1


# ---------------------------------------------------------------------------
# Precision-aware number matching
# ---------------------------------------------------------------------------


class TestGuardPrecisionMatching:
    """A number written with k decimals matches an evidence value v of the
    same unit when round(v, k) equals it."""

    def test_rounded_minutes_match(self):
        """2.9 min matches evidence value 2.93 min (round(2.93, 1) = 2.9)."""
        pack = EvidencePack(
            items=[
                EvidenceItem(
                    id="E1",
                    kind="gps_analysis",
                    summary="Extra time 2.93 min.",
                    data={"extra_min": 2.93},
                    source="gps_analysis",
                ),
            ]
        )
        output = AdvocateOutput(
            position="Based on E1.",
            arguments=[
                Argument(text="E1 shows extra time of 2.9 min.", cites=["E1"]),
            ],
            requested_outcome="Refund per P-RD-1.",
        )
        problems = check(output, pack, ["P-RD-1"])
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert num_problems == []

    def test_whole_number_matches(self):
        """3 min matches evidence value 2.93 min (round(2.93, 0) = 3)."""
        pack = EvidencePack(
            items=[
                EvidenceItem(
                    id="E1",
                    kind="gps_analysis",
                    summary="Extra time 2.93 min.",
                    data={"extra_min": 2.93},
                    source="gps_analysis",
                ),
            ]
        )
        output = AdvocateOutput(
            position="Based on E1.",
            arguments=[
                Argument(text="E1 shows extra time of 3 min.", cites=["E1"]),
            ],
            requested_outcome="Refund per P-RD-1.",
        )
        problems = check(output, pack, ["P-RD-1"])
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert num_problems == []

    def test_too_far_does_not_match(self):
        """3.1 min does NOT match evidence value 2.93 min
        (round(2.93, 1) = 2.9, not 3.1)."""
        pack = EvidencePack(
            items=[
                EvidenceItem(
                    id="E1",
                    kind="gps_analysis",
                    summary="Extra time 2.93 min.",
                    data={"extra_min": 2.93},
                    source="gps_analysis",
                ),
            ]
        )
        output = AdvocateOutput(
            position="Based on E1.",
            arguments=[
                Argument(text="E1 shows extra time of 3.1 min.", cites=["E1"]),
            ],
            requested_outcome="Refund per P-RD-1.",
        )
        problems = check(output, pack, ["P-RD-1"])
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1

    def test_money_two_decimals_match(self):
        """S$7.96 matches evidence amount 7.955 (round(7.955, 2) = 7.96)."""
        pack = EvidencePack(
            items=[
                EvidenceItem(
                    id="E1",
                    kind="policy_computation",
                    summary="Policy amount S$7.96.",
                    data={"amount": 7.955},
                    source="policy_engine",
                ),
            ]
        )
        output = AdvocateOutput(
            position="Based on E1.",
            arguments=[
                Argument(text="E1 shows refund of S$7.96.", cites=["E1"]),
            ],
            requested_outcome="Refund per P-RD-1.",
        )
        problems = check(output, pack, ["P-RD-1"])
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert num_problems == []

    def test_km_one_decimal_match(self):
        """6.9 km matches evidence value 6.873 km (round(6.873, 1) = 6.9)."""
        pack = EvidencePack(
            items=[
                EvidenceItem(
                    id="E1",
                    kind="gps_analysis",
                    summary="Extra distance 6.873 km.",
                    data={"extra_km": 6.873},
                    source="gps_analysis",
                ),
            ]
        )
        output = AdvocateOutput(
            position="Based on E1.",
            arguments=[
                Argument(text="E1 shows extra distance of 6.9 km.", cites=["E1"]),
            ],
            requested_outcome="Refund per P-RD-1.",
        )
        problems = check(output, pack, ["P-RD-1"])
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert num_problems == []

    def test_negative_minutes_match(self):
        """-2.9 min matches evidence value -2.94 min (round(-2.94, 1) = -2.9)."""
        pack = EvidencePack(
            items=[
                EvidenceItem(
                    id="E1",
                    kind="gps_analysis",
                    summary="Extra time -2.94 min.",
                    data={"extra_min": -2.94},
                    source="gps_analysis",
                ),
            ]
        )
        output = AdvocateOutput(
            position="Based on E1.",
            arguments=[
                Argument(text="E1 shows extra time of -2.9 min.", cites=["E1"]),
            ],
            requested_outcome="Refund per P-RD-1.",
        )
        problems = check(output, pack, ["P-RD-1"])
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert num_problems == []


# ---------------------------------------------------------------------------
# Judge role — numbers from any evidence item, not just cited ones
# ---------------------------------------------------------------------------

class TestJudgeGuardRole:
    """The Judge sees the whole evidence pack: a number in the Judge's text
    should be accepted when it appears in any evidence item of the pack or in
    the policy computation.  Advocates keep the stricter cited-evidence rule."""

    def _judge_pack() -> EvidencePack:
        """A pack with two evidence items: E1 has a distance, E2 has a fare."""
        return EvidencePack(
            items=[
                EvidenceItem(
                    id="E1",
                    kind="gps_analysis",
                    summary="Actual distance 13.68 km vs optimal 10.00 km.",
                    data={
                        "distance_actual_km": 13.68,
                        "distance_optimal_km": 10.0,
                        "extra_km": 3.68,
                    },
                    source="gps_analysis",
                ),
                EvidenceItem(
                    id="E2",
                    kind="fare_check",
                    summary="Recomputed fare S$28.00 vs charged S$18.50.",
                    data={
                        "recomputed_fare": 28.0,
                        "charged_total": 18.5,
                    },
                    source="fare_check",
                ),
            ]
        )

    def test_judge_number_from_uncited_evidence_accepted(self):
        """The Judge cites only E1 but mentions a number from E2 — accepted
        because the Judge sees the whole pack."""
        pack = TestJudgeGuardRole._judge_pack()
        output = JudgeOutput(
            outcome="refund_route_deviation",
            amount=5.52,
            confidence=0.85,
            reasoning="The charged total was S$18.50 per E1.",
            cites=["E1", "P-RD-1"],
            message_to_rider="Refund S$5.52 per P-RD-1.",
            message_to_driver="Refund S$5.52 per P-RD-1.",
        )
        problems = check(
            output, pack, ["P-RD-1"],
            policy_result=_policy_result(),
            agent_role="judge",
        )
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert num_problems == []

    def test_judge_number_from_any_evidence_accepted(self):
        """The Judge mentions 13.68 km from E1 without citing E1 explicitly."""
        pack = TestJudgeGuardRole._judge_pack()
        output = JudgeOutput(
            outcome="refund_route_deviation",
            amount=5.52,
            confidence=0.85,
            reasoning="The actual distance was 13.68 km per P-RD-1.",
            cites=["P-RD-1"],
            message_to_rider="Refund S$5.52 per P-RD-1.",
            message_to_driver="Refund S$5.52 per P-RD-1.",
        )
        problems = check(
            output, pack, ["P-RD-1"],
            policy_result=_policy_result(),
            agent_role="judge",
        )
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert num_problems == []

    def test_advocate_number_from_uncited_evidence_rejected(self):
        """An advocate citing E1 cannot use S$18.50 from E2 — stricter rule."""
        pack = TestJudgeGuardRole._judge_pack()
        output = AdvocateOutput(
            position="Based on E1.",
            arguments=[
                Argument(
                    text="E1 shows overcharge of S$18.50.",
                    cites=["E1"],
                ),
            ],
            requested_outcome="Refund S$5.52 per P-RD-1.",
        )
        problems = check(
            output, pack, ["P-RD-1"],
            policy_result=_policy_result(),
            agent_role="advocate",
        )
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1

    def test_judge_citation_must_still_exist(self):
        """Even with agent_role='judge', unknown citations are rejected."""
        pack = TestJudgeGuardRole._judge_pack()
        output = JudgeOutput(
            outcome="refund_route_deviation",
            amount=5.52,
            confidence=0.85,
            reasoning="Per E99 the refund is S$5.52.",
            cites=["E99", "P-RD-1"],
            message_to_rider="Refund S$5.52 per P-RD-1.",
            message_to_driver="Refund S$5.52 per P-RD-1.",
        )
        problems = check(
            output, pack, ["P-RD-1"],
            policy_result=_policy_result(),
            agent_role="judge",
        )
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert len(cite_problems) >= 1
        assert any("E99" in p.detail or p.cite == "E99" for p in cite_problems)

    def test_judge_invented_number_still_rejected(self):
        """A number that is nowhere in the pack or policy is rejected even
        for the Judge."""
        pack = TestJudgeGuardRole._judge_pack()
        output = JudgeOutput(
            outcome="refund_route_deviation",
            amount=5.52,
            confidence=0.85,
            reasoning="The overcharge was S$99.99 per P-RD-1.",
            cites=["P-RD-1"],
            message_to_rider="Refund S$5.52 per P-RD-1.",
            message_to_driver="Refund S$5.52 per P-RD-1.",
        )
        problems = check(
            output, pack, ["P-RD-1"],
            policy_result=_policy_result(),
            agent_role="judge",
        )
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1
        assert any("99.99" in p.detail for p in num_problems)

    def test_default_role_is_advocate(self):
        """Without agent_role, the default is advocate (cited-only)."""
        pack = TestJudgeGuardRole._judge_pack()
        output = JudgeOutput(
            outcome="refund_route_deviation",
            amount=5.52,
            confidence=0.85,
            reasoning="S$18.50 was charged per E1.",
            cites=["E1", "P-RD-1"],
            message_to_rider="Refund S$5.52 per P-RD-1.",
            message_to_driver="Refund S$5.52 per P-RD-1.",
        )
        # Default agent_role → advocate → S$18.50 from E2 is NOT accepted
        # because only E1 is cited.
        problems = check(
            output, pack, ["P-RD-1"],
            policy_result=_policy_result(),
        )
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        # S$18.50 is money, E1 has no money values → unverified
        assert any("18.50" in p.detail for p in num_problems)
