"""Tests for the guard — must-fail and must-pass cases.

Covers:
- Unknown citation → fail (E#, §N.M, PC-YYYY-NNN, DISP-YYYY-NNNN)
- Invented number (not in evidence) → fail
- Number with the wrong unit (S$3.25 backed by 3.25 km) → fail
- Valid output (cites exist, numbers match) → pass
- Forbidden certainty words → fail
- v2.1 §N.M section citations accepted
- DISP-YYYY-NNNN precedent IDs accepted

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
        applicable_clauses=["§1.3"],
        outcome="refund_full_excess_distance",
        amount=3.20,
        computation=(
            "excess_km=3.68 > 1.0 km; "
            "no traffic justification on recommended route (§1.4); "
            "refund = 3.68 km × S$0.80/km = S$3.20 "
            "(full excess distance fare, §1.3). "
            "Tolls, time, surge not refunded."
        ),
    )


# Valid v2.1 policy IDs (sections) — also accept legacy P- IDs for backward compat
VALID_POLICY_IDS = ["§1.3", "§2.3", "§3.3", "§4.2", "§5.1", "§6.1", "P-RD-1"]


# ---------------------------------------------------------------------------
# Must-pass: valid output
# ---------------------------------------------------------------------------


class TestGuardPasses:
    def test_valid_output_no_problems(self):
        """All cites exist, all numbers match cited evidence → no problems."""
        output = AdvocateOutput(
            position="The rider requests a refund based on E1, §1.3.",
            arguments=[
                Argument(
                    text="Evidence E1 shows the route deviation of 3.68 km.",
                    cites=["E1"],
                ),
            ],
            requested_outcome="Refund of S$3.20 per §1.3.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        assert problems == []

    def test_amount_from_policy_engine_accepted(self):
        """The amount S$3.20 from the policy result is valid."""
        output = JudgeOutput(
            outcome="refund_full_excess_distance",
            amount=3.20,
            confidence=0.85,
            reasoning="The policy engine computes S$3.20 per §1.3.",
            cites=["§1.3"],
            message_to_rider="Refund of S$3.20 per §1.3.",
            message_to_driver="Refund of S$3.20 per §1.3.",
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
            requested_outcome="Refund S$3.20 per §1.3.",
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

    def test_cited_section_id_accepted(self):
        """A v2.1 §N.M section citation in text must be accepted."""
        output = AdvocateOutput(
            position="Per §1.3 the deviation refund applies.",
            arguments=[Argument(text="§1.4 also cited here.", cites=["§1.4"])],
            requested_outcome="Refund S$3.20.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        assert problems == []

    def test_legacy_p_id_still_accepted(self):
        """Legacy P-RD-1 citations are still accepted (backward compat)."""
        output = AdvocateOutput(
            position="Per P-RD-1 the deviation refund applies.",
            arguments=[Argument(text="Deviation of 3.68 km per E1.", cites=["E1"])],
            requested_outcome="Refund S$3.20.",
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
            requested_outcome="Refund S$3.20 per §1.3.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert len(cite_problems) >= 1
        assert any("E99" in p.detail or p.cite == "E99" for p in cite_problems)

    def test_unknown_section_citation(self):
        """§9.9 is not a valid v2.1 section → fail."""
        output = AdvocateOutput(
            position="Per §9.9 the refund applies.",
            arguments=[Argument(text="§9.9 cited.", cites=["§9.9"])],
            requested_outcome="Refund S$3.20 per §1.3.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert len(cite_problems) >= 1
        assert any("§9.9" in p.detail or p.cite == "§9.9" for p in cite_problems)

    def test_unknown_cite_in_cites_list(self):
        output = AdvocateOutput(
            position="Claim based on E1.",
            arguments=[
                Argument(text="Deviation of 3.68 km per E1.", cites=["E1", "E50"]),
            ],
            requested_outcome="Refund S$3.20 per §1.3.",
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
            requested_outcome="Refund S$3.20 per §1.3.",
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
            requested_outcome="Refund S$9.99 per §1.3.",
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
            requested_outcome="Refund S$3.20 per §1.3.",
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
        """S$3.68 must NOT be backed by 3.68 km from the evidence."""
        output = AdvocateOutput(
            position="Rider claims based on E1.",
            arguments=[
                Argument(
                    text="Evidence E1 shows a charge of S$3.68.",
                    cites=["E1"],
                ),
            ],
            requested_outcome="Refund S$3.20 per §1.3.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1
        assert any("3.68" in p.detail and "money" in p.detail for p in num_problems)

    def test_km_backed_by_money_value(self):
        """18.50 km must NOT be backed by S$18.50 from the fare evidence."""
        output = AdvocateOutput(
            position="Rider claims based on E2.",
            arguments=[
                Argument(
                    text="Evidence E2 shows the trip was 18.50 km.",
                    cites=["E2"],
                ),
            ],
            requested_outcome="Refund S$3.20 per §1.3.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1
        assert any("18.50" in p.detail and "km" in p.detail for p in num_problems)


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
            requested_outcome="Refund S$3.20 per §1.3.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS, policy_result=_policy_result())
        word_problems = [p for p in problems if p.kind == "forbidden_word"]
        assert len(word_problems) >= 1
        assert any("guaranteed" in p.detail for p in word_problems)


# ---------------------------------------------------------------------------
# Precision-aware number matching
# ---------------------------------------------------------------------------


class TestGuardPrecisionMatching:
    def test_rounded_minutes_match(self):
        """2.9 min matches evidence value 2.93 min."""
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
            requested_outcome="Refund per §1.3.",
        )
        problems = check(output, pack, ["§1.3"])
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert num_problems == []

    def test_too_far_does_not_match(self):
        """3.1 min does NOT match evidence value 2.93 min."""
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
            requested_outcome="Refund per §1.3.",
        )
        problems = check(output, pack, ["§1.3"])
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1

    def test_money_two_decimals_match(self):
        """S$3.20 matches evidence amount 3.20."""
        pack = EvidencePack(
            items=[
                EvidenceItem(
                    id="E1",
                    kind="policy_computation",
                    summary="Policy amount S$3.20.",
                    data={"amount": 3.20},
                    source="policy_engine",
                ),
            ]
        )
        output = AdvocateOutput(
            position="Based on E1.",
            arguments=[
                Argument(text="E1 shows refund of S$3.20.", cites=["E1"]),
            ],
            requested_outcome="Refund per §1.3.",
        )
        problems = check(output, pack, ["§1.3"])
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert num_problems == []


# ---------------------------------------------------------------------------
# Judge role — numbers from any evidence item
# ---------------------------------------------------------------------------


class TestJudgeGuardRole:
    _judge_pack = EvidencePack(
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
        """The Judge cites only E1 but mentions a number from E2 — accepted."""
        output = JudgeOutput(
            outcome="refund_full_excess_distance",
            amount=3.20,
            confidence=0.85,
            reasoning="The charged total was S$18.50 per E1.",
            cites=["E1", "§1.3"],
            message_to_rider="Refund S$3.20 per §1.3.",
            message_to_driver="Refund S$3.20 per §1.3.",
        )
        problems = check(
            output, self._judge_pack, ["§1.3"],
            policy_result=_policy_result(),
            agent_role="judge",
        )
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert num_problems == []

    def test_advocate_number_from_uncited_evidence_rejected(self):
        """An advocate citing E1 cannot use S$18.50 from E2."""
        output = AdvocateOutput(
            position="Based on E1.",
            arguments=[
                Argument(
                    text="E1 shows overcharge of S$18.50.",
                    cites=["E1"],
                ),
            ],
            requested_outcome="Refund S$3.20 per §1.3.",
        )
        problems = check(
            output, self._judge_pack, ["§1.3"],
            policy_result=_policy_result(),
            agent_role="advocate",
        )
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1

    def test_judge_invented_number_still_rejected(self):
        """A number that is nowhere in the pack or policy is rejected."""
        output = JudgeOutput(
            outcome="refund_full_excess_distance",
            amount=3.20,
            confidence=0.85,
            reasoning="The overcharge was S$99.99 per §1.3.",
            cites=["§1.3"],
            message_to_rider="Refund S$3.20 per §1.3.",
            message_to_driver="Refund S$3.20 per §1.3.",
        )
        problems = check(
            output, self._judge_pack, ["§1.3"],
            policy_result=_policy_result(),
            agent_role="judge",
        )
        num_problems = [p for p in problems if p.kind == "unverified_number"]
        assert len(num_problems) >= 1
        assert any("99.99" in p.detail for p in num_problems)


# ---------------------------------------------------------------------------
# v2.1 section citation validation
# ---------------------------------------------------------------------------


class TestSectionCitationValidation:
    """The guard must accept valid §N.M citations and reject invalid ones."""

    def test_valid_section_accepted(self):
        """§1.3 is a valid section citation."""
        output = AdvocateOutput(
            position="Per §1.3 refund applies.",
            arguments=[Argument(text="See §1.4.", cites=["§1.4"])],
            requested_outcome="Refund.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS)
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert len(cite_problems) == 0

    def test_invalid_section_rejected(self):
        """§9.9 is not a valid section citation."""
        output = AdvocateOutput(
            position="Per §9.9 refund applies.",
            arguments=[],
            requested_outcome="Refund.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS)
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert len(cite_problems) >= 1

    def test_top_level_section_accepted(self):
        """§1 (top-level section) is valid."""
        output = AdvocateOutput(
            position="Per §1 the deviation rules apply.",
            arguments=[],
            requested_outcome="Refund.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS)
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert len(cite_problems) == 0


# ---------------------------------------------------------------------------
# DISP-YYYY-NNNN precedent ID validation
# ---------------------------------------------------------------------------


class TestPrecedentIdValidation:
    """The guard must accept DISP-YYYY-NNNN precedent IDs."""

    def test_valid_disp_precedent_accepted(self):
        """DISP-2026-0089 is a valid Ryde precedent ID."""
        from ridecourt.data import Precedent

        prec = Precedent(
            case_id="DISP-2026-0089",
            category="route_deviation",
            summary="Partial refund S$3.20 for traffic-justified deviation.",
            outcome="refund_partial_excess_distance",
            amount=3.20,
        )
        output = AdvocateOutput(
            position="Per DISP-2026-0089 a partial refund applies.",
            arguments=[
                Argument(
                    text="DISP-2026-0089 shows S$3.20 refund.",
                    cites=["DISP-2026-0089"],
                ),
            ],
            requested_outcome="Refund S$3.20.",
        )
        problems = check(
            output, _pack(), VALID_POLICY_IDS,
            precedents=[prec],
        )
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert len(cite_problems) == 0

    def test_unknown_disp_precedent_rejected(self):
        """DISP-2026-9999 is not in the precedent list → fail."""
        output = AdvocateOutput(
            position="Per DISP-2026-9999 a refund applies.",
            arguments=[],
            requested_outcome="Refund.",
        )
        problems = check(output, _pack(), VALID_POLICY_IDS)
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert len(cite_problems) >= 1

    def test_pc_precedent_still_accepted(self):
        """PC-2025-001 (seed precedent) is still accepted."""
        from ridecourt.data import Precedent

        prec = Precedent(
            case_id="PC-2025-001",
            category="route_deviation",
            summary="Full refund S$2.40 for unjustified deviation.",
            outcome="refund_full_excess_distance",
            amount=2.40,
        )
        output = AdvocateOutput(
            position="Per PC-2025-001 a refund applies.",
            arguments=[Argument(text="PC-2025-001 shows S$2.40.", cites=["PC-2025-001"])],
            requested_outcome="Refund S$2.40.",
        )
        problems = check(
            output, _pack(), VALID_POLICY_IDS,
            precedents=[prec],
        )
        cite_problems = [p for p in problems if p.kind == "unknown_cite"]
        assert len(cite_problems) == 0
