"""Tests for the mock LLM provider and the LLM layer interface.

All tests run with ``LLM_PROVIDER=mock`` — no real Gemini calls are made.

Run with: .venv/bin/python -m pytest tests/test_llm_mock.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from ridecourt.data import EvidenceItem, EvidencePack, PolicyResult
from ridecourt.llm import (
    CaseBudget,
    LLMMeta,
    TokenBudgetExceeded,
    complete_json,
    set_case_budget,
)


# ---------------------------------------------------------------------------
# Schema fixtures — shapes an agent might return
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


class RiskOutput(BaseModel):
    risk: Literal["low", "medium", "high"]
    signals: list[Argument] = Field(default_factory=list)


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
        outcome="refund_route_deviation",
        amount=5.52,
        computation=(
            "extra_km=3.68 x S$0.75/km = S$2.76; "
            "extra_min=4.0 x S$0.25/min = S$1.00; "
            "sum x surge 1.0 = S$5.52"
        ),
    )


# ---------------------------------------------------------------------------
# Mock provider tests
# ---------------------------------------------------------------------------


class TestMockProvider:
    def test_returns_pydantic_instance(self):
        result, meta = complete_json(
            "system prompt", "user prompt", AdvocateOutput, "rider_advocate"
        )
        assert isinstance(result, AdvocateOutput)
        assert isinstance(meta, LLMMeta)

    def test_meta_has_tokens_and_latency(self):
        _, meta = complete_json("s", "u", AdvocateOutput, "judge")
        assert meta.tokens > 0
        assert meta.latency_ms >= 0
        assert meta.provider == "mock"
        assert meta.agent_name == "judge"

    def test_mock_is_deterministic(self):
        """Same inputs → same outputs (deterministic)."""
        r1, _ = complete_json("s", "u", AdvocateOutput, "rider_advocate")
        r2, _ = complete_json("s", "u", AdvocateOutput, "rider_advocate")
        assert r1.model_dump() == r2.model_dump()

    def test_cites_real_evidence_ids(self):
        """Mock output must cite IDs that exist in the evidence pack."""
        pack = _pack()
        pr = _policy_result()
        result, _ = complete_json(
            "s", "u", AdvocateOutput, "rider_advocate",
            evidence_pack=pack, policy_result=pr,
        )
        valid_ids = {"E1", "E2", "§1.3"}
        # Collect all cites from arguments.
        all_cites: list[str] = []
        for arg in result.arguments:
            all_cites.extend(arg.cites)
        # At least one cite, and all must be valid.
        assert len(all_cites) >= 1
        for cite in all_cites:
            assert cite in valid_ids, f"mock cited {cite} which is not in the pack"

    def test_judge_schema_with_amount_from_policy(self):
        """The judge's amount should come from the policy result."""
        pack = _pack()
        pr = _policy_result()
        result, _ = complete_json(
            "s", "u", JudgeOutput, "judge",
            evidence_pack=pack, policy_result=pr,
        )
        assert isinstance(result, JudgeOutput)
        assert result.amount == pr.amount
        assert 0 <= result.confidence <= 1

    def test_risk_schema_literal_field(self):
        """Literal fields should get a valid enum value."""
        result, _ = complete_json("s", "u", RiskOutput, "fraud_agent")
        assert isinstance(result, RiskOutput)
        assert result.risk in ("low", "medium", "high")

    def test_numbers_in_mock_text_match_evidence(self):
        """Numbers in the mock agent text must come from the evidence pack."""
        pack = _pack()
        pr = _policy_result()
        result, _ = complete_json(
            "s", "u", AdvocateOutput, "rider_advocate",
            evidence_pack=pack, policy_result=pr,
        )
        # The mock argument text mentions extra_km = 3.68 km.
        texts = [arg.text for arg in result.arguments]
        joined = " ".join(texts)
        # 3.68 km should appear (from E1 data).
        assert "3.68" in joined


# ---------------------------------------------------------------------------
# Token budget tests
# ---------------------------------------------------------------------------


class TestCaseBudget:
    def test_budget_tracks_tokens(self):
        budget = CaseBudget(budget=1000)
        budget.charge(100)
        assert budget.spent == 100
        budget.charge(200)
        assert budget.spent == 300

    def test_budget_raises_when_exceeded(self):
        budget = CaseBudget(budget=100)
        budget.charge(50)
        with pytest.raises(TokenBudgetExceeded) as exc_info:
            budget.charge(60)
        assert exc_info.value.spent == 110
        assert exc_info.value.budget == 100

    def test_active_budget_charged_on_complete_json(self):
        """complete_json should charge the active case budget."""
        budget = CaseBudget(budget=100000)
        set_case_budget(budget)
        try:
            complete_json("s", "u", AdvocateOutput, "rider_advocate")
            assert budget.spent > 0
        finally:
            set_case_budget(None)

    def test_budget_exceeds_via_complete_json(self):
        """When the budget is tiny, complete_json raises TokenBudgetExceeded."""
        budget = CaseBudget(budget=1)
        set_case_budget(budget)
        try:
            with pytest.raises(TokenBudgetExceeded):
                complete_json("s", "u", AdvocateOutput, "rider_advocate")
        finally:
            set_case_budget(None)


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


class TestProviderSelection:
    def test_unknown_provider_raises(self):
        original = os.environ.get("LLM_PROVIDER")
        os.environ["LLM_PROVIDER"] = "nonexistent_provider"
        try:
            with pytest.raises(Exception):
                complete_json("s", "u", AdvocateOutput, "test_agent")
        finally:
            if original is not None:
                os.environ["LLM_PROVIDER"] = original
            else:
                del os.environ["LLM_PROVIDER"]
