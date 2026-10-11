"""Test for scripts/evaluate.py — runs the evaluation on 3 cases with mock.

Verifies the script runs without error, produces a results JSON with the
expected structure, and the Markdown table contains the expected headers.

Run with: .venv/bin/python -m pytest tests/test_evaluate.py -q
"""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LLM_PROVIDER", "mock")

# Make scripts/ importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as evaluate_mod  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="module")
def results() -> dict:
    """Run the evaluation on 3 cases with the mock provider."""
    return evaluate_mod.run_evaluation(
        provider="mock", limit=3, repeats=3,
    )


class TestEvaluateScript:
    """Smoke test: the evaluation script runs and produces valid output."""

    def test_runs_without_error(self, results):
        """The evaluation completes and returns a dict."""
        assert isinstance(results, dict)

    def test_overall_structure(self, results):
        """The results dict has the expected top-level keys."""
        assert "provider" in results
        assert "date" in results
        assert "repeats" in results
        assert "overall" in results
        assert "per_category" in results
        assert "cases" in results

    def test_provider_is_mock(self, results):
        assert results["provider"] == "mock"

    def test_repeats_is_3(self, results):
        assert results["repeats"] == 3

    def test_cases_count(self, results):
        """Exactly 3 cases were evaluated (--limit 3)."""
        assert len(results["cases"]) == 3

    def test_each_case_has_runs(self, results):
        """Each case was run 3 times (repeats=3)."""
        for case in results["cases"]:
            assert len(case["runs"]) == 3

    def test_each_case_has_metrics(self, results):
        """Each case has the expected metric fields."""
        for case in results["cases"]:
            assert "dispute_id" in case
            assert "category" in case
            assert "outcome" in case
            assert "amount" in case
            assert "escalated" in case
            assert "consistent" in case
            assert "tokens" in case
            assert "label_outcome" in case
            assert "label_amount" in case
            assert "outcome_match" in case
            assert "amount_match" in case

    def test_overall_has_aggregates(self, results):
        """The overall dict has the aggregate metrics."""
        o = results["overall"]
        assert "total" in o
        assert "outcome_accuracy" in o
        assert "amount_accuracy" in o
        assert "escalation_rate" in o
        assert "safety_all_escalated" in o
        assert "consistency" in o
        assert "median_time_ms" in o
        assert "p90_time_ms" in o
        assert "median_tokens" in o

    def test_markdown_table_has_headers(self, results):
        """The Markdown table has the expected columns."""
        table = evaluate_mod.markdown_table(results)
        assert "| Category |" in table
        assert "| N |" in table
        assert "Outcome acc." in table
        assert "Amount acc." in table
        assert "Escalation rate" in table
        assert "Consistency" in table

    def test_markdown_table_has_overall_row(self, results):
        """The Markdown table includes an Overall summary row."""
        table = evaluate_mod.markdown_table(results)
        assert "**Overall**" in table

    def test_markdown_table_has_safety_line(self, results):
        """The Markdown table mentions the safety escalation check."""
        table = evaluate_mod.markdown_table(results)
        assert "Safety cases all escalated" in table
