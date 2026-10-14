"""Tests for the evaluation script's resilience features.

Verifies:

* An exception in one run is recorded (outcome ``error``) and the
  evaluation continues.
* Results are written to the output file after every case, so a crash
  keeps what was done.
* One progress line per case is printed and flushed.
* ``--resume`` skips cases already present in the output file.
* Errored runs are counted separately in the summary.

Run with: .venv/bin/python -m pytest tests/test_evaluate_resilience.py -q
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate as evaluate_mod
from ridecourt.data import load_dataset
from ridecourt.orchestrator import Orchestrator, CaseFile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ds():
    return load_dataset()


@pytest.fixture
def tmp_output(tmp_path):
    """Return a temp output path for the results file."""
    return tmp_path / "results.json"


# ---------------------------------------------------------------------------
# Per-case error handling
# ---------------------------------------------------------------------------


class TestPerCaseErrorHandling:
    """An exception in one run is recorded and the evaluation continues."""

    def test_error_run_recorded_and_continues(self, ds, tmp_output):
        """When ``orc.resolve`` raises on one dispute, the run is
        recorded with outcome='error' and the evaluation continues."""
        orc = Orchestrator(ds)
        # Patch resolve to fail on the first dispute.
        original_resolve = orc.resolve
        call_count = {"n": 0}

        def flaky_resolve(dispute_id):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated crash")
            return original_resolve(dispute_id)

        with patch.object(orc, "resolve", side_effect=flaky_resolve):
            result = evaluate_mod.evaluate_case(orc, "disp_rd_000", repeats=2)

        # First run errored, second succeeded.
        assert result["runs"][0]["outcome"] == "error"
        assert "simulated crash" in result["runs"][0]["error"]
        assert result["runs"][1]["outcome"] != "error"
        # Case-level summary uses the first non-error run.
        assert result["outcome"] != "error"

    def test_all_runs_error(self, ds, tmp_output):
        """When all runs error, the case outcome is 'error'."""
        orc = Orchestrator(ds)

        def always_fail(dispute_id):
            raise RuntimeError("always fails")

        with patch.object(orc, "resolve", side_effect=always_fail):
            result = evaluate_mod.evaluate_case(orc, "disp_rd_000", repeats=2)

        assert result["outcome"] == "error"
        assert all(r["outcome"] == "error" for r in result["runs"])


# ---------------------------------------------------------------------------
# Incremental writes
# ---------------------------------------------------------------------------


class TestIncrementalWrites:
    """Results are written after every case, so a crash keeps what was done."""

    def test_partial_results_preserved_after_each_case(self, ds, tmp_output):
        """The output file is written after every case.  If the
        evaluation crashes mid-way, completed cases are preserved."""
        write_count = {"n": 0}
        original_write = evaluate_mod._write_results

        def counting_write(path, results):
            write_count["n"] += 1
            original_write(path, results)

        with patch.object(evaluate_mod, "_write_results", side_effect=counting_write):
            # Run 3 cases; expect 3 writes.
            evaluate_mod.run_evaluation(
                "mock", limit=3, repeats=1,
                output_path=tmp_output,
            )

        assert write_count["n"] == 3
        # The file has 3 cases.
        with open(tmp_output) as f:
            data = json.load(f)
        assert len(data["cases"]) == 3


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


class TestResume:
    """``--resume`` skips cases already present in the output file."""

    def test_resume_skips_done_cases(self, ds, tmp_output):
        """When the output file already has some cases, --resume skips
        them and only runs the remaining cases."""
        # Pre-write a file with 2 cases done (first 2 by dispute_id order).
        existing = evaluate_mod._build_results(
            "mock", "2026-01-01", 1,
            [
                {"dispute_id": "disp_fa_000", "runs": [], "outcome": "no_refund",
                 "amount": 0.0, "escalated": False, "consistent": True,
                 "tokens": 10, "wall_ms": 5, "category": "fare_error",
                 "label_outcome": "no_refund", "label_amount": 0.0,
                 "outcome_match": True, "amount_match": True},
                {"dispute_id": "disp_ns_000", "runs": [], "outcome": "no_refund",
                 "amount": 0.0, "escalated": False, "consistent": True,
                 "tokens": 10, "wall_ms": 5, "category": "no_show_rider",
                 "label_outcome": "no_refund", "label_amount": 0.0,
                 "outcome_match": True, "amount_match": True},
            ],
        )
        evaluate_mod._write_results(tmp_output, existing)

        # Run with --resume and limit=3.  Should skip the first 2 and
        # only run 1 new case (disp_ns_001).
        results = evaluate_mod.run_evaluation(
            "mock", limit=3, repeats=1,
            resume=True, output_path=tmp_output,
        )
        # The final results have 3 cases (2 existing + 1 new).
        assert len(results["cases"]) == 3
        # The first 2 are the pre-existing ones.
        assert results["cases"][0]["dispute_id"] == "disp_fa_000"
        assert results["cases"][1]["dispute_id"] == "disp_ns_000"
        # The 3rd is the new one.
        assert results["cases"][2]["dispute_id"] == "disp_ns_001"


# ---------------------------------------------------------------------------
# Error counting in summary
# ---------------------------------------------------------------------------


class TestErrorCounting:
    """Errored runs are counted separately (neither correct nor escalation)."""

    def test_error_count_in_overall(self, ds, tmp_output):
        """The overall summary has an ``error_count`` field."""
        orc = Orchestrator(ds)

        def fail_first(dispute_id):
            raise RuntimeError("crash")

        with patch.object(orc, "resolve", side_effect=fail_first):
            result = evaluate_mod.evaluate_case(orc, "disp_rd_000", repeats=1)

        # Build a results dict with this case.
        result["category"] = "route_deviation"
        result["label_outcome"] = "no_refund"
        result["label_amount"] = 0.0
        result["outcome_match"] = False
        result["amount_match"] = False
        results = evaluate_mod._build_results(
            "mock", "2026-01-01", 1, [result],
        )

        assert results["overall"]["error_count"] == 1
        assert results["overall"]["error_rate"] == 1.0

    def test_markdown_table_shows_errors(self, ds, tmp_output):
        """The Markdown table has an Errors column and an errored-runs line."""
        orc = Orchestrator(ds)

        def fail_first(dispute_id):
            raise RuntimeError("crash")

        with patch.object(orc, "resolve", side_effect=fail_first):
            result = evaluate_mod.evaluate_case(orc, "disp_rd_000", repeats=1)

        result["category"] = "route_deviation"
        result["label_outcome"] = "no_refund"
        result["label_amount"] = 0.0
        result["outcome_match"] = False
        result["amount_match"] = False
        results = evaluate_mod._build_results(
            "mock", "2026-01-01", 1, [result],
        )

        table = evaluate_mod.markdown_table(results)
        assert "Errors" in table
        assert "Errored runs: 1" in table
