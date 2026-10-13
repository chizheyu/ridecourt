#!/usr/bin/env python3
"""Evaluation script — runs every labelled dispute through the pipeline.

Reports (per SPEC section 9):

- Ruling accuracy vs the policy label, per category (outcome and amount
  within S$0.01).
- Consistency: the same case run ``--repeats`` times gives the same
  outcome and amount.
- Escalation rate and whether every safety case escalated.
- Median and p90 wall-clock time to ruling.
- LLM tokens per case (median).
- Errored runs counted separately (neither correct nor an escalation).

Usage (mock, offline)::

    .venv/bin/python scripts/evaluate.py --provider mock

Usage (Gemini)::

    .venv/bin/python scripts/evaluate.py --provider gemini

Options:

    --provider mock|gemini   LLM provider to use (default: mock)
    --limit N                Only run the first N labelled disputes
    --repeats N              Consistency repeats per case (default: 3)
    --resume                 Skip cases already present in the output file

Writes ``eval/results-<date>-<provider>.json`` after every case (so a
crash keeps what was done) and prints a Markdown table to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path when running directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ridecourt.data import load_dataset
from ridecourt.orchestrator import Orchestrator, CaseFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    "route_deviation",
    "no_show_driver",
    "no_show_rider",
    "fare_error",
    "safety",
]

CATEGORY_LABELS = {
    "route_deviation": "route_deviation",
    "no_show_driver": "no_show_driver",
    "no_show_rider": "no_show_rider",
    "fare_error": "fare_error",
    "safety": "safety",
}

AMOUNT_TOLERANCE = 0.01  # S$0.01


def _ruling_outcome(cf: CaseFile) -> str:
    """The effective outcome for a resolved case.

    Escalated cases (including safety) have outcome ``"escalate"`` so they
    can be compared to the label.
    """
    if cf.escalated:
        return "escalate"
    if cf.ruling is not None:
        return cf.ruling.outcome
    return "error"


def _ruling_amount(cf: CaseFile) -> float:
    """The effective amount for a resolved case."""
    if cf.escalated:
        return 0.0
    if cf.ruling is not None:
        return round(cf.ruling.amount, 2)
    return 0.0


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile of a sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[f]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


# ---------------------------------------------------------------------------
# Output file management
# ---------------------------------------------------------------------------


def _output_path(provider: str, date_str: str) -> Path:
    """Return the output file path for this provider + date."""
    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir / f"results-{date_str}-{provider}.json"


def _load_existing(path: Path) -> dict[str, Any] | None:
    """Load an existing results file, or None if it doesn't exist."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_results(path: Path, results: dict[str, Any]) -> None:
    """Write the full results dict to *path* (atomic-ish: write then flush)."""
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _existing_case_ids(results: dict[str, Any]) -> set[str]:
    """Return the set of dispute_ids already present in the results."""
    return {c["dispute_id"] for c in results.get("cases", [])}


# ---------------------------------------------------------------------------
# Single-case evaluation
# ---------------------------------------------------------------------------


def evaluate_case(
    orc: Orchestrator,
    dispute_id: str,
    repeats: int,
) -> dict[str, Any]:
    """Run one dispute *repeats* times and return per-case metrics.

    If a run raises an exception, the run is recorded with
    ``outcome="error"`` and the error message, and the next run/case
    proceeds.  An errored run is neither correct nor an escalation.
    """
    runs: list[dict[str, Any]] = []
    for _ in range(repeats):
        t0 = time.monotonic()
        try:
            cf = orc.resolve(dispute_id)
            wall_ms = int((time.monotonic() - t0) * 1000)
            runs.append({
                "outcome": _ruling_outcome(cf),
                "amount": _ruling_amount(cf),
                "escalated": cf.escalated,
                "escalation_reason": cf.escalation_reason,
                "guard_problems": cf.guard_problems,
                "tokens": cf.total_tokens,
                "latency_ms": cf.total_latency_ms,
                "wall_ms": wall_ms,
            })
        except Exception as exc:
            wall_ms = int((time.monotonic() - t0) * 1000)
            runs.append({
                "outcome": "error",
                "amount": 0.0,
                "escalated": False,
                "escalation_reason": "",
                "tokens": 0,
                "latency_ms": 0,
                "wall_ms": wall_ms,
                "error": str(exc),
            })

    outcomes = [r["outcome"] for r in runs]
    amounts = [r["amount"] for r in runs]
    consistent = len(set(outcomes)) == 1 and len(set(amounts)) == 1

    # Use the first non-error run for the case-level summary, if any.
    first_ok = next((r for r in runs if r["outcome"] != "error"), None)
    summary_outcome = first_ok["outcome"] if first_ok else "error"
    summary_amount = first_ok["amount"] if first_ok else 0.0
    summary_escalated = first_ok["escalated"] if first_ok else False
    summary_tokens = first_ok["tokens"] if first_ok else 0
    summary_wall = first_ok["wall_ms"] if first_ok else runs[0]["wall_ms"]

    return {
        "dispute_id": dispute_id,
        "runs": runs,
        "outcome": summary_outcome,
        "amount": summary_amount,
        "escalated": summary_escalated,
        "consistent": consistent,
        "tokens": summary_tokens,
        "wall_ms": summary_wall,
    }


# ---------------------------------------------------------------------------
# Progress printing
# ---------------------------------------------------------------------------


def _print_progress(
    case_num: int,
    total: int,
    dispute_id: str,
    result: dict[str, Any],
    label_outcome: str,
    label_amount: float,
) -> None:
    """Print and flush one progress line for a case."""
    outcome = result["outcome"]
    amount = result["amount"]
    outcome_match = outcome == label_outcome
    amount_match = abs(amount - label_amount) <= AMOUNT_TOLERANCE
    seconds = result["wall_ms"] / 1000.0
    label_match = "OK" if (outcome_match and amount_match) else "MISS"
    print(
        f"  [{case_num}/{total}] {dispute_id} "
        f"outcome={outcome} "
        f"amount=S${amount:.2f} "
        f"label_match={label_match} "
        f"seconds={seconds:.1f}",
        file=sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    provider: str,
    limit: int | None,
    repeats: int,
    *,
    resume: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run the full evaluation and return a structured results dict.

    Results are written to the output file after every case so a crash
    keeps what was done.  When *resume* is True, cases already present
    in the output file are skipped.
    """
    os.environ["LLM_PROVIDER"] = provider

    ds = load_dataset()

    # Only evaluate disputes that have a label.
    labelled = [
        d for d in ds.disputes.values()
        if ds.labels.get(d.dispute_id) is not None
    ]
    # Sort by dispute_id for deterministic ordering.
    labelled.sort(key=lambda d: d.dispute_id)
    if limit:
        labelled = labelled[:limit]

    date_str = date.today().isoformat()
    if output_path is None:
        output_path = _output_path(provider, date_str)

    # --- Resume: load existing results and skip done cases ------------
    case_results: list[dict[str, Any]] = []
    if resume:
        existing = _load_existing(output_path)
        if existing is not None:
            case_results = list(existing.get("cases", []))
            done_ids = _existing_case_ids(existing)
            labelled = [d for d in labelled if d.dispute_id not in done_ids]
            print(
                f"Resume: {len(case_results)} cases already done, "
                f"{len(labelled)} remaining.",
                file=sys.stderr, flush=True,
            )

    orc = Orchestrator(ds)

    total = len(labelled)
    for i, dispute in enumerate(labelled):
        did = dispute.dispute_id
        label = ds.labels[did]
        try:
            result = evaluate_case(orc, did, repeats)
            result["category"] = dispute.category
            result["label_outcome"] = label.outcome
            result["label_amount"] = label.amount
            result["outcome_match"] = result["outcome"] == label.outcome
            result["amount_match"] = (
                abs(result["amount"] - label.amount) <= AMOUNT_TOLERANCE
            )
        except Exception as exc:
            # Should not happen — evaluate_case catches per-run errors —
            # but guard against metadata-level failures too.
            result = {
                "dispute_id": did,
                "runs": [],
                "outcome": "error",
                "amount": 0.0,
                "escalated": False,
                "consistent": False,
                "tokens": 0,
                "wall_ms": 0,
                "category": dispute.category,
                "label_outcome": label.outcome,
                "label_amount": label.amount,
                "outcome_match": False,
                "amount_match": False,
                "error": str(exc),
            }
        case_results.append(result)
        _print_progress(
            i + 1, total, did, result, label.outcome, label.amount,
        )

        # Write after every case so a crash keeps what was done.
        partial = _build_results(provider, date_str, repeats, case_results)
        _write_results(output_path, partial)

    return _build_results(provider, date_str, repeats, case_results)


def _build_results(
    provider: str,
    date_str: str,
    repeats: int,
    case_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the full results dict from a list of per-case results."""
    # --- Per-category aggregates ----------------------------------------
    per_category: dict[str, dict[str, Any]] = {}
    for cat in CATEGORY_ORDER:
        cat_cases = [c for c in case_results if c["category"] == cat]
        if not cat_cases:
            continue
        n = len(cat_cases)
        outcome_ok = sum(1 for c in cat_cases if c["outcome_match"])
        amount_ok = sum(1 for c in cat_cases if c["amount_match"])
        both_ok = sum(
            1 for c in cat_cases if c["outcome_match"] and c["amount_match"]
        )
        escalated = sum(1 for c in cat_cases if c["escalated"])
        errored = sum(1 for c in cat_cases if c["outcome"] == "error")
        consistent = sum(1 for c in cat_cases if c["consistent"])
        per_category[cat] = {
            "n": n,
            "outcome_accuracy": round(outcome_ok / n, 4),
            "amount_accuracy": round(amount_ok / n, 4),
            "both_accuracy": round(both_ok / n, 4),
            "escalation_rate": round(escalated / n, 4),
            "error_rate": round(errored / n, 4) if errored else 0.0,
            "consistency": round(consistent / n, 4),
        }

    # --- Overall aggregates --------------------------------------------
    total = len(case_results)
    all_outcome_ok = sum(1 for c in case_results if c["outcome_match"])
    all_amount_ok = sum(1 for c in case_results if c["amount_match"])
    all_both_ok = sum(
        1 for c in case_results if c["outcome_match"] and c["amount_match"]
    )
    all_escalated = sum(1 for c in case_results if c["escalated"])
    all_errored = sum(1 for c in case_results if c["outcome"] == "error")
    all_consistent = sum(1 for c in case_results if c["consistent"])

    # Safety check: every safety case must escalate.
    safety_cases = [c for c in case_results if c["category"] == "safety"]
    safety_all_escalated = (
        all(c["escalated"] for c in safety_cases) if safety_cases else True
    )

    # Timing (wall clock per case, first non-error run).
    wall_times = sorted(
        c["wall_ms"] for c in case_results if c["wall_ms"] > 0
    )
    median_ms = statistics.median(wall_times) if wall_times else 0
    p90_ms = _percentile(wall_times, 90)

    # Tokens per case (first non-error run).
    tokens = [c["tokens"] for c in case_results if c["tokens"] > 0]
    median_tokens = statistics.median(tokens) if tokens else 0

    overall = {
        "total": total,
        "outcome_accuracy": round(all_outcome_ok / total, 4) if total else 0.0,
        "amount_accuracy": round(all_amount_ok / total, 4) if total else 0.0,
        "both_accuracy": round(all_both_ok / total, 4) if total else 0.0,
        "escalation_rate": round(all_escalated / total, 4) if total else 0.0,
        "error_count": all_errored,
        "error_rate": round(all_errored / total, 4) if total else 0.0,
        "safety_all_escalated": safety_all_escalated,
        "consistency": round(all_consistent / total, 4) if total else 0.0,
        "median_time_ms": median_ms,
        "p90_time_ms": p90_ms,
        "median_tokens": median_tokens,
    }

    return {
        "provider": provider,
        "date": date_str,
        "repeats": repeats,
        "overall": overall,
        "per_category": per_category,
        "cases": case_results,
    }


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------


def markdown_table(results: dict[str, Any]) -> str:
    """Render the evaluation results as a Markdown table."""
    overall = results["overall"]
    per_cat = results["per_category"]

    lines: list[str] = []
    lines.append(
        "| Category | N | Outcome acc. | Amount acc. | Both acc. | "
        "Escalation rate | Errors | Consistency |"
    )
    lines.append(
        "|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|"
    )

    for cat in CATEGORY_ORDER:
        if cat not in per_cat:
            continue
        c = per_cat[cat]
        lines.append(
            f"| {cat} | {c['n']} | "
            f"{c['outcome_accuracy']:.1%} | "
            f"{c['amount_accuracy']:.1%} | "
            f"{c['both_accuracy']:.1%} | "
            f"{c['escalation_rate']:.1%} | "
            f"{c.get('error_rate', 0.0):.1%} | "
            f"{c['consistency']:.1%} |"
        )

    # Overall row.
    o = overall
    lines.append(
        f"| **Overall** | {o['total']} | "
        f"**{o['outcome_accuracy']:.1%}** | "
        f"**{o['amount_accuracy']:.1%}** | "
        f"**{o['both_accuracy']:.1%}** | "
        f"**{o['escalation_rate']:.1%}** | "
        f"**{o.get('error_count', 0)}** | "
        f"**{o['consistency']:.1%}** |"
    )
    lines.append("")

    # Summary lines.
    lines.append(
        f"- Safety cases all escalated: "
        f"{'yes' if o['safety_all_escalated'] else 'NO'}"
    )
    lines.append(
        f"- Errored runs: {o.get('error_count', 0)} "
        f"(neither correct nor an escalation)"
    )
    lines.append(
        f"- Median time to ruling: {o['median_time_ms']} ms"
    )
    lines.append(
        f"- p90 time to ruling: {o['p90_time_ms']} ms"
    )
    lines.append(
        f"- Median tokens per case: {o['median_tokens']}"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RideCourt evaluation on labelled synthetic cases.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini", "openai_compatible"],
        default="mock",
        help="LLM provider to use (default: mock)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N labelled disputes",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Consistency repeats per case (default: 3)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip cases already present in the output file",
    )
    args = parser.parse_args()

    date_str = date.today().isoformat()
    out_path = _output_path(args.provider, date_str)

    print(f"Running evaluation (provider={args.provider}) ...", file=sys.stderr, flush=True)
    results = run_evaluation(
        args.provider, args.limit, args.repeats,
        resume=args.resume, output_path=out_path,
    )

    print(f"Results written to {out_path}", file=sys.stderr, flush=True)

    # Print Markdown table to stdout.
    print(markdown_table(results))


if __name__ == "__main__":
    main()
