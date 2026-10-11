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

Usage (mock, offline)::

    .venv/bin/python scripts/evaluate.py --provider mock

Usage (Gemini)::

    .venv/bin/python scripts/evaluate.py --provider gemini

Options:
    --provider mock|gemini   LLM provider to use (default: mock)
    --limit N                Only run the first N labelled disputes
    --repeats N              Consistency repeats per case (default: 3)

Writes ``eval/results-<date>-<provider>.json`` and prints a Markdown
table to stdout.
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
# Single-case evaluation
# ---------------------------------------------------------------------------


def evaluate_case(
    orc: Orchestrator,
    dispute_id: str,
    repeats: int,
) -> dict[str, Any]:
    """Run one dispute *repeats* times and return per-case metrics."""
    runs: list[dict[str, Any]] = []
    for _ in range(repeats):
        t0 = time.monotonic()
        cf = orc.resolve(dispute_id)
        wall_ms = int((time.monotonic() - t0) * 1000)
        runs.append({
            "outcome": _ruling_outcome(cf),
            "amount": _ruling_amount(cf),
            "escalated": cf.escalated,
            "escalation_reason": cf.escalation_reason,
            "tokens": cf.total_tokens,
            "latency_ms": cf.total_latency_ms,
            "wall_ms": wall_ms,
        })

    outcomes = [r["outcome"] for r in runs]
    amounts = [r["amount"] for r in runs]
    consistent = len(set(outcomes)) == 1 and len(set(amounts)) == 1

    return {
        "dispute_id": dispute_id,
        "runs": runs,
        "outcome": outcomes[0],
        "amount": amounts[0],
        "escalated": runs[0]["escalated"],
        "consistent": consistent,
        "tokens": runs[0]["tokens"],
        "wall_ms": runs[0]["wall_ms"],
    }


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    provider: str,
    limit: int | None,
    repeats: int,
) -> dict[str, Any]:
    """Run the full evaluation and return a structured results dict."""
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

    orc = Orchestrator(ds)

    case_results: list[dict[str, Any]] = []
    for i, dispute in enumerate(labelled):
        did = dispute.dispute_id
        label = ds.labels[did]
        result = evaluate_case(orc, did, repeats)
        result["category"] = dispute.category
        result["label_outcome"] = label.outcome
        result["label_amount"] = label.amount
        result["outcome_match"] = result["outcome"] == label.outcome
        result["amount_match"] = (
            abs(result["amount"] - label.amount) <= AMOUNT_TOLERANCE
        )
        case_results.append(result)
        if (i + 1) % 10 == 0 or i == len(labelled) - 1:
            print(
                f"  [{i + 1}/{len(labelled)}] {did} "
                f"outcome={'OK' if result['outcome_match'] else 'MISS'} "
                f"amount={'OK' if result['amount_match'] else 'MISS'}",
                file=sys.stderr,
            )

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
        consistent = sum(1 for c in cat_cases if c["consistent"])
        per_category[cat] = {
            "n": n,
            "outcome_accuracy": round(outcome_ok / n, 4),
            "amount_accuracy": round(amount_ok / n, 4),
            "both_accuracy": round(both_ok / n, 4),
            "escalation_rate": round(escalated / n, 4),
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
    all_consistent = sum(1 for c in case_results if c["consistent"])

    # Safety check: every safety case must escalate.
    safety_cases = [c for c in case_results if c["category"] == "safety"]
    safety_all_escalated = (
        all(c["escalated"] for c in safety_cases) if safety_cases else True
    )

    # Timing (wall clock per case, first run).
    wall_times = sorted(c["wall_ms"] for c in case_results)
    median_ms = statistics.median(wall_times) if wall_times else 0
    p90_ms = _percentile(wall_times, 90)

    # Tokens per case (first run).
    tokens = [c["tokens"] for c in case_results]
    median_tokens = statistics.median(tokens) if tokens else 0

    overall = {
        "total": total,
        "outcome_accuracy": round(all_outcome_ok / total, 4) if total else 0.0,
        "amount_accuracy": round(all_amount_ok / total, 4) if total else 0.0,
        "both_accuracy": round(all_both_ok / total, 4) if total else 0.0,
        "escalation_rate": round(all_escalated / total, 4) if total else 0.0,
        "safety_all_escalated": safety_all_escalated,
        "consistency": round(all_consistent / total, 4) if total else 0.0,
        "median_time_ms": median_ms,
        "p90_time_ms": p90_ms,
        "median_tokens": median_tokens,
    }

    return {
        "provider": provider,
        "date": date.today().isoformat(),
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
        "Escalation rate | Consistency |"
    )
    lines.append(
        "|:--|:--:|:--:|:--:|:--:|:--:|:--:|"
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
        f"**{o['consistency']:.1%}** |"
    )
    lines.append("")

    # Summary lines.
    lines.append(
        f"- Safety cases all escalated: "
        f"{'yes' if o['safety_all_escalated'] else 'NO'}"
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
    args = parser.parse_args()

    print(f"Running evaluation (provider={args.provider}) ...", file=sys.stderr)
    results = run_evaluation(args.provider, args.limit, args.repeats)

    # Write JSON results file.
    eval_dir = Path("eval")
    eval_dir.mkdir(parents=True, exist_ok=True)
    filename = f"results-{results['date']}-{args.provider}.json"
    out_path = eval_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results written to {out_path}", file=sys.stderr)

    # Print Markdown table to stdout.
    print(markdown_table(results))


if __name__ == "__main__":
    main()
