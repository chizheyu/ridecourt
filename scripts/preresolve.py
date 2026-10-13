#!/usr/bin/env python3
"""Pre-resolve labelled cases and write results to the configured store.

Resolves every labelled case (or a given list) in the current process with
the configured LLM, writes results to the configured store (in-memory by
default, Firestore when ``STATE_BACKEND=firestore``), one case at a time,
and skips cases already resolved unless ``--force``.

Usage (mock, offline)::

    .venv/bin/python scripts/preresolve.py --provider mock

Usage (Gemini)::

    LLM_PROVIDER=gemini \\
    GOOGLE_GENAI_USE_VERTEXAI=true \\
    GOOGLE_CLOUD_PROJECT=your-project \\
    GOOGLE_CLOUD_LOCATION=global \\
    MODEL=gemini-3.8-flash \\
    .venv/bin/python scripts/preresolve.py --provider gemini

Options::

    --provider mock|gemini|openai_compatible   LLM provider (default: mock)
    --limit N                                  Only run the first N disputes
    --force                                    Re-resolve even if already done
    dispute_ids...                             Positional: resolve only these IDs

Prints one progress line per case (to stderr) and continues after a failed
case. Exits 0 when all cases have been attempted.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

# Ensure the project root is on sys.path when running directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ridecourt.data import load_dataset
from ridecourt.orchestrator import CaseFile, Orchestrator
from ridecourt.serialize import serialize_case_file, serialize_events
from ridecourt.store import get_store


def _print_progress(
    case_num: int,
    total: int,
    dispute_id: str,
    outcome: str,
    amount: float,
    escalated: bool,
    skipped: bool,
    error: str | None,
    seconds: float,
) -> None:
    """Print one progress line for a case."""
    if skipped:
        print(
            f"  [{case_num}/{total}] {dispute_id} SKIPPED (already resolved)",
            file=sys.stderr,
            flush=True,
        )
        return
    if error:
        print(
            f"  [{case_num}/{total}] {dispute_id} ERROR: {error}",
            file=sys.stderr,
            flush=True,
        )
        return
    status = "escalated" if escalated else "ruled"
    print(
        f"  [{case_num}/{total}] {dispute_id} "
        f"outcome={outcome} "
        f"amount=S${amount:.2f} "
        f"status={status} "
        f"seconds={seconds:.1f}",
        file=sys.stderr,
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-resolve labelled cases and write to the store.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini", "openai_compatible"],
        default=os.environ.get("LLM_PROVIDER", "mock"),
        help="LLM provider to use (default: from LLM_PROVIDER env or mock)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N labelled disputes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-resolve cases even if already in the store",
    )
    parser.add_argument(
        "dispute_ids",
        nargs="*",
        help="Specific dispute IDs to resolve (default: all labelled)",
    )
    args = parser.parse_args()

    os.environ["LLM_PROVIDER"] = args.provider

    ds = load_dataset()
    orc = Orchestrator(ds)
    store = get_store()

    # Determine which cases to resolve.
    if args.dispute_ids:
        dispute_ids = list(args.dispute_ids)
    else:
        # All labelled disputes, sorted for determinism.
        dispute_ids = sorted(
            did for did, d in ds.disputes.items()
            if ds.labels.get(did) is not None
        )

    if args.limit:
        dispute_ids = dispute_ids[:args.limit]

    total = len(dispute_ids)
    print(
        f"Preresolve: {total} cases, provider={args.provider}, "
        f"store={store.__class__.__name__}, force={args.force}",
        file=sys.stderr, flush=True,
    )

    for i, did in enumerate(dispute_ids, start=1):
        # Skip already-resolved cases unless --force.
        if not args.force:
            existing = store.load_case(did)
            if existing is not None:
                _print_progress(i, total, did, "", 0, False, True, None, 0)
                continue

        t0 = time.monotonic()
        try:
            cf: CaseFile = orc.resolve(did)
            seconds = time.monotonic() - t0

            # Write to the store.
            store.save_case(cf.dispute_id, serialize_case_file(cf))
            store.save_events(cf.dispute_id, serialize_events(cf))
            if cf.escalated:
                store.set_status(cf.dispute_id, "escalated")
            else:
                store.set_status(cf.dispute_id, "ruled")

            outcome = cf.ruling.outcome if cf.ruling else "escalated"
            amount = cf.ruling.amount if cf.ruling else 0.0
            _print_progress(
                i, total, did, outcome, amount, cf.escalated,
                False, None, seconds,
            )
        except Exception as exc:
            seconds = time.monotonic() - t0
            _print_progress(i, total, did, "", 0, False, False, str(exc), seconds)

    print("Preresolve complete.", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
