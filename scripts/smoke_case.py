#!/usr/bin/env python3
"""Smoke test — resolve one case end-to-end and print a summary.

Usage (mock, offline)::

    .venv/bin/python scripts/smoke_case.py disp_rd_000

Usage (real Gemini)::

    LLM_PROVIDER=gemini \\
    GOOGLE_GENAI_USE_VERTEXAI=true \\
    GOOGLE_CLOUD_PROJECT=no-fine-print-2026 \\
    GOOGLE_CLOUD_LOCATION=global \\
    MODEL=gemini-3.8-flash \\
    .venv/bin/python scripts/smoke_case.py disp_rd_000

Prints: event sequence, guard problems (if any), the ruling (or escalation),
total tokens, and total wall-clock seconds.
"""

from __future__ import annotations

import os
import sys
import time

# Ensure the project root is on sys.path when running directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ridecourt.data import load_dataset
from ridecourt.orchestrator import Orchestrator


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: scripts/smoke_case.py <dispute_id>")
        sys.exit(1)

    dispute_id = sys.argv[1]

    ds = load_dataset()
    if ds.dispute(dispute_id) is None:
        print(f"Error: dispute '{dispute_id}' not found.")
        sys.exit(1)

    orc = Orchestrator(ds)

    print(f"Resolving {dispute_id} (provider={ds.dispute(dispute_id).category}) ...")
    print()

    t0 = time.monotonic()
    cf = orc.resolve(dispute_id)
    wall_s = time.monotonic() - t0

    # --- Event sequence --------------------------------------------------
    print("=== Event Sequence ===")
    for ev in cf.events.events:
        cites_str = f" cites={ev.cites}" if ev.cites else ""
        tokens_str = f" tokens={ev.tokens}" if ev.tokens else ""
        latency_str = f" latency={ev.latency_ms}ms" if ev.latency_ms else ""
        from_str = f" {ev.from_agent}→{ev.to_agent}" if ev.from_agent else ""
        print(f"  {ev.type}{from_str}{cites_str}{tokens_str}{latency_str}")
    print()

    # --- Guard problems --------------------------------------------------
    if cf.guard_problems:
        print("=== Guard Problems ===")
        for p in cf.guard_problems:
            print(f"  [{p.get('kind', '?')}] {p.get('detail', '')} (field={p.get('field', '')})")
        print()
    else:
        print("=== Guard Problems ===")
        print("  (none)")
        print()

    # --- Ruling / Escalation ---------------------------------------------
    if cf.escalated:
        print("=== ESCALATED ===")
        print(f"  reason: {cf.escalation_reason}")
        if cf.ruling is not None:
            print(f"  judge outcome: {cf.ruling.outcome}")
            print(f"  judge confidence: {cf.ruling.confidence}")
        print(f"  policy outcome: {cf.policy_result.outcome}")
        print(f"  policy amount: S${cf.policy_result.amount:.2f}")
    else:
        print("=== RULING ===")
        print(f"  outcome: {cf.ruling.outcome}")
        print(f"  amount:  S${cf.ruling.amount:.2f}")
        print(f"  confidence: {cf.ruling.confidence}")
        print(f"  cites: {cf.ruling.cites}")
        print()
        print("  message_to_rider:")
        print(f"    {cf.ruling.message_to_rider}")
        print("  message_to_driver:")
        print(f"    {cf.ruling.message_to_driver}")
    print()

    # --- Label comparison (if available) ---------------------------------
    label = ds.label(dispute_id)
    if label is not None:
        print("=== Label Comparison ===")
        print(f"  label outcome: {label.outcome}")
        print(f"  label amount:  S${label.amount:.2f}")
        if cf.ruling is not None:
            match_o = cf.ruling.outcome == label.outcome
            match_a = abs(cf.ruling.amount - label.amount) < 0.01
            print(f"  outcome match: {'YES' if match_o else 'NO'}")
            print(f"  amount match:  {'YES' if match_a else 'NO'}")
        print()

    # --- Totals ----------------------------------------------------------
    print("=== Totals ===")
    print(f"  total tokens:  {cf.total_tokens}")
    print(f"  total latency: {cf.total_latency_ms} ms")
    print(f"  wall clock:    {wall_s:.1f} s")
    print()


if __name__ == "__main__":
    main()
