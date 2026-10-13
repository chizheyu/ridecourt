"""Sweep test — no generated text may contain a negative quantity described
as "extra", "longer", "over", or "more".

Defect 3: the GPS evidence summary prints "(-16.7 min extra)" when the trip
was shorter than the optimal route. The data fields may stay signed, but the
text a person reads must not show a negative "extra".

Walks all 48 cases and checks every summary, computation, and escalation text
the system generates.
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import re

import pytest

from ridecourt.agents.judge import neutral_case_summary
from ridecourt.data import load_dataset
from ridecourt.orchestrator import Orchestrator
from ridecourt.policy import evaluate
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack, policy_evidence_item

# Match a minus sign followed by digits near "extra", "longer", "over", "more".
# e.g. "(-16.7 min extra)" or "-5.2 min longer" or "-3.0 km over"
_NEG_EXTRA_RE = re.compile(
    r"-\d+(?:\.\d+)?\s*(?:km\s*)?(?:min\s*)?(?:extra|longer|over|more)",
    re.IGNORECASE,
)

# Also catch negative-zero deviation like "-0.0%" or "-0.00 km extra"
_NEG_ZERO_RE = re.compile(r"-\d*\.?0+\s*(?:%|km\s*extra|min\s*extra)")


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


def _all_generated_texts(ds, did):
    """Collect every human-readable text string the system generates."""
    disp = ds.dispute(did)
    trip = ds.trip(disp.trip_id)
    pack = build_evidence_pack(disp, ds)
    facts = facts_from_evidence_pack(pack)
    pr = evaluate(disp, trip, ds.profiles, facts, trip.fare)

    texts: list[tuple[str, str]] = []

    # Evidence summaries.
    for item in pack.items:
        texts.append((f"{did}/{item.kind}/summary", item.summary))

    # Policy computation.
    texts.append((f"{did}/policy/computation", pr.computation))

    # Policy evidence item summary.
    next_id = len(pack.items) + 1
    pe = policy_evidence_item(pr, facts, next_id)
    texts.append((f"{did}/policy_evidence/summary", pe.summary))

    # Escalation summary (neutral_case_summary).
    esc = neutral_case_summary(disp.claim_text, pack, pr)
    texts.append((f"{did}/escalation/summary", esc))

    return texts


def test_no_negative_extra_in_evidence_or_policy(ds):
    """No generated evidence summary or policy computation may contain a
    negative quantity described as extra/longer/over/more.

    Fails on current code: safety cases have "(-21.6 min extra)".
    """
    failures = []
    for did in ds.disputes:
        if ds.dispute(did) is None or ds.trip(ds.dispute(did).trip_id) is None:
            continue
        for label, text in _all_generated_texts(ds, did):
            if _NEG_EXTRA_RE.search(text):
                failures.append(f"{label}: {text}")
            if _NEG_ZERO_RE.search(text):
                failures.append(f"{label} (neg-zero): {text}")

    assert not failures, (
        f"Found {len(failures)} text(s) with negative extra/longer/over/more:\n"
        + "\n".join(failures[:20])
    )


def test_no_negative_extra_in_full_pipeline(ds):
    """Run the full orchestrator pipeline (mock) on all cases and check
    every event, ruling, and message text for negative extras.
    """
    orc = Orchestrator(ds)
    failures = []
    for did in ds.disputes:
        if ds.dispute(did) is None or ds.trip(ds.dispute(did).trip_id) is None:
            continue
        cf = orc.resolve(did)
        # Check ruling messages.
        if cf.ruling:
            for label, text in [
                ("reasoning", cf.ruling.reasoning),
                ("message_to_rider", cf.ruling.message_to_rider),
                ("message_to_driver", cf.ruling.message_to_driver),
            ]:
                if _NEG_EXTRA_RE.search(text):
                    failures.append(f"{did} ruling.{label}: {text}")
        # Check escalation summary.
        if cf.escalation:
            if _NEG_EXTRA_RE.search(cf.escalation.summary):
                failures.append(f"{did} escalation.summary: {cf.escalation.summary}")
        # Check event content.
        if cf.events:
            for ev in cf.events.events:
                content = ev.content
                if isinstance(content, dict):
                    for k, v in content.items():
                        if isinstance(v, str):
                            if _NEG_EXTRA_RE.search(v):
                                failures.append(f"{did} event.{k}: {v}")
    assert not failures, (
        f"Found {len(failures)} text(s) with negative extra in pipeline:\n"
        + "\n".join(failures[:20])
    )
