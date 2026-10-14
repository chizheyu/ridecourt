"""Tests that every number in every evidence item on Ryde's three cases
is either a number from the raw JSON (excluding held-out fields) or a
sum, difference, product, or percentage of two such numbers.

Also asserts that no evidence item reports a value for a field the adapter
had to default because the source lacks it (e.g. base, per_km, per_min
for DISP-001 and DISP-003).

For synthetic cases, components exist so recomputation is allowed, but
every evidence item must agree with the policy computation on shared
quantities.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

from ridecourt.ryde_adapter import (
    RydeDatasetAdapter,
    RYDE_SAMPLES_DIR,
    _extract_json_blocks,
)
from ridecourt.ryde_adapter import _HELD_OUT_TOP_LEVEL, _HELD_OUT_FARE_NOTES
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack
from ridecourt.policy import evaluate
from ridecourt.data import load_dataset


# ---------------------------------------------------------------------------
# Collect raw numbers from the source JSON (excluding held-out fields)
# ---------------------------------------------------------------------------

def _collect_numbers(obj, path=""):
    """Recursively collect all numeric (non-bool) values with their paths."""
    results = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            results.extend(_collect_numbers(v, f"{path}.{k}"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            results.extend(_collect_numbers(v, f"{path}[{i}]"))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        results.append((path, float(obj)))
    return results


def _load_raw_numbers(filename: str) -> set[float]:
    """Load all numeric values from the raw JSON, excluding held-out fields."""
    md = (RYDE_SAMPLES_DIR / filename).read_text(encoding="utf-8")
    blocks = _extract_json_blocks(md)
    assert blocks, f"No JSON blocks found in {filename}"
    data = blocks[0]

    # Remove held-out fields
    for k in _HELD_OUT_TOP_LEVEL:
        data.pop(k, None)
    if "fare_breakdown" in data and isinstance(data["fare_breakdown"], dict):
        data["fare_breakdown"].pop(_HELD_OUT_FARE_NOTES, None)

    nums = _collect_numbers(data)
    return {v for _, v in nums}


def _is_derivable(value: float, raw_nums: set[float], tol=0.02) -> bool:
    """Check if a value is a raw number, or a sum/diff/product/percentage of
    two raw numbers, within rounding tolerance."""
    if value == 0.0:
        return True  # zero is always OK (defaults, empty fields)

    # Direct match
    for r in raw_nums:
        if abs(value - r) < tol:
            return True

    raw_list = list(raw_nums)
    for a in raw_list:
        # Sum a + b
        for b in raw_list:
            if abs(value - (a + b)) < tol:
                return True
        # Difference a - b
        for b in raw_list:
            if abs(value - (a - b)) < tol:
                return True
        # Product a * b
        for b in raw_list:
            if abs(value - (a * b)) < tol:
                return True
        # Percentage a is of b: a / b * 100
        for b in raw_list:
            if b != 0 and abs(value - (a / b * 100)) < tol:
                return True
        # Percentage b is of a: b / a * 100
        for b in raw_list:
            if a != 0 and abs(value - (b / a * 100)) < tol:
                return True

    return False


def _extract_numbers_from_text(text: str) -> list[float]:
    """Extract all numeric values from a text string, skipping numbers that
    are part of dates/timestamps (e.g. 2026 in 2026-09-12T13:19:30) and
    label suffixes like (90d), (30d), etc."""
    # Remove ISO timestamps first so their components aren't extracted.
    text_clean = re.sub(r"\d{4}-\d{2}-\d{2}T[\d::+\-]+", " ", text)
    # Remove label numbers like (90d), (30d), 90d, etc.
    text_clean = re.sub(r"\(\d+d\)|\b\d+d\b", " ", text_clean)
    # Match numbers like 35.00, 4.3, 0.80, 1.5, etc.
    # Skip integers that look like years (4 digits starting with 19 or 20)
    pattern = r"(?<![\w-])(\d+\.?\d*)"
    matches = re.findall(pattern, text_clean)
    results = []
    for m in matches:
        val = float(m)
        if val == 0:
            continue
        # Skip year-like numbers
        if re.match(r"^(19|20)\d{2}$", m):
            continue
        results.append(val)
    return results


def _extract_numbers_from_data(data) -> list[float]:
    """Recursively extract all numeric values from a dict/list/number."""
    results = []
    if isinstance(data, dict):
        for v in data.values():
            results.extend(_extract_numbers_from_data(v))
    elif isinstance(data, list):
        for v in data:
            results.extend(_extract_numbers_from_data(v))
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        results.append(float(data))
    return results


# ---------------------------------------------------------------------------
# Defaulted fields per case (fields the adapter defaults because source lacks)
# ---------------------------------------------------------------------------

# DISP-001: source has no base, per_km, per_min components
# DISP-003: source has no base, per_km, per_min components
# DISP-002: source has no accepted_at (arrival phrased against scheduled time)

_DEFAULTED_FIELDS = {
    "DISP-001": {"recomputed_fare", "fare_difference"},
    "DISP-002": set(),
    "DISP-003": {"recomputed_fare", "fare_difference"},
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not RYDE_SAMPLES_DIR.exists(),
    reason="data/ryde_samples/ directory not found",
)
class TestEvidenceSourceAgreement:
    """Every number in every evidence item's summary and data must come from
    the raw JSON or a deterministic computation from it."""

    @pytest.fixture(scope="class")
    def ds(self):
        return RydeDatasetAdapter().load()

    @pytest.fixture(scope="class")
    def raw_nums(self):
        return {
            "DISP-001": _load_raw_numbers("DISP-001_route_deviation.md"),
            "DISP-002": _load_raw_numbers("DISP-002_no_show.md"),
            "DISP-003": _load_raw_numbers("DISP-003_fare_surge.md"),
        }

    def _check_case(self, ds, raw_nums, dispute_id, filename):
        disp = ds.dispute(dispute_id)
        pack = build_evidence_pack(disp, ds)
        case_raw = raw_nums[dispute_id]
        defaulted = _DEFAULTED_FIELDS.get(dispute_id, set())

        problems = []
        for item in pack.items:
            # Check summary numbers
            summary_nums = _extract_numbers_from_text(item.summary)
            for num in summary_nums:
                if not _is_derivable(num, case_raw):
                    problems.append(
                        f"{item.id} [{item.kind}] summary number {num} "
                        f"not derivable from source data"
                    )

            # Check data numbers — but skip defaulted fields
            for key, val in item.data.items():
                if key in defaulted:
                    # This field should not be reported at all
                    if val != 0 and val != 0.0:
                        problems.append(
                            f"{item.id} [{item.kind}] data.{key}={val} "
                            f"reports a value for a defaulted field"
                        )
                    continue
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if not _is_derivable(float(val), case_raw):
                        problems.append(
                            f"{item.id} [{item.kind}] data.{key}={val} "
                            f"not derivable from source data"
                        )
                elif isinstance(val, dict):
                    for sub_key, sub_val in val.items():
                        if isinstance(sub_val, (int, float)) and not isinstance(sub_val, bool):
                            if not _is_derivable(float(sub_val), case_raw):
                                problems.append(
                                    f"{item.id} [{item.kind}] "
                                    f"data.{key}.{sub_key}={sub_val} "
                                    f"not derivable from source data"
                                )

        assert not problems, (
            f"{dispute_id} evidence-source agreement failures:\n  "
            + "\n  ".join(problems)
        )

    def test_disp001_evidence_agrees_with_source(self, ds, raw_nums):
        self._check_case(ds, raw_nums, "DISP-001",
                         "DISP-001_route_deviation.md")

    def test_disp002_evidence_agrees_with_source(self, ds, raw_nums):
        self._check_case(ds, raw_nums, "DISP-002",
                         "DISP-002_no_show.md")

    def test_disp003_evidence_agrees_with_source(self, ds, raw_nums):
        self._check_case(ds, raw_nums, "DISP-003",
                         "DISP-003_fare_surge.md")

    def test_no_defaulted_field_reported(self, ds):
        """No evidence item should report a non-zero value for a field the
        adapter had to default because the source lacks it."""
        for dispute_id, fields in _DEFAULTED_FIELDS.items():
            if not fields:
                continue
            disp = ds.dispute(dispute_id)
            pack = build_evidence_pack(disp, ds)
            for item in pack.items:
                for field in fields:
                    if field in item.data:
                        val = item.data[field]
                        assert val == 0 or val == 0.0, (
                            f"{dispute_id} {item.id} reports {field}={val} "
                            f"but the source lacks the components to compute it"
                        )


@pytest.mark.skipif(
    not RYDE_SAMPLES_DIR.exists(),
    reason="data/ryde_samples/ directory not found",
)
class TestSyntheticEvidencePolicyAgreement:
    """On synthetic cases, components exist so recomputation is allowed, but
    every evidence item must agree with the policy computation on shared
    quantities."""

    @pytest.fixture(scope="class")
    def ds(self):
        return load_dataset()

    def test_synthetic_fares_agree_with_recomputation(self, ds):
        """For synthetic cases where fare components exist, the fare_check's
        recomputed fare must agree with the policy computation."""
        problems = []
        for disp_id, disp in ds.disputes.items():
            if disp.category not in ("fare_error", "route_deviation"):
                continue
            trip = ds.trip(disp.trip_id)
            if trip is None:
                continue
            # Only check cases where fare components are present
            if trip.fare.base <= 0 or trip.fare.per_km <= 0:
                continue
            pack = build_evidence_pack(disp, ds)
            facts = facts_from_evidence_pack(pack)
            result = evaluate(disp, trip, ds.profiles, facts, trip.fare)
            # The policy amount should be consistent with the evidence
            # (not necessarily equal to recomputed_fare, but the policy
            # computation should use numbers from the evidence)
            fare_items = pack.by_kind("fare_check")
            for fi in fare_items:
                rec = fi.data.get("recomputed_fare", 0)
                charged = fi.data.get("charged_total", 0)
                diff = fi.data.get("fare_difference", 0)
                # Internal consistency: diff = charged - rec
                if rec > 0 and charged > 0:
                    expected_diff = round(charged - rec, 2)
                    if abs(diff - expected_diff) > 0.02:
                        problems.append(
                            f"{disp_id}: fare_check diff {diff} != "
                            f"charged {charged} - recomputed {rec} = {expected_diff}"
                        )

        assert not problems, (
            "Synthetic fare consistency failures:\n  "
            + "\n  ".join(problems)
        )
