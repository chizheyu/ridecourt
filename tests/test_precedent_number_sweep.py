"""Sweep test: every number in the three Ryde precedent summaries in
``data/synthetic/precedents.json`` must match the policy file v2.1 §7.

Never change Ryde's numbers to make arithmetic work — the test compares
the summary text with the policy file's precedent tables.
"""

import re

import pytest

from ridecourt.data import load_dataset


def _extract_numbers(text: str) -> set[str]:
    """Extract all numeric values (with optional S$ prefix) from text."""
    # Match S$3.20, $5.00, 1.8 km, 6 min, 91%, etc.
    pattern = r"(?:S?\$)?(\d+\.?\d*)\s*(?:km|min|%|x)?"
    matches = re.findall(pattern, text)
    return {m for m in matches if m != ""}


class TestPrecedentNumbersMatchPolicy:
    """Every number in the three Ryde precedent summaries must match the
    policy file v2.1 §7 tables."""

    def _read_policy_precedents(self) -> dict[str, dict[str, str]]:
        """Parse the policy file's §7 precedent tables."""
        from pathlib import Path
        policy_path = Path(__file__).resolve().parent.parent / "data" / "ryde_samples" / "ryde_dispute_policy_v2.1.md"
        text = policy_path.read_text(encoding="utf-8")
        precedents: dict[str, dict[str, str]] = {}
        blocks = re.split(r"### Precedent \d+\.\d+:", text)
        for block in blocks[1:]:
            fields: dict[str, str] = {}
            for line in block.strip().splitlines():
                line = line.strip()
                if not line or line.startswith("|---"):
                    continue
                m = re.match(r"\|\s*\*\*(.+?)\*\*\s*\|\s*(.+?)\s*\|$", line)
                if m:
                    fields[m.group(1).strip().lower()] = m.group(2).strip()
            case_id = fields.get("case id", "")
            if case_id:
                precedents[case_id] = fields
        return precedents

    @pytest.mark.parametrize("case_id", [
        "DISP-2026-0089",
        "DISP-2026-0142",
        "DISP-2026-0211",
    ])
    def test_precedent_summary_matches_policy(self, case_id: str):
        """Every number in the precedent summary must appear in the policy file."""
        ds = load_dataset()
        precedent = next(
            (p for p in ds.precedents if p.case_id == case_id), None,
        )
        assert precedent is not None, f"{case_id} not found in precedents"

        policy_fields = self._read_policy_precedents().get(case_id, {})
        assert policy_fields, f"{case_id} not found in policy file §7"

        # Build a single text from all policy field values
        policy_text = " ".join(policy_fields.values())
        policy_numbers = _extract_numbers(policy_text)

        summary_numbers = _extract_numbers(precedent.summary)

        # Every number in the summary must appear in the policy file
        missing = summary_numbers - policy_numbers
        assert not missing, (
            f"{case_id}: numbers {missing} in summary but not in policy file. "
            f"Summary: {precedent.summary}"
        )

    def test_precedent_amounts_match_policy(self):
        """The amount field of each Ryde precedent must match the policy."""
        ds = load_dataset()
        policy = self._read_policy_precedents()
        for case_id in ["DISP-2026-0089", "DISP-2026-0142", "DISP-2026-0211"]:
            precedent = next(p for p in ds.precedents if p.case_id == case_id)
            amount_str = policy[case_id].get("refund amount", "$0")
            policy_amount = float(amount_str.replace("$", "").replace("S", "").strip())
            assert precedent.amount == policy_amount, (
                f"{case_id}: amount {precedent.amount} != policy {policy_amount}"
            )
