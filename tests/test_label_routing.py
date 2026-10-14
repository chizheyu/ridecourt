"""Sweep test: the engine's outcome, amount, and routing must match every
label in labels.json (synthetic) and the Ryde adapter labels (official).

For official cases, the label's `ryde_outcome` holds Ryde's original wording
separately; our `outcome` uses our own outcome names.
"""

import pytest

from ridecourt.data import load_dataset
from ridecourt.tools import build_evidence_pack, facts_from_evidence_pack
from ridecourt.policy import evaluate
from ridecourt.ryde_adapter import RydeDatasetAdapter
from ridecourt.precedent import retrieve_precedents


def _run_case(ds, did: str):
    """Build evidence pack, facts, and evaluate for one case."""
    disp = ds.dispute(did)
    trip = ds.trip(disp.trip_id)
    pack = build_evidence_pack(disp, ds)
    facts = facts_from_evidence_pack(pack)
    # Retrieve precedents before evaluate so they feed into confidence.
    matches = retrieve_precedents(disp, ds.precedents, top_k=3)
    best_sim = matches[0].similarity if matches else 0.0
    pr = evaluate(
        disp, trip, ds.profiles, facts, trip.fare,
        has_precedent_match=len(matches) > 0,
        precedent_similarity=best_sim,
    )
    return pr


class TestSyntheticLabels:
    """Every synthetic label's routing and deciding_section must match
    the engine's output."""

    @pytest.fixture(scope="class")
    def ds(self):
        return load_dataset()

    def test_all_labels_have_routing(self, ds):
        """Every label must have routing and deciding_section."""
        for did, label in ds.labels.items():
            assert label.routing in ("ai_final", "human", ""), (
                f"{did}: routing='{label.routing}' is not valid"
            )
            assert label.deciding_section, (
                f"{did}: deciding_section is empty"
            )

    def test_engine_routing_matches_labels(self, ds):
        """The engine's routing must match every label's routing."""
        failures = []
        for did, label in ds.labels.items():
            if not did.startswith("disp_"):
                continue
            pr = _run_case(ds, did)
            if pr.routing != label.routing:
                failures.append(
                    f"{did}: engine routing={pr.routing} "
                    f"but label routing={label.routing} "
                    f"(reason: {pr.routing_reason})"
                )
        assert not failures, "\n".join(failures)

    def test_engine_outcome_matches_labels(self, ds):
        """The engine's outcome must match every label's outcome."""
        failures = []
        for did, label in ds.labels.items():
            if not did.startswith("disp_"):
                continue
            pr = _run_case(ds, did)
            if pr.outcome != label.outcome:
                failures.append(
                    f"{did}: engine outcome={pr.outcome} "
                    f"but label outcome={label.outcome}"
                )
        assert not failures, "\n".join(failures)

    def test_engine_amount_matches_labels(self, ds):
        """The engine's amount must match every label's amount."""
        failures = []
        for did, label in ds.labels.items():
            if not did.startswith("disp_"):
                continue
            pr = _run_case(ds, did)
            if abs(pr.amount - label.amount) > 0.02:
                failures.append(
                    f"{did}: engine amount={pr.amount:.2f} "
                    f"but label amount={label.amount:.2f}"
                )
        assert not failures, "\n".join(failures)


class TestOfficialLabels:
    """Official Ryde labels must have routing, deciding_section, ryde_outcome."""

    @pytest.fixture(scope="class")
    def ds(self):
        return RydeDatasetAdapter().load()

    def test_official_labels_have_routing(self, ds):
        for did in ["DISP-001", "DISP-002", "DISP-003"]:
            label = ds.labels.get(did)
            assert label is not None, f"{did} not found in labels"
            assert label.routing in ("ai_final", "human")
            assert label.deciding_section
            assert label.ryde_outcome

    def test_disp001_label(self, ds):
        label = ds.labels["DISP-001"]
        assert label.outcome == "refund_partial_excess_distance"
        assert label.amount == 3.44
        assert label.routing == "human"
        assert label.deciding_section == "§6.2"
        assert label.ryde_outcome == "partial_refund of excess distance fare S$3.44"

    def test_disp002_label(self, ds):
        label = ds.labels["DISP-002"]
        assert label.outcome == "cancellation_fee_upheld"
        assert label.amount == 5.0
        assert label.routing == "ai_final"
        assert label.deciding_section == "§6.3"

    def test_disp003_label(self, ds):
        label = ds.labels["DISP-003"]
        assert label.outcome == "refund_surge_differential"
        assert label.amount == 4.50
        assert label.routing == "ai_final"
        assert label.deciding_section == "§6.1"

    def test_engine_matches_official_labels(self, ds):
        """The engine's outcome, amount, and routing must match the
        official labels."""
        for did in ["DISP-001", "DISP-002", "DISP-003"]:
            pr = _run_case(ds, did)
            label = ds.labels[did]
            assert pr.outcome == label.outcome, (
                f"{did}: engine outcome={pr.outcome} != label={label.outcome}"
            )
            assert abs(pr.amount - label.amount) < 0.02, (
                f"{did}: engine amount={pr.amount} != label={label.amount}"
            )
            assert pr.routing == label.routing, (
                f"{did}: engine routing={pr.routing} != label={label.routing}"
            )
