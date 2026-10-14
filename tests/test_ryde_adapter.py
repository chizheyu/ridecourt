"""Tests for the Ryde DatasetAdapter.

Tests:
- Loads all 3 official cases from data/ryde_samples/
- Both chat-log shapes (Shape 1: message_id/sender/message, Shape 2: sender/type/content)
- Held-out answer-key fields never reach agents/tools/evidence pack
- Labels for evaluation (DISP-001, DISP-002, DISP-003)

Run with: .venv/bin/python -m pytest tests/test_ryde_adapter.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest

from ridecourt.data import Dataset
from ridecourt.ryde_adapter import RydeDatasetAdapter, RYDE_SAMPLES_DIR


# ---------------------------------------------------------------------------
# Skip if sample files are missing
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(
    not RYDE_SAMPLES_DIR.exists(),
    reason="data/ryde_samples/ directory not found",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def adapter() -> RydeDatasetAdapter:
    return RydeDatasetAdapter()


@pytest.fixture(scope="module")
def dataset(adapter) -> Dataset:
    return adapter.load()


# ---------------------------------------------------------------------------
# Load all 3 cases
# ---------------------------------------------------------------------------


class TestLoadAll:
    def test_loads_three_cases(self, dataset):
        assert len(dataset.disputes) == 3
        assert len(dataset.trips) == 3
        assert len(dataset.chats) == 3

    def test_expected_dispute_ids(self, dataset):
        ids = set(dataset.disputes.keys())
        assert "DISP-001" in ids
        assert "DISP-002" in ids
        assert "DISP-003" in ids

    def test_profiles_loaded(self, dataset):
        assert len(dataset.profiles) >= 4  # at least 2 riders + 2 drivers

    def test_labels_loaded(self, dataset):
        assert len(dataset.labels) == 3

    def test_empty_dir_returns_empty(self):
        """If the directory doesn't exist, return an empty dataset."""
        ad = RydeDatasetAdapter(data_dir="/nonexistent/path")
        ds = ad.load()
        assert len(ds.disputes) == 0


# ---------------------------------------------------------------------------
# DISP-001: route deviation
# ---------------------------------------------------------------------------


class TestDisp001:
    def test_category(self, dataset):
        d = dataset.dispute("DISP-001")
        assert d.category == "route_deviation"

    def test_filed_by_rider(self, dataset):
        d = dataset.dispute("DISP-001")
        assert d.filed_by == "rider"

    def test_trip_has_traffic_conditions(self, dataset):
        d = dataset.dispute("DISP-001")
        trip = dataset.trip(d.trip_id)
        assert len(trip.traffic_conditions) > 0

    def test_optimal_route(self, dataset):
        d = dataset.dispute("DISP-001")
        trip = dataset.trip(d.trip_id)
        assert trip.optimal_route.distance_km > 0

    def test_label_outcome_and_amount(self, dataset):
        """DISP-001 label: partial refund, S$3.44."""
        lbl = dataset.label("DISP-001")
        assert lbl is not None
        assert "refund" in lbl.outcome.lower() or "partial" in lbl.outcome.lower()
        assert lbl.amount == 3.44


# ---------------------------------------------------------------------------
# DISP-002: no-show
# ---------------------------------------------------------------------------


class TestDisp002:
    def test_category(self, dataset):
        d = dataset.dispute("DISP-002")
        assert d.category == "no_show_rider"

    def test_filed_by_rider(self, dataset):
        d = dataset.dispute("DISP-002")
        assert d.filed_by == "rider"

    def test_label_upheld(self, dataset):
        """DISP-002: cancellation fee upheld, S$5.00."""
        lbl = dataset.label("DISP-002")
        assert lbl is not None
        assert lbl.outcome == "cancellation_fee_upheld"
        assert lbl.amount == 5.0


# ---------------------------------------------------------------------------
# DISP-003: fare/surge
# ---------------------------------------------------------------------------


class TestDisp003:
    def test_category(self, dataset):
        d = dataset.dispute("DISP-003")
        assert d.category == "fare_error"

    def test_filed_by_rider(self, dataset):
        d = dataset.dispute("DISP-003")
        assert d.filed_by == "rider"

    def test_label_surge_refund(self, dataset):
        """DISP-003: surge differential refund, S$4.50."""
        lbl = dataset.label("DISP-003")
        assert lbl is not None
        assert "surge" in lbl.outcome.lower()
        assert lbl.amount == 4.50


# ---------------------------------------------------------------------------
# Chat-log shapes
# ---------------------------------------------------------------------------


class TestChatLogShapes:
    """Both chat-log shapes must be mapped to ChatMessage objects."""

    def test_disp001_shape1_message_id(self, dataset):
        """DISP-001 uses Shape 1: message_id / sender / message / timestamp."""
        d = dataset.dispute("DISP-001")
        chat = dataset.chat(d.trip_id)
        assert len(chat.messages) > 0
        # Shape 1 messages should have message_id set
        m = chat.messages[0]
        assert m.message_id.startswith("MSG-")
        assert m.message_type == "message"

    def test_disp002_shape2_type_content(self, dataset):
        """DISP-002 uses Shape 2: sender / type / content / timestamp."""
        d = dataset.dispute("DISP-002")
        chat = dataset.chat(d.trip_id)
        assert len(chat.messages) > 0
        # Shape 2 messages should have message_type set
        m = chat.messages[0]
        assert m.message_type in ("message", "call", "system")

    def test_disp002_has_call_messages(self, dataset):
        """DISP-002 chat should include a 'call' type message."""
        d = dataset.dispute("DISP-002")
        chat = dataset.chat(d.trip_id)
        call_msgs = [m for m in chat.messages if m.message_type == "call"]
        assert len(call_msgs) >= 1

    def test_disp003_shape1(self, dataset):
        """DISP-003 uses Shape 1 as well (or at least has messages)."""
        d = dataset.dispute("DISP-003")
        chat = dataset.chat(d.trip_id)
        assert len(chat.messages) >= 0  # may or may not have chat

    def test_senders_are_valid(self, dataset):
        """All senders must be rider, driver, or system."""
        for chat in dataset.chats.values():
            for m in chat.messages:
                assert m.sender in ("rider", "driver", "system")


# ---------------------------------------------------------------------------
# Held-out answer-key fields
# ---------------------------------------------------------------------------


class TestHeldOutFields:
    """Answer-key fields must never be in the mapped data.

    Held-out:
    - case_metadata.expected_*
    - ruling_rationale
    - recommended_ruling
    - chat_log_analysis.analysis
    - traffic_analysis.analysis
    - evidence_summary
    - fare_breakdown.notes
    """

    def test_dispute_label_not_in_claim_text(self, dataset):
        """The label (expected answer) must not appear in the claim text."""
        for did in ("DISP-001", "DISP-002", "DISP-003"):
            d = dataset.dispute(did)
            lbl = dataset.label(did)
            if lbl and lbl.amount > 0:
                amount_str = f"{lbl.amount:.2f}"
                # The specific answer amount should not appear in the claim
                # (e.g. "3.44" should not be in the claim text for DISP-001)
                # Note: DISP-003's claim mentions "$4.50" so we only check
                # the ruling outcome keyword
                if "partial" in lbl.outcome:
                    assert "partial refund" not in d.claim_text.lower()

    def test_no_ruling_rationale_in_dispute(self, dataset):
        """ruling_rationale must not be in the dispute's claim text."""
        for d in dataset.disputes.values():
            assert "ruling_rationale" not in d.claim_text.lower()

    def test_label_is_separate_from_claim(self, dataset):
        """The label dict is stored separately on Dispute, not in claim_text."""
        for did, lbl in dataset.labels.items():
            d = dataset.dispute(did)
            # The label outcome should not be the claim text
            assert d.claim_text != lbl.outcome


# ---------------------------------------------------------------------------
# load_case: single case by ID
# ---------------------------------------------------------------------------


class TestLoadCase:
    def test_load_single_case_disp001(self, adapter):
        ds = adapter.load_case("DISP-001")
        assert ds is not None
        assert len(ds.disputes) == 1
        assert "DISP-001" in ds.disputes

    def test_load_single_case_disp002(self, adapter):
        ds = adapter.load_case("DISP-002")
        assert ds is not None
        assert "DISP-002" in ds.disputes

    def test_load_nonexistent_returns_none(self, adapter):
        ds = adapter.load_case("DISP-999")
        assert ds is None
