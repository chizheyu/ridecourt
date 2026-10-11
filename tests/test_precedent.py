"""Tests for the Precedent Agent — TF-IDF retrieval of similar past rulings.

Covers:
- TF-IDF retrieval returns top 3 for route_deviation
- Same-category precedents score higher than different-category
- Similarity reason is human-readable
- Override-sourced precedents are retrieved (feedback loop)
- PrecedentMatch has source field

Run with: LLM_PROVIDER=mock python -m pytest tests/test_precedent.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest

from ridecourt.data import Dispute, Precedent, load_dataset
from ridecourt.precedent import retrieve_precedents, PrecedentMatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


# ---------------------------------------------------------------------------
# TF-IDF retrieval
# ---------------------------------------------------------------------------


class TestPrecedentRetrieval:
    def test_returns_top_3(self, ds):
        dispute = ds.dispute("disp_rd_000")
        matches = retrieve_precedents(dispute, ds.precedents, top_k=3)
        assert len(matches) <= 3
        assert len(matches) > 0

    def test_same_category_scores_higher(self, ds):
        """A route_deviation dispute should match route_deviation precedents
        higher than other categories."""
        dispute = ds.dispute("disp_rd_000")
        matches = retrieve_precedents(dispute, ds.precedents, top_k=3)

        # The top match should be a route_deviation precedent.
        assert matches[0].category == "route_deviation"

    def test_no_show_matches_no_show_precedents(self, ds):
        dispute = ds.dispute("disp_ns_d_000")
        matches = retrieve_precedents(dispute, ds.precedents, top_k=3)

        # At least one match should be a no_show_driver precedent.
        categories = [m.category for m in matches]
        assert "no_show_driver" in categories

    def test_similarity_sorted_descending(self, ds):
        dispute = ds.dispute("disp_rd_000")
        matches = retrieve_precedents(dispute, ds.precedents, top_k=3)

        for i in range(len(matches) - 1):
            assert matches[i].similarity >= matches[i + 1].similarity

    def test_reason_is_readable(self, ds):
        dispute = ds.dispute("disp_rd_000")
        matches = retrieve_precedents(dispute, ds.precedents, top_k=3)

        for m in matches:
            assert isinstance(m.reason, str)
            assert len(m.reason) > 0

    def test_empty_precedents_returns_empty(self):
        dispute = Dispute(
            dispute_id="test",
            trip_id="test",
            filed_by="rider",
            category="route_deviation",
            claim_text="Test dispute",
            filed_at="2025-01-01T00:00:00",
        )
        matches = retrieve_precedents(dispute, [], top_k=3)
        assert matches == []

    def test_source_field_default_seed(self, ds):
        dispute = ds.dispute("disp_rd_000")
        matches = retrieve_precedents(dispute, ds.precedents, top_k=3)
        # Seed precedents have source="seed".
        for m in matches:
            assert m.source in ("seed", "override")


# ---------------------------------------------------------------------------
# Override-sourced precedents (learning feedback loop)
# ---------------------------------------------------------------------------


class TestOverridePrecedents:
    def test_override_precedent_is_retrieved(self, ds):
        """When an override is stored as a precedent, it should be
        retrievable for similar future cases."""
        dispute = ds.dispute("disp_rd_000")

        # Simulate an override precedent.
        override_prec = Precedent(
            case_id="override-test-001",
            category="route_deviation",
            summary="Driver took a longer route. Refund S$5.00.",
            outcome="refund",
            amount=5.00,
            source="override",
            reason="Reviewer judged the deviation was unjustified.",
        )

        all_precedents = list(ds.precedents) + [override_prec]
        matches = retrieve_precedents(dispute, all_precedents, top_k=3)

        # The override precedent should be in the top 3.
        override_matches = [m for m in matches if m.source == "override"]
        assert len(override_matches) >= 1
        assert override_matches[0].case_id == "override-test-001"

    def test_override_precedent_has_source_field(self, ds):
        dispute = ds.dispute("disp_rd_000")
        override_prec = Precedent(
            case_id="override-test-002",
            category="route_deviation",
            summary="Driver took a longer route. Refund S$5.00.",
            outcome="refund",
            amount=5.00,
            source="override",
            reason="Reviewer judged the deviation was unjustified.",
        )

        all_precedents = list(ds.precedents) + [override_prec]
        matches = retrieve_precedents(dispute, all_precedents, top_k=3)

        for m in matches:
            assert hasattr(m, "source")
            assert m.source in ("seed", "override")


# ---------------------------------------------------------------------------
# Precedent Agent (LLM wrapper)
# ---------------------------------------------------------------------------


class TestPrecedentAgent:
    def test_agent_build(self, ds):
        from ridecourt.agents.precedent import PrecedentAgent, PrecedentOutput

        dispute = ds.dispute("disp_rd_000")
        agent = PrecedentAgent()
        output, meta, matches = agent.build(dispute, ds.precedents)

        assert isinstance(output, PrecedentOutput)
        assert len(output.matches) > 0
        assert len(output.matches) <= 3
        assert len(matches) == len(output.matches)
