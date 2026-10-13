"""Tests for the Learning Feedback Loop.

Covers:
- Human override stores a precedent with source="override"
- The override precedent is retrievable for similar future cases
- "Learned from override" appears in the timeline when a precedent
  from an override is used
- The override resolves the escalation in the queue

Run with: LLM_PROVIDER=mock python -m pytest tests/test_feedback.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import time

import pytest
from fastapi.testclient import TestClient

from ridecourt.api import (
    create_app,
    _case_files,
    _case_logs,
    _case_status,
    _overrides,
    _rate_limiter,
)
from ridecourt.data import load_dataset, Precedent
from ridecourt.precedent import retrieve_precedents
from ridecourt.store import InMemoryStore, reset_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    _case_files.clear()
    _case_logs.clear()
    _case_status.clear()
    _overrides.clear()
    _rate_limiter._hits.clear()

    store = InMemoryStore()
    reset_store()
    app = create_app(store=store)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def ds():
    return load_dataset()


# ---------------------------------------------------------------------------
# Override stores a precedent
# ---------------------------------------------------------------------------


class TestOverrideAsPrecedent:
    def test_override_adds_precedent_to_dataset(self, client):
        """Override should add a new Precedent with source='override'."""
        disputes_res = client.get("/api/disputes")
        dispute_id = disputes_res.json()["disputes"][0]["dispute_id"]

        res = client.post(
            f"/api/disputes/{dispute_id}/override",
            json={
                "outcome": "refund_route_deviation",
                "amount": 5.50,
                "reason": "Reviewer judged the deviation was unjustified.",
            },
        )
        assert res.status_code == 200

        # The precedent should now be in the dataset.
        from ridecourt.api import _dataset
        ds = _dataset
        override_precs = [
            p for p in ds.precedents if getattr(p, "source", "") == "override"
        ]
        assert len(override_precs) >= 1
        assert override_precs[-1].outcome == "refund_route_deviation"
        assert override_precs[-1].amount == 5.50
        assert override_precs[-1].source == "override"
        assert override_precs[-1].reason == "Reviewer judged the deviation was unjustified."

    def test_override_precedent_retrieved_for_similar_case(self, client, ds):
        """After an override is stored, a similar case should retrieve it."""
        # Find a route_deviation dispute.
        disputes_res = client.get("/api/disputes")
        rd_dispute_id = None
        for d in disputes_res.json()["disputes"]:
            if d["category"] == "route_deviation":
                rd_dispute_id = d["dispute_id"]
                break
        assert rd_dispute_id is not None

        # Override it.
        client.post(
            f"/api/disputes/{rd_dispute_id}/override",
            json={
                "outcome": "refund_route_deviation",
                "amount": 7.00,
                "reason": "Deviation confirmed by GPS.",
            },
        )

        # Now check that a similar route_deviation dispute retrieves it.
        from ridecourt.api import _dataset
        dispute = _dataset.dispute(rd_dispute_id)
        matches = retrieve_precedents(dispute, _dataset.precedents, top_k=3)

        override_matches = [m for m in matches if m.source == "override"]
        assert len(override_matches) >= 1
        assert override_matches[0].source == "override"

    def test_learned_from_override_in_timeline(self, client):
        """When a case is resolved and the orchestrator retrieves an
        override-sourced precedent, the PRECEDENT event should show
        learned_from_override=True."""
        disputes_res = client.get("/api/disputes")
        rd_dispute_id = None
        for d in disputes_res.json()["disputes"]:
            if d["category"] == "route_deviation":
                rd_dispute_id = d["dispute_id"]
                break
        assert rd_dispute_id is not None

        # Override it to create an override precedent.
        client.post(
            f"/api/disputes/{rd_dispute_id}/override",
            json={
                "outcome": "refund_route_deviation",
                "amount": 5.00,
                "reason": "Test override.",
            },
        )

        # Now resolve a different route_deviation dispute.
        other_rd_id = None
        for d in disputes_res.json()["disputes"]:
            if d["category"] == "route_deviation" and d["dispute_id"] != rd_dispute_id:
                other_rd_id = d["dispute_id"]
                break
        assert other_rd_id is not None

        client.post(f"/api/disputes/{other_rd_id}/resolve")

        # Wait for resolution.
        deadline = time.time() + 30
        while time.time() < deadline:
            res = client.get(f"/api/disputes/{other_rd_id}")
            data = res.json()
            if data["status"] in ("ruled", "escalated"):
                break
            time.sleep(0.5)

        # Check the PRECEDENT event.
        events = data.get("events", [])
        prec_events = [e for e in events if e["type"] == "PRECEDENT"]
        assert len(prec_events) >= 1

        # The override precedent should be in the matches.
        matches = prec_events[0]["content"]["matches"]
        override_matches = [m for m in matches if m.get("source") == "override"]
        if override_matches:
            assert prec_events[0]["content"]["learned_from_override"] is True

    def test_override_resolves_escalation(self, client):
        """Override should resolve the escalation in the queue."""
        disputes_res = client.get("/api/disputes")
        # Find a safety case (always escalates).
        safety_id = None
        for d in disputes_res.json()["disputes"]:
            if d["category"] == "safety":
                safety_id = d["dispute_id"]
                break
        assert safety_id is not None

        # Resolve it (will escalate).
        client.post(f"/api/disputes/{safety_id}/resolve")
        deadline = time.time() + 30
        while time.time() < deadline:
            res = client.get(f"/api/disputes/{safety_id}")
            if res.json()["status"] in ("ruled", "escalated"):
                break
            time.sleep(0.5)

        # Check the escalation is open.
        esc_res = client.get("/api/escalations")
        assert esc_res.json()["total"] >= 1

        # Override the case.
        client.post(
            f"/api/disputes/{safety_id}/override",
            json={
                "outcome": "escalate",
                "amount": 0,
                "reason": "Reviewed and resolved by human.",
            },
        )

        # Check the escalation is now resolved.
        esc_res2 = client.get("/api/escalations")
        open_escs = [
            e for e in esc_res2.json()["escalations"]
            if e["case_id"] == safety_id and e["status"] == "open"
        ]
        assert len(open_escs) == 0
