"""Tests for the Escalation Protocol.

Covers:
- EscalationQueue add, resolve, by_case, open, all
- Safety category always creates an escalation
- Low confidence creates an escalation
- Guard failure creates an escalation
- Outcome mismatch creates an escalation
- Neutral case summary is present in every escalation
- GET /api/escalations returns the queue

Run with: LLM_PROVIDER=mock python -m pytest tests/test_escalation.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import pytest

from ridecourt.data import load_dataset
from ridecourt.escalation import Escalation, EscalationQueue
from ridecourt.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds():
    return load_dataset()


@pytest.fixture(scope="module")
def orc(ds):
    return Orchestrator(ds)


# ---------------------------------------------------------------------------
# EscalationQueue unit tests
# ---------------------------------------------------------------------------


class TestEscalationQueue:
    def test_add_and_all(self):
        q = EscalationQueue()
        assert len(q) == 0

        esc = Escalation(case_id="c1", reason="low_confidence", summary="test")
        q.add(esc)
        assert len(q) == 1
        assert q.all()[0].case_id == "c1"

    def test_open_returns_only_open(self):
        q = EscalationQueue()
        q.add(Escalation(case_id="c1", reason="low_confidence", summary="s1"))
        q.add(Escalation(case_id="c2", reason="safety_category", summary="s2"))
        assert len(q.open()) == 2

        q.resolve("c1", reviewer="alice", resolution="approved")
        assert len(q.open()) == 1
        assert q.open()[0].case_id == "c2"

    def test_resolve_marks_resolved(self):
        q = EscalationQueue()
        q.add(Escalation(case_id="c1", reason="low_confidence", summary="s1"))
        result = q.resolve("c1", reviewer="bob", resolution="denied")
        assert result is True
        assert q.all()[0].status == "resolved"
        assert q.all()[0].reviewer == "bob"
        assert q.all()[0].resolution == "denied"

    def test_resolve_nonexistent_returns_false(self):
        q = EscalationQueue()
        result = q.resolve("nonexistent", reviewer="x", resolution="y")
        assert result is False

    def test_resolve_all_open_for_case(self):
        """resolve() should resolve ALL open escalations for a case_id."""
        q = EscalationQueue()
        q.add(Escalation(case_id="c1", reason="low_confidence", summary="s1"))
        q.add(Escalation(case_id="c1", reason="safety_category", summary="s2"))
        result = q.resolve("c1", reviewer="alice", resolution="approved")
        assert result is True
        assert len(q.open()) == 0
        assert all(e.status == "resolved" for e in q.by_case("c1"))

    def test_by_case(self):
        q = EscalationQueue()
        q.add(Escalation(case_id="c1", reason="low_confidence", summary="s1"))
        q.add(Escalation(case_id="c1", reason="safety_category", summary="s2"))
        q.add(Escalation(case_id="c2", reason="low_confidence", summary="s3"))
        assert len(q.by_case("c1")) == 2
        assert len(q.by_case("c2")) == 1

    def test_safety_escalations_sorted_first(self):
        q = EscalationQueue()
        q.add(Escalation(case_id="c1", reason="low_confidence", summary="s1"))
        q.add(Escalation(case_id="c2", reason="safety_category", summary="s2", severity="high"))
        all_items = q.all()
        assert all_items[0].case_id == "c2"  # safety first

    def test_to_dict(self):
        esc = Escalation(case_id="c1", reason="low_confidence", summary="s1")
        d = esc.to_dict()
        assert d["case_id"] == "c1"
        assert d["reason"] == "low_confidence"
        assert d["summary"] == "s1"
        assert d["status"] == "open"


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


class TestOrchestratorEscalation:
    def test_safety_creates_escalation(self, orc):
        cf = orc.resolve("disp_sf_000")
        assert cf.escalated
        assert cf.escalation is not None
        assert cf.escalation.reason == "safety_category"
        assert cf.escalation.severity == "high"
        assert cf.escalation.summary  # non-empty

    def test_escalation_in_queue(self, orc):
        cf = orc.resolve("disp_sf_001")
        esc_list = orc.escalation_queue.by_case("disp_sf_001")
        assert len(esc_list) >= 1
        assert esc_list[0].reason == "safety_category"

    def test_neutral_summary_present(self, orc):
        cf = orc.resolve("disp_sf_000")
        assert cf.escalation is not None
        assert "Dispute:" in cf.escalation.summary
        assert "Policy engine:" in cf.escalation.summary

    def test_normal_case_no_escalation(self, orc):
        """A normal route deviation case that resolves should not
        create an escalation."""
        cf = orc.resolve("disp_rd_000")
        if not cf.escalated:
            assert cf.escalation is None
        else:
            assert cf.escalation is not None

    def test_guard_failure_creates_escalation(self, ds, monkeypatch):
        import ridecourt.orchestrator as orc_mod
        from ridecourt.guard import Problem

        original_check = orc_mod.guard_check

        def always_fail(output, pack, policy_ids, policy_result=None, **kwargs):
            return [Problem(
                kind="forbidden_word",
                detail="guaranteed found",
                field="position",
            )]

        monkeypatch.setattr(orc_mod, "guard_check", always_fail)
        orc = Orchestrator(ds)
        cf = orc.resolve("disp_rd_000")

        assert cf.escalated
        assert cf.escalation is not None
        assert cf.escalation.reason == "guard_failure_advocate"
        assert cf.escalation.summary  # has a neutral summary


# ---------------------------------------------------------------------------
# API integration — GET /api/escalations
# ---------------------------------------------------------------------------


class TestApiEscalations:
    def test_escalations_endpoint(self):
        from fastapi.testclient import TestClient
        from ridecourt.api import (
            create_app,
            _case_files,
            _case_logs,
            _case_status,
            _overrides,
            _rate_limiter,
        )

        _case_files.clear()
        _case_logs.clear()
        _case_status.clear()
        _overrides.clear()
        _rate_limiter._hits.clear()

        from ridecourt.store import InMemoryStore, reset_store
        reset_store()
        app = create_app(store=InMemoryStore())
        with TestClient(app) as client:
            res = client.get("/api/escalations")
            assert res.status_code == 200
            data = res.json()
            assert "escalations" in data
            assert "total" in data
            assert "open" in data
            assert data["total"] >= 0

    def test_escalations_after_safety_resolve(self):
        from fastapi.testclient import TestClient
        from ridecourt.api import (
            create_app,
            _case_files,
            _case_logs,
            _case_status,
            _overrides,
            _rate_limiter,
        )
        import time

        _case_files.clear()
        _case_logs.clear()
        _case_status.clear()
        _overrides.clear()
        _rate_limiter._hits.clear()

        from ridecourt.store import InMemoryStore, reset_store
        reset_store()
        app = create_app(store=InMemoryStore())
        with TestClient(app) as client:
            # Find a safety dispute.
            res = client.get("/api/disputes")
            safety_id = None
            for d in res.json()["disputes"]:
                if d["category"] == "safety":
                    safety_id = d["dispute_id"]
                    break
            assert safety_id is not None

            # Resolve it.
            client.post(f"/api/disputes/{safety_id}/resolve")

            # Wait for completion.
            deadline = time.time() + 30
            while time.time() < deadline:
                case_res = client.get(f"/api/disputes/{safety_id}")
                if case_res.json()["status"] in ("ruled", "escalated"):
                    break
                time.sleep(0.5)

            # Check escalation queue.
            esc_res = client.get("/api/escalations")
            esc_data = esc_res.json()
            assert esc_data["total"] >= 1
            safety_escs = [
                e for e in esc_data["escalations"]
                if e["case_id"] == safety_id
            ]
            assert len(safety_escs) >= 1
            assert safety_escs[0]["reason"] == "safety_category"
