"""Tests for the persistence layer.

Uses ``InMemoryStore`` as a fake store. The key pattern: create two app
instances that share the same ``InMemoryStore`` — the second app sees
state written by the first, simulating cross-process visibility.

Covers:
- A resolved case survives creating a new app instance on the same store
- An override and its learned precedent survive a new app instance
- The daily resolve counter survives a new app instance
- A case written by another process (directly to the store) appears in
  the queue
- ``scripts/preresolve.py`` skips already-resolved cases

Run with: LLM_PROVIDER=mock python -m pytest tests/test_persistence.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import time
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure scripts/ is on sys.path so we can import preresolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from ridecourt.api import (
    create_app,
    _case_files,
    _case_logs,
    _case_status,
    _daily_cap,
    _overrides,
    _rate_limiter,
)
from ridecourt.store import InMemoryStore, reset_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(store: InMemoryStore) -> TestClient:
    """Create a TestClient on the given store, with clean in-memory state."""
    _case_files.clear()
    _case_logs.clear()
    _case_status.clear()
    _overrides.clear()
    _rate_limiter._hits.clear()
    _daily_cap._day = None
    _daily_cap._count = 0
    _daily_cap._loaded_from_store = False
    _daily_cap.cap = 300

    reset_store()
    app = create_app(store=store)
    return TestClient(app)


def _resolve_and_wait(client: TestClient, dispute_id: str, timeout: float = 30) -> dict:
    """Resolve a dispute and poll until done, returning the case data."""
    client.post(f"/api/disputes/{dispute_id}/resolve")
    deadline = time.time() + timeout
    while time.time() < deadline:
        res = client.get(f"/api/disputes/{dispute_id}")
        data = res.json()
        if data["status"] in ("ruled", "escalated", "error"):
            return data
        time.sleep(0.5)
    raise TimeoutError(f"Case {dispute_id} did not resolve in {timeout}s")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStateSurvivesNewAppInstance:
    """State must survive creating a new app instance on the same store."""

    def test_resolved_case_survives(self):
        """A resolved case must appear as ruled on a new app instance."""
        store = InMemoryStore()
        client1 = _make_client(store)

        # Get the first dispute ID.
        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]

        # Resolve it on app1.
        data = _resolve_and_wait(client1, did)
        assert data["status"] in ("ruled", "escalated")
        assert data.get("ruling") is not None or data.get("escalated") is True

        # Create a second app instance on the same store.
        client2 = _make_client(store)

        # The case must show as ruled on app2.
        case_res = client2.get(f"/api/disputes/{did}")
        assert case_res.status_code == 200
        case_data = case_res.json()
        assert case_data["status"] in ("ruled", "escalated"), (
            f"Case {did} should be ruled/escalated on a new app instance, "
            f"got status={case_data['status']}"
        )
        # The case file data must be loaded from the store.
        assert case_data.get("evidence_pack") is not None, (
            "Evidence pack must be loaded from the store on a new app instance."
        )
        assert case_data.get("ruling") is not None or case_data.get("escalated") is True

        # The queue must show the case as ruled/escalated.
        queue_res = client2.get("/api/disputes")
        queue_item = next(
            d for d in queue_res.json()["disputes"] if d["dispute_id"] == did
        )
        assert queue_item["status"] in ("ruled", "escalated")

    def test_override_and_precedent_survive(self):
        """An override and its learned precedent must survive a new app
        instance, so the learning loop is visible after a restart."""
        store = InMemoryStore()
        client1 = _make_client(store)

        # Get a route_deviation dispute to override.
        res = client1.get("/api/disputes")
        rd_id = None
        for d in res.json()["disputes"]:
            if d["category"] == "route_deviation":
                rd_id = d["dispute_id"]
                break
        assert rd_id is not None

        # Override it.
        client1.post(
            f"/api/disputes/{rd_id}/override",
            json={
                "outcome": "refund_route_deviation",
                "amount": 7.50,
                "reason": "Reviewer confirmed the deviation.",
            },
        )

        # Create a second app instance on the same store.
        client2 = _make_client(store)

        # The override status must survive.
        case_res = client2.get(f"/api/disputes/{rd_id}")
        assert case_res.json()["status"] == "overridden"

        # The learned precedent must be in the dataset so future cases can
        # retrieve it.
        from ridecourt.api import _dataset
        override_precs = [
            p for p in _dataset.precedents if getattr(p, "source", "") == "override"
        ]
        assert len(override_precs) >= 1, (
            "Override-sourced precedent must survive a new app instance."
        )

    def test_daily_counter_survives(self):
        """The daily resolve counter must survive a restart so the cost
        guard does not reset when the instance restarts."""
        store = InMemoryStore()
        client1 = _make_client(store)

        # Resolve one case on app1 (consumes 1 from the daily cap).
        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]
        _resolve_and_wait(client1, did)

        # The counter must be 1.
        assert _daily_cap.snapshot()["resolves_today"] == 1

        # Create a second app instance on the same store.
        client2 = _make_client(store)

        # The counter must still be 1 on the new instance.
        snap = _daily_cap.snapshot()
        assert snap["resolves_today"] == 1, (
            f"Daily counter should be 1 after restart, got {snap['resolves_today']}. "
            "The cost guard must not reset when the instance restarts."
        )
        assert snap["remaining_resolves_today"] == 299


class TestCrossProcessVisibility:
    """A case written by another process must appear in the queue."""

    def test_case_from_other_process_in_queue(self):
        """A case written directly to the store must appear in the queue
        of a new app instance that didn't resolve it."""
        store = InMemoryStore()
        client1 = _make_client(store)

        # Get a dispute ID.
        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]

        # Simulate another process writing a resolved case to the store
        # directly (e.g. via preresolve.py).
        store.save_case(did, {
            "dispute_id": did,
            "dispute": {"dispute_id": did, "trip_id": "trip_rd_000",
                        "filed_by": "rider", "category": "route_deviation",
                        "claim_text": "test", "filed_at": "2026-01-01T00:00:00Z"},
            "evidence_pack": {"items": []},
            "ruling": {"outcome": "no_refund", "amount": 0.0,
                       "confidence": 0.9, "reasoning": "test",
                       "cites": [], "message_to_rider": "test",
                       "message_to_driver": "test"},
            "escalated": False,
        })
        store.save_events(did, [{"type": "RULING", "content": {"outcome": "no_refund"}}])
        store.set_status(did, "ruled")

        # Create a new app instance — it should see the case as ruled.
        client2 = _make_client(store)

        # The queue must show it as ruled.
        queue_res = client2.get("/api/disputes")
        queue_item = next(
            d for d in queue_res.json()["disputes"] if d["dispute_id"] == did
        )
        assert queue_item["status"] == "ruled", (
            f"A case written to the store by another process must appear as "
            f"ruled in the queue. Got {queue_item['status']}."
        )

        # The case file endpoint must return the stored data.
        case_res = client2.get(f"/api/disputes/{did}")
        case_data = case_res.json()
        assert case_data["status"] == "ruled"
        assert case_data.get("ruling") is not None
        assert case_data["ruling"]["outcome"] == "no_refund"


class TestPreresolveSkipsDoneCases:
    """preresolve.py must skip cases that are already in the store."""

    def test_preresolve_skips_done(self):
        """After preresolve resolves a case, a second run must skip it."""
        store = InMemoryStore()
        client1 = _make_client(store)

        # Get a dispute ID.
        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]

        # Resolve it directly via the orchestrator + store.
        from ridecourt.data import load_dataset
        from ridecourt.orchestrator import Orchestrator
        from ridecourt.serialize import serialize_case_file, serialize_events

        ds = load_dataset()
        orc = Orchestrator(ds)
        cf = orc.resolve(did)
        store.save_case(cf.dispute_id, serialize_case_file(cf))
        store.save_events(cf.dispute_id, serialize_events(cf))
        store.set_status(cf.dispute_id, "ruled" if not cf.escalated else "escalated")

        # Now run preresolve — it should skip this case.
        # Capture stderr from preresolve.main.
        import io
        from contextlib import redirect_stderr

        # Set up argparse args for preresolve.
        old_argv = sys.argv
        sys.argv = [
            "preresolve.py",
            "--provider", "mock",
            did,
        ]
        try:
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                import preresolve
                preresolve.main()
            output = err_buf.getvalue()
            # The case should have been skipped.
            assert "SKIPPED" in output, (
                f"preresolve should skip an already-resolved case. "
                f"Output: {output}"
            )
        finally:
            sys.argv = old_argv

    def test_preresolve_force_re_resolves(self):
        """With --force, preresolve re-resolves even done cases."""
        store = InMemoryStore()
        client1 = _make_client(store)

        # Get a dispute ID.
        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]

        # Resolve it directly.
        from ridecourt.data import load_dataset
        from ridecourt.orchestrator import Orchestrator
        from ridecourt.serialize import serialize_case_file, serialize_events

        ds = load_dataset()
        orc = Orchestrator(ds)
        cf = orc.resolve(did)
        store.save_case(cf.dispute_id, serialize_case_file(cf))
        store.save_events(cf.dispute_id, serialize_events(cf))
        store.set_status(cf.dispute_id, "ruled")

        # Run preresolve with --force.
        import io
        from contextlib import redirect_stderr

        old_argv = sys.argv
        sys.argv = [
            "preresolve.py",
            "--provider", "mock",
            "--force",
            did,
        ]
        try:
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                import preresolve
                preresolve.main()
            output = err_buf.getvalue()
            assert "SKIPPED" not in output, (
                f"preresolve --force should not skip. Output: {output}"
            )
            assert "ruled" in output or "escalated" in output, (
                f"preresolve --force should re-resolve. Output: {output}"
            )
        finally:
            sys.argv = old_argv
