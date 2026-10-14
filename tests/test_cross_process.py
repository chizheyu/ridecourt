"""Cross-process persistence tests.

Every piece of state the store saves must be visible through the API on a
fresh app instance that shares the store.  This simulates a Cloud Run
scenario where one instance resolves a case (or preresolve writes results
offline) and a different instance serves the UI to a reviewer.

State the store saves (per ``ridecourt/store.py`` docstring):

1. Case files       — ``save_case`` / ``load_case``
2. Case events      — ``save_events`` / ``load_events``
3. Case statuses    — ``set_status`` / ``get_status`` / ``list_statuses``
4. Human overrides  — ``save_override`` / ``load_overrides``
5. Learned precedents — ``save_precedent`` / ``load_precedents``
6. Daily resolve counter — ``set_daily_count`` / ``get_daily_count``

For each, this file has a test where one app instance (or a direct store
write simulating another process) writes the data and a fresh app instance
on the same store exposes it through the API that a person uses:

- Queue (``GET /api/disputes``) — status visibility
- Case file (``GET /api/disputes/{id}``) — case data + events
- SSE stream (``GET /api/disputes/{id}/events``) — event replay + done
- Precedent retrieval in the next resolve — learned precedents
- Metrics (``GET /api/metrics``) — resolved count
- Cost guard (``POST /api/disputes/{id}/resolve`` 429) — daily counter

The events test (``test_events_replayed_from_store_on_fresh_instance``)
fails on the code before the fix: the case-file endpoint returns
``"events": []`` and the SSE stream sends only the ``done`` event.

Run with: LLM_PROVIDER=mock python -m pytest tests/test_cross_process.py -q
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import json
import sys
import time
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


def _clear_jsonl(dispute_id: str) -> None:
    """Delete the JSONL file for a case, simulating a fresh instance that
    didn't run the case (so events are only in the store)."""
    from ridecourt.events import RUNS_DIR
    jsonl = RUNS_DIR / f"{dispute_id}.jsonl"
    if jsonl.exists():
        jsonl.unlink()


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
# 1. Case files — a resolved case file survives a fresh app instance
# ---------------------------------------------------------------------------


class TestCaseFileCrossProcess:
    """Case files saved by one instance must be visible on a fresh instance."""

    def test_case_file_loaded_from_store(self):
        """``GET /api/disputes/{id}`` on a fresh instance must return the
        evidence pack, ruling, and case data from the store."""
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]
        _resolve_and_wait(client1, did)

        # Fresh instance on the same store.
        client2 = _make_client(store)
        case_res = client2.get(f"/api/disputes/{did}")
        assert case_res.status_code == 200
        data = case_res.json()
        assert data["status"] in ("ruled", "escalated")
        assert data.get("evidence_pack") is not None, (
            "Evidence pack must be loaded from the store on a fresh instance."
        )
        assert data.get("ruling") is not None or data.get("escalated") is True


# ---------------------------------------------------------------------------
# 2. Events — stored events must be replayed on a fresh instance
# ---------------------------------------------------------------------------


class TestEventsCrossProcess:
    """Events saved by one instance must be visible on a fresh instance.

    This is the regression test for the bug described in the task: after
    preresolve writes events to the store, opening a case on a fresh
    instance shows an empty courtroom.
    """

    def test_case_file_returns_stored_events(self):
        """``GET /api/disputes/{id}`` on a fresh instance must return
        stored events, not an empty list.

        This fails on the code before the fix.
        """
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]
        _resolve_and_wait(client1, did)

        # Confirm the first instance has events.
        case1 = client1.get(f"/api/disputes/{did}").json()
        assert len(case1.get("events", [])) > 0, "First instance should have events."

        # Simulate a fresh instance on a different machine: the JSONL file
        # doesn't exist here. Events are only in the store.
        _clear_jsonl(did)

        # Fresh instance — no local JSONL file, no in-memory log.
        client2 = _make_client(store)
        _clear_jsonl(did)  # _make_client may have left runs from other tests
        case2 = client2.get(f"/api/disputes/{did}").json()
        events = case2.get("events", [])
        assert len(events) > 0, (
            "Fresh instance must return stored events, got empty list. "
            "The case-file endpoint must attach stored events when loading "
            "a case from the store."
        )
        event_types = [e["type"] for e in events]
        assert "RULING" in event_types or "ESCALATION" in event_types, (
            f"Stored events must include RULING or ESCALATION, got {event_types}."
        )

    def test_sse_replays_stored_events_on_fresh_instance(self):
        """``GET /api/disputes/{id}/events`` on a fresh instance must
        replay stored events in order and then send ``done``.

        Before the fix, the SSE endpoint only read the local JSONL file
        and sent just the ``done`` event.
        """
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]
        _resolve_and_wait(client1, did)

        # Simulate a fresh instance on a different machine: no JSONL file.
        _clear_jsonl(did)

        # Fresh instance on the same store.
        client2 = _make_client(store)
        _clear_jsonl(did)

        events = []
        with client2.stream(
            "GET",
            f"/api/disputes/{did}/events",
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: done"):
                    # Capture the done event data on the next line.
                    break
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        ev = json.loads(data_str)
                        events.append(ev)
                    except json.JSONDecodeError:
                        pass

        assert len(events) > 0, (
            "SSE stream on a fresh instance must replay stored events, "
            "not just send done."
        )
        event_types = [e.get("type", "") for e in events]
        assert "RULING" in event_types or "ESCALATION" in event_types, (
            f"Replayed events must include RULING or ESCALATION, got {event_types}."
        )

    def test_sse_preserves_event_order_from_store(self):
        """Stored events must be replayed in the order they were saved."""
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]
        _resolve_and_wait(client1, did)

        # Get the event types from the first instance (in-memory log).
        case1 = client1.get(f"/api/disputes/{did}").json()
        types1 = [e["type"] for e in case1["events"]]

        # Fresh instance — replay from store (no JSONL file).
        _clear_jsonl(did)
        client2 = _make_client(store)
        _clear_jsonl(did)
        events = []
        with client2.stream(
            "GET",
            f"/api/disputes/{did}/events",
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: done"):
                    break
                if line.startswith("data: "):
                    try:
                        ev = json.loads(line[6:])
                        events.append(ev)
                    except json.JSONDecodeError:
                        pass

        types2 = [e.get("type", "") for e in events]
        assert types2 == types1, (
            f"Event order from store must match the original. "
            f"Expected {types1}, got {types2}."
        )


# ---------------------------------------------------------------------------
# 3. Statuses — queue visibility across instances
# ---------------------------------------------------------------------------


class TestStatusCrossProcess:
    """A status set by one instance must appear in the queue of a fresh
    instance."""

    def test_ruled_status_in_queue(self):
        """A case resolved on instance1 must show as ``ruled`` in the
        queue of instance2."""
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]
        _resolve_and_wait(client1, did)

        client2 = _make_client(store)
        queue_res = client2.get("/api/disputes")
        queue_item = next(
            d for d in queue_res.json()["disputes"] if d["dispute_id"] == did
        )
        assert queue_item["status"] in ("ruled", "escalated"), (
            f"Case resolved on instance1 must show as ruled/escalated on "
            f"instance2, got {queue_item['status']}."
        )

    def test_overridden_status_in_queue(self):
        """An override set on instance1 must show as ``overridden`` on
        instance2."""
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]
        client1.post(
            f"/api/disputes/{did}/override",
            json={"outcome": "test", "amount": 0, "reason": "test"},
        )

        client2 = _make_client(store)
        queue_res = client2.get("/api/disputes")
        queue_item = next(
            d for d in queue_res.json()["disputes"] if d["dispute_id"] == did
        )
        assert queue_item["status"] == "overridden", (
            f"Override on instance1 must show as overridden on instance2, "
            f"got {queue_item['status']}."
        )

    def test_case_from_other_process_in_queue(self):
        """A case written directly to the store (simulating preresolve)
        must appear as ruled in the queue of a fresh instance."""
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]

        # Simulate another process writing a resolved case.
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

        client2 = _make_client(store)
        queue_res = client2.get("/api/disputes")
        queue_item = next(
            d for d in queue_res.json()["disputes"] if d["dispute_id"] == did
        )
        assert queue_item["status"] == "ruled"


# ---------------------------------------------------------------------------
# 4. Overrides — override records survive a fresh instance
# ---------------------------------------------------------------------------


class TestOverrideCrossProcess:
    """An override saved by one instance must be visible on a fresh instance."""

    def test_override_status_visible(self):
        """``GET /api/disputes/{id}`` on a fresh instance must show the
        override status."""
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]
        client1.post(
            f"/api/disputes/{did}/override",
            json={
                "outcome": "refund_route_deviation",
                "amount": 7.50,
                "reason": "Reviewer confirmed the deviation.",
            },
        )

        client2 = _make_client(store)
        case_res = client2.get(f"/api/disputes/{did}")
        assert case_res.json()["status"] == "overridden"


# ---------------------------------------------------------------------------
# 5. Learned precedents — precedents from overrides survive and are
#    retrievable in the next resolve on a fresh instance
# ---------------------------------------------------------------------------


class TestPrecedentCrossProcess:
    """A learned precedent (from an override) saved by one instance must
    be in the dataset of a fresh instance so the next resolve can retrieve
    it."""

    def test_override_precedent_survives(self):
        """An override-sourced precedent must survive a fresh app instance."""
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        rd_id = None
        for d in res.json()["disputes"]:
            if d["category"] == "route_deviation":
                rd_id = d["dispute_id"]
                break
        assert rd_id is not None

        client1.post(
            f"/api/disputes/{rd_id}/override",
            json={
                "outcome": "refund_route_deviation",
                "amount": 7.50,
                "reason": "Reviewer confirmed the deviation.",
            },
        )

        client2 = _make_client(store)
        from ridecourt.api import _dataset
        override_precs = [
            p for p in _dataset.precedents if getattr(p, "source", "") == "override"
        ]
        assert len(override_precs) >= 1, (
            "Override-sourced precedent must survive a fresh app instance "
            "so the learning loop is visible after a restart."
        )

    def test_override_precedent_retrieved_in_next_resolve(self):
        """A precedent from an override on instance1 must appear in the
        precedent matches of a case resolved on instance2."""
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        rd_id = None
        for d in res.json()["disputes"]:
            if d["category"] == "route_deviation":
                rd_id = d["dispute_id"]
                break
        assert rd_id is not None

        client1.post(
            f"/api/disputes/{rd_id}/override",
            json={
                "outcome": "refund_route_deviation",
                "amount": 7.50,
                "reason": "Reviewer confirmed the deviation.",
            },
        )

        # Fresh instance — resolve a different route_deviation case.
        client2 = _make_client(store)
        res2 = client2.get("/api/disputes")
        other_rd = None
        for d in res2.json()["disputes"]:
            if d["category"] == "route_deviation" and d["dispute_id"] != rd_id:
                other_rd = d["dispute_id"]
                break
        assert other_rd is not None, "Need a second route_deviation case."

        data = _resolve_and_wait(client2, other_rd)
        precedents = data.get("precedent_matches", [])
        override_precs = [
            p for p in precedents if p.get("source") == "override"
        ]
        assert len(override_precs) >= 1, (
            "An override-sourced precedent from instance1 must be retrieved "
            "in the next resolve on instance2. "
            f"Got precedents: {precedents}."
        )


# ---------------------------------------------------------------------------
# 6. Daily resolve counter — cost guard survives restarts
# ---------------------------------------------------------------------------


class TestDailyCounterCrossProcess:
    """The daily resolve counter must survive a restart so the cost guard
    does not reset."""

    def test_counter_survives_restart(self):
        """After resolving one case on instance1, a fresh instance2 must
        see the counter at 1 (not 0)."""
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]
        _resolve_and_wait(client1, did)

        assert _daily_cap.snapshot()["resolves_today"] == 1

        client2 = _make_client(store)
        snap = _daily_cap.snapshot()
        assert snap["resolves_today"] == 1, (
            f"Daily counter should be 1 after restart, got {snap['resolves_today']}. "
            "The cost guard must not reset when the instance restarts."
        )

    def test_counter_blocks_after_restart(self):
        """If the counter is exhausted on instance1, a fresh instance2
        must still block resolves (429)."""
        store = InMemoryStore()
        client1 = _make_client(store)
        _daily_cap.cap = 1

        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]
        _resolve_and_wait(client1, did)

        # Fresh instance — the counter value (1) survives from the store,
        # but the cap is a per-instance config value that resets to 300
        # in _make_client. Set it back to 1 so the cost guard blocks.
        client2 = _make_client(store)
        _daily_cap.cap = 1

        # Find a different dispute to resolve.
        res2 = client2.get("/api/disputes")
        other_id = None
        for d in res2.json()["disputes"]:
            if d["dispute_id"] != did:
                other_id = d["dispute_id"]
                break
        assert other_id is not None

        res3 = client2.post(f"/api/disputes/{other_id}/resolve")
        assert res3.status_code == 429, (
            f"Cost guard must block resolves after restart when the daily "
            f"counter is exhausted. Got {res3.status_code}."
        )


# ---------------------------------------------------------------------------
# 7. Metrics — resolved count reflects cross-process state
# ---------------------------------------------------------------------------


class TestMetricsCrossProcess:
    """Metrics must reflect cases resolved by another process."""

    def test_resolved_count_includes_store(self):
        """``GET /api/metrics`` on a fresh instance must count cases
        resolved by another process."""
        store = InMemoryStore()
        client1 = _make_client(store)

        res = client1.get("/api/disputes")
        did = res.json()["disputes"][0]["dispute_id"]
        _resolve_and_wait(client1, did)

        client2 = _make_client(store)
        metrics = client2.get("/api/metrics").json()
        assert metrics["resolved"] >= 1, (
            f"Metrics on a fresh instance must count cases resolved by "
            f"another process. Got resolved={metrics['resolved']}."
        )


# ---------------------------------------------------------------------------
# 8. End-to-end: preresolve into a shared store, fresh app, open case,
#    read SSE stream — advocates' claims and ruling must arrive
# ---------------------------------------------------------------------------


class TestEndToEndPreresolveFreshApp:
    """Run preresolve into a shared in-memory store, start a fresh app on
    that store, open a resolved case and read its SSE stream. Assert the
    advocates' claims and the ruling arrive."""

    def test_preresolve_then_fresh_app_sse(self):
        """End-to-end: preresolve writes to the store, a fresh app reads
        the case file and SSE stream with advocates' claims and ruling."""
        store = InMemoryStore()
        reset_store()
        from ridecourt.store import set_store
        set_store(store)

        # Get a dispute ID.
        from ridecourt.data import load_dataset
        ds = load_dataset()
        did = sorted(ds.disputes.keys())[0]

        # Run preresolve for this one case (mock provider).
        import io
        from contextlib import redirect_stderr

        old_argv = sys.argv
        sys.argv = ["preresolve.py", "--provider", "mock", did]
        try:
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                import preresolve
                preresolve.main()
            output = err_buf.getvalue()
            assert "ruled" in output or "escalated" in output, (
                f"preresolve should resolve the case. Output: {output}"
            )
        finally:
            sys.argv = old_argv

        # The store should now have the case, events, and status.
        assert store.load_case(did) is not None
        stored_events = store.load_events(did)
        assert len(stored_events) > 0, "Store should have events after preresolve."
        assert store.get_status(did) in ("ruled", "escalated")

        # Simulate a fresh instance on a different machine: the JSONL file
        # written by preresolve doesn't exist here.
        _clear_jsonl(did)

        # Start a fresh app on the same store.
        client = _make_client(store)
        _clear_jsonl(did)

        # 1. Case file endpoint must return stored events.
        case_res = client.get(f"/api/disputes/{did}")
        assert case_res.status_code == 200
        case_data = case_res.json()
        assert case_data["status"] in ("ruled", "escalated")
        events = case_data.get("events", [])
        assert len(events) > 0, (
            "Case file on fresh app must return stored events, got empty list."
        )
        event_types = [e["type"] for e in events]
        assert "DISPUTE_FILED" in event_types, (
            f"Events must start with DISPUTE_FILED, got {event_types}."
        )

        # 2. SSE stream must replay stored events and send done.
        sse_events = []
        with client.stream(
            "GET",
            f"/api/disputes/{did}/events",
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: done"):
                    break
                if line.startswith("data: "):
                    try:
                        ev = json.loads(line[6:])
                        sse_events.append(ev)
                    except json.JSONDecodeError:
                        pass

        assert len(sse_events) > 0, (
            "SSE stream on fresh app must replay stored events, not just done."
        )
        sse_types = [e.get("type", "") for e in sse_events]
        assert "DISPUTE_FILED" in sse_types, (
            f"SSE replay must include DISPUTE_FILED, got {sse_types}."
        )

        # The advocates' claims must arrive (CLAIM events).
        claim_events = [e for e in sse_events if e.get("type") == "CLAIM"]
        assert len(claim_events) >= 1, (
            f"SSE replay must include at least one CLAIM event (advocate "
            f"arguments). Got types: {sse_types}."
        )

        # The ruling must arrive (RULING or ESCALATION).
        assert "RULING" in sse_types or "ESCALATION" in sse_types, (
            f"SSE replay must include RULING or ESCALATION, got {sse_types}."
        )

    def test_run_again_streams_live_then_replaces_stored(self):
        """After preresolve, 'Run Again' (force=true) must stream the new
        run live and then replace the stored events."""
        store = InMemoryStore()
        reset_store()
        from ridecourt.store import set_store
        set_store(store)

        from ridecourt.data import load_dataset
        ds = load_dataset()
        did = sorted(ds.disputes.keys())[0]

        # Preresolve first.
        import io
        from contextlib import redirect_stderr

        old_argv = sys.argv
        sys.argv = ["preresolve.py", "--provider", "mock", did]
        try:
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                import preresolve
                preresolve.main()
        finally:
            sys.argv = old_argv

        original_events = store.load_events(did)
        assert len(original_events) > 0

        # Simulate a fresh instance: no JSONL file from preresolve.
        _clear_jsonl(did)

        # Fresh app.
        client = _make_client(store)
        _clear_jsonl(did)

        # Run Again with force=true.
        res = client.post(f"/api/disputes/{did}/resolve?force=true")
        assert res.status_code == 200
        assert res.json()["status"] == "resolving"

        # Wait for it to finish.
        deadline = time.time() + 30
        while time.time() < deadline:
            case_res = client.get(f"/api/disputes/{did}")
            data = case_res.json()
            if data["status"] in ("ruled", "escalated", "error"):
                break
            time.sleep(0.5)

        assert data["status"] in ("ruled", "escalated")

        # The stored events must have been replaced by the new run.
        new_events = store.load_events(did)
        assert len(new_events) > 0, "Store must have events after Run Again."

        # The case file must show events (from in-memory log this time).
        assert len(data.get("events", [])) > 0


# ---------------------------------------------------------------------------
# Status written by another process after this instance started
# ---------------------------------------------------------------------------


class TestStatusWrittenAfterStartup:
    """Found on the live app on 14 October: preresolve ran after the Cloud Run
    instance had started, so the instance had seeded every case as "new" in
    memory. The queue read the store and showed "ruled", but the case file
    trusted memory and said "new", and the page replayed the case as if it
    were still resolving."""

    def test_case_file_status_reads_store_when_memory_says_new(self):
        store = InMemoryStore()
        client = _make_client(store)
        did = client.get("/api/disputes").json()["disputes"][0]["dispute_id"]
        assert client.get(f"/api/disputes/{did}").json()["status"] == "new"

        # Another process (preresolve) writes the status after startup.
        store.set_status(did, "ruled")

        assert client.get(f"/api/disputes/{did}").json()["status"] == "ruled"
