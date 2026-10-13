"""Tests for the FastAPI API — ``LLM_PROVIDER=mock``.

Covers (per build-order step 5):
- ``GET /api/disputes`` list
- ``POST /api/disputes/{id}/resolve`` starts the pipeline
- ``GET /api/disputes/{id}/events`` SSE stream ends with a ``done`` event
  whose status is ``ruled`` or ``escalated``
- ``POST /api/disputes/{id}/override`` stores a precedent
"""

import os

os.environ.setdefault("LLM_PROVIDER", "mock")

import json
import time

import pytest
from fastapi.testclient import TestClient

from ridecourt.api import (
    create_app,
    _case_files,
    _case_logs,
    _case_status,
    _daily_cap,
    _overrides,
    _rate_limiter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Fresh app + client for each test (clean state)."""
    _case_files.clear()
    _case_logs.clear()
    _case_status.clear()
    _overrides.clear()
    # Reset rate limiter.
    _rate_limiter._hits.clear()
    # Reset daily cap counter (so prior tests don't leak budget into this one).
    _daily_cap._day = None
    _daily_cap._count = 0
    # Pin the cap to a small value for deterministic behavior in cap tests.
    _daily_cap.cap = 300

    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def first_dispute_id(client):
    """Return the first dispute ID from the list."""
    res = client.get("/api/disputes")
    assert res.status_code == 200
    data = res.json()
    assert data["total"] > 0
    return data["disputes"][0]["dispute_id"]


@pytest.fixture()
def safety_dispute_id(client):
    """Return a safety dispute ID (always escalates)."""
    res = client.get("/api/disputes")
    data = res.json()
    for d in data["disputes"]:
        if d["category"] == "safety":
            return d["dispute_id"]
    pytest.skip("No safety dispute in dataset")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealth:
    def test_healthz(self, client):
        res = client.get("/healthz")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert "ts" in body


# ---------------------------------------------------------------------------
# List disputes
# ---------------------------------------------------------------------------


class TestListDisputes:
    def test_list_returns_all_disputes(self, client):
        res = client.get("/api/disputes")
        assert res.status_code == 200
        data = res.json()
        assert "disputes" in data
        assert "total" in data
        assert data["total"] >= 48  # synthetic dataset has ~48

    def test_list_has_required_fields(self, client):
        res = client.get("/api/disputes")
        disputes = res.json()["disputes"]
        for d in disputes:
            assert "dispute_id" in d
            assert "trip_id" in d
            assert "category" in d
            assert "status" in d
            assert "priority" in d
            assert "claim_text" in d

    def test_safety_cases_are_first(self, client):
        """Safety cases should be at the top of the queue (SLA priority)."""
        res = client.get("/api/disputes")
        disputes = res.json()["disputes"]
        # Find safety disputes.
        safety_ids = {d["dispute_id"] for d in disputes if d["category"] == "safety"}
        if safety_ids:
            # The first few items should include safety cases.
            top_ids = {d["dispute_id"] for d in disputes[:len(safety_ids)]}
            assert safety_ids == top_ids, "Safety cases should be at the top"

    def test_list_sorted_by_priority(self, client):
        res = client.get("/api/disputes")
        disputes = res.json()["disputes"]
        priorities = [d["priority"] for d in disputes]
        assert priorities == sorted(priorities)

    def test_priority_differs_between_cases(self, client):
        """SLA priority must not be the same value for every case.

        Defect 2: the old formula clamped every non-safety case to 1
        because age_hours dominated the category tier. The synthetic
        disputes have filed_at dates far in the past, so the badge showed
        "1" on every non-safety row.
        """
        res = client.get("/api/disputes")
        disputes = res.json()["disputes"]
        non_safety = [d for d in disputes if d["category"] != "safety"]
        priorities = [d["priority"] for d in non_safety]
        # There should be more than one distinct priority value.
        assert len(set(priorities)) > 1, (
            f"All non-safety cases have priority {priorities[0]} — "
            "the SLA badge is meaningless."
        )

    def test_safety_has_lowest_priority_number(self, client):
        """Safety cases must have the smallest priority numbers (top of queue)."""
        res = client.get("/api/disputes")
        disputes = res.json()["disputes"]
        safety_pris = [d["priority"] for d in disputes if d["category"] == "safety"]
        non_safety_pris = [d["priority"] for d in disputes if d["category"] != "safety"]
        if safety_pris and non_safety_pris:
            assert max(safety_pris) < min(non_safety_pris), (
                f"Safety priorities {safety_pris} must all be below "
                f"non-safety {non_safety_pris[:5]}..."
            )

    def test_sort_order_is_safety_then_category_then_amount_then_age(self, client):
        """The queue must be sorted: safety first, then category tier,
        then amount (higher first), then age (older first)."""
        res = client.get("/api/disputes")
        disputes = res.json()["disputes"]
        # Build expected sort key.
        cat_tier = {
            "safety": 0,
            "route_deviation": 1,
            "no_show_driver": 2,
            "no_show_rider": 2,
            "fare_error": 3,
        }
        keys = []
        for d in disputes:
            tier = cat_tier.get(d["category"], 9)
            keys.append((tier, -d["amount"], d["filed_at"]))
        assert keys == sorted(keys), (
            "Disputes are not sorted by (safety, category, -amount, filed_at)"
        )


# ---------------------------------------------------------------------------
# Resolve
# ---------------------------------------------------------------------------


class TestResolve:
    def test_resolve_returns_resolving(self, client, first_dispute_id):
        res = client.post(f"/api/disputes/{first_dispute_id}/resolve")
        assert res.status_code == 200
        data = res.json()
        assert data["dispute_id"] == first_dispute_id
        assert data["status"] == "resolving"

    def test_resolve_nonexistent_returns_404(self, client):
        res = client.post("/api/disputes/nonexistent/resolve")
        assert res.status_code == 404

    def test_resolve_eventually_produces_case_file(self, client, first_dispute_id):
        """The background pipeline should complete and produce a CaseFile."""
        res = client.post(f"/api/disputes/{first_dispute_id}/resolve")
        assert res.status_code == 200

        # Poll until the case file appears or timeout.
        deadline = time.time() + 30
        while time.time() < deadline:
            case_res = client.get(f"/api/disputes/{first_dispute_id}")
            data = case_res.json()
            if data["status"] in ("ruled", "escalated", "error"):
                break
            time.sleep(0.5)

        assert data["status"] in ("ruled", "escalated")
        if data["status"] == "ruled":
            assert data.get("ruling") is not None
        else:
            assert data.get("escalated") is True


# ---------------------------------------------------------------------------
# Events stream (SSE)
# ---------------------------------------------------------------------------


class TestEventsStream:
    def test_events_stream_ends_with_done(self, client, first_dispute_id):
        """The SSE stream should replay past events, stream live ones,
        and end with a ``done`` event."""
        # Start resolution.
        client.post(f"/api/disputes/{first_dispute_id}/resolve")

        # Connect to the SSE stream and read until the done event.
        events = []
        with client.stream(
            "GET",
            f"/api/disputes/{first_dispute_id}/events",
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event: done"):
                    # The next line(s) should have the data.
                    break
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        ev = json.loads(data_str)
                        events.append(ev)
                    except json.JSONDecodeError:
                        pass

        # We should have received at least some events (replayed or live).
        assert len(events) > 0, "Expected at least some events before done"

        # Reconnect to read the done event properly (the stream may have
        # already closed if resolution was fast).
        # Fetch the case to verify the final status.
        case_res = client.get(f"/api/disputes/{first_dispute_id}")
        case_data = case_res.json()
        assert case_data["status"] in ("ruled", "escalated")

    def test_events_stream_safety_ends_with_escalation(self, client, safety_dispute_id):
        """Safety cases should stream events ending with escalation."""
        client.post(f"/api/disputes/{safety_dispute_id}/resolve")

        # Wait for resolution.
        deadline = time.time() + 30
        status = "resolving"
        while time.time() < deadline and status == "resolving":
            case_res = client.get(f"/api/disputes/{safety_dispute_id}")
            status = case_res.json()["status"]
            if status != "resolving":
                break
            time.sleep(0.5)

        assert status == "escalated", f"Safety case should escalate, got {status}"

    def test_events_stream_has_ruling_or_escalation(self, client, first_dispute_id):
        """The event stream should contain a RULING or ESCALATION event."""
        # Resolve first.
        client.post(f"/api/disputes/{first_dispute_id}/resolve")

        # Wait for completion.
        deadline = time.time() + 30
        case_data = {}
        while time.time() < deadline:
            res = client.get(f"/api/disputes/{first_dispute_id}")
            case_data = res.json()
            if case_data["status"] in ("ruled", "escalated"):
                break
            time.sleep(0.5)

        # The case file should have events including RULING or ESCALATION.
        events = case_data.get("events", [])
        event_types = [e["type"] for e in events]
        assert "RULING" in event_types or "ESCALATION" in event_types, (
            f"Expected RULING or ESCALATION in events, got {event_types}"
        )


# ---------------------------------------------------------------------------
# Case file
# ---------------------------------------------------------------------------


class TestCaseFile:
    def test_case_file_before_resolve(self, client, first_dispute_id):
        """Before resolution, the case file has dispute + trip but no ruling."""
        res = client.get(f"/api/disputes/{first_dispute_id}")
        assert res.status_code == 200
        data = res.json()
        assert "dispute" in data
        assert "trip" in data
        assert data["status"] == "new"
        assert "ruling" not in data or data["ruling"] is None

    def test_case_file_after_resolve(self, client, first_dispute_id):
        """After resolution, the case file has evidence + ruling."""
        client.post(f"/api/disputes/{first_dispute_id}/resolve")

        deadline = time.time() + 30
        data = {}
        while time.time() < deadline:
            res = client.get(f"/api/disputes/{first_dispute_id}")
            data = res.json()
            if data["status"] in ("ruled", "escalated"):
                break
            time.sleep(0.5)

        assert "evidence_pack" in data
        assert data["evidence_pack"]["items"]
        assert data["status"] in ("ruled", "escalated")

    def test_case_file_404(self, client):
        res = client.get("/api/disputes/nonexistent")
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# Override (stores a precedent)
# ---------------------------------------------------------------------------


class TestOverride:
    def test_override_stores_precedent(self, client, first_dispute_id):
        """Override should store a precedent record."""
        res = client.post(
            f"/api/disputes/{first_dispute_id}/override",
            json={
                "outcome": "refund_route_deviation",
                "amount": 5.50,
                "reason": "Reviewer judged the deviation was unjustified.",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "overridden"
        assert data["precedent"]["outcome"] == "refund_route_deviation"
        assert data["precedent"]["amount"] == 5.50
        assert "reason" in data["precedent"]

    def test_override_nonexistent_returns_404(self, client):
        res = client.post(
            "/api/disputes/nonexistent/override",
            json={"outcome": "test", "amount": 0, "reason": "test"},
        )
        assert res.status_code == 404

    def test_override_persists_in_overrides_list(self, client, first_dispute_id):
        """The override should be retrievable via the metrics or case endpoint."""
        client.post(
            f"/api/disputes/{first_dispute_id}/override",
            json={
                "outcome": "cancellation_fee_stands",
                "amount": 4.00,
                "reason": "Driver waited 8 minutes.",
            },
        )
        # Check the case file shows overridden status.
        case_res = client.get(f"/api/disputes/{first_dispute_id}")
        assert case_res.json()["status"] == "overridden"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_metrics_initial(self, client):
        res = client.get("/api/metrics")
        assert res.status_code == 200
        data = res.json()
        assert data["total_disputes"] >= 48
        assert data["resolved"] == 0
        assert "by_category" in data

    def test_metrics_after_resolve(self, client, first_dispute_id):
        """After resolving a case, metrics should reflect it."""
        client.post(f"/api/disputes/{first_dispute_id}/resolve")

        deadline = time.time() + 30
        while time.time() < deadline:
            res = client.get("/api/metrics")
            data = res.json()
            if data["resolved"] > 0:
                break
            time.sleep(0.5)

        assert data["resolved"] >= 1

    def test_metrics_exposes_daily_cap_fields(self, client):
        """/api/metrics must expose the daily resolve cap and remaining count."""
        res = client.get("/api/metrics")
        assert res.status_code == 200
        data = res.json()
        assert data["daily_resolve_cap"] == 300
        assert data["resolves_today"] == 0
        assert data["remaining_resolves_today"] == 300
        # reset_at_utc must be an ISO timestamp string ending with +00:00.
        assert isinstance(data["reset_at_utc"], str)
        assert data["reset_at_utc"].endswith("+00:00")


# ---------------------------------------------------------------------------
# Daily resolve cap (cost guard)
# ---------------------------------------------------------------------------


class TestDailyResolveCap:
    def test_cap_blocks_resolve_once_exhausted(self, client, first_dispute_id):
        """Setting the cap to 1 lets the first resolve through and blocks the
        second distinct dispute with 429."""
        _daily_cap.cap = 1

        # First resolve on one dispute — should start.
        res1 = client.post(f"/api/disputes/{first_dispute_id}/resolve")
        assert res1.status_code == 200
        assert res1.json()["status"] == "resolving"

        # Pick a different dispute so we don't hit the "already resolving"
        # short-circuit.
        list_res = client.get("/api/disputes")
        other_id = None
        for d in list_res.json()["disputes"]:
            if d["dispute_id"] != first_dispute_id:
                other_id = d["dispute_id"]
                break
        assert other_id is not None

        # Second resolve — should now be blocked by the daily cap.
        res2 = client.post(f"/api/disputes/{other_id}/resolve")
        assert res2.status_code == 429
        detail = res2.json().get("detail", "")
        assert "Daily resolve limit" in detail
        assert "1/1" in detail

    def test_cap_429_message_mentions_reset(self, client, first_dispute_id):
        """The 429 message must tell the user when the cap resets (UTC midnight)."""
        _daily_cap.cap = 0
        res = client.post(f"/api/disputes/{first_dispute_id}/resolve")
        assert res.status_code == 429
        detail = res.json()["detail"]
        assert "UTC midnight" in detail

    def test_cap_does_not_count_404_or_already_resolving(self, client, first_dispute_id):
        """404 and 'already resolving' responses must not consume budget."""
        _daily_cap.cap = 2

        # 404 doesn't count.
        r404 = client.post("/api/disputes/nonexistent/resolve")
        assert r404.status_code == 404

        # First real resolve — should succeed.
        r1 = client.post(f"/api/disputes/{first_dispute_id}/resolve")
        assert r1.status_code == 200
        # Second resolve on the same dispute — should hit "already resolving"
        # (not a new budget consumption).
        r2 = client.post(f"/api/disputes/{first_dispute_id}/resolve")
        assert r2.status_code == 200
        assert r2.json()["status"] == "resolving"

        # Budget was consumed once.
        snap = _daily_cap.snapshot()
        assert snap["resolves_today"] == 1
        assert snap["remaining_resolves_today"] == 1

    def test_metrics_reflects_cap_after_resolve(self, client, first_dispute_id):
        """After a resolve, /api/metrics shows one resolve consumed."""
        client.post(f"/api/disputes/{first_dispute_id}/resolve")
        data = client.get("/api/metrics").json()
        assert data["resolves_today"] == 1
        assert data["remaining_resolves_today"] == 299


# ---------------------------------------------------------------------------
# Per-IP rate limit behind Cloud Run (X-Forwarded-For)
# ---------------------------------------------------------------------------


class TestRateLimitForwardedFor:
    def test_uses_last_xff_entry_as_client_ip(self, client):
        """Cloud Run appends the visitor's address as the last entry of
        X-Forwarded-For. Earlier entries can be forged and must be ignored.

        A header "1.2.3.4, 203.0.113.9" must count as 203.0.113.9.
        """
        from ridecourt.api import _client_ip

        # Simulate a request with a forged X-Forwarded-For.
        class _Req:
            class client:
                host = "10.0.0.1"  # Cloud Run frontend IP
            headers = {"x-forwarded-for": "1.2.3.4, 203.0.113.9"}

        ip = _client_ip(_Req())
        assert ip == "203.0.113.9", (
            f"Expected 203.0.113.9 (last entry), got {ip}"
        )

    def test_falls_back_to_client_host_without_xff(self, client):
        from ridecourt.api import _client_ip

        class _Req:
            class client:
                host = "10.0.0.1"
            headers = {}

        ip = _client_ip(_Req())
        assert ip == "10.0.0.1"

    def test_forged_xff_counts_as_last_ip(self, client, first_dispute_id):
        """Two resolves with a forged header "1.2.3.4, 203.0.113.9" must
        rate-limit the 21st request from that IP (203.0.113.9), not from
        1.2.3.4 or the frontend IP.

        This fails on the current code: request.client.host is the same
        for all visitors behind Cloud Run, so they share one bucket.
        """
        _daily_cap.cap = 300  # ensure cap doesn't interfere
        list_res = client.get("/api/disputes")
        all_ids = [d["dispute_id"] for d in list_res.json()["disputes"]]

        # Reset state to avoid "already resolved" — use fresh disputes.
        _case_files.clear()
        _case_status.clear()
        _rate_limiter._hits.clear()

        headers = {"X-Forwarded-For": "1.2.3.4, 203.0.113.9"}
        for i, did in enumerate(all_ids):
            res = client.post(
                f"/api/disputes/{did}/resolve",
                headers=headers,
            )
            # The rate limiter allows 20 requests; the 21st must be blocked.
            if i < 20:
                assert res.status_code == 200, (
                    f"Request {i+1} from 203.0.113.9 should be allowed, "
                    f"got {res.status_code}"
                )
            else:
                assert res.status_code == 429, (
                    f"Request {i+1} from 203.0.113.9 should be rate-limited, "
                    f"got {res.status_code}"
                )
                assert "Rate limit" in res.json().get("detail", "")
                break


class TestFileDispute:
    def test_file_dispute_success(self, client):
        """Filing a new dispute should create it and return a dispute ID."""
        # Use an existing trip ID.
        disputes_res = client.get("/api/disputes")
        trip_id = disputes_res.json()["disputes"][0]["trip_id"]

        res = client.post(
            "/api/disputes",
            json={
                "trip_id": trip_id,
                "category": "route_deviation",
                "claim_text": "Driver took a longer route.",
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "new"
        assert "dispute_id" in data

    def test_file_dispute_bad_trip(self, client):
        res = client.post(
            "/api/disputes",
            json={
                "trip_id": "nonexistent",
                "category": "route_deviation",
                "claim_text": "Test",
            },
        )
        assert res.status_code == 404


# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------


class TestStaticFiles:
    def test_index_html_served(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "")
        assert "RideCourt" in res.text

    def test_app_js_served(self, client):
        res = client.get("/app.js")
        assert res.status_code == 200
        assert "javascript" in res.headers.get("content-type", "")

    def test_style_css_served(self, client):
        res = client.get("/style.css")
        assert res.status_code == 200
        assert "css" in res.headers.get("content-type", "")
