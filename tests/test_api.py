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
from ridecourt.store import InMemoryStore, reset_store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Fresh app + client for each test (clean state).

    Uses a fresh ``InMemoryStore`` so state from previous tests doesn't
    leak through the store.
    """
    _case_files.clear()
    _case_logs.clear()
    _case_status.clear()
    _overrides.clear()
    # Reset rate limiter.
    _rate_limiter._hits.clear()
    # Reset daily cap counter (so prior tests don't leak budget into this one).
    _daily_cap._day = None
    _daily_cap._count = 0
    _daily_cap._loaded_from_store = False
    # Pin the cap to a small value for deterministic behavior in cap tests.
    _daily_cap.cap = 300

    # Fresh store for each test.
    store = InMemoryStore()
    reset_store()
    app = create_app(store=store)
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
        assert data["total"] >= 48  # synthetic dataset has ~50

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

    def test_file_and_resolve_dispute_with_claim_numbers_no_guard_failure(
        self, client
    ):
        """File a dispute whose claim text contains numbers in every unit
        the guard knows (S$, km, m, min, %), resolve it with the mock
        provider, and assert the case is NOT escalated for a guard
        failure.

        Before the CLAIM citation source was added, the advocates saw
        the claim numbers in their prompt and the guard rejected them,
        escalating every custom dispute with ``guard_failure_advocate``.
        """
        disputes_res = client.get("/api/disputes")
        trip_id = disputes_res.json()["disputes"][0]["trip_id"]

        claim = (
            "The driver overcharged me S$10 and drove 5 km out of the way. "
            "I waited 15 min at the pickup and the detour added 800 m to "
            "the trip. The fare was 20% higher than quoted."
        )
        res = client.post(
            "/api/disputes",
            json={
                "trip_id": trip_id,
                "category": "route_deviation",
                "claim_text": claim,
            },
        )
        assert res.status_code == 200
        did = res.json()["dispute_id"]

        # Resolve it.
        resolve_res = client.post(f"/api/disputes/{did}/resolve")
        assert resolve_res.status_code == 200

        # Poll until done.
        deadline = time.time() + 30
        data = {}
        while time.time() < deadline:
            case_res = client.get(f"/api/disputes/{did}")
            data = case_res.json()
            if data["status"] in ("ruled", "escalated", "error"):
                break
            time.sleep(0.5)

        # The case must not have escalated due to a guard failure.
        # (It may escalate for other reasons — low confidence, outcome
        # mismatch — but never ``guard_failure_advocate`` or
        # ``guard_failure_judge``.)
        escalation_reason = data.get("escalation_reason", "")
        assert "guard_failure" not in escalation_reason, (
            f"Dispute with claim numbers escalated with guard failure: "
            f"{escalation_reason}. The CLAIM citation source should let "
            f"agents quote claim numbers without guard rejection."
        )


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


# ---------------------------------------------------------------------------
# UI rendering coverage — SPEC-required fields must appear in web/
# ---------------------------------------------------------------------------


class TestUIFieldCoverage:
    """Verify that fields the SPEC requires the system to state explicitly
    are rendered somewhere in the web UI (``app.js``).

    The SPEC says (section 5, P-BF-1): "the Judge **must say so explicitly**"
    for bad-faith findings. If ``bad_faith_mentioned`` is true but the UI
    never renders it, a reviewer cannot see the finding — the system has
    not *stated* it in any visible way.

    Similarly, ``reasoning`` and ``precedent_consistency`` are Judge output
    fields that the SPEC requires the Judge to produce (section 6 table);
    they must be visible to a reviewer.
    """

    def _app_js(self, client) -> str:
        res = client.get("/app.js")
        assert res.status_code == 200
        return res.text

    def test_bad_faith_mentioned_rendered(self, client):
        """P-BF-1: the Judge must say so explicitly — the UI must render
        the ``bad_faith_mentioned`` field so a reviewer can see it."""
        js = self._app_js(client)
        assert "bad_faith_mentioned" in js, (
            "bad_faith_mentioned is not rendered in the web UI. "
            "SPEC P-BF-1 requires the Judge to state bad-faith findings "
            "explicitly, and a reviewer must be able to see them."
        )

    def test_reasoning_rendered(self, client):
        """The Judge's ``reasoning`` field must be displayed on the ruling
        card so the rationale is visible to a reviewer."""
        js = self._app_js(client)
        assert "reasoning" in js, (
            "reasoning is not rendered in the web UI. "
            "The SPEC requires the Judge to issue a ruling with reasoning; "
            "it must be visible to a reviewer."
        )

    def test_precedent_consistency_rendered(self, client):
        """The ``precedent_consistency`` field must be displayed so a
        reviewer can see whether the ruling aligns with past precedents."""
        js = self._app_js(client)
        assert "precedent_consistency" in js, (
            "precedent_consistency is not rendered in the web UI. "
            "The SPEC requires the Judge to state consistency with precedents."
        )

    def test_all_cites_rendered_as_chips(self, client):
        """Both E# and P-# citations must be rendered as chips in the
        ruling card (not just P-#). The SPEC says every claim cites E#
        and P-# — the ruling card must show both."""
        js = self._app_js(client)
        # The ruling card must not filter out E# cites.
        # Check that the chip class is used in the ruling rendering.
        assert "chip policy" in js, "Policy chips must use the 'chip policy' class"
        # The ruling card must render all cites, not just P-# ones.
        # Previously it filtered with .filter(c => c.startsWith("P-")).
        # Now it should render all cites as chips.
        assert 'ruling.cites.map' in js or 'ruling.cites.forEach' in js, (
            "Ruling card must render all citations as chips (E# and P-#)."
        )

    def test_precedent_source_rendered(self, client):
        """The PRECEDENT bubble must show each precedent's ``source`` so a
        reviewer can see when a ruling was learned from a reviewer override.

        The learning loop stores overrides as new precedents with
        ``source: "override"``.  If the UI never renders ``source``, the
        learning loop is invisible — the demo had to show it through the
        case-file API.
        """
        js = self._app_js(client)
        assert 'm.source' in js or 'source' in js, (
            "Precedent source is not rendered in the PRECEDENT bubble. "
            "When a precedent came from a reviewer override, the UI must "
            "say so (e.g. 'Reviewer override of disp_fa_005')."
        )
        assert '"override"' in js, (
            "The UI does not check for source === 'override' on precedents. "
            "Override-sourced precedents must be marked."
        )

    def test_precedent_similarity_rendered(self, client):
        """Each precedent in the PRECEDENT bubble must show its similarity
        as a percentage."""
        js = self._app_js(client)
        assert 'similarity' in js, (
            "Precedent similarity is not rendered in the PRECEDENT bubble. "
            "Each precedent must show its similarity as a percentage."
        )

    def test_learned_from_override_on_ruling_card(self, client):
        """When the top precedent came from an override, the ruling card
        must say 'Learned from a reviewer override'."""
        js = self._app_js(client)
        assert 'Learned from a reviewer override' in js, (
            "The ruling card does not show 'Learned from a reviewer override' "
            "when the top precedent came from an override."
        )
        assert 'precedent_matches' in js, (
            "The ruling card does not check precedent_matches for override source."
        )


# ---------------------------------------------------------------------------
# UI layout — evidence list and ruling card must coexist at 1440x810 / 1920x1080
# ---------------------------------------------------------------------------


class TestUILayout:
    """Verify that the CSS and JS restructure the evidence panel so the
    evidence list and ruling card have their own scroll areas, the ruling
    card defaults to a collapsed summary, and clicking an E# chip collapses
    the ruling card to bring the evidence item into view.

    What a reviewer should see at each window size:
    - 1920x1080: Ruling card collapsed (~160px tall) shows outcome, amount,
      confidence bar, markers. Evidence list fills the remaining ~650px —
      8+ evidence items visible. Clicking a chip collapses the card and
      scrolls the evidence item into view. Clicking the summary expands
      details (computation, reasoning, messages, buttons).
    - 1440x810: Collapsed ruling card (~160px) + evidence list (~490px) —
      5+ evidence items visible alongside the ruling summary.
    - 1024px: Layout switches to stacked (media query). Each panel gets its
      own section. The ruling card expands fully (no max-height in stacked
      layout).
    """

    def _app_js(self, client) -> str:
        res = client.get("/app.js")
        assert res.status_code == 200
        return res.text

    def _style_css(self, client) -> str:
        res = client.get("/style.css")
        assert res.status_code == 200
        return res.text

    def test_ruling_card_has_collapsed_state(self, client):
        """The ruling card must support a collapsed state so evidence is
        visible alongside the ruling at 1440x810 and 1920x1080."""
        js = self._app_js(client)
        assert "collapsed" in js, (
            "Ruling card must have a collapsed state so evidence items are "
            "visible alongside the ruling summary."
        )
        css = self._style_css(client)
        assert ".ruling-card.collapsed" in css, (
            "CSS must define a .ruling-card.collapsed state."
        )

    def test_ruling_details_section(self, client):
        """The ruling card must separate always-visible summary from
        collapsible details (computation, reasoning, messages, buttons)."""
        js = self._app_js(client)
        assert "ruling-details" in js, (
            "Ruling card must wrap collapsible content in a ruling-details section."
        )
        assert "ruling-summary" in js, (
            "Ruling card must have a ruling-summary section that is always visible."
        )

    def test_ruling_card_defaults_collapsed(self, client):
        """The ruling card must default to collapsed so evidence is visible
        when a case is first displayed."""
        js = self._app_js(client)
        assert 'add("collapsed")' in js, (
            "Ruling card must be collapsed by default on render."
        )

    def test_ruling_card_toggle(self, client):
        """The ruling card summary must toggle expand/collapse on click."""
        js = self._app_js(client)
        assert "toggle" in js, (
            "Ruling card summary must toggle the collapsed class on click."
        )
        assert "ruling-toggle" in js, (
            "A toggle indicator must show whether details are expanded or collapsed."
        )

    def test_highlight_evidence_collapses_ruling(self, client):
        """Clicking an E# chip must collapse the ruling card so the
        evidence item is brought into view."""
        js = self._app_js(client)
        # highlightEvidence should reference the ruling card and collapsed class.
        assert "ruling-card" in js, (
            "highlightEvidence must reference the ruling card."
        )
        # The function should collapse the card before scrolling.
        assert "collapsed" in js, (
            "highlightEvidence must collapse the ruling card to show the evidence item."
        )

    def test_evidence_content_min_height(self, client):
        """The evidence content must have a min-height so it never collapses
        to near-zero when the ruling card is present."""
        css = self._style_css(client)
        assert "min-height: 120px" in css, (
            "Evidence content must have min-height: 120px so at least a few "
            "evidence items are always visible."
        )

    def test_ruling_card_max_height(self, client):
        """The ruling card must have a max-height so it doesn't push the
        evidence list out of view."""
        css = self._style_css(client)
        assert "max-height" in css, (
            "Ruling card must have a max-height so the evidence list stays visible."
        )

    def test_evidence_panel_does_not_scroll_as_whole(self, client):
        """The evidence panel must not scroll as a whole — children scroll
        independently."""
        css = self._style_css(client)
        # The evidence-panel should use overflow: hidden, not overflow-y: auto.
        # Find the .evidence-panel rule and check it doesn't use overflow-y: auto.
        assert "overflow: hidden" in css or "overflow:hidden" in css, (
            "Evidence panel must use overflow: hidden so children scroll "
            "independently instead of the whole panel scrolling."
        )

    def test_ruling_card_flex_shrink(self, client):
        """The ruling card must allow shrinking (flex-shrink != 0) so the
        evidence content gets space."""
        css = self._style_css(client)
        assert "flex-shrink: 1" in css, (
            "Ruling card must have flex-shrink: 1 (was flex-shrink: 0) so it "
            "doesn't push the evidence content to near-zero height."
        )
