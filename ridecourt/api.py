"""FastAPI app — RideCourt API (SPEC section 7) and static web UI.

Endpoints:

* ``GET  /api/disputes``             — list with status and SLA priority
* ``POST /api/disputes``             — file a new dispute
* ``POST /api/disputes/{id}/resolve`` — run the pipeline in the background
* ``GET  /api/disputes/{id}/events``  — SSE event stream (replay → live → done)
* ``GET  /api/disputes/{id}``         — case file (evidence, arguments, ruling)
* ``POST /api/disputes/{id}/override`` — human override, stored as a precedent
* ``GET  /api/metrics``              — evaluation and operating metrics
* ``GET  /healthz``                  — health check

Static files from ``web/`` are served at ``/``.
A per-IP rate limit (20 resolves/hour) and a global daily resolve cap
(``DAILY_RESOLVE_CAP``, default 300, resets at UTC midnight) protect the
model budget. The cap is exposed in ``GET /api/metrics``.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .data import Dataset, Dispute, Precedent, load_dataset
from .events import Event, EventLog, RUNS_DIR
from .orchestrator import CaseFile, Orchestrator


# ---------------------------------------------------------------------------
# In-memory state (single process)
# ---------------------------------------------------------------------------

_WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# One global dataset + orchestrator (loaded once at startup).
_dataset: Dataset | None = None
_orchestrator: Orchestrator | None = None

# Active case files keyed by dispute_id (populated when resolve completes).
_case_files: dict[str, CaseFile] = {}

# Active event logs keyed by dispute_id (so SSE can replay before/during/after
# resolve).
_case_logs: dict[str, EventLog] = {}

# Resolution status keyed by dispute_id.
_case_status: dict[str, str] = {}  # "new" | "resolving" | "ruled" | "escalated"

# Human overrides stored as precedents (in-memory; persisted to JSON on disk).
_overrides: list[dict] = []


def _ds() -> Dataset:
    global _dataset
    if _dataset is None:
        _dataset = load_dataset()
    return _dataset


def _orc() -> Orchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator(_ds())
    return _orchestrator


def _init_status() -> None:
    """Seed case status for all known disputes as ``new``."""
    ds = _ds()
    for did in ds.disputes:
        if did not in _case_status:
            _case_status[did] = "new"


# ---------------------------------------------------------------------------
# Rate limiter — per-IP, sliding window
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple per-IP sliding-window rate limiter."""

    def __init__(self, max_requests: int = 20, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, ip: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        now = time.time()
        cutoff = now - self.window_seconds
        dq = self._hits[ip]

        # Drop expired entries.
        while dq and dq[0] < cutoff:
            dq.popleft()

        if len(dq) >= self.max_requests:
            return False

        dq.append(now)
        return True


_rate_limiter = RateLimiter(max_requests=20, window_seconds=3600)


def _client_ip(request: Request) -> str:
    """Extract the visitor's IP address from a request.

    Behind Cloud Run, ``request.client.host`` is the address of Google's
    front end, not the visitor.  Cloud Run appends the visitor's address as
    the **last** entry of ``X-Forwarded-For``; earlier entries can be
    forged and must be ignored.

    Falls back to ``request.client.host`` when the header is absent.
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[-1]
    if request.client:
        return request.client.host
    return "unknown"


# ---------------------------------------------------------------------------
# Global daily cap on resolves (cost guard for the public demo)
# ---------------------------------------------------------------------------


class DailyCap:
    """Global, persistent across-requests counter that resets at UTC midnight.

    Each successful ``record()`` increments the count. ``remaining()`` returns
    how many resolves are left today. When the count reaches the cap, the
    ``is_exhausted()`` check returns True and the API responds with 429.
    """

    def __init__(self, cap: int | None = None):
        # Read from env at construction time so tests can override.
        env_cap = os.getenv("DAILY_RESOLVE_CAP")
        if cap is None:
            cap = int(env_cap) if env_cap else 300
        self.cap = cap
        self._lock = threading.Lock()
        self._day: date | None = None
        self._count: int = 0

    def _maybe_rollover(self, today: date) -> None:
        if self._day != today:
            self._day = today
            self._count = 0

    def is_exhausted(self) -> bool:
        today = datetime.now(timezone.utc).date()
        with self._lock:
            self._maybe_rollover(today)
            return self._count >= self.cap

    def remaining(self) -> int:
        today = datetime.now(timezone.utc).date()
        with self._lock:
            self._maybe_rollover(today)
            return max(0, self.cap - self._count)

    def record(self) -> int:
        """Increment the counter; returns the new count for the day."""
        today = datetime.now(timezone.utc).date()
        with self._lock:
            self._maybe_rollover(today)
            self._count += 1
            return self._count

    def snapshot(self) -> dict[str, Any]:
        today = datetime.now(timezone.utc).date()
        with self._lock:
            self._maybe_rollover(today)
            return {
                "daily_resolve_cap": self.cap,
                "resolves_today": self._count,
                "remaining_resolves_today": max(0, self.cap - self._count),
                "reset_at_utc": datetime(
                    today.year, today.month, today.day, tzinfo=timezone.utc
                ).isoformat(),
            }


_daily_cap = DailyCap()


# ---------------------------------------------------------------------------
# SLA priority
# ---------------------------------------------------------------------------

_CATEGORY_PRIORITY = {
    "safety": 0,
    "route_deviation": 1,
    "no_show_driver": 2,
    "no_show_rider": 2,
    "fare_error": 3,
}

_STATUS_PRIORITY = {
    "new": 0,
    "resolving": 1,
    "ruled": 2,
    "escalated": 2,
}


def _sla_priority(dispute_id: str) -> int:
    """Return a priority tier for a single dispute (lower = higher priority).

    0 = safety, then category tier (1-3).  The actual queue ordering and
    rank assignment happens in ``list_disputes`` which sorts by a
    multi-key tuple and assigns sequential ranks.
    """
    ds = _ds()
    disp = ds.dispute(dispute_id)
    if disp is None:
        return 99
    return _CATEGORY_PRIORITY.get(disp.category, 9)


def _sla_sort_key(disp, amount: float):
    """Sort key: (safety_first, category_tier, -amount, filed_at).

    Lower is higher priority:
    - Safety cases (tier 0) come first.
    - Then category tier (route_deviation=1, no_show=2, fare_error=3).
    - Then higher amount at stake (negated so larger amounts sort first).
    - Then older filed_at (ascending = older first).
    """
    tier = _CATEGORY_PRIORITY.get(disp.category, 9)
    return (tier, -amount, disp.filed_at)


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------


class FileDisputeRequest(BaseModel):
    trip_id: str
    category: str
    claim_text: str
    filed_by: str = "rider"


class OverrideRequest(BaseModel):
    outcome: str
    amount: float = 0.0
    reason: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    app = FastAPI(title="RideCourt", version="0.1.0")

    # Initialise dataset + status at startup.
    _ds()
    _init_status()

    # Serve web/ as static files at /.
    if _WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_WEB_DIR)), name="static")

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}

    # ------------------------------------------------------------------
    # List disputes
    # ------------------------------------------------------------------
    @app.get("/api/disputes")
    async def list_disputes():
        ds = _ds()
        items = []
        for did, disp in ds.disputes.items():
            trip = ds.trip(disp.trip_id)
            status = _case_status.get(did, "new")
            amount = 0.0
            if did in _case_files:
                cf = _case_files[did]
                if cf.ruling:
                    amount = cf.ruling.amount
                elif cf.policy_result:
                    amount = cf.policy_result.amount
            elif trip:
                amount = trip.fare.total_charged

            items.append({
                "dispute_id": did,
                "trip_id": disp.trip_id,
                "category": disp.category,
                "filed_by": disp.filed_by,
                "claim_text": disp.claim_text,
                "filed_at": disp.filed_at,
                "status": status,
                "priority": 0,  # assigned after sorting
                "amount": round(amount, 2),
                "pickup": trip.pickup.model_dump() if trip else None,
                "dropoff": trip.dropoff.model_dump() if trip else None,
                "_sort_amount": amount,
            })

        # Sort by: safety first, then category tier, then amount
        # (higher first), then age (older first).
        items.sort(key=lambda x: _sla_sort_key(
            ds.dispute(x["dispute_id"]), x["_sort_amount"],
        ))

        # Assign sequential ranks as the priority badge value.
        for rank, item in enumerate(items, start=1):
            item["priority"] = rank
            del item["_sort_amount"]

        return {"disputes": items, "total": len(items)}

    # ------------------------------------------------------------------
    # File a new dispute
    # ------------------------------------------------------------------
    @app.post("/api/disputes")
    async def file_dispute(req: FileDisputeRequest):
        ds = _ds()
        trip = ds.trip(req.trip_id)
        if trip is None:
            raise HTTPException(status_code=404, detail=f"Trip {req.trip_id} not found")

        # Generate a new dispute ID.
        existing_ids = [d for d in ds.disputes if d.startswith("disp_usr_")]
        next_n = len(existing_ids)
        did = f"disp_usr_{next_n:03d}"

        disp = Dispute(
            dispute_id=did,
            trip_id=req.trip_id,
            filed_by=req.filed_by,  # type: ignore[arg-type]
            category=req.category,  # type: ignore[arg-type]
            claim_text=req.claim_text,
            filed_at=datetime.now(timezone.utc).isoformat(),
        )
        ds.disputes[did] = disp
        _case_status[did] = "new"

        return {
            "dispute_id": did,
            "status": "new",
            "message": "Dispute filed successfully.",
        }

    # ------------------------------------------------------------------
    # Resolve a dispute (background)
    # ------------------------------------------------------------------
    @app.post("/api/disputes/{dispute_id}/resolve")
    async def resolve_dispute(dispute_id: str, request: Request):
        ds = _ds()
        if dispute_id not in ds.disputes:
            raise HTTPException(status_code=404, detail="Dispute not found")

        # Global daily cap (cost guard) — checked before the per-IP limit so
        # the cap dominates when exhausted.
        cap_snapshot = _daily_cap.snapshot()
        if cap_snapshot["remaining_resolves_today"] <= 0:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Daily resolve limit reached "
                    f"({cap_snapshot['resolves_today']}/{cap_snapshot['daily_resolve_cap']}). "
                    f"The cap resets at {cap_snapshot['reset_at_utc']} (UTC midnight)."
                ),
            )

        # Rate limit per IP.
        client_ip = _client_ip(request)
        if not _rate_limiter.check(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded (20 resolves/hour). Try again later.",
            )

        # Don't restart if already resolving.
        if _case_status.get(dispute_id) == "resolving":
            return {
                "dispute_id": dispute_id,
                "status": "resolving",
                "message": "Already resolving. Subscribe to /events for updates.",
            }

        # If already done, return the cached result.
        if dispute_id in _case_files:
            return {
                "dispute_id": dispute_id,
                "status": _case_status.get(dispute_id, "done"),
                "message": "Already resolved. Fetch /api/disputes/{id} for the case file.",
            }

        # Count this resolve against the daily cap. We only record here so
        # that 404s, rate-limit rejects, "already resolving", and "already
        # resolved" responses don't consume budget.
        _daily_cap.record()

        # Remove any stale log from a previous run.
        _case_logs.pop(dispute_id, None)
        _case_status[dispute_id] = "resolving"

        # Run the orchestrator in a background thread. The orchestrator
        # creates its own EventLog internally (with the same case_id), which
        # writes to ``runs/{case_id}.jsonl`` in real-time. The SSE endpoint
        # polls that file for live events.
        def _run():
            orc = _orc()
            try:
                cf = orc.resolve(dispute_id)
                _case_files[dispute_id] = cf
                # Store the orchestrator's log for the case file endpoint.
                _case_logs[dispute_id] = cf.events
                if cf.escalated:
                    _case_status[dispute_id] = "escalated"
                else:
                    _case_status[dispute_id] = "ruled"
            except Exception as exc:
                _case_status[dispute_id] = "error"
                # Emit an error event to the JSONL file so SSE can pick it up.
                try:
                    err_log = EventLog(case_id=dispute_id)
                    err_log.emit(
                        "ESCALATION",
                        from_agent="orchestrator",
                        content={"reason": "system_error", "error": str(exc)},
                    )
                    err_log.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        return {
            "dispute_id": dispute_id,
            "status": "resolving",
            "message": "Pipeline started. Subscribe to /api/disputes/{id}/events for live updates.",
        }

    # ------------------------------------------------------------------
    # SSE event stream — polls the JSONL file written by the orchestrator's
    # EventLog in real-time. Replays past events, streams live ones, then
    # sends a final "done" event.
    # ------------------------------------------------------------------
    @app.get("/api/disputes/{dispute_id}/events")
    async def dispute_events(dispute_id: str, request: Request):
        ds = _ds()
        if dispute_id not in ds.disputes:
            raise HTTPException(status_code=404, detail="Dispute not found")

        jsonl_path = RUNS_DIR / f"{dispute_id}.jsonl"

        async def event_stream():
            sent_count = 0
            deadline = time.time() + 120  # 2-minute max stream

            while time.time() < deadline:
                # Check if the case is done.
                status = _case_status.get(dispute_id, "new")
                done = status in ("ruled", "escalated", "error", "overridden")

                # Read all lines from the JSONL file.
                lines_read = 0
                if jsonl_path.exists():
                    try:
                        with open(jsonl_path, encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                lines_read += 1
                                if lines_read > sent_count:
                                    try:
                                        ev = json.loads(line)
                                        ev_type = ev.get("type", "UNKNOWN")
                                        yield f"event: {ev_type}\ndata: {json.dumps(ev, default=str)}\n\n"
                                    except json.JSONDecodeError:
                                        pass
                                    sent_count = lines_read
                    except OSError:
                        pass

                if done:
                    # Send the final "done" event.
                    done_data = {
                        "type": "done",
                        "case_id": dispute_id,
                        "status": status,
                    }
                    cf = _case_files.get(dispute_id)
                    if cf:
                        done_data["escalated"] = cf.escalated
                        if cf.ruling:
                            done_data["ruling"] = cf.ruling.model_dump()
                        if cf.policy_result:
                            done_data["policy"] = cf.policy_result.model_dump()
                    yield f"event: done\ndata: {json.dumps(done_data, default=str)}\n\n"
                    break

                # Brief sleep to avoid busy-waiting, then yield control.
                await asyncio.sleep(0.15)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ------------------------------------------------------------------
    # Case file
    # ------------------------------------------------------------------
    @app.get("/api/disputes/{dispute_id}")
    async def get_case(dispute_id: str):
        ds = _ds()
        disp = ds.dispute(dispute_id)
        if disp is None:
            raise HTTPException(status_code=404, detail="Dispute not found")

        trip = ds.trip(disp.trip_id)
        chat = ds.chat(disp.trip_id)

        result: dict[str, Any] = {
            "dispute": disp.model_dump(),
            "trip": trip.model_dump() if trip else None,
            "chat": chat.model_dump() if chat else None,
            "status": _case_status.get(dispute_id, "new"),
        }

        cf = _case_files.get(dispute_id)
        if cf:
            result["evidence_pack"] = cf.evidence_pack.model_dump()
            result["rider_case"] = cf.rider_case.model_dump() if cf.rider_case else None
            result["driver_case"] = cf.driver_case.model_dump() if cf.driver_case else None
            result["rider_rebuttal"] = (
                cf.rider_rebuttal.model_dump() if cf.rider_rebuttal else None
            )
            result["driver_rebuttal"] = (
                cf.driver_rebuttal.model_dump() if cf.driver_rebuttal else None
            )
            result["policy_result"] = (
                cf.policy_result.model_dump() if cf.policy_result else None
            )
            result["ruling"] = cf.ruling.model_dump() if cf.ruling else None
            result["escalated"] = cf.escalated
            result["escalation_reason"] = cf.escalation_reason
            result["total_tokens"] = cf.total_tokens
            result["total_latency_ms"] = cf.total_latency_ms
            # Stretch: risk + precedents
            result["risk_output"] = (
                cf.risk_output.model_dump() if cf.risk_output else None
            )
            result["risk_assessment"] = (
                {
                    "risk": cf.risk_assessment.risk,
                    "claimant_id": cf.risk_assessment.claimant_id,
                    "signals": [
                        {"text": s.text, "cites": s.cites, "feature": s.feature}
                        for s in cf.risk_assessment.signals
                    ],
                }
                if cf.risk_assessment else None
            )
            result["precedent_matches"] = [
                {
                    "case_id": m.case_id,
                    "category": m.category,
                    "summary": m.summary,
                    "outcome": m.outcome,
                    "amount": m.amount,
                    "similarity": m.similarity,
                    "reason": m.reason,
                    "source": m.source,
                }
                for m in cf.precedent_matches
            ] if cf.precedent_matches else []
            result["escalation"] = (
                cf.escalation.to_dict() if cf.escalation else None
            )

        # Include events if available — from the in-memory log or the JSONL file.
        log = _case_logs.get(dispute_id)
        if log:
            result["events"] = [e.model_dump() for e in log.events]
        else:
            # Fallback: read events from the JSONL file (during/after resolution).
            jsonl_path = RUNS_DIR / f"{dispute_id}.jsonl"
            if jsonl_path.exists():
                events_list = []
                try:
                    with open(jsonl_path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                events_list.append(json.loads(line))
                            except json.JSONDecodeError:
                                pass
                except OSError:
                    pass
                result["events"] = events_list

        return result

    # ------------------------------------------------------------------
    # Human override (stored as a precedent)
    # ------------------------------------------------------------------
    @app.post("/api/disputes/{dispute_id}/override")
    async def override_dispute(dispute_id: str, req: OverrideRequest):
        ds = _ds()
        disp = ds.dispute(dispute_id)
        if disp is None:
            raise HTTPException(status_code=404, detail="Dispute not found")

        override_record = {
            "case_id": dispute_id,
            "category": disp.category,
            "summary": disp.claim_text,
            "outcome": req.outcome,
            "amount": req.amount,
            "reason": req.reason,
            "overridden_at": datetime.now(timezone.utc).isoformat(),
        }
        _overrides.append(override_record)
        _case_status[dispute_id] = "overridden"

        # Learning feedback loop: store the override as a new precedent
        # so it can be retrieved for similar future cases.
        new_precedent = Precedent(
            case_id=f"override-{dispute_id}",
            category=disp.category,  # type: ignore[arg-type]
            summary=disp.claim_text,
            outcome=req.outcome,
            amount=req.amount,
            source="override",
            reason=req.reason,
        )
        ds.precedents.append(new_precedent)

        # Also persist to the overrides list for the escalation queue.
        # Resolve the escalation if one exists.
        orc = _orc()
        orc.escalation_queue.resolve(
            dispute_id,
            reviewer="human_reviewer",
            resolution=req.reason,
        )

        # Emit a HUMAN_OVERRIDE event if the case log exists.
        log = _case_logs.get(dispute_id)
        if log:
            log.emit(
                "HUMAN_OVERRIDE",
                from_agent="human_reviewer",
                content={**override_record, "source": "override"},
            )

        return {
            "dispute_id": dispute_id,
            "status": "overridden",
            "precedent": override_record,
            "message": "Override stored as a precedent for future cases.",
        }

    # ------------------------------------------------------------------
    # Escalation queue
    # ------------------------------------------------------------------
    @app.get("/api/escalations")
    async def escalations():
        orc = _orc()
        q = orc.escalation_queue
        return {
            "escalations": [e.to_dict() for e in q.all()],
            "total": len(q),
            "open": len(q.open()),
        }

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @app.get("/api/metrics")
    async def metrics():
        ds = _ds()
        total = len(ds.disputes)
        resolved = len(_case_files)
        escalated = sum(1 for cf in _case_files.values() if cf.escalated)
        ruled = sum(1 for cf in _case_files.values() if not cf.escalated)
        overridden = len(_overrides)

        # Per-category breakdown.
        by_category: dict[str, dict[str, int]] = {}
        for did, disp in ds.disputes.items():
            cat = disp.category
            if cat not in by_category:
                by_category[cat] = {"total": 0, "resolved": 0, "escalated": 0}
            by_category[cat]["total"] += 1
            if did in _case_files:
                by_category[cat]["resolved"] += 1
                if _case_files[did].escalated:
                    by_category[cat]["escalated"] += 1

        # Token and latency stats.
        total_tokens = sum(cf.total_tokens for cf in _case_files.values())
        total_latency = sum(cf.total_latency_ms for cf in _case_files.values())
        median_latency = 0
        if _case_files:
            latencies = sorted(cf.total_latency_ms for cf in _case_files.values())
            n = len(latencies)
            median_latency = latencies[n // 2] if n % 2 == 1 else (
                (latencies[n // 2 - 1] + latencies[n // 2]) / 2
            )

        return {
            "total_disputes": total,
            "resolved": resolved,
            "ruled": ruled,
            "escalated": escalated,
            "overridden": overridden,
            "pending": total - resolved,
            "total_tokens": total_tokens,
            "total_latency_ms": total_latency,
            "median_latency_ms": median_latency,
            "by_category": by_category,
            **_daily_cap.snapshot(),
        }

    # ------------------------------------------------------------------
    # Static file serving for the web UI
    # ------------------------------------------------------------------
    @app.get("/")
    async def index():
        index_path = _WEB_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path), media_type="text/html")
        return HTML_FALLBACK

    @app.get("/app.js")
    async def app_js():
        js_path = _WEB_DIR / "app.js"
        if js_path.exists():
            return FileResponse(str(js_path), media_type="application/javascript")
        raise HTTPException(status_code=404, detail="app.js not found")

    @app.get("/style.css")
    async def style_css():
        css_path = _WEB_DIR / "style.css"
        if css_path.exists():
            return FileResponse(str(css_path), media_type="text/css")
        raise HTTPException(status_code=404, detail="style.css not found")

    return app


HTML_FALLBACK = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>RideCourt</title></head>
<body><h1>RideCourt</h1><p>Web UI not found. Create web/index.html.</p>
</body></html>"""


# Module-level app instance for uvicorn: ``uvicorn ridecourt.api:app``
app = create_app()
