"""Typed event log — every inter-agent message on a case.

Per SPEC section 3, each event has::

    {id, case_id, ts, from_agent, to_agent, type, content,
     cites: [E#, P-#], tokens, latency_ms}

Event types: ``DISPUTE_FILED, EVIDENCE_REQUEST, EVIDENCE, CLAIM,
REBUTTAL, RISK_SIGNAL, PRECEDENT, RULING, ESCALATION,
HUMAN_OVERRIDE``.

The log lives in memory (for the current run) and is appended to a
JSONL file under ``runs/`` so a run can be replayed.

An async subscriber API (``subscribe``) yields events as they are
appended, for SSE streaming by the API layer.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

EventType = Literal[
    "DISPUTE_FILED",
    "EVIDENCE_REQUEST",
    "EVIDENCE",
    "CLAIM",
    "REBUTTAL",
    "RISK_SIGNAL",
    "PRECEDENT",
    "RULING",
    "ESCALATION",
    "HUMAN_OVERRIDE",
    "GUARD_CHECK",
    "POLICY_EVAL",
]


class Event(BaseModel):
    """A single typed message on the case event log."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    case_id: str = ""
    ts: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    from_agent: str = ""
    to_agent: str = ""
    type: EventType
    content: Any = None
    cites: list[str] = Field(default_factory=list)
    tokens: int = 0
    latency_ms: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Event log — in-memory + append-only JSONL
# ---------------------------------------------------------------------------

RUNS_DIR = Path(os.environ.get("RUNS_DIR", "runs"))


def _ensure_runs_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR


class EventLog:
    """Per-case typed event log with in-memory list, JSONL persistence,
    and an async subscriber API for streaming.

    Usage::

        log = EventLog(case_id="disp_rd_000")
        log.append(Event(type="DISPUTE_FILED", ...))
        async for ev in log.subscribe():
            ...
    """

    def __init__(self, case_id: str, *, runs_dir: Path | str | None = None):
        self.case_id = case_id
        self._events: list[Event] = []
        self._subscribers: list[asyncio.Queue[Event]] = []
        self._closed = False
        self._runs_dir = Path(runs_dir) if runs_dir else _ensure_runs_dir()
        self._jsonl_path = self._runs_dir / f"{case_id}.jsonl"
        # Truncate the file for a fresh run (append-only within a run).
        self._jsonl_path.write_text("")

    # -- core append -------------------------------------------------------

    def append(self, event: Event) -> Event:
        """Add *event* to the log, persist to JSONL, and notify subscribers."""
        if event.case_id == "":
            event.case_id = self.case_id
        self._events.append(event)
        self._write_jsonl(event)
        self._notify(event)
        return event

    def _write_jsonl(self, event: Event) -> None:
        line = json.dumps(event.model_dump(), default=str)
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _notify(self, event: Event) -> None:
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass  # subscriber too slow — drop

    # -- read access -------------------------------------------------------

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def types(self) -> list[str]:
        return [e.type for e in self._events]

    def by_type(self, etype: str) -> list[Event]:
        return [e for e in self._events if e.type == etype]

    def total_tokens(self) -> int:
        return sum(e.tokens for e in self._events)

    def total_latency_ms(self) -> int:
        return sum(e.latency_ms for e in self._events)

    # -- async subscriber API ---------------------------------------------

    def subscribe(self, *, maxsize: int = 256) -> asyncio.Queue[Event]:
        """Return a queue that receives events as they are appended.

        Call ``close()`` to signal end-of-stream (sentinel ``None``).
        """
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=maxsize)
        self._subscribers.append(q)
        # Replay existing events into the new subscriber.
        for ev in self._events:
            q.put_nowait(ev)
        return q

    async def stream(self) -> AsyncIterator[Event]:
        """Async generator yielding events, including replays, until close."""
        q = self.subscribe()
        while True:
            ev = await q.get()
            if ev is None:
                break
            yield ev

    def close(self) -> None:
        """Signal end-of-stream to all subscribers."""
        self._closed = True
        for q in self._subscribers:
            try:
                q.put_nowait(None)  # type: ignore[arg-type]
            except asyncio.QueueFull:
                pass
        self._subscribers.clear()

    # -- replay ------------------------------------------------------------

    @classmethod
    def replay(cls, jsonl_path: Path | str, *, case_id: str = "") -> list[Event]:
        """Load events from a JSONL file for replay/debugging."""
        path = Path(jsonl_path)
        events: list[Event] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                events.append(Event(**data))
        return events

    # -- convenience -------------------------------------------------------

    def emit(
        self,
        type: EventType,
        *,
        from_agent: str = "",
        to_agent: str = "",
        content: Any = None,
        cites: list[str] | None = None,
        tokens: int = 0,
        latency_ms: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> Event:
        """Shortcut: build and append an Event in one call."""
        return self.append(
            Event(
                case_id=self.case_id,
                type=type,
                from_agent=from_agent,
                to_agent=to_agent,
                content=content,
                cites=cites or [],
                tokens=tokens,
                latency_ms=latency_ms,
                meta=meta or {},
            )
        )


# ---------------------------------------------------------------------------
# Convenience: standalone helpers
# ---------------------------------------------------------------------------


def now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_ms(start: float) -> int:
    """Latency in ms since *start* (a ``time.monotonic()`` value)."""
    return int((time.monotonic() - start) * 1000)
