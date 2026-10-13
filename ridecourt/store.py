"""Persistence layer for RideCourt case state.

Two implementations:

- **InMemoryStore** (default): used by tests and local runs. A shared
  singleton — two ``create_app()`` calls on the same ``InMemoryStore``
  see the same data, simulating cross-process state for tests.
- **FirestoreStore** (``STATE_BACKEND=firestore``): the project's default
  Firestore database in native mode, ``asia-southeast1``. Collection
  names are prefixed with ``ridecourt_`` because another app uses the
  same database.

State persisted:

- Case files (serialized as dicts, without events)
- Case events (stored separately — see ``save_events`` / ``load_events``)
- Case statuses (new, resolving, ruled, escalated, overridden, error)
- Human overrides (each stored as a learned precedent)
- Daily resolve counter (the cost guard survives restarts)

Not persisted (stays in memory):

- Per-visitor hourly rate limit (ephemeral; Cloud Run instance scope)
- Live ``EventLog`` objects (reconstructed from stored events on load)

Firestore documents are limited to 1 MB. Mock case files are ~12 KB and
events ~7 KB; real Gemini responses would be larger but still well under
1 MB for a single case. Events are stored in their own document to keep
each doc small and to allow splitting into a subcollection later if a
case ever approaches the limit.
"""

from __future__ import annotations

import abc
import os
import threading
import time
from datetime import date, datetime, timezone
from typing import Any


class Store(abc.ABC):
    """Abstract persistence interface for case state."""

    # --- Case files ------------------------------------------------------

    @abc.abstractmethod
    def save_case(self, dispute_id: str, case_data: dict[str, Any]) -> None:
        """Save a full case file (serialized as a dict, without events)."""

    @abc.abstractmethod
    def load_case(self, dispute_id: str) -> dict[str, Any] | None:
        """Load a case file by dispute_id, or ``None`` if not stored."""

    @abc.abstractmethod
    def list_resolved_ids(self) -> list[str]:
        """Return dispute IDs with a terminal status (ruled, escalated,
        overridden, error). Used by ``preresolve`` to skip done cases.
        """

    # --- Case status -----------------------------------------------------

    @abc.abstractmethod
    def get_status(self, dispute_id: str) -> str | None:
        """Return the stored status for a dispute, or ``None``."""

    @abc.abstractmethod
    def set_status(self, dispute_id: str, status: str) -> None:
        """Set the status for a dispute."""

    @abc.abstractmethod
    def list_statuses(self) -> dict[str, str]:
        """Return all stored statuses as ``{dispute_id: status}``.

        Used for cross-process queue visibility: another process may have
        resolved a case, and this process needs to show it as ruled."""

    # --- Events ----------------------------------------------------------

    @abc.abstractmethod
    def save_events(self, dispute_id: str, events: list[dict[str, Any]]) -> None:
        """Save the event list for a case (stored separately from the
        case file to stay under the 1 MB Firestore document limit)."""

    @abc.abstractmethod
    def load_events(self, dispute_id: str) -> list[dict[str, Any]]:
        """Load events for a case. Returns ``[]`` if none stored."""

    # --- Overrides (learned precedents) ---------------------------------

    @abc.abstractmethod
    def save_override(self, override: dict[str, Any]) -> None:
        """Append an override record (stored as a learned precedent)."""

    @abc.abstractmethod
    def load_overrides(self) -> list[dict[str, Any]]:
        """Return all stored override records."""

    # --- Daily resolve counter ------------------------------------------

    @abc.abstractmethod
    def get_daily_count(self) -> tuple[date, int]:
        """Return ``(day, count)`` for the daily resolve counter.

        If the stored day differs from today, returns ``(today, 0)``.
        """

    @abc.abstractmethod
    def increment_daily_count(self) -> tuple[date, int]:
        """Atomically increment the daily counter and return
        ``(day, count)``. Resets to 1 if the day changed."""

    @abc.abstractmethod
    def set_daily_count(self, day: date, count: int) -> None:
        """Set the daily counter to ``(day, count)``.

        Used by the API layer to keep the in-memory ``DailyCap`` in sync
        with the store when a resolve is counted."""

    # --- Precedents (learned from overrides) -----------------------------

    @abc.abstractmethod
    def save_precedent(self, precedent: dict[str, Any]) -> None:
        """Save a learned precedent (from a human override)."""

    @abc.abstractmethod
    def load_precedents(self) -> list[dict[str, Any]]:
        """Return all stored learned precedents."""


# ---------------------------------------------------------------------------
# In-memory implementation (default — used by tests and local runs)
# ---------------------------------------------------------------------------


class InMemoryStore(Store):
    """Thread-safe in-memory store.

    When the same ``InMemoryStore`` instance is passed to multiple
    ``create_app()`` calls, both app instances share the same state —
    this is how the persistence tests simulate cross-process visibility
    without a real database.
    """

    def __init__(self) -> None:
        self._cases: dict[str, dict[str, Any]] = {}
        self._statuses: dict[str, str] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._overrides: list[dict[str, Any]] = []
        self._precedents: list[dict[str, Any]] = []
        self._day: date | None = None
        self._count: int = 0
        self._lock = threading.Lock()

    # --- Case files ------------------------------------------------------

    def save_case(self, dispute_id: str, case_data: dict[str, Any]) -> None:
        with self._lock:
            self._cases[dispute_id] = case_data

    def load_case(self, dispute_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._cases.get(dispute_id)
            return dict(data) if data else None

    def list_resolved_ids(self) -> list[str]:
        with self._lock:
            return sorted(
                did
                for did, s in self._statuses.items()
                if s in ("ruled", "escalated", "overridden", "error")
            )

    # --- Case status -----------------------------------------------------

    def get_status(self, dispute_id: str) -> str | None:
        with self._lock:
            return self._statuses.get(dispute_id)

    def set_status(self, dispute_id: str, status: str) -> None:
        with self._lock:
            self._statuses[dispute_id] = status

    def list_statuses(self) -> dict[str, str]:
        with self._lock:
            return dict(self._statuses)

    # --- Events ----------------------------------------------------------

    def save_events(self, dispute_id: str, events: list[dict[str, Any]]) -> None:
        with self._lock:
            self._events[dispute_id] = list(events)

    def load_events(self, dispute_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events.get(dispute_id, []))

    # --- Overrides -------------------------------------------------------

    def save_override(self, override: dict[str, Any]) -> None:
        with self._lock:
            self._overrides.append(dict(override))

    def load_overrides(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(o) for o in self._overrides]

    # --- Daily counter ---------------------------------------------------

    def get_daily_count(self) -> tuple[date, int]:
        today = datetime.now(timezone.utc).date()
        with self._lock:
            if self._day != today:
                self._day = today
                self._count = 0
            return (self._day, self._count)

    def increment_daily_count(self) -> tuple[date, int]:
        today = datetime.now(timezone.utc).date()
        with self._lock:
            if self._day != today:
                self._day = today
                self._count = 0
            self._count += 1
            return (self._day, self._count)

    def set_daily_count(self, day: date, count: int) -> None:
        with self._lock:
            self._day = day
            self._count = count

    # --- Precedents ------------------------------------------------------

    def save_precedent(self, precedent: dict[str, Any]) -> None:
        with self._lock:
            self._precedents.append(dict(precedent))

    def load_precedents(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(p) for p in self._precedents]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: Store | None = None


def get_store() -> Store:
    """Return the configured store instance.

    ``STATE_BACKEND=memory`` (or unset) → ``InMemoryStore`` (default)

    ``STATE_BACKEND=firestore`` → ``FirestoreStore`` (lazy import so
    tests never need ``google-cloud-firestore`` installed)
    """
    global _store
    if _store is not None:
        return _store

    backend = os.environ.get("STATE_BACKEND", "memory").lower()
    if backend == "firestore":
        from .store_firestore import FirestoreStore

        _store = FirestoreStore()
    else:
        _store = InMemoryStore()
    return _store


def set_store(store: Store) -> None:
    """Override the module-level store (used by tests to inject a shared
    ``InMemoryStore``)."""
    global _store
    _store = store


def reset_store() -> None:
    """Reset the module-level store to ``None`` (used by test fixtures)."""
    global _store
    _store = None
