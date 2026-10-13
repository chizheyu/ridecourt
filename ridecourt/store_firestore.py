"""Firestore-backed implementation of ``Store``.

Used when ``STATE_BACKEND=firestore`` is set. The Firestore database is
the project's default native-mode database in ``asia-southeast1``.
Collection names are prefixed with ``ridecourt_`` because another app
uses the same database.

Firestore documents are limited to 1 MB. Case files (without events) are
~12 KB with mock data and well under 1 MB with real Gemini responses.
Events are stored in a separate document to keep each doc small and to
allow splitting into a subcollection later if needed.

This module is never instantiated in tests — ``google-cloud-firestore``
is imported lazily so tests run without it installed.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from typing import Any

from .store import Store

_COLLECTION_CASES = "ridecourt_cases"
_COLLECTION_EVENTS = "ridecourt_events"
_COLLECTION_STATUS = "ridecourt_status"
_COLLECTION_OVERRIDES = "ridecourt_overrides"
_COLLECTION_PRECEDENTS = "ridecourt_precedents"
_COLLECTION_COUNTERS = "ridecourt_counters"
_COUNTER_DOC = "daily_resolve"

_STATUS_TERMINAL = ("ruled", "escalated", "overridden", "error")


class FirestoreStore(Store):
    """Firestore-backed store.

    The Firestore client is created on first use (lazy) so that the module
    can be imported without credentials in tests.
    """

    def __init__(self, project: str | None = None) -> None:
        self._project = project or os.environ.get(
            "GOOGLE_CLOUD_PROJECT",
            os.environ.get("FIRESTORE_PROJECT", ""),
        )
        self._db: Any = None

    @property
    def db(self) -> Any:
        """Lazy Firestore client."""
        if self._db is None:
            from google.cloud import firestore

            self._db = firestore.Client(project=self._project or None)
        return self._db

    # --- Case files ------------------------------------------------------

    def save_case(self, dispute_id: str, case_data: dict[str, Any]) -> None:
        ref = self.db.collection(_COLLECTION_CASES).document(dispute_id)
        ref.set(case_data)

    def load_case(self, dispute_id: str) -> dict[str, Any] | None:
        ref = self.db.collection(_COLLECTION_CASES).document(dispute_id)
        doc = ref.get()
        if not doc.exists:
            return None
        return doc.to_dict()

    def list_resolved_ids(self) -> list[str]:
        col = self.db.collection(_COLLECTION_CASES)
        ids: list[str] = []
        for status in _STATUS_TERMINAL:
            # Query by the 'escalated' field presence and status field.
            docs = col.where("escalated", "==", True).stream()
            ids.extend(d.id for d in docs)
            docs = col.where("escalated", "==", False).stream()
            ids.extend(d.id for d in docs)
        return sorted(set(ids))

    # --- Case status -----------------------------------------------------

    def get_status(self, dispute_id: str) -> str | None:
        ref = self.db.collection(_COLLECTION_STATUS).document(dispute_id)
        doc = ref.get()
        if not doc.exists:
            return None
        return doc.to_dict().get("status")

    def set_status(self, dispute_id: str, status: str) -> None:
        ref = self.db.collection(_COLLECTION_STATUS).document(dispute_id)
        ref.set({"status": status, "updated_at": _now_iso()})

    def list_statuses(self) -> dict[str, str]:
        col = self.db.collection(_COLLECTION_STATUS)
        result: dict[str, str] = {}
        for doc in col.stream():
            status = doc.to_dict().get("status")
            if status:
                result[doc.id] = status
        return result

    # --- Events ----------------------------------------------------------

    def save_events(self, dispute_id: str, events: list[dict[str, Any]]) -> None:
        ref = self.db.collection(_COLLECTION_EVENTS).document(dispute_id)
        # Store as a single document with an 'events' array field.
        # Mock events are ~7 KB; real Gemini responses are larger but still
        # well under 1 MB for a single case (~11 events, each a few hundred
        # chars). If a case ever approaches 1 MB, split into a subcollection.
        ref.set({"events": events})

    def load_events(self, dispute_id: str) -> list[dict[str, Any]]:
        ref = self.db.collection(_COLLECTION_EVENTS).document(dispute_id)
        doc = ref.get()
        if not doc.exists:
            return []
        return doc.to_dict().get("events", [])

    # --- Overrides -------------------------------------------------------

    def save_override(self, override: dict[str, Any]) -> None:
        case_id = override.get("case_id", "")
        ref = self.db.collection(_COLLECTION_OVERRIDES).document(case_id)
        ref.set(override)

    def load_overrides(self) -> list[dict[str, Any]]:
        col = self.db.collection(_COLLECTION_OVERRIDES)
        return [doc.to_dict() for doc in col.stream()]

    # --- Daily counter ---------------------------------------------------

    def get_daily_count(self) -> tuple[date, int]:
        ref = self.db.collection(_COLLECTION_COUNTERS).document(_COUNTER_DOC)
        doc = ref.get()
        if not doc.exists:
            today = _today()
            return (today, 0)
        data = doc.to_dict()
        stored_day = data.get("day")
        today = _today()
        if stored_day != today.isoformat():
            return (today, 0)
        return (today, int(data.get("count", 0)))

    def increment_daily_count(self) -> tuple[date, int]:
        import threading as _t

        lock = _t.Lock()
        with lock:
            day, count = self.get_daily_count()
            count += 1
            self.set_daily_count(day, count)
            return (day, count)

    def set_daily_count(self, day: date, count: int) -> None:
        ref = self.db.collection(_COLLECTION_COUNTERS).document(_COUNTER_DOC)
        ref.set({"day": day.isoformat(), "count": count})

    # --- Precedents ------------------------------------------------------

    def save_precedent(self, precedent: dict[str, Any]) -> None:
        case_id = precedent.get("case_id", "")
        ref = self.db.collection(_COLLECTION_PRECEDENTS).document(case_id)
        ref.set(precedent)

    def load_precedents(self) -> list[dict[str, Any]]:
        col = self.db.collection(_COLLECTION_PRECEDENTS)
        return [doc.to_dict() for doc in col.stream()]


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
