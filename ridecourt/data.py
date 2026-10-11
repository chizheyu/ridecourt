"""Pydantic models for RideCourt data and a loader for the synthetic dataset.

The loader supports an adapter interface so the organiser's real dataset can be
plugged in later without changing downstream code.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Primitive value objects
# ---------------------------------------------------------------------------


class GeoPoint(BaseModel):
    lat: float
    lon: float
    address: str = ""


class GPSPoint(BaseModel):
    ts: str  # ISO 8601
    lat: float
    lon: float
    speed_kmh: float = 0.0


class OptimalRoute(BaseModel):
    distance_km: float
    duration_min: float
    points: list[GeoPoint] = Field(default_factory=list)


class Fare(BaseModel):
    base: float = 3.0
    per_km: float = 0.75
    per_min: float = 0.25
    distance_km: float = 0.0
    duration_min: float = 0.0
    surge_multiplier: float = 1.0
    promo_code: str = ""
    discount: float = 0.0  # absolute discount
    booking_fee: float = 1.0
    cancellation_fee: float = 4.0
    total_charged: float = 0.0


# ---------------------------------------------------------------------------
# Core records
# ---------------------------------------------------------------------------

TripStatus = Literal[
    "completed",
    "cancelled_by_rider",
    "cancelled_by_driver",
    "no_show_rider",
    "no_show_driver",
]


class Trip(BaseModel):
    trip_id: str
    rider_id: str
    driver_id: str
    pickup: GeoPoint
    dropoff: GeoPoint
    requested_at: str
    accepted_at: str
    driver_arrived_at: Optional[str] = None
    trip_started_at: Optional[str] = None
    trip_ended_at: Optional[str] = None
    status: TripStatus = "completed"
    gps_trace: list[GPSPoint] = Field(default_factory=list)
    optimal_route: OptimalRoute = Field(default_factory=OptimalRoute)
    fare: Fare = Field(default_factory=Fare)


class ChatMessage(BaseModel):
    ts: str
    sender: Literal["rider", "driver", "system"]
    text: str


class CallAttempt(BaseModel):
    ts: str
    sender: Literal["rider", "driver"]
    recipient: Literal["rider", "driver"]
    duration_s: int = 0
    answered: bool = False


class ChatLog(BaseModel):
    trip_id: str
    messages: list[ChatMessage] = Field(default_factory=list)
    calls: list[CallAttempt] = Field(default_factory=list)


class Profile(BaseModel):
    account_age_days: int
    rating: float
    trips_completed: int
    disputes_filed_90d: int = 0
    disputes_upheld_90d: int = 0
    cancellations_30d: int = 0
    prior_flags: list[str] = Field(default_factory=list)


DisputeCategory = Literal[
    "route_deviation",
    "no_show_driver",
    "no_show_rider",
    "fare_error",
    "safety",
]


class Dispute(BaseModel):
    dispute_id: str
    trip_id: str
    filed_by: Literal["rider", "driver"]
    category: DisputeCategory
    claim_text: str
    filed_at: str
    label: Optional[dict] = None  # {outcome, amount} hidden from agents


class Precedent(BaseModel):
    case_id: str
    category: DisputeCategory
    summary: str
    outcome: str
    amount: float = 0.0


class Label(BaseModel):
    dispute_id: str
    outcome: str
    amount: float = 0.0
    applicable_clauses: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Facts (computed by tools, consumed by policy)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Evidence pack (built by deterministic tools, consumed by agents + policy)
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """A single piece of computed evidence, cited by agents as ``E#``."""

    id: str  # E1, E2, ...
    kind: str  # gps_analysis | arrival_check | chat_scan | fare_check | history_profile
    summary: str  # one-line human-readable summary (numbers must match data)
    data: dict  # structured values — single source of truth for numbers
    source: str  # tool name that produced this item


class EvidencePack(BaseModel):
    items: list[EvidenceItem] = Field(default_factory=list)

    def by_id(self, eid: str) -> EvidenceItem | None:
        return next((i for i in self.items if i.id == eid), None)

    def by_kind(self, kind: str) -> list[EvidenceItem]:
        return [i for i in self.items if i.kind == kind]


class Facts(BaseModel):
    distance_actual_km: float = 0.0
    distance_optimal_km: float = 0.0
    duration_actual_min: float = 0.0
    duration_optimal_min: float = 0.0
    deviation_pct: float = 0.0
    detour_segment: dict | None = None  # {start_ts, end_ts}
    unexpected_stops: list[dict] = Field(default_factory=list)  # {start_ts, end_ts, duration_min}
    rider_requested_change: bool = False
    rider_requested_change_quotes: list[str] = Field(default_factory=list)
    road_closure: bool = False
    road_closure_quotes: list[str] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    abusive_language: list[str] = Field(default_factory=list)
    threatening_language: list[str] = Field(default_factory=list)
    contact_attempts: int = 0
    min_distance_to_pickup_m: float = float("inf")
    arrival_ts: str | None = None
    arrival_vs_eta_min: float = 0.0  # negative = early, positive = late
    wait_minutes_within_150m: float = 0.0
    wait_minutes_past_eta: float = 0.0  # kept for policy engine no-show clauses
    recomputed_fare: float = 0.0
    charged_total: float = 0.0
    fare_difference: float = 0.0  # charged - recomputed
    promo_applied: bool = False
    promo_valid_at_booking: bool = True
    rider_disputes_filed_90d: int = 0
    rider_disputes_upheld_90d: int = 0
    rider_upheld_rate: float = 0.0
    rider_account_age_days: int = 0
    rider_rating: float = 0.0
    rider_prior_flags: list[str] = Field(default_factory=list)
    driver_disputes_filed_90d: int = 0
    driver_disputes_upheld_90d: int = 0
    driver_upheld_rate: float = 0.0
    driver_account_age_days: int = 0
    driver_rating: float = 0.0
    driver_prior_flags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Policy result
# ---------------------------------------------------------------------------


class PolicyResult(BaseModel):
    applicable_clauses: list[str] = Field(default_factory=list)
    outcome: str = "no_refund"
    amount: float = 0.0
    computation: str = ""


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


class Dataset:
    """In-memory dataset loaded from JSON files."""

    def __init__(
        self,
        trips: list[Trip],
        chats: list[ChatLog],
        profiles: dict[str, Profile],
        disputes: list[Dispute],
        precedents: list[Precedent],
        labels: dict[str, Label],
    ):
        self.trips = {t.trip_id: t for t in trips}
        self.chats = {c.trip_id: c for c in chats}
        self.profiles = profiles
        self.disputes = {d.dispute_id: d for d in disputes}
        self.precedents = precedents
        self.labels = labels

    # convenience accessors ------------------------------------------------

    def trip(self, trip_id: str) -> Optional[Trip]:
        return self.trips.get(trip_id)

    def chat(self, trip_id: str) -> Optional[ChatLog]:
        return self.chats.get(trip_id)

    def dispute(self, dispute_id: str) -> Optional[Dispute]:
        return self.disputes.get(dispute_id)

    def label(self, dispute_id: str) -> Optional[Label]:
        return self.labels.get(dispute_id)


def _load_json(path: Path) -> list | dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset(data_dir: Path | str | None = None) -> Dataset:
    """Load the synthetic (or adapted) dataset from *data_dir*.

    If *data_dir* is None, uses the default ``data/synthetic`` location.
    """
    d = Path(data_dir) if data_dir else DATA_DIR
    trips = [Trip(**t) for t in _load_json(d / "trips.json")]
    chats = [ChatLog(**c) for c in _load_json(d / "chats.json")]
    profiles_raw = _load_json(d / "profiles.json")
    profiles = {pid: Profile(**p) for pid, p in profiles_raw.items()}
    disputes = [Dispute(**dd) for dd in _load_json(d / "disputes.json")]
    precedents = [Precedent(**p) for p in _load_json(d / "precedents.json")]
    labels_raw = _load_json(d / "labels.json")
    labels = {
        lbl["dispute_id"]: Label(**lbl) for lbl in labels_raw
    }
    return Dataset(trips, chats, profiles, disputes, precedents, labels)


# ---------------------------------------------------------------------------
# Adapter interface (for the organiser's real dataset)
# ---------------------------------------------------------------------------


class DatasetAdapter:
    """Subclass and implement ``load`` to plug in an external dataset.

    The adapter should return a ``Dataset`` object so the rest of the codebase
    is agnostic to the source format.
    """

    def load(self) -> Dataset:  # pragma: no cover - interface
        raise NotImplementedError
