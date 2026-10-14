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
    cancellation_fee: float = 5.0  # v2.1 §2.2: S$5.00
    total_charged: float = 0.0
    # v2.1 §3: surge pricing at quote and trip-start time
    quote_surge_multiplier: float = 1.0
    trip_start_surge_multiplier: float = 1.0
    surgeable_subtotal: float = 0.0  # excludes booking fee
    upfront_quote: float = 0.0
    rider_notified_surge: bool = False
    # v2.1 §1.3: excess distance fare components
    excess_distance_km: float = 0.0
    excess_distance_fare: float = 0.0
    # v2.1 §4.3: damage claim amount
    damage_claim_amount: float = 0.0
    # Whether base/per_km/per_min come from the source data (vs. defaults).
    # When False, recompute_fare must not be used — the source lacks the
    # individual rate components.
    has_fare_components: bool = False


# ---------------------------------------------------------------------------
# v2.1-specific data models
# ---------------------------------------------------------------------------


class TrafficIncident(BaseModel):
    """A single traffic condition entry (v2.1 §1.4)."""

    incident_id: str = ""
    road: str = ""
    direction: str = ""
    stretch: str = ""
    incident_type: str = ""  # vehicle_fire, heavy_congestion, traffic_flow
    severity: str = "low"  # low, moderate, high
    timestamp: str = ""
    clear_time: str | None = None
    description: str = ""
    on_recommended_route: bool = False  # whether incident is on the recommended route


class DamageEvidence(BaseModel):
    """Property damage evidence (v2.1 §4.2)."""

    photo_source: str = ""  # "in_app_camera" | "external_upload"
    has_timestamp: bool = False
    submitted_within_24h: bool = False
    linked_to_trip: bool = False
    damage_type: str = ""  # "cleaning" | "repair"
    claim_amount: float = 0.0
    has_certified_quote: bool = False
    quote_within_7_days: bool = False


class AppEvent(BaseModel):
    """A single app event from the app events log."""

    timestamp: str
    event_type: str
    details: str = ""


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
    scheduled_pickup_at: Optional[str] = None  # v2.1 §2.1
    cancelled_at: Optional[str] = None  # when the trip was cancelled (no-show)
    status: TripStatus = "completed"
    gps_trace: list[GPSPoint] = Field(default_factory=list)
    optimal_route: OptimalRoute = Field(default_factory=OptimalRoute)
    fare: Fare = Field(default_factory=Fare)
    # v2.1 §1.4: traffic conditions for route deviation analysis
    traffic_conditions: list[TrafficIncident] = Field(default_factory=list)
    # v2.1 §4: property damage evidence
    damage_evidence: Optional[DamageEvidence] = None
    # v2.1: app events log
    app_events: list[AppEvent] = Field(default_factory=list)


class ChatMessage(BaseModel):
    ts: str
    sender: Literal["rider", "driver", "system"]
    text: str
    # v2.1 second chat shape: type field ("message" | "call" | "system")
    message_type: str = "message"  # "message" | "call" | "system"
    message_id: str = ""  # MSG-001 style ID from Ryde format


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
    # v2.1 §6.3: fraud flags raise the confidence threshold to 95%
    fraud_flags: int = 0
    fraud_flag_reason: str = ""
    # v2.1 §2.5: 3+ no-show cancellations in 30 days → flag for review
    no_show_cancellations_30d: int = 0
    source: str = "synthetic"  # "synthetic" or "ryde"


DisputeCategory = Literal[
    "route_deviation",
    "no_show_driver",
    "no_show_rider",
    "fare_error",
    "safety",
    "property_damage",
    "promo_code",
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
    source: str = "seed"  # "seed" or "override" (learned from human override)
    reason: str = ""  # reviewer's reason (for override-sourced precedents)
    # Structured deciding facts used for precedent alignment scoring.
    # e.g. {"rider_informed": false, "traffic_justified": true, "contact_made": true}
    deciding_facts: dict = Field(default_factory=dict)


class Label(BaseModel):
    dispute_id: str
    outcome: str
    amount: float = 0.0
    applicable_clauses: list[str] = Field(default_factory=list)
    routing: str = ""  # "ai_final" or "human"
    deciding_section: str = ""  # e.g. "§6.1", "§6.2", "§6.3"
    ryde_outcome: str = ""  # Ryde's original outcome wording (official cases only)


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
    # Route deviation (§1)
    distance_actual_km: float = 0.0
    distance_optimal_km: float = 0.0
    duration_actual_min: float = 0.0
    duration_optimal_min: float = 0.0
    extra_min: float = 0.0  # duration_actual - duration_optimal
    deviation_pct: float = 0.0
    detour_segment: dict | None = None  # {start_ts, end_ts}
    unexpected_stops: list[dict] = Field(default_factory=list)
    rider_requested_change: bool = False
    rider_requested_change_quotes: list[str] = Field(default_factory=list)
    road_closure: bool = False
    road_closure_quotes: list[str] = Field(default_factory=list)
    # v2.1 §1.2: rider informed of deviation before/at deviation point
    rider_informed_deviation: bool = False
    deviation_point_ts: str | None = None  # first GPS point off recommended route
    # v2.1 §1.4: traffic justification (moderate or high severity on recommended route)
    traffic_justified: bool = False
    traffic_incidents_on_route: list[dict] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    abusive_language: list[str] = Field(default_factory=list)
    threatening_language: list[str] = Field(default_factory=list)
    # No-show (§2)
    contact_attempts: int = 0
    driver_chat_messages: int = 0  # v2.1 §2.3: at least 1 chat message
    driver_call_attempts: int = 0  # v2.1 §2.3: at least 1 call
    min_distance_to_pickup_m: float = float("inf")
    arrival_ts: str | None = None
    arrival_vs_eta_min: float = 0.0  # negative = early, positive = late
    wait_minutes_within_150m: float = 0.0
    wait_minutes_past_eta: float = 0.0
    # v2.1 §2.1: free wait from later of scheduled pickup and arrival
    free_wait_expired: bool = False
    # minutes from the start of the free wait to the cancellation; None when
    # the data has no cancellation time
    free_wait_minutes: Optional[float] = None
    driver_no_show: bool = False  # §2.4: driver not at pickup by scheduled+5
    driver_made_contact: bool = False  # §2.3: both chat AND call
    rider_no_show_cancellations_30d: int = 0  # §2.5
    # Fare/surge (§3)
    recomputed_fare: float = 0.0
    charged_total: float = 0.0
    fare_difference: float = 0.0  # charged - recomputed
    promo_applied: bool = False
    promo_valid_at_booking: bool = True
    # v2.1 §3: surge pricing
    quote_surge_multiplier: float = 1.0
    trip_start_surge_multiplier: float = 1.0
    surgeable_subtotal: float = 0.0
    upfront_quote: float = 0.0
    rider_notified_surge: bool = False
    surge_increased: bool = False
    surge_decreased: bool = False
    surge_differential: float = 0.0
    # v2.1 §4: property damage
    damage_claim_amount: float = 0.0
    damage_type: str = ""  # "cleaning" | "repair"
    photo_source: str = ""  # "in_app_camera" | "external_upload"
    has_timestamp: bool = False
    submitted_within_24h: bool = False
    linked_to_trip: bool = False
    has_certified_quote: bool = False
    quote_within_7_days: bool = False
    # v2.1 §5: safety indicators
    safety_indicators_in_chat: bool = False
    safety_indicators_in_claim: bool = False
    # Profiles
    rider_disputes_filed_90d: int = 0
    rider_disputes_upheld_90d: int = 0
    rider_upheld_rate: float = 0.0
    rider_account_age_days: int = 0
    rider_rating: float = 0.0
    rider_prior_flags: list[str] = Field(default_factory=list)
    rider_fraud_flags: int = 0
    driver_disputes_filed_90d: int = 0
    driver_disputes_upheld_90d: int = 0
    driver_upheld_rate: float = 0.0
    driver_account_age_days: int = 0
    driver_rating: float = 0.0
    driver_prior_flags: list[str] = Field(default_factory=list)
    driver_fraud_flags: int = 0


# ---------------------------------------------------------------------------
# Policy result
# ---------------------------------------------------------------------------


class PolicyResult(BaseModel):
    applicable_clauses: list[str] = Field(default_factory=list)
    outcome: str = "no_refund"
    amount: float = 0.0
    computation: str = ""
    # v2.1 §6: routing and confidence
    routing: str = "ai_final"  # "ai_final" | "human"
    routing_reason: str = ""  # e.g. "§6.2: deviation 4.3 km > 2 km"
    confidence: float = 0.0
    confidence_factors: dict = Field(default_factory=dict)
    dispute_value: float = 0.0
    actions: list[dict] = Field(default_factory=list)  # e.g. flag account, suspend account


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
    trips_raw = _load_json(d / "trips.json")
    trips = []
    for t in trips_raw:
        fare_raw = t.get("fare", {})
        if "base" in fare_raw and "per_km" in fare_raw and "per_min" in fare_raw:
            fare_raw = {**fare_raw, "has_fare_components": True}
            t = {**t, "fare": fare_raw}
        trips.append(Trip(**t))
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


def load_app_dataset(data_dir: Path | str | None = None) -> Dataset:
    """The dataset the app, the evaluation and pre-resolve use: the synthetic
    cases plus Ryde's sample cases when ``data/ryde_samples/`` is present.

    Ryde's cases keep their own IDs (``DISP-001`` ...) and are listed in
    ``ds.sample_ids`` so the queue can show them first.
    """
    ds = load_dataset(data_dir)
    ds.sample_ids = []
    from .ryde_adapter import RYDE_SAMPLES_DIR, RydeDatasetAdapter

    if not RYDE_SAMPLES_DIR.exists():
        return ds
    ryde = RydeDatasetAdapter().load()
    ds.trips.update(ryde.trips)
    ds.chats.update(ryde.chats)
    ds.profiles.update(ryde.profiles)
    ds.disputes.update(ryde.disputes)
    ds.labels.update(ryde.labels)
    known = {p.case_id for p in ds.precedents}
    ds.precedents.extend(p for p in ryde.precedents if p.case_id not in known)
    ds.sample_ids = sorted(ryde.disputes)
    return ds
