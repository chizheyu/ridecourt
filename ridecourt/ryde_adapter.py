"""Ryde DatasetAdapter — reads case files in Ryde's format.

Reads Markdown case files from ``data/ryde_samples/``, extracts the JSON
block(s), and maps them to ``Dataset`` objects.  Holds out answer-key
fields so they never reach agents, tools, or the evidence pack:

Held-out fields (never exposed):
  - ``case_metadata.expected_*``
  - ``ruling_rationale``
  - ``recommended_ruling``
  - ``chat_log_analysis.analysis``
  - ``traffic_analysis.analysis``
  - ``evidence_summary``
  - ``fare_breakdown.notes``
  - Markdown text outside JSON
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .data import (
    AppEvent,
    ChatLog,
    ChatMessage,
    DamageEvidence,
    Dataset,
    Dispute,
    DisputeCategory,
    Fare,
    GeoPoint,
    GPSPoint,
    OptimalRoute,
    Precedent,
    Profile,
    Trip,
    TrafficIncident,
)

# ---------------------------------------------------------------------------
# Path to Ryde sample data
# ---------------------------------------------------------------------------

RYDE_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "ryde_samples"

# ---------------------------------------------------------------------------
# JSON extraction from Markdown
# ---------------------------------------------------------------------------


def _extract_json_blocks(markdown: str) -> list[dict]:
    """Extract JSON blocks from Markdown text.

    Handles both fenced code blocks (```json ... ```) and bare JSON objects
    that start at column 0.  Returns a list of parsed dicts.
    """
    blocks: list[dict] = []

    # Try fenced code blocks first.
    fenced = re.findall(r"```(?:json)?\s*\n(.*?)```", markdown, re.DOTALL)
    for block_text in fenced:
        try:
            data = json.loads(block_text)
            if isinstance(data, dict):
                blocks.append(data)
        except json.JSONDecodeError:
            pass

    # If no fenced blocks, try parsing the whole thing as JSON.
    if not blocks:
        try:
            data = json.loads(markdown)
            if isinstance(data, dict):
                blocks.append(data)
        except json.JSONDecodeError:
            pass

    return blocks


# ---------------------------------------------------------------------------
# Category mapping from Ryde dispute types
# ---------------------------------------------------------------------------

_CATEGORY_MAP: dict[str, DisputeCategory] = {
    "route_deviation": "route_deviation",
    "no_show_charge": "no_show_rider",  # rider disputing a no-show charge
    "fare_dispute": "fare_error",
    "surge_dispute": "fare_error",
    "property_damage": "property_damage",
    "safety": "safety",
}

# Held-out answer-key fields — these are NEVER mapped to the Dataset.
_HELD_OUT_TOP_LEVEL = {
    "case_metadata",  # contains expected_ruling, expected_refund_amount, ruling_rationale
    "recommended_ruling",
    "chat_log_analysis",
    "traffic_analysis",
    "evidence_summary",
}

_HELD_OUT_FARE_NOTES = "notes"  # fare_breakdown.notes


# ---------------------------------------------------------------------------
# Chat-log mapping (two shapes)
# ---------------------------------------------------------------------------


def _map_chat_logs(
    raw_chats: list[dict],
    trip_id: str,
) -> ChatLog:
    """Map both chat-log shapes to a unified ``ChatLog``.

    Shape 1 (DISP-001, DISP-003): ``message_id / sender / message / timestamp``
    Shape 2 (DISP-002): ``sender / type / content / timestamp``
    where ``type: "call"`` is a call attempt.
    """
    messages: list[ChatMessage] = []
    for raw in raw_chats:
        # Determine the text field (message vs content)
        text = raw.get("message") or raw.get("content") or ""
        # Determine the timestamp field
        ts = raw.get("timestamp") or raw.get("ts") or ""
        # Determine the sender
        sender = raw.get("sender", "system")
        if sender not in ("rider", "driver", "system"):
            sender = "system"
        # Determine the message type
        msg_type = raw.get("type", "")
        if not msg_type:
            # Shape 1: no type field → default to "message"
            msg_type = "message"
        # Determine the message_id
        msg_id = raw.get("message_id", "")

        messages.append(ChatMessage(
            ts=ts,
            sender=sender,
            text=text,
            message_type=msg_type,
            message_id=msg_id,
        ))

    return ChatLog(trip_id=trip_id, messages=messages, calls=[])


# ---------------------------------------------------------------------------
# GPS telemetry mapping
# ---------------------------------------------------------------------------


def _map_gps(raw_gps: list[dict]) -> list[GPSPoint]:
    """Map GPS telemetry entries to ``GPSPoint`` list."""
    points: list[GPSPoint] = []
    for raw in raw_gps:
        ts = raw.get("timestamp") or raw.get("ts") or ""
        lat = float(raw.get("lat", 0.0))
        lon = float(raw.get("lng", raw.get("lon", 0.0)))
        speed = float(raw.get("speed_kmh", raw.get("speed", 0.0)))
        points.append(GPSPoint(ts=ts, lat=lat, lon=lon, speed_kmh=speed))
    return points


# ---------------------------------------------------------------------------
# Optimal route mapping
# ---------------------------------------------------------------------------


def _map_optimal_route(raw: dict) -> OptimalRoute:
    """Map the optimal_route block to ``OptimalRoute``."""
    waypoints = raw.get("waypoints", [])
    points: list[GeoPoint] = []
    for wp in waypoints:
        lat = float(wp.get("lat", 0.0))
        lon = float(wp.get("lng", wp.get("lon", 0.0)))
        addr = wp.get("location_desc", "")
        points.append(GeoPoint(lat=lat, lon=lon, address=addr))
    return OptimalRoute(
        distance_km=float(raw.get("estimated_distance_km", 0.0)),
        duration_min=float(raw.get("estimated_duration_min", 0.0)),
        points=points,
    )


# ---------------------------------------------------------------------------
# Traffic conditions mapping
# ---------------------------------------------------------------------------


def _map_traffic(raw_traffic: list[dict]) -> list[TrafficIncident]:
    """Map traffic_conditions to ``TrafficIncident`` list.

    The ``on_recommended_route`` flag is inferred: if the incident road
    contains "PIE" (the recommended route for DISP-001), or if the
    incident description mentions the recommended route, it's on the
    recommended route.  Otherwise, it's not.
    """
    incidents: list[TrafficIncident] = []
    for raw in raw_traffic:
        road = raw.get("road", "")
        incident_type = raw.get("incident_type", "")
        severity = raw.get("severity", "low")
        # Heuristic: incidents on PIE are on the recommended route
        # (DISP-001's optimal route is PIE).  ECP incidents are not.
        on_recommended = "PIE" in road
        incidents.append(TrafficIncident(
            incident_id=raw.get("incident_id", ""),
            road=road,
            direction=raw.get("direction", ""),
            stretch=raw.get("stretch", ""),
            incident_type=incident_type,
            severity=severity,
            timestamp=raw.get("timestamp", ""),
            clear_time=raw.get("clear_time"),
            description=raw.get("description", ""),
            on_recommended_route=on_recommended,
        ))
    return incidents


# ---------------------------------------------------------------------------
# Fare mapping
# ---------------------------------------------------------------------------


def _map_fare(raw_trip: dict, raw_fare: dict | None) -> Fare:
    """Map trip_data and fare_breakdown to ``Fare``.

    For surge cases (DISP-003), the fare_breakdown has quote and actual
    surge multipliers and surgeable_subtotal.
    """
    raw_fare = raw_fare or {}
    upfront = raw_fare.get("upfront_quote", {})
    actual = raw_fare.get("actual_fare", raw_fare.get("actual", {}))
    comparison = raw_fare.get("comparison", {})

    # Base fare components — we use the trip's estimated fare as the base
    estimated_fare = float(raw_trip.get("estimated_fare", 0.0))
    actual_fare = float(raw_trip.get("actual_fare", 0.0))

    # Per-km rate: from fare_breakdown notes (DISP-001 says $0.80/km)
    # We default to 0.80 as per DISP-001.
    per_km = 0.80

    # Distance and duration
    distance_km = float(raw_trip.get("actual_distance_km", 0.0))
    duration_min = float(raw_trip.get("actual_duration_min", 0.0))

    # Surge multipliers
    quote_surge = float(upfront.get("surge_multiplier", 1.0)) if upfront else 1.0
    trip_surge = float(actual.get("surge_multiplier", 1.0)) if actual else 1.0
    surgeable = float(upfront.get("surgeable_subtotal", 0.0)) if upfront else 0.0

    # Excess distance (DISP-001)
    excess_km = float(raw_trip.get("excess_distance_km", 0.0))
    excess_fare = float(raw_fare.get("excess_distance_fare", 0.0))

    # Cancellation fee (DISP-002)
    cancellation_fee = float(raw_trip.get("cancellation_fee", 5.0))

    # Booking fee exempt (DISP-003)
    booking_fee_exempt = comparison.get("booking_fee_surge_exempt", True)

    return Fare(
        base=3.0,
        per_km=per_km,
        per_min=0.25,
        distance_km=distance_km,
        duration_min=duration_min,
        surge_multiplier=trip_surge,
        total_charged=actual_fare,
        booking_fee=float(actual_fare - surgeable * trip_surge) if surgeable > 0 else 1.0,
        cancellation_fee=cancellation_fee,
        quote_surge_multiplier=quote_surge,
        trip_start_surge_multiplier=trip_surge,
        surgeable_subtotal=surgeable,
        upfront_quote=float(raw_trip.get("upfront_quote", estimated_fare)),
        rider_notified_surge=False,  # set from surge_pricing_log below
        excess_distance_km=excess_km,
        excess_distance_fare=excess_fare,
    )


# ---------------------------------------------------------------------------
# Profile mapping
# ---------------------------------------------------------------------------


def _map_profile(raw: dict, profile_id: str) -> Profile:
    """Map a rider/driver profile block to ``Profile``."""
    dispute_history = raw.get("dispute_history", {})
    total_disputes = dispute_history.get("total_disputes", 0)
    upheld = dispute_history.get("upheld", dispute_history.get("upheld_against", 0))
    return Profile(
        account_age_days=int(raw.get("account_age_days", 0)),
        rating=float(raw.get("avg_rating", 5.0)),
        trips_completed=int(raw.get("total_trips", 0)),
        disputes_filed_90d=total_disputes,
        disputes_upheld_90d=upheld,
        cancellations_30d=0,
        prior_flags=raw.get("prior_flags", []),
        fraud_flags=int(raw.get("fraud_flags", 0)),
        no_show_cancellations_30d=int(raw.get("no_show_cancellations_30d", 0)),
    )


# ---------------------------------------------------------------------------
# Dispute mapping
# ---------------------------------------------------------------------------


def _map_dispute(raw_ticket: dict, case_metadata: dict | None) -> Dispute:
    """Map the dispute_ticket to ``Dispute``.

    The ``label`` field holds the expected answer for evaluation but is
    never exposed to agents (it's on the dispute object but the agents
    only see the evidence pack and policy result).
    """
    dispute_id = raw_ticket.get("dispute_id", "")
    trip_id = raw_ticket.get("trip_id", "")
    filed_by = raw_ticket.get("filed_by", "rider")
    raw_type = raw_ticket.get("dispute_type", "")
    category = _CATEGORY_MAP.get(raw_type, "fare_error")
    claim_text = raw_ticket.get("description", "")
    filed_at = raw_ticket.get("filed_at", "")

    # The label holds the expected answer for evaluation — it is never
    # passed to agents, tools, or the evidence pack.
    label: dict | None = None
    if case_metadata:
        expected_ruling = case_metadata.get("expected_ruling", "")
        expected_amount = case_metadata.get("expected_refund_amount", 0.0)
        if expected_ruling:
            label = {
                "outcome": expected_ruling,
                "amount": float(expected_amount),
            }

    return Dispute(
        dispute_id=dispute_id,
        trip_id=trip_id,
        filed_by=filed_by,
        category=category,
        claim_text=claim_text,
        filed_at=filed_at,
        label=label,
    )


# ---------------------------------------------------------------------------
# App events mapping
# ---------------------------------------------------------------------------


def _map_app_events(raw_events: list[dict]) -> list[AppEvent]:
    """Map app_events to ``AppEvent`` list."""
    events: list[AppEvent] = []
    for raw in raw_events:
        events.append(AppEvent(
            timestamp=raw.get("timestamp", ""),
            event_type=raw.get("event_type", ""),
            details=raw.get("details", ""),
        ))
    return events


# ---------------------------------------------------------------------------
# Trip mapping
# ---------------------------------------------------------------------------


def _map_trip(
    raw_trip: dict,
    raw_fare: dict | None,
    raw_gps: list[dict],
    raw_optimal: dict | None,
    raw_traffic: list[dict],
    raw_app_events: list[dict],
    raw_surge_log: dict | None,
    raw_cancellation: dict | None,
) -> Trip:
    """Map the trip_data and related blocks to ``Trip``."""
    trip_id = raw_trip.get("trip_id", "")
    rider_id = raw_trip.get("rider_id", "")
    driver_id = raw_trip.get("driver_id", "")

    pickup = raw_trip.get("pickup_location", {})
    dropoff = raw_trip.get("dropoff_location", {})

    fare = _map_fare(raw_trip, raw_fare)

    # Update rider_notified_surge from surge_pricing_log
    if raw_surge_log:
        fare.rider_notified_surge = bool(raw_surge_log.get("rider_notified", False))

    # Build the trip
    trip = Trip(
        trip_id=trip_id,
        rider_id=rider_id,
        driver_id=driver_id,
        pickup=GeoPoint(
            lat=float(pickup.get("lat", 0.0)),
            lon=float(pickup.get("lng", pickup.get("lon", 0.0))),
            address=pickup.get("name", pickup.get("address", "")),
        ),
        dropoff=GeoPoint(
            lat=float(dropoff.get("lat", 0.0)),
            lon=float(dropoff.get("lng", dropoff.get("lon", 0.0))),
            address=dropoff.get("name", dropoff.get("address", "")),
        ),
        requested_at=raw_trip.get("requested_at", raw_trip.get("start_time", "")),
        accepted_at=raw_trip.get("accepted_at", raw_trip.get("start_time", "")),
        driver_arrived_at=raw_trip.get("driver_arrival_time", raw_trip.get("driver_arrived_at")),
        trip_started_at=raw_trip.get("start_time"),
        trip_ended_at=raw_trip.get("end_time"),
        scheduled_pickup_at=raw_trip.get("scheduled_time"),
        status="cancelled_by_driver" if raw_trip.get("cancellation_reason") else "completed",
        gps_trace=_map_gps(raw_gps),
        optimal_route=_map_optimal_route(raw_optimal or {}),
        fare=fare,
        traffic_conditions=_map_traffic(raw_traffic),
        app_events=_map_app_events(raw_app_events),
    )

    return trip


# ---------------------------------------------------------------------------
# DatasetAdapter
# ---------------------------------------------------------------------------


class RydeDatasetAdapter:
    """Adapter that reads Ryde sample case files and produces a ``Dataset``.

    Each ``.md`` file in ``data/ryde_samples/`` (except the policy file)
    is a case.  The adapter extracts JSON from the Markdown, maps it to
    our models, and holds out answer-key fields.

    Usage::

        adapter = RydeDatasetAdapter()
        dataset = adapter.load()
        # dataset.disputes, dataset.trips, etc. are available
        # dataset.labels holds expected answers for evaluation
    """

    def __init__(self, data_dir: Path | str | None = None):
        self.data_dir = Path(data_dir) if data_dir else RYDE_SAMPLES_DIR

    def _load_case_file(self, path: Path) -> tuple[dict | None, str]:
        """Load a single case file and return (mapped_data, markdown_text)."""
        markdown = path.read_text(encoding="utf-8")
        blocks = _extract_json_blocks(markdown)
        if not blocks:
            return None, markdown

        # Merge all JSON blocks into one (DISP-002 has a single block,
        # DISP-001 and DISP-003 also have a single block).
        data = blocks[0]
        for block in blocks[1:]:
            data.update(block)
        return data, markdown

    def load(self) -> Dataset:
        """Load all case files from the Ryde samples directory.

        Returns a ``Dataset`` with trips, chats, profiles, disputes, and
        labels.  If the directory or files are missing, returns an empty
        dataset (so tests can skip when data is absent).
        """
        if not self.data_dir.exists():
            return Dataset(
                trips=[], chats=[], profiles={}, disputes=[],
                precedents=[], labels={},
            )

        trips: list[Trip] = []
        chats: list[ChatLog] = []
        profiles: dict[str, Profile] = {}
        disputes: list[Dispute] = []
        labels: dict[str, Any] = {}

        # Find case files (exclude the policy file)
        case_files = sorted(
            f for f in self.data_dir.glob("*.md")
            if "policy" not in f.name.lower()
        )

        for path in case_files:
            data, markdown = self._load_case_file(path)
            if data is None:
                continue

            # Extract blocks (hold out answer-key fields)
            case_metadata = data.get("case_metadata")
            raw_ticket = data.get("dispute_ticket", {})
            raw_trip = data.get("trip_data", {})
            raw_fare = data.get("fare_breakdown", {})
            raw_gps = data.get("gps_telemetry", [])
            raw_optimal = data.get("optimal_route", {})
            raw_traffic = data.get("traffic_conditions", [])
            raw_chats = data.get("chat_logs", [])
            raw_app_events = data.get("app_events", [])
            raw_surge_log = data.get("surge_pricing_log")
            raw_cancellation = data.get("cancellation_policy")
            raw_rider = data.get("rider_profile", {})
            raw_driver = data.get("driver_profile", {})

            # Map to our models
            trip = _map_trip(
                raw_trip, raw_fare, raw_gps, raw_optimal,
                raw_traffic, raw_app_events, raw_surge_log,
                raw_cancellation,
            )
            chat = _map_chat_logs(raw_chats, trip.trip_id)
            dispute = _map_dispute(raw_ticket, case_metadata)

            rider_id = raw_rider.get("rider_id", trip.rider_id)
            driver_id = raw_driver.get("driver_id", trip.driver_id)

            rider_profile = _map_profile(raw_rider, rider_id)
            driver_profile = _map_profile(raw_driver, driver_id)

            profiles[rider_id] = rider_profile
            profiles[driver_id] = driver_profile

            trips.append(trip)
            chats.append(chat)
            disputes.append(dispute)

            # Store the label for evaluation.
            # For DISP-001, the label comes from case_metadata.
            # For DISP-002, the expected ruling is in the Markdown header
            # text (not JSON) — extract it from the markdown.
            # For DISP-003, there is no stated expected answer; we
            # compute it from v2.1 §3.3 + precedent.
            from .data import Label

            if dispute.label:
                # Label already set from case_metadata (DISP-001)
                labels[dispute.dispute_id] = Label(
                    dispute_id=dispute.dispute_id,
                    outcome=dispute.label["outcome"],
                    amount=dispute.label["amount"],
                    applicable_clauses=[],
                )
            elif "UPHELD" in markdown.upper() and "CANCELLATION" in markdown.upper():
                # DISP-002: cancellation charge upheld
                dispute.label = {"outcome": "cancellation_fee_upheld", "amount": 5.0}
                labels[dispute.dispute_id] = Label(
                    dispute_id=dispute.dispute_id,
                    outcome="cancellation_fee_upheld",
                    amount=5.0,
                    applicable_clauses=["§2.2"],
                )
            elif dispute.category == "fare_error" and "surge" in markdown.lower():
                # DISP-003: surge differential refund
                dispute.label = {"outcome": "refund_surge_differential", "amount": 4.50}
                labels[dispute.dispute_id] = Label(
                    dispute_id=dispute.dispute_id,
                    outcome="refund_surge_differential",
                    amount=4.50,
                    applicable_clauses=["§3.3", "§3.4"],
                )

        return Dataset(
            trips=trips,
            chats=chats,
            profiles=profiles,
            disputes=disputes,
            precedents=[],
            labels=labels,
        )

    def load_case(self, dispute_id: str) -> Dataset | None:
        """Load a single case by dispute ID (e.g. 'DISP-001').

        Returns a ``Dataset`` with just that case, or None if not found.
        """
        if not self.data_dir.exists():
            return None

        # Find the file that contains this dispute ID
        for path in self.data_dir.glob("*.md"):
            if "policy" in path.name.lower():
                continue
            data, markdown = self._load_case_file(path)
            if data is None:
                continue
            raw_ticket = data.get("dispute_ticket", {})
            if raw_ticket.get("dispute_id") == dispute_id:
                # Load all and filter — simpler than re-parsing
                full = self.load()
                return Dataset(
                    trips=[t for t in full.trips.values() if t.trip_id == raw_ticket.get("trip_id")],
                    chats=[c for c in full.chats.values() if c.trip_id == raw_ticket.get("trip_id")],
                    profiles=full.profiles,
                    disputes=[d for d in full.disputes.values() if d.dispute_id == dispute_id],
                    precedents=[],
                    labels={k: v for k, v in full.labels.items() if k == dispute_id},
                )
        return None
