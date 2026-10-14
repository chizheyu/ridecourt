"""Policy engine v2.1 — pure functions implementing Ryde Dispute Resolution
Policy v2.1 sections §1–§6.

Numbers come from the ``Facts`` object computed by ``ridecourt.tools``.
Every function returns a ``PolicyResult`` with the applicable section
citations, outcome, refund amount, computation string, and §6 routing
metadata (routing, routing_reason, confidence, dispute_value).

Agents cite sections as ``§1.3``, ``§2.3``, etc.  The guard validates
these against the list of valid sections in ``VALID_SECTIONS``.
"""

from __future__ import annotations

from typing import Sequence

from .data import Facts, Profile, PolicyResult

# ---------------------------------------------------------------------------
# v2.1 thresholds
# ---------------------------------------------------------------------------

# §1 Route Deviation
RD_TOLERANCE_KM = 1.0          # §1.3: deviation ≤ 1 km → within tolerance
RD_DEVIATION_THRESHOLD_KM = 2.0  # §6.2: deviation > 2 km → human

# §2 No-Show
NS_FREE_WAIT_MIN = 5.0         # §2.1: free wait time
NS_CANCELLATION_FEE = 5.0      # §2.2: S$5.00
NS_DRIVER_CREDIT = 5.0         # §2.4: S$5.00 goodwill credit
NS_REPEAT_FLAG_THRESHOLD = 3    # §2.5: 3+ no-shows in 30d → flag

# §3 Surge
# (no fixed threshold — computed from fare components)

# §4 Property Damage
PD_CLEANING_MAX = 50.0         # §4.3: up to S$50
PD_REPAIR_QUOTE_DAYS = 7       # §4.3: certified quote within 7 days

# §6 Ruling Thresholds
RT_UNDER_10 = 10.0             # §6.1: under S$10 → AI final
RT_10_TO_50 = 50.0             # §6.1: S$10–S$50
RT_CONFIDENCE_80 = 0.80       # §6.1: 80%+ confidence for S$10–S$50
RT_CONFIDENCE_95 = 0.95       # §6.3: 95% when fraud flag ≥ 1

# Confidence weights (§6.3)
W_EVIDENCE = 0.30
W_CLARITY = 0.25
W_PRECEDENT = 0.25
W_CREDIBILITY = 0.20

# ---------------------------------------------------------------------------
# Valid section citations (for guard validation)
# ---------------------------------------------------------------------------

VALID_SECTIONS: frozenset[str] = frozenset({
    "§1", "§1.1", "§1.2", "§1.3", "§1.4",
    "§2", "§2.1", "§2.2", "§2.3", "§2.4", "§2.5",
    "§3", "§3.1", "§3.2", "§3.3", "§3.4", "§3.5",
    "§4", "§4.1", "§4.2", "§4.3", "§4.4",
    "§5", "§5.1", "§5.2", "§5.3",
    "§6", "§6.1", "§6.2", "§6.3",
    "§7",
})


# ---------------------------------------------------------------------------
# §1 Route Deviation
# ---------------------------------------------------------------------------

def section_1_route_deviation(
    facts: Facts,
    per_km_rate: float = 0.80,
) -> PolicyResult:
    """§1: Route deviation refund.

    deviation = actual_route_distance − recommended_route_distance
    ≤ 1 km tolerance → no refund (§1.3)
    Traffic-justified (moderate/high on recommended route) + rider informed → no refund
    Traffic-justified + rider NOT informed → partial refund of excess distance fare
    NOT traffic-justified → full refund of excess distance fare
    Tolls, time, surge not refunded (§1.3)
    """
    excess_km = max(0.0, round(facts.distance_actual_km - facts.distance_optimal_km, 4))

    # Within 1 km tolerance → no refund regardless
    if excess_km <= RD_TOLERANCE_KM:
        return PolicyResult(
            applicable_clauses=["§1.3"],
            outcome="no_refund_within_tolerance",
            amount=0.0,
            computation=(
                f"excess_km={excess_km:.2f} ≤ {RD_TOLERANCE_KM:.1f} km tolerance (§1.3) — "
                "no refund."
            ),
        )

    # Compute excess distance fare refund
    refund = round(excess_km * per_km_rate, 2)

    if not facts.traffic_justified:
        # No traffic justification → full refund of excess distance fare
        comp = (
            f"excess_km={excess_km:.2f} > {RD_TOLERANCE_KM:.1f} km; "
            f"no traffic justification on recommended route (§1.4); "
            f"refund = {excess_km:.2f} km × S${per_km_rate:.2f}/km = S${refund:.2f} "
            f"(full excess distance fare, §1.3). "
            "Tolls, time, surge not refunded."
        )
        return PolicyResult(
            applicable_clauses=["§1.3", "§1.4"],
            outcome="refund_full_excess_distance",
            amount=refund,
            computation=comp,
        )

    # Traffic-justified deviation
    if facts.rider_informed_deviation:
        # Rider informed + traffic justified → no refund
        comp = (
            f"excess_km={excess_km:.2f} > {RD_TOLERANCE_KM:.1f} km; "
            f"traffic-justified (§1.4) and rider informed (§1.2) — "
            "no refund."
        )
        return PolicyResult(
            applicable_clauses=["§1.2", "§1.3", "§1.4"],
            outcome="no_refund_justified_informed",
            amount=0.0,
            computation=comp,
        )

    # Traffic-justified but rider NOT informed → partial refund of excess distance fare
    comp = (
        f"excess_km={excess_km:.2f} > {RD_TOLERANCE_KM:.1f} km; "
        f"traffic-justified (§1.4) but rider NOT informed (§1.2) — "
        f"refund = {excess_km:.2f} km × S${per_km_rate:.2f}/km = S${refund:.2f} "
        f"(partial excess distance fare, §1.3). "
        "Tolls, time, surge not refunded."
    )
    return PolicyResult(
        applicable_clauses=["§1.2", "§1.3", "§1.4"],
        outcome="refund_partial_excess_distance",
        amount=refund,
        computation=comp,
    )


# ---------------------------------------------------------------------------
# §2 No-Show & Cancellation
# ---------------------------------------------------------------------------

def section_2_no_show(
    facts: Facts,
    category: str,
) -> PolicyResult:
    """§2: No-show and cancellation.

    §2.1: free wait 5 min from later of scheduled pickup and arrival
    §2.2: S$5.00 cancellation fee to rider
    §2.3: driver must make ≥1 chat AND ≥1 call before cancelling;
          if not, fee voided
    §2.4: driver no-show → S$5.00 credit if no contact
    §2.5: 3+ no-show cancellations in 30d → flag account
    """
    # §2.5: flag account for review (an action, not a refund)
    actions: list[dict] = []
    if facts.rider_no_show_cancellations_30d >= NS_REPEAT_FLAG_THRESHOLD:
        actions.append({
            "action": "flag_account_review",
            "description": (
                f"Rider has {facts.rider_no_show_cancellations_30d} no-show "
                f"cancellations in 30d (≥ {NS_REPEAT_FLAG_THRESHOLD}, §2.5) — "
                "account flagged for review."
            ),
        })

    if category == "no_show_driver":
        # §2.4: driver no-show
        # If driver didn't show AND didn't attempt contact → S$5 credit
        if facts.driver_no_show:
            if facts.driver_chat_messages == 0 and facts.driver_call_attempts == 0:
                credit = NS_DRIVER_CREDIT
                comp = (
                    f"Driver no-show (§2.4): driver did not arrive and "
                    f"made 0 contact attempts — S${credit:.2f} goodwill credit. "
                    f"Cancellation fee voided."
                )
                return PolicyResult(
                    applicable_clauses=["§2.4"],
                    outcome="driver_no_show_credit",
                    amount=credit,
                    computation=comp,
                    actions=actions,
                )
            else:
                comp = (
                    f"Driver no-show (§2.4): driver did not arrive but "
                    f"made {facts.driver_chat_messages} chat + "
                    f"{facts.driver_call_attempts} calls — "
                    f"no credit (contact attempted). "
                    f"Cancellation fee voided."
                )
                return PolicyResult(
                    applicable_clauses=["§2.4"],
                    outcome="driver_no_show_no_credit",
                    amount=0.0,
                    computation=comp,
                    actions=actions,
                )
        # Driver did show → not a driver no-show
        comp = "Driver arrived — not a driver no-show (§2.4)."
        return PolicyResult(
            applicable_clauses=["§2.4"],
            outcome="no_driver_no_show",
            amount=0.0,
            computation=comp,
            actions=actions,
        )

    # §2.3: rider no-show — check driver's contact obligations
    if not facts.driver_made_contact:
        # Driver failed both-contact obligation → fee voided
        comp = (
            f"Driver contact obligation not met (§2.3): "
            f"{facts.driver_chat_messages} chat message(s), "
            f"{facts.driver_call_attempts} call(s) — "
            f"need ≥1 of each. Cancellation fee voided (§2.3)."
        )
        return PolicyResult(
            applicable_clauses=["§2.3"],
            outcome="cancellation_fee_voided",
            amount=0.0,
            computation=comp,
            actions=actions,
        )

    # §2.2: cancellation fee stands
    fee = NS_CANCELLATION_FEE
    comp = (
        f"Free wait {NS_FREE_WAIT_MIN:.0f} min expired (§2.1); "
        f"driver made contact: {facts.driver_chat_messages} chat + "
        f"{facts.driver_call_attempts} call (§2.3 met); "
        f"cancellation fee S${fee:.2f} upheld (§2.2)."
    )
    return PolicyResult(
        applicable_clauses=["§2.1", "§2.2", "§2.3"],
        outcome="cancellation_fee_upheld",
        amount=fee,
        computation=comp,
        actions=actions,
    )


# ---------------------------------------------------------------------------
# §3 Fare & Surge Pricing
# ---------------------------------------------------------------------------

def section_3_surge(facts: Facts) -> PolicyResult:
    """§3: Surge pricing refund.

    §3.1: upfront quote binding at quote-time multiplier
    §3.2: rider must be notified of surge increase
    §3.3: un-notified increase → full refund of surge differential
    §3.4: surge_differential = surgeable_subtotal × (trip_mult − quote_mult)
    §3.5: booking fee exempt
    """
    quote_mult = facts.quote_surge_multiplier
    trip_mult = facts.trip_start_surge_multiplier
    surgeable = facts.surgeable_subtotal

    # No surge increase → no refund
    if not facts.surge_increased:
        if facts.surge_decreased:
            comp = (
                f"Surge decreased {quote_mult}x → {trip_mult}x (§3.3) — "
                "lower fare applies automatically."
            )
        else:
            comp = (
                f"Surge unchanged at {quote_mult}x (§3.3) — "
                "quoted fare applies."
            )
        return PolicyResult(
            applicable_clauses=["§3.1", "§3.3"],
            outcome="no_surge_refund",
            amount=0.0,
            computation=comp,
        )

    # Surge increased
    if facts.rider_notified_surge:
        comp = (
            f"Surge increased {quote_mult}x → {trip_mult}x (§3.2); "
            f"rider notified and proceeded — no refund (§3.3)."
        )
        return PolicyResult(
            applicable_clauses=["§3.1", "§3.2", "§3.3"],
            outcome="no_surge_refund_notified",
            amount=0.0,
            computation=comp,
        )

    # Surge increased, rider NOT notified → full refund of differential
    differential = round(surgeable * (trip_mult - quote_mult), 2)
    comp = (
        f"Surge increased {quote_mult}x → {trip_mult}x (§3.2); "
        f"rider NOT notified — full refund of surge differential (§3.3): "
        f"surgeable_subtotal S${surgeable:.2f} × "
        f"({trip_mult} − {quote_mult}) = S${differential:.2f} (§3.4). "
        f"Booking fee exempt (§3.5)."
    )
    return PolicyResult(
        applicable_clauses=["§3.2", "§3.3", "§3.4", "§3.5"],
        outcome="refund_surge_differential",
        amount=differential,
        computation=comp,
    )


# ---------------------------------------------------------------------------
# §4 Property Damage
# ---------------------------------------------------------------------------

def section_4_property_damage(facts: Facts, filed_by: str = "driver") -> PolicyResult:
    """§4: Property damage.

    §4.2: verifiable photo (in-app camera, timestamp, 24h, linked to trip)
    §4.3: cleaning up to S$50 with photo; repair needs certified quote
    §4.4: rider dispute → human review
    """
    # §4.4: rider disputing a damage charge → always human
    if filed_by == "rider":
        return PolicyResult(
            applicable_clauses=["§4.4"],
            outcome="human_review_rider_dispute",
            amount=0.0,
            computation=(
                "Rider disputes the damage charge (§4.4) — "
                "routed to human review."
            ),
            routing="human",
            routing_reason="§4.4: rider dispute requires human review",
        )

    # Repair claim → always human (§4.3, §6.2)
    if facts.damage_type == "repair" or facts.has_certified_quote:
        return PolicyResult(
            applicable_clauses=["§4.3"],
            outcome="human_review_repair",
            amount=0.0,
            computation=(
                "Repair claim (§4.3) — requires certified workshop quote "
                f"within {PD_REPAIR_QUOTE_DAYS} days. Routed to human review."
            ),
            routing="human",
            routing_reason="§4.3: repair claim requires human review",
        )

    # Cleaning claim
    claim_amount = min(facts.damage_claim_amount, PD_CLEANING_MAX)

    # Check verifiability (§4.2)
    verifiable = (
        facts.photo_source == "in_app_camera"
        and facts.has_timestamp
        and facts.submitted_within_24h
        and facts.linked_to_trip
    )

    if not verifiable:
        return PolicyResult(
            applicable_clauses=["§4.2", "§4.4"],
            outcome="claim_rejected_insufficient_evidence",
            amount=0.0,
            computation=(
                "Damage evidence not verifiable (§4.2): "
                f"source={facts.photo_source}, "
                f"timestamp={facts.has_timestamp}, "
                f"24h={facts.submitted_within_24h}, "
                f"linked={facts.linked_to_trip}. "
                "Claim rejected (§4.4)."
            ),
        )

    # Verifiable cleaning claim ≤ S$50
    comp = (
        f"Cleaning claim verifiable (§4.2): in-app camera, timestamp, "
        f"within 24h, linked to trip. "
        f"Claim amount S${claim_amount:.2f} ≤ S${PD_CLEANING_MAX:.2f} (§4.3)."
    )
    return PolicyResult(
        applicable_clauses=["§4.2", "§4.3"],
        outcome="damage_claim_approved",
        amount=claim_amount,
        computation=comp,
    )


# ---------------------------------------------------------------------------
# §5 Safety Incident
# ---------------------------------------------------------------------------

def section_5_safety() -> PolicyResult:
    """§5: Safety incidents always escalate to human review."""
    return PolicyResult(
        applicable_clauses=["§5.1", "§5.2"],
        outcome="escalate_safety",
        amount=0.0,
        computation=(
            "Safety incident (§5.1) — automatically escalated to human review. "
            "AI agents cannot issue rulings on safety incidents (§5.2)."
        ),
        routing="human",
        routing_reason="§5.1: safety incident — always human review",
    )


# ---------------------------------------------------------------------------
# §6 Ruling Thresholds & Routing
# ---------------------------------------------------------------------------

def compute_dispute_value(category: str, facts: Facts, policy_amount: float) -> float:
    """§6.1: Compute the dispute value for routing.

    The dispute value is the amount at stake — what the rider is disputing.

    For fare/surge: actual fare minus upfront quote (the amount the rider
    was overcharged relative to the quote, regardless of the refund policy
    decides).
    For route deviation: actual fare minus upfront quote/estimate (the
    excess the rider is disputing).
    For no-show: the cancellation fee.
    For property damage: the claimed amount.
    """
    if category == "property_damage":
        return min(facts.damage_claim_amount, PD_CLEANING_MAX)
    if category in ("no_show_rider", "no_show_driver"):
        return NS_CANCELLATION_FEE
    # route_deviation, fare_error — use actual charged minus upfront quote
    # when available. This is the amount at stake, not the refund the policy
    # engine computed. A S$0 refund with S$60 at stake still routes to human.
    if facts.charged_total > 0 and facts.upfront_quote > 0:
        return max(0.0, round(facts.charged_total - facts.upfront_quote, 2))
    # Fallback when no upfront quote: use charged minus recomputed
    if facts.charged_total > 0 and facts.recomputed_fare > 0:
        return max(0.0, round(facts.charged_total - facts.recomputed_fare, 2))
    return policy_amount


def _numeric_condition_score(v: float, t: float) -> float:
    """Score for a numeric condition with value v and threshold t.

    Score = min(1.0, 0.6 + 2 × |v − t| / t).

    At the threshold the score is 0.6; the further past the threshold
    the value is, the closer to 1.0.
    """
    if t == 0:
        return 1.0
    return min(1.0, 0.6 + 2.0 * abs(v - t) / t)


def _case_deciding_facts(facts: Facts, category: str) -> dict:
    """Extract the structured deciding facts for a case, matching the
    precedent's deciding_facts schema."""
    df: dict[str, bool | str] = {}
    if category == "route_deviation":
        df["rider_informed"] = bool(facts.rider_informed_deviation)
        df["traffic_justified"] = bool(facts.traffic_justified)
        df["deviation_exceeds_tolerance"] = (
            facts.distance_actual_km - facts.distance_optimal_km > RD_TOLERANCE_KM
        )
    elif category in ("no_show_rider", "no_show_driver"):
        df["driver_made_contact"] = bool(facts.driver_made_contact)
        df["free_wait_expired"] = bool(facts.free_wait_expired or facts.wait_minutes_past_eta > 0)
        if category == "no_show_driver":
            df["driver_no_show"] = bool(facts.driver_no_show)
    elif category == "fare_error":
        df["surge_increased"] = bool(facts.surge_increased)
        df["rider_notified_surge"] = bool(facts.rider_notified_surge)
    elif category == "property_damage":
        df["photo_verifiable"] = (
            facts.photo_source == "in_app_camera"
            and facts.has_timestamp
            and facts.submitted_within_24h
            and facts.linked_to_trip
        )
        df["damage_type"] = facts.damage_type or "cleaning"
    elif category == "safety":
        df["safety_incident"] = True
    return df


def _facts_match(a: dict, b: dict) -> bool:
    """Check whether two deciding-facts dicts match on their shared keys."""
    shared = set(a.keys()) & set(b.keys())
    if not shared:
        return False
    return all(a[k] == b[k] for k in shared)


_RIDER_WIN_KEYWORDS = frozenset({"refund", "voided", "credit"})
_DRIVER_WIN_PREFIXES = ("no_refund", "fee_upheld", "upheld")


def _outcome_direction(outcome: str) -> str:
    """Classify an outcome as favouring 'rider' or 'driver'.

    Rider wins: refund, voided, credit.
    Driver wins: no_refund, upheld, fee_upheld.
    """
    ol = outcome.lower().replace("-", "_")
    for prefix in _DRIVER_WIN_PREFIXES:
        if prefix in ol:
            return "driver"
    if any(kw in ol for kw in _RIDER_WIN_KEYWORDS):
        return "rider"
    return "driver"


def compute_confidence(
    facts: Facts,
    category: str,
    policy_amount: float,
    has_precedent_match: bool = False,
    precedent_similarity: float = 0.0,
    outcome: str = "",
    retrieved_precedents: list | None = None,
) -> tuple[float, dict]:
    """§6.3: Confidence = weighted average of 4 factors.

    0.30 × evidence_completeness + 0.25 × policy_clarity
    + 0.25 × precedent_alignment + 0.20 × party_credibility

    Each factor is computed from real inputs and has a one-line reason.
    """
    # --- Factor 1: Evidence completeness ---
    # Checks that the key data fields needed for this category are present
    # (non-zero, non-default values that indicate real source data exists).
    evidence_checks: list[tuple[bool, str]] = []
    if category == "route_deviation":
        evidence_checks.append((
            facts.distance_actual_km > 0,
            "actual distance recorded",
        ))
        evidence_checks.append((
            facts.distance_optimal_km > 0,
            "optimal route distance recorded",
        ))
        evidence_checks.append((
            facts.deviation_point_ts is not None,
            "deviation point timestamp identified",
        ))
        evidence_checks.append((
            len(facts.traffic_incidents_on_route) > 0 or facts.traffic_justified is not None,
            "traffic data available",
        ))
        evidence_checks.append((
            facts.rider_informed_deviation is not None or len(facts.disagreements) > 0 or len(facts.agreements) > 0,
            "rider notification status determined",
        ))
    elif category in ("no_show_rider", "no_show_driver"):
        evidence_checks.append((
            facts.arrival_ts is not None,
            "driver arrival timestamp recorded",
        ))
        evidence_checks.append((
            facts.min_distance_to_pickup_m < float("inf"),
            "GPS proximity to pickup measured",
        ))
        evidence_checks.append((
            facts.driver_chat_messages > 0 or facts.driver_call_attempts > 0 or facts.contact_attempts > 0,
            "driver contact attempts recorded",
        ))
        evidence_checks.append((
            facts.wait_minutes_within_150m > 0 or facts.wait_minutes_past_eta > 0,
            "wait time measured",
        ))
    elif category == "fare_error":
        evidence_checks.append((
            facts.charged_total > 0,
            "actual fare recorded",
        ))
        evidence_checks.append((
            facts.upfront_quote > 0 or facts.recomputed_fare > 0,
            "quote or recomputed fare available",
        ))
        evidence_checks.append((
            facts.quote_surge_multiplier > 0,
            "quote surge multiplier recorded",
        ))
        evidence_checks.append((
            facts.trip_start_surge_multiplier > 0,
            "trip-start surge multiplier recorded",
        ))
    else:
        evidence_checks.append((
            facts.charged_total > 0 or policy_amount > 0,
            "dispute value or fare recorded",
        ))

    passed = sum(1 for ok, _ in evidence_checks if ok)
    evidence_completeness = passed / max(len(evidence_checks), 1)
    evidence_reason = "; ".join(
        r for ok, r in evidence_checks if not ok
    ) or "all key evidence fields present"

    # --- Factor 2: Policy clarity ---
    # Minimum over the conditions the applied rule depends on:
    # - numeric condition: min(1.0, 0.6 + 2 × |v − t| / t)
    # - boolean from system record: 1.0
    # - boolean from free text: 0.9
    condition_scores: list[tuple[str, float, str]] = []

    if category == "route_deviation":
        excess_km = max(0.0, facts.distance_actual_km - facts.distance_optimal_km)
        condition_scores.append((
            f"excess_km {excess_km:.2f} vs threshold {RD_TOLERANCE_KM:.1f}",
            _numeric_condition_score(excess_km, RD_TOLERANCE_KM),
            "numeric",
        ))
        if facts.traffic_justified is not None:
            source = "system" if facts.traffic_incidents_on_route else "inferred"
            score = 1.0 if source == "system" else 0.9
            condition_scores.append((
                f"traffic_justified={facts.traffic_justified} ({source} record)",
                score,
                source,
            ))
        # rider_informed is inferred from chat (free text) → 0.9
        condition_scores.append((
            f"rider_informed={facts.rider_informed_deviation} (inferred from chat)",
            0.9,
            "free_text",
        ))
    elif category in ("no_show_rider", "no_show_driver"):
        # Free wait time: numeric condition (wait vs 5-minute threshold)
        wait = facts.wait_minutes_past_eta
        condition_scores.append((
            f"wait {wait:.1f} min vs threshold {NS_FREE_WAIT_MIN:.0f} min",
            _numeric_condition_score(wait, NS_FREE_WAIT_MIN),
            "numeric",
        ))
        # Contact obligations: boolean from system records (chat log, call log)
        if category == "no_show_rider":
            condition_scores.append((
                f"driver_made_contact={facts.driver_made_contact} "
                f"(chat + call records)",
                1.0,
                "system",
            ))
    elif category == "fare_error":
        if facts.surge_increased:
            # Surge change: numeric (trip_mult - quote_mult vs 0 threshold)
            delta = facts.trip_start_surge_multiplier - facts.quote_surge_multiplier
            condition_scores.append((
                f"surge change {delta:.2f}x vs threshold 0.0",
                _numeric_condition_score(delta, 0.01),
                "numeric",
            ))
        # Rider notification: boolean from system record (surge log / app event)
        condition_scores.append((
            f"rider_notified_surge={facts.rider_notified_surge} (surge log)",
            1.0,
            "system",
        ))
    else:
        # Default: no specific conditions to evaluate
        condition_scores.append(("standard policy application", 1.0, "default"))

    clarity = min(s for _, s, _ in condition_scores) if condition_scores else 1.0
    clarity_reason = "; ".join(
        f"{desc} → {s:.2f}" for desc, s, _ in condition_scores
    )

    # --- Factor 3: Precedent alignment ---
    # Among the top 3 retrieved precedents in the same category:
    # - matching deciding_facts + same outcome → agrees
    # - matching deciding_facts + different outcome → conflict
    # - different deciding_facts → distinguished (neither helps nor hurts)
    case_df = _case_deciding_facts(facts, category)
    precedent_alignment = 0.85  # default: none agree, none conflict
    precedent_reason_parts: list[str] = []

    if retrieved_precedents:
        agrees: list[str] = []
        conflicts: list[str] = []
        distinguished: list[str] = []
        for pm in retrieved_precedents[:3]:
            # Only consider precedents in the same category
            if pm.category != category:
                continue
            prec_df = getattr(pm, "deciding_facts", {}) or {}
            if not prec_df:
                # No structured facts → distinguished
                distinguished.append(pm.case_id)
                continue
            if _facts_match(case_df, prec_df):
                if _outcome_direction(pm.outcome) == _outcome_direction(outcome):
                    agrees.append(pm.case_id)
                else:
                    conflicts.append(pm.case_id)
            else:
                distinguished.append(pm.case_id)

        if conflicts:
            precedent_alignment = 0.5
        elif agrees:
            precedent_alignment = 1.0
        elif distinguished:
            # All same-category precedents are distinguished:
            # no agreement, no conflict → neutral (1.0)
            precedent_alignment = 1.0
        else:
            # No same-category precedents retrieved at all
            precedent_alignment = 0.85

        for cid in agrees:
            precedent_reason_parts.append(
                f"{cid} agrees (matching facts, same outcome)"
            )
        for cid in conflicts:
            precedent_reason_parts.append(
                f"{cid} conflicts (matching facts, different outcome)"
            )
        for cid in distinguished:
            precedent_reason_parts.append(
                f"{cid} distinguished (different deciding facts)"
            )
        if not precedent_reason_parts:
            precedent_reason_parts.append("no same-category precedents retrieved")
    else:
        if has_precedent_match and precedent_similarity >= 0.4:
            precedent_alignment = 1.0
            precedent_reason_parts.append(
                f"strong precedent match (similarity {precedent_similarity:.2f})"
            )
        elif has_precedent_match:
            precedent_alignment = 0.85
            precedent_reason_parts.append(
                f"precedent match found (similarity {precedent_similarity:.2f})"
            )
        else:
            precedent_alignment = 0.85
            precedent_reason_parts.append("no matching precedent found")

    precedent_reason = "; ".join(precedent_reason_parts)

    # --- Factor 4: Party credibility ---
    # 1.0 unless the ruling favours, or rests on the uncorroborated word of,
    # a party with one of these signals; then 0.6 with the signal named:
    # - a fraud flag
    # - a statement contradicted by system records
    # - three or more upheld disputes of the same type in 90 days
    # Account age and rating alone never lower it.
    rider_wins = any(kw in outcome for kw in ("refund", "voided", "credit"))
    winning_party = "rider" if rider_wins else "driver"

    if winning_party == "rider":
        fraud = facts.rider_fraud_flags
        upheld = facts.rider_disputes_upheld_90d
        party = "rider"
    else:
        fraud = facts.driver_fraud_flags
        upheld = facts.driver_disputes_upheld_90d
        party = "driver"

    credibility = 1.0
    credibility_reason = f"winning party ({party}) has no adverse signal"

    if fraud > 0:
        credibility = 0.6
        credibility_reason = f"winning party ({party}) has fraud flag ({fraud})"
    elif upheld >= 3:
        credibility = 0.6
        credibility_reason = (
            f"winning party ({party}) has {upheld} upheld disputes of same type in 90d"
        )

    confidence = (
        W_EVIDENCE * evidence_completeness
        + W_CLARITY * clarity
        + W_PRECEDENT * precedent_alignment
        + W_CREDIBILITY * credibility
    )
    confidence = round(confidence, 4)

    factors = {
        "evidence_completeness": round(evidence_completeness, 4),
        "evidence_reason": evidence_reason,
        "policy_clarity": round(clarity, 4),
        "policy_clarity_reason": clarity_reason,
        "precedent_alignment": round(precedent_alignment, 4),
        "precedent_reason": precedent_reason,
        "party_credibility": round(credibility, 4),
        "credibility_reason": credibility_reason,
    }
    return confidence, factors


def section_6_routing(
    category: str,
    facts: Facts,
    policy_amount: float,
    confidence: float,
    confidence_factors: dict,
    has_fraud_flag: bool = False,
    has_precedent_match: bool = False,
) -> tuple[str, str]:
    """§6: Determine routing (ai_final vs human).

    §6.1: <S$10 AI final; S$10–50 needs 80%+ confidence; >S$50 human
    §6.2: safety always human; route deviation needs ≤2 km + 80% confidence;
          repair damage always human; no-show AI final
    §6.3: fraud flag ≥1 → confidence threshold raised
    Strictest rule wins.
    """
    dispute_value = compute_dispute_value(category, facts, policy_amount)
    reasons: list[str] = []

    # §6.3: fraud flag raises confidence threshold
    if has_fraud_flag:
        if confidence < RT_CONFIDENCE_95:
            reasons.append(
                f"§6.3: fraud flag ≥1, confidence {confidence:.2f} < "
                f"{RT_CONFIDENCE_95:.2f} — human review"
            )

    # §6.1: monetary thresholds
    if dispute_value > RT_10_TO_50:
        reasons.append(
            f"§6.1: dispute value S${dispute_value:.2f} > S${RT_10_TO_50:.2f} — human review"
        )
    elif RT_UNDER_10 < dispute_value <= RT_10_TO_50:
        if confidence < RT_CONFIDENCE_80:
            reasons.append(
                f"§6.1: dispute value S${dispute_value:.2f} in S$10–S$50 range, "
                f"confidence {confidence:.2f} < {RT_CONFIDENCE_80:.2f} — human review"
            )

    # §6.2: category-specific thresholds
    if category == "safety":
        reasons.append("§6.2: safety incident — always human review")
    elif category == "route_deviation":
        excess_km = max(0.0, facts.distance_actual_km - facts.distance_optimal_km)
        if excess_km > RD_DEVIATION_THRESHOLD_KM:
            reasons.append(
                f"§6.2: route deviation {excess_km:.1f} km > "
                f"{RD_DEVIATION_THRESHOLD_KM:.0f} km — human review"
            )
        if confidence < RT_CONFIDENCE_80:
            reasons.append(
                f"§6.2: route deviation confidence {confidence:.2f} < "
                f"{RT_CONFIDENCE_80:.2f} — human review"
            )
    elif category == "property_damage":
        if facts.damage_type == "repair" or facts.has_certified_quote:
            reasons.append("§6.2: repair damage claim — always human review")
        if facts.damage_claim_amount > PD_CLEANING_MAX:
            reasons.append(
                f"§6.2: damage claim S${facts.damage_claim_amount:.2f} > "
                f"S${PD_CLEANING_MAX:.2f} — human review"
            )

    # Strictest rule wins: any reason → human
    if reasons:
        return "human", "; ".join(reasons)

    return "ai_final", (
        f"§6.1: dispute value S${dispute_value:.2f} ≤ S${RT_UNDER_10:.0f}; "
        f"confidence {confidence:.2f} ≥ {RT_CONFIDENCE_80:.2f} — AI ruling final"
    )


# ---------------------------------------------------------------------------
# Top-level evaluate
# ---------------------------------------------------------------------------

def evaluate(
    dispute,
    trip,
    profiles: dict[str, Profile],
    facts: Facts,
    fare_params=None,
    *,
    has_precedent_match: bool = False,
    precedent_similarity: float = 0.0,
    retrieved_precedents: list | None = None,
) -> PolicyResult:
    """Apply the relevant v2.1 policy section(s) to a dispute.

    Returns a single ``PolicyResult`` with the outcome, amount, computation,
    and §6 routing metadata.

    For non-safety categories, the claimant's fraud flags are checked and
    added to the result's actions and routing consideration.
    """
    category = dispute.category
    fare_params = fare_params or trip.fare

    # §5: safety always escalates
    if category == "safety":
        result = section_5_safety()
        # Still compute routing for consistency
        result.dispute_value = 0.0
        result.confidence = 0.0
        result.confidence_factors = {}
        return result

    # Determine the claimant for fraud check
    claimant_id = trip.rider_id if dispute.filed_by == "rider" else trip.driver_id
    claimant_profile = profiles.get(claimant_id)

    # Apply the category-specific section
    if category == "route_deviation":
        per_km = getattr(fare_params, "per_km", 0.80)
        result = section_1_route_deviation(facts, per_km_rate=per_km)
    elif category in ("no_show_driver", "no_show_rider"):
        result = section_2_no_show(facts, category)
    elif category == "fare_error":
        result = section_3_surge(facts)
    elif category == "property_damage":
        result = section_4_property_damage(facts, filed_by=dispute.filed_by)
    elif category == "promo_code":
        # Uncovered by v2.1 → human
        result = PolicyResult(
            applicable_clauses=[],
            outcome="human_review_uncovered",
            amount=0.0,
            computation="Promo code disputes not covered by policy v2.1 — human review.",
            routing="human",
            routing_reason="not covered by policy v2.1",
        )
        result.dispute_value = 0.0
        result.confidence = 0.0
        result.confidence_factors = {}
        return result
    else:
        # Unknown category → human
        result = PolicyResult(
            applicable_clauses=[],
            outcome="human_review_uncovered",
            amount=0.0,
            computation=f"Category '{category}' not covered by policy v2.1 — human review.",
            routing="human",
            routing_reason="not covered by policy v2.1",
        )
        result.dispute_value = 0.0
        result.confidence = 0.0
        result.confidence_factors = {}
        return result

    # Check for safety indicators in claim/chat (§5.2)
    if facts.safety_indicators_in_claim or facts.safety_indicators_in_chat:
        result.routing = "human"
        result.routing_reason = "§5.2: safety indicators detected — human review"
        result.applicable_clauses = list(result.applicable_clauses) + ["§5.2"]
        result.dispute_value = compute_dispute_value(category, facts, result.amount)
        result.confidence = 0.0
        result.confidence_factors = {}
        return result

    # Check fraud flags (§6.3)
    has_fraud_flag = False
    if claimant_profile is not None:
        if claimant_profile.fraud_flags > 0:
            has_fraud_flag = True
            result.actions = list(result.actions) + [{
                "action": "fraud_flag_notice",
                "description": (
                    f"Claimant {claimant_id} has {claimant_profile.fraud_flags} "
                    f"fraud flag(s) — confidence threshold raised (§6.3)."
                ),
            }]
    # Also check the other party
    other_id = trip.driver_id if dispute.filed_by == "rider" else trip.rider_id
    other_profile = profiles.get(other_id)
    if other_profile is not None and other_profile.fraud_flags > 0:
        has_fraud_flag = True

    # §6: compute routing and confidence
    confidence, conf_factors = compute_confidence(
        facts, category, result.amount, has_precedent_match,
        precedent_similarity=precedent_similarity,
        outcome=result.outcome,
        retrieved_precedents=retrieved_precedents,
    )

    routing, routing_reason = section_6_routing(
        category, facts, result.amount, confidence, conf_factors,
        has_fraud_flag=has_fraud_flag,
        has_precedent_match=has_precedent_match,
    )

    result.routing = routing
    result.routing_reason = routing_reason
    result.confidence = confidence
    result.confidence_factors = conf_factors
    result.dispute_value = compute_dispute_value(category, facts, result.amount)

    # If the section already set routing=human (e.g. repair claim), keep it
    if result.routing == "human" and routing == "ai_final":
        # Section-level human routing takes precedence
        pass  # routing already "human"

    return result
