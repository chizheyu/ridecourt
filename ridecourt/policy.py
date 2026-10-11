"""Policy engine — pure functions, one per clause, plus ``evaluate``.

Numbers come from the ``Facts`` object computed by ``ridecourt.tools``.
The functions return a ``PolicyResult`` with the applicable clauses, outcome,
refund amount, and a human-readable computation string.
"""

from __future__ import annotations

from typing import Sequence

from .data import Facts, Profile, PolicyResult

# ---------------------------------------------------------------------------
# Thresholds (from SPEC section 5)
# ---------------------------------------------------------------------------

RD_DEVIATION_PCT = 0.15  # 15 %
RD_DEVIATION_KM = 1.5  # 1.5 km
NS_PICKUP_RADIUS_M = 150.0  # 150 m
NS_WAIT_MIN = 5.0  # 5 minutes
FARE_TOLERANCE = 0.50  # S$0.50
NS_DRIVER_CREDIT = 3.0  # S$3 goodwill credit
BF_REPEAT_THRESHOLD = 3  # 3 upheld disputes same type in 90d

# ---------------------------------------------------------------------------
# Individual clause functions
# ---------------------------------------------------------------------------

def clause_rd_1(facts: Facts) -> PolicyResult:
    """P-RD-1: route deviation refund."""
    extra_km = max(0.0, facts.distance_actual_km - facts.distance_optimal_km)
    extra_min = max(0.0, facts.duration_actual_min - facts.duration_optimal_min)
    pct_over = (
        extra_km / facts.distance_optimal_km
        if facts.distance_optimal_km > 0
        else 0.0
    )

    exceeds_pct = pct_over > RD_DEVIATION_PCT
    exceeds_km = extra_km > RD_DEVIATION_KM

    # Even if the numbers say "deviation", rider-requested change or a
    # documented road closure means P-RD-1 does not apply (handled by P-RD-2
    # and the road-closure logic in evaluate()).
    if not exceeds_pct or not exceeds_km:
        return PolicyResult(
            applicable_clauses=["P-RD-1"],
            outcome="no_deviation",
            amount=0.0,
            computation=(
                f"extra_km={extra_km:.2f} ({pct_over:.1%} over), "
                f"threshold {RD_DEVIATION_PCT:.0%}/{RD_DEVIATION_KM} km — "
                "deviation not significant."
            ),
        )

    return PolicyResult(
        applicable_clauses=["P-RD-1"],
        outcome="no_refund",  # evaluate() decides the actual refund
        amount=0.0,
        computation=(
            f"extra_km={extra_km:.2f} ({pct_over:.1%} over), "
            f"extra_min={extra_min:.1f}; deviation significant."
        ),
    )


def clause_rd_2(facts: Facts) -> PolicyResult:
    """P-RD-2: rider-requested stops / route changes are not deviations."""
    if facts.rider_requested_change:
        return PolicyResult(
            applicable_clauses=["P-RD-2"],
            outcome="requested_change",
            amount=0.0,
            computation="Rider requested a stop/route change in chat — no deviation.",
        )
    return PolicyResult(
        applicable_clauses=["P-RD-2"],
        outcome="not_applicable",
        amount=0.0,
        computation="No rider-requested change detected.",
    )


def clause_ns_1(facts: Facts) -> PolicyResult:
    """P-NS-1: driver no-show."""
    too_far = facts.min_distance_to_pickup_m > NS_PICKUP_RADIUS_M
    waited = facts.wait_minutes_past_eta > NS_WAIT_MIN

    if too_far or waited:
        # refund cancellation fee + S$3 credit
        # The cancellation fee is known from trip.fare, but facts carries
        # charged_total as the rider's total. For no-show, the "charged_total"
        # is the cancellation fee the rider was charged.
        cancellation_fee = max(facts.charged_total, 0.0)
        refund = round(cancellation_fee + NS_DRIVER_CREDIT, 2)
        reason = (
            f"driver min distance {facts.min_distance_to_pickup_m:.0f} m "
            f"> {NS_PICKUP_RADIUS_M:.0f} m"
            if too_far
            else f"rider waited {facts.wait_minutes_past_eta:.1f} min "
            f"> {NS_WAIT_MIN:.0f} min"
        )
        return PolicyResult(
            applicable_clauses=["P-NS-1"],
            outcome="refund_and_credit",
            amount=refund,
            computation=(
                f"{reason}; refund cancellation fee S${cancellation_fee:.2f} "
                f"+ credit S${NS_DRIVER_CREDIT:.2f} = S${refund:.2f}"
            ),
        )
    return PolicyResult(
        applicable_clauses=["P-NS-1"],
        outcome="no_show_not_proven",
        amount=0.0,
        computation=(
            f"driver within {NS_PICKUP_RADIUS_M:.0f} m "
            f"(min {facts.min_distance_to_pickup_m:.0f} m), "
            f"wait {facts.wait_minutes_past_eta:.1f} min "
            f"≤ {NS_WAIT_MIN:.0f} min — driver did show."
        ),
    )


def clause_ns_2(facts: Facts) -> PolicyResult:
    """P-NS-2: rider no-show / cancellation fee stands."""
    within = facts.min_distance_to_pickup_m <= NS_PICKUP_RADIUS_M
    waited = facts.wait_minutes_past_eta >= NS_WAIT_MIN
    contacted = facts.contact_attempts >= 1

    if within and waited and contacted:
        cancellation_fee = facts.charged_total
        return PolicyResult(
            applicable_clauses=["P-NS-2"],
            outcome="cancellation_fee_stands",
            amount=round(cancellation_fee, 2),
            computation=(
                f"driver within {NS_PICKUP_RADIUS_M:.0f} m "
                f"(min {facts.min_distance_to_pickup_m:.0f} m), "
                f"waited {facts.wait_minutes_past_eta:.1f} min "
                f"≥ {NS_WAIT_MIN:.0f} min, "
                f"contact attempts {facts.contact_attempts} ≥ 1 — "
                f"cancellation fee S${cancellation_fee:.2f} stands."
            ),
        )
    return PolicyResult(
        applicable_clauses=["P-NS-2"],
        outcome="cancellation_fee_invalid",
        amount=0.0,
        computation=(
            f"within={within}, waited={waited} ({facts.wait_minutes_past_eta:.1f} min), "
            f"contacts={facts.contact_attempts} — conditions not met."
        ),
    )


def clause_fa_1(facts: Facts) -> PolicyResult:
    """P-FA-1: fare / surge / promo errors."""
    # If promo is invalid, it should not be applied → recompute without it.
    recomputed = facts.recomputed_fare
    if not facts.promo_valid_at_booking:
        # Invalid promo means the discount should be removed; the recomputed
        # fare already reflects the correct value (tools handle this).
        recomputed = facts.recomputed_fare  # tools already computed correctly

    diff = round(facts.charged_total - recomputed, 2)
    abs_diff = abs(diff)

    if abs_diff > FARE_TOLERANCE:
        if diff > 0:
            # rider was overcharged
            outcome = "refund_overcharge"
            computation = (
                f"charged S${facts.charged_total:.2f} − recomputed "
                f"S${recomputed:.2f} = S${diff:.2f} > S${FARE_TOLERANCE:.2f}; "
                "refund the difference."
            )
        else:
            # undercharged — note but don't collect
            outcome = "undercharge_noted"
            computation = (
                f"recomputed S${recomputed:.2f} > charged "
                f"S${facts.charged_total:.2f} by S${-diff:.2f}; "
                "undercharge noted, not collected."
            )
        return PolicyResult(
            applicable_clauses=["P-FA-1"],
            outcome=outcome,
            amount=round(max(diff, 0.0), 2),
            computation=computation,
        )

    return PolicyResult(
        applicable_clauses=["P-FA-1"],
        outcome="fare_correct",
        amount=0.0,
        computation=(
            f"charged S${facts.charged_total:.2f} vs recomputed "
            f"S${recomputed:.2f} (diff S${diff:.2f} ≤ "
            f"S${FARE_TOLERANCE:.2f}) — fare correct."
        ),
    )


def clause_sf_1(facts: Facts) -> PolicyResult:
    """P-SF-1: safety incidents always escalate."""
    return PolicyResult(
        applicable_clauses=["P-SF-1"],
        outcome="escalate",
        amount=0.0,
        computation="Safety incident — escalate immediately to human reviewer.",
    )


def clause_bf_1(profiles: dict[str, Profile], claimant_id: str) -> PolicyResult:
    """P-BF-1: bad faith / credibility."""
    profile = profiles.get(claimant_id)
    if profile is None:
        return PolicyResult(
            applicable_clauses=["P-BF-1"],
            outcome="profile_not_found",
            amount=0.0,
            computation=f"No profile for {claimant_id}.",
        )
    if profile.disputes_upheld_90d >= BF_REPEAT_THRESHOLD:
        return PolicyResult(
            applicable_clauses=["P-BF-1"],
            outcome="bad_faith_flagged",
            amount=0.0,
            computation=(
                f"{claimant_id} has {profile.disputes_upheld_90d} upheld "
                f"disputes in 90d (≥ {BF_REPEAT_THRESHOLD}) — credibility lowered."
            ),
        )
    return PolicyResult(
        applicable_clauses=["P-BF-1"],
        outcome="no_bad_faith",
        amount=0.0,
        computation=(
            f"{claimant_id} has {profile.disputes_upheld_90d} upheld disputes "
            f"in 90d (< {BF_REPEAT_THRESHOLD}) — no bad-faith flag."
        ),
    )


# ---------------------------------------------------------------------------
# Refund computation for route deviation
# ---------------------------------------------------------------------------

def _route_deviation_refund(facts: Facts, fare_params) -> float:
    """Compute the route deviation refund amount.

    refund = (extra_km × per_km + extra_min × per_min) × surge_multiplier
    rounded to cents.

    Both ``extra_km`` and ``extra_min`` are clamped to ``max(0, …)`` so a
    detour that is shorter in time than the optimal route does not produce
    a negative contribution (P-RD-1).

    *fare_params* is a simple namespace with ``per_km``, ``per_min``, and
    ``surge_multiplier``. We pass the trip's Fare object in practice.
    """
    extra_km = max(0.0, facts.distance_actual_km - facts.distance_optimal_km)
    extra_min = max(0.0, facts.duration_actual_min - facts.duration_optimal_min)
    if extra_km <= 0 and extra_min <= 0:
        return 0.0
    per_km = getattr(fare_params, "per_km", 0.75)
    per_min = getattr(fare_params, "per_min", 0.25)
    surge = getattr(fare_params, "surge_multiplier", 1.0)
    raw = (extra_km * per_km + extra_min * per_min) * surge
    return round(raw, 2)


# ---------------------------------------------------------------------------
# Top-level evaluate
# ---------------------------------------------------------------------------

def evaluate(
    dispute,
    trip,
    profiles: dict[str, Profile],
    facts: Facts,
    fare_params=None,
) -> PolicyResult:
    """Apply the relevant policy clause(s) to a dispute.

    Returns a single ``PolicyResult``.  For route deviation the refund
    formula from P-RD-1 is computed here (the clause function only checks
    whether the deviation is significant).
    """
    category = dispute.category
    fare_params = fare_params or trip.fare

    if category == "safety":
        return clause_sf_1(facts)

    if category == "route_deviation":
        # Check exclusions first (P-RD-2, road closure).
        if facts.rider_requested_change:
            res = clause_rd_2(facts)
            res.outcome = "no_refund_rider_requested"
            res.computation = (
                "P-RD-2 applies: rider requested a route change — "
                "no deviation refund."
            )
            return res
        if facts.road_closure:
            return PolicyResult(
                applicable_clauses=["P-RD-1"],
                outcome="no_refund_road_closure",
                amount=0.0,
                computation=(
                    "Road closure documented in chat — deviation justified, "
                    "no refund."
                ),
            )

        # Standard deviation check
        check = clause_rd_1(facts)
        if check.outcome == "no_deviation":
            return check

        # Deviation is significant — compute refund
        refund = _route_deviation_refund(facts, fare_params)
        extra_km = max(0.0, facts.distance_actual_km - facts.distance_optimal_km)
        extra_min = max(0.0, facts.extra_min)
        per_km = fare_params.per_km
        per_min = fare_params.per_min
        surge = fare_params.surge_multiplier
        comp = (
            f"extra_km=max(0,{facts.distance_actual_km - facts.distance_optimal_km:.2f})"
            f"={extra_km:.2f} × S${per_km:.2f}/km = S${extra_km * per_km:.2f}; "
            f"extra_min=max(0,{facts.extra_min:.1f})={extra_min:.1f} "
            f"× S${per_min:.2f}/min = S${extra_min * per_min:.2f}; "
            f"sum × surge {surge:.1f} = S${refund:.2f}"
        )
        return PolicyResult(
            applicable_clauses=["P-RD-1"],
            outcome="refund_route_deviation",
            amount=refund,
            computation=comp,
        )

    if category == "no_show_driver":
        return clause_ns_1(facts)

    if category == "no_show_rider":
        return clause_ns_2(facts)

    if category == "fare_error":
        return clause_fa_1(facts)

    # Fallback
    return PolicyResult(
        applicable_clauses=[],
        outcome="unknown_category",
        amount=0.0,
        computation=f"Unknown category: {category}",
    )
