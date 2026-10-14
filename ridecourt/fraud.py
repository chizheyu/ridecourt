"""Deterministic fraud and bad-faith feature computation.

All features are computed from the evidence pack and profiles — never from
LLM text.  The LLM only explains the signals.

Features (per SPEC P-BF-1):
- dispute_rate: disputes_filed_90d / max(account_age_days/90, 1)
- upheld_rate: disputes_upheld_90d / disputes_filed_90d
- repeat_upheld: disputes_upheld_90d >= 3 (P-BF-1 threshold)
- gps_contradiction: for no-show cases, min_distance_to_pickup_m > 150 m
  while the claimant asserts the driver came / didn't come
- prior_flags: from profile (e.g. excessive_disputes, gps_anomaly)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .data import EvidencePack, Facts, Profile


RiskLevel = Literal["low", "medium", "high"]


@dataclass
class FraudSignal:
    """A single bad-faith signal with deterministic evidence citation."""

    text: str
    cites: list[str] = field(default_factory=list)
    feature: str = ""  # machine-readable feature name


@dataclass
class FraudAssessment:
    """Deterministic fraud assessment — the LLM only explains these."""

    risk: RiskLevel
    signals: list[FraudSignal]
    claimant_id: str = ""


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------


def _find_history_items(
    pack: EvidencePack, party: str
) -> list[EvidencePack]:
    """Return evidence items of kind history_profile for the given party."""
    return [
        i for i in pack.by_kind("history_profile")
        if i.data.get("party") == party
    ]


def _find_arrival_item(pack: EvidencePack) -> list:
    """Return arrival_check evidence items."""
    return pack.by_kind("arrival_check")


# ---------------------------------------------------------------------------
# Main entry: compute_fraud_signals
# ---------------------------------------------------------------------------

# Thresholds
RISK_UPHELD_THRESHOLD = 3  # P-BF-1: 3+ upheld disputes in 90d
RISK_HIGH_UPHELD = 5
RISK_GPS_CONTRADICTION_M = 150.0  # same as NS_PICKUP_RADIUS_M
RISK_DISPUTE_RATE_HIGH = 4.0  # > 4 disputes per 90-day-window = high


def compute_fraud_signals(
    pack: EvidencePack,
    facts: Facts,
    profiles: dict[str, Profile],
    claimant_id: str,
    claimant_party: str,
    claim_text: str = "",
    category: str = "",
) -> FraudAssessment:
    """Compute deterministic fraud signals for the claimant.

    Parameters
    ----------
    pack
        The evidence pack (used for citation IDs).
    facts
        The Facts object (has min_distance_to_pickup_m etc).
    profiles
        Profile dict from the dataset.
    claimant_id
        The rider_id or driver_id of the filing party.
    claimant_party
        ``"rider"`` or ``"driver"``.
    claim_text
        The dispute claim text (used for GPS contradiction logic).
    category
        The dispute category.
    """
    signals: list[FraudSignal] = []
    risk_score = 0  # 0=low, 1=medium, 2+=high

    profile = profiles.get(claimant_id)

    # Find the history_profile evidence item for this party so we can cite it.
    hist_items = _find_history_items(pack, claimant_party)
    hist_eid = hist_items[0].id if hist_items else "E1"

    # --- Feature 1: repeat upheld disputes (P-BF-1) ---
    if profile is not None:
        upheld = profile.disputes_upheld_90d
        filed = profile.disputes_filed_90d

        if upheld >= RISK_UPHELD_THRESHOLD:
            if upheld >= RISK_HIGH_UPHELD:
                risk_score += 2
            else:
                risk_score += 1
            signals.append(FraudSignal(
                text=(
                    f"{claimant_id} has {upheld} upheld disputes in 90d "
                    f"(≥ {RISK_UPHELD_THRESHOLD} threshold, §6.3) — "
                    "credibility lowered."
                ),
                cites=[hist_eid, "§6.3"],
                feature="repeat_upheld",
            ))

        # --- Feature 2: high dispute rate ---
        age_windows = max(profile.account_age_days / 90.0, 1.0)
        dispute_rate = filed / age_windows
        if dispute_rate > RISK_DISPUTE_RATE_HIGH and filed >= 3:
            risk_score += 1
            signals.append(FraudSignal(
                text=(
                    f"{claimant_id} has a high dispute rate "
                    f"({dispute_rate:.1f} per 90-day window, "
                    f"{filed} filed in 90d)."
                ),
                cites=[hist_eid],
                feature="high_dispute_rate",
            ))

        # --- Feature 3: prior flags (including fraud flags) ---
        if profile.prior_flags:
            flags_str = ", ".join(profile.prior_flags)
            risk_score += 1
            signals.append(FraudSignal(
                text=(
                    f"{claimant_id} has excessive_disputes / prior flags: {flags_str}."
                ),
                cites=[hist_eid, "§6.3"],
                feature="prior_flags",
            ))

        # --- Feature 3b: fraud flags (§6.3: raises confidence threshold) ---
        if profile.fraud_flags > 0:
            risk_score += 2
            signals.append(FraudSignal(
                text=(
                    f"{claimant_id} has {profile.fraud_flags} fraud flag(s) "
                    f"(§6.3) — confidence threshold raised."
                ),
                cites=[hist_eid, "§6.3"],
                feature="fraud_flags",
            ))

    # --- Feature 4: GPS contradicting the claim ---
    # For no-show cases: if the driver claims the rider didn't show but
    # GPS shows the driver was far away, or vice versa.
    arrival_items = _find_arrival_item(pack)
    arrival_eid = arrival_items[0].id if arrival_items else "E1"
    min_dist = facts.min_distance_to_pickup_m

    if category in ("no_show_driver", "no_show_rider") and min_dist != float("inf"):
        if min_dist > RISK_GPS_CONTRADICTION_M:
            # GPS shows the driver was never within 150 m of pickup.
            # If the claimant is the driver claiming no-show by rider,
            # this contradicts their own claim.
            if claimant_party == "driver":
                risk_score += 2
                signals.append(FraudSignal(
                    text=(
                        f"GPS shows driver was {min_dist:.0f} m from pickup "
                        f"(> {RISK_GPS_CONTRADICTION_M:.0f} m), contradicting "
                        "the driver's no-show claim."
                    ),
                    cites=[arrival_eid, "§2.4"],
                    feature="gps_contradiction",
                ))
            elif claimant_party == "rider":
                # Rider claims driver no-show — GPS confirms. Not a
                # contradiction, but note the evidence.
                pass

    # --- Determine risk level ---
    if risk_score >= 3:
        risk: RiskLevel = "high"
    elif risk_score >= 1:
        risk = "medium"
    else:
        risk = "low"

    return FraudAssessment(
        risk=risk,
        signals=signals,
        claimant_id=claimant_id,
    )
