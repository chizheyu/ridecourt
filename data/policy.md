# RideCourt Policy v2.1

Our summary of the Ryde Dispute Resolution Policy v2.1, organised by the
original section numbers.  Agents cite sections as `§1.3`, `§2.3`, `§6.2`
and so on — every section that exists below is a valid citation.  The guard
rejects any `§` citation whose section does not exist in this document.
All monetary amounts are in SGD.

---

## §1 Route Deviation

### §1.1 General Principle

A driver may reroute from the GPS-recommended path to respond to real-time
traffic, road closures, or congestion.  The rider must be informed of any
deviation.

### §1.2 Notification Requirement

The driver must inform the rider of a route deviation via in-app chat or
verbally (with the trip call recording as evidence) **before or at the
point of deviation**.  A brief reason ("traffic on PIE, taking alternate
route") is sufficient.

### §1.3 Refund Eligibility Matrix

| Scenario | Rider Informed? | Deviation > 1 km? | Refund |
|:--|:--|:--|:--|
| Driver reroutes for traffic, rider informed and agrees | Yes | Yes | No refund |
| Driver reroutes for traffic, rider **not** informed | No | Yes | Partial refund of excess distance fare |
| Driver reroutes for traffic, rider not informed | No | No (≤ 1 km) | No refund (within tolerance) |
| Driver reroutes **without** traffic justification | N/A | Yes | Full refund of excess distance fare |
| Driver reroutes without justification, deviation ≤ 1 km | N/A | No | No refund (within tolerance) |

**Refund amount** = excess distance × per-km distance rate.
Tolls, time, and surge are **not** refunded.  Only the excess distance fare
is refunded.

### §1.4 Evidence Requirements

- Traffic justification is verified via historical traffic API data for the
  route segment at the trip time.
- Route deviation distance = `actual_route_distance − recommended_route_distance`.
- If the driver's in-app chat or call recording confirms rider
  acknowledgment, the deviation is considered agreed.

---

## §2 No-Show & Cancellation

### §2.1 Free Wait Time

The driver must wait at the pickup location for a **free wait time of 5
minutes**, counted from the later of the scheduled pickup time and the
driver's arrival.  After 5 minutes the driver may cancel as a "rider
no-show".

### §2.2 Cancellation Fee

A **no-show cancellation fee of S$5.00** is charged to the rider when the
driver cancels after the free wait period.

### §2.3 Driver Contact Obligation

Before cancelling for no-show, the driver must attempt to contact the rider
with **at least 1 in-app chat message AND at least 1 phone call** (via the
in-app calling system).  If the driver fails to meet **both** contact
obligations, the cancellation fee is **voided** and the rider is not
charged.

### §2.4 Driver No-Show

If the driver fails to arrive at the pickup location by the scheduled
pickup time + 5 minutes, the rider may cancel at no charge.  If the driver
fails to arrive **and** fails to attempt contact, the rider is eligible
for a **S$5.00 goodwill credit**.

### §2.5 Repeat No-Show Flagging

If a rider accumulates **3 or more no-show cancellations within a 30-day
rolling window**, the rider's account is **flagged for review** (an action
on the ruling, not a refund).  Flagged accounts are subject to mandatory
upfront payment and temporary suspension pending Trust & Safety review.

---

## §3 Fare & Surge Pricing

### §3.1 Upfront Quote Binding

The upfront quote presented at booking is **binding at the surge multiplier
in effect at quote time** (typically 1.0x in normal conditions).  The quoted
fare includes base fare, distance fare, time fare, booking fee, and any
applicable surge.

### §3.2 Surge Change Notification

If the surge multiplier **increases** between quote and trip start, the
rider must be notified via an **in-app notification** before or at pickup.
The notification must state the new surge multiplier, the revised estimated
fare, and an option to cancel without penalty.

### §3.3 Refund Eligibility

| Scenario | Rider Notified? | Refund |
|:--|:--|:--|
| Surge increases, rider notified, rider proceeds | Yes | No refund — higher fare applies |
| Surge increases, rider **not** notified | No | **Full refund of the surge differential** |
| Surge decreases between quote and trip start | N/A | Lower fare applies automatically |
| Surge unchanged | N/A | Quoted fare applies |

### §3.4 Surge Differential Calculation

```
surge_differential = surgeable_subtotal × (trip_start_multiplier − quote_multiplier)
```

Only the difference attributable to a surge multiplier change is
refundable.  The booking fee is exempt.

### §3.5 Surge-Exempt Components

The **booking fee** is exempt from surge pricing and remains fixed
regardless of multiplier.  All other components (base fare, distance fare,
time fare) are subject to the surge multiplier.

---

## §4 Property Damage

### §4.1 Driver Claim Rights

Drivers may file claims for cleaning or repair fees resulting from
rider-caused damage, including spills, vomiting, food/beverage stains,
physical damage to interior, or damage to exterior caused by rider
behavior during boarding/alighting.

### §4.2 Evidence Requirements

| Requirement | Details |
|:--|:--|
| Photo evidence | Must be submitted within 24 hours of trip completion |
| Timestamps | Photos must be automatically timestamped by the in-app camera |
| Verifiability | Photos must be taken through the Ryde driver app's built-in camera, not external uploads |
| Trip association | Damage must be linked to the specific trip via GPS telemetry and trip timestamp |

### §4.3 Fee Structure

| Damage Type | Maximum Fee | Conditions |
|:--|:--|:--|
| Cleaning fee | Up to **S$50.00** | Photo evidence of soiling/stain; no external quote required |
| Repair fee | Cost of repair | Requires a written quote from a certified workshop submitted within 7 days |

### §4.4 Dispute Handling

If the rider disputes the damage claim, the dispute is routed to **human
review**.  The driver's photo evidence and the rider's account are both
reviewed.  If evidence is insufficient (no timestamp, external photo, or
damage not linked to trip), the claim is **rejected**.

---

## §5 Safety Incident

### §5.1 Automatic Escalation

Any dispute flagged as safety-related is **automatically escalated to human
review**.  Safety-related disputes include reckless/dangerous driving,
physical or verbal harassment, assault or threat of assault, inappropriate
behavior, traffic accidents, or a rider/driver feeling unsafe.

### §5.2 AI Agent Limitations

AI agents may gather evidence, summarize accounts, and prepare case files
for human reviewers.  AI agents **cannot** issue rulings, determine fault,
or recommend resolutions on safety incidents.  AI agents must flag any
dispute that contains safety-related keywords or indicators for immediate
human escalation, regardless of the dispute type selected by the filer.

### §5.3 Interim Measures

Upon escalation, the system may apply interim measures pending human
review: temporary suspension of the accused party's account, and prevention
of new trip acceptance (for drivers) or booking (for riders).  These
measures are lifted if the human reviewer determines no safety violation
occurred.

---

## §6 Ruling Thresholds

### §6.1 Monetary Thresholds

| Dispute Value | Ruling Path |
|:--|:--|
| Under S$10 | AI agent ruling is **final** |
| S$10 – S$50 | AI agent ruling with **80%+ confidence**; if confidence < 80%, escalate |
| Above S$50 | **Always human review** |

### §6.2 Category Thresholds

| Dispute Category | Ruling Path |
|:--|:--|
| Safety incident | **Always human review** (§5) |
| Fare dispute < S$10 | AI agent ruling is final |
| Fare dispute S$10 – S$50 | AI agent ruling if confidence ≥ 80%, else escalate |
| Fare dispute > S$50 | Always human review |
| Route deviation | AI agent ruling if confidence ≥ 80% **and** deviation ≤ 2 km, else escalate |
| Property damage (cleaning ≤ S$50) | AI agent ruling if photo evidence is verifiable |
| Property damage (repair > S$50) | Always human review |
| No-show / cancellation | AI agent ruling is final (fee is fixed at S$5.00) |

### §6.3 Confidence Calibration

Agent confidence is calculated from four factors: evidence completeness,
policy clarity, precedent alignment, and party credibility.  If any party
has a fraud flag ≥ 1, the confidence threshold is raised to **95%**
regardless of dispute value.

---

## §7 Precedent Rulings

Past rulings on similar cases are referenced for consistency.  Agents
search precedent rulings by dispute type and key facts and cite the
relevant precedent in their ruling rationale.

- **DISP-2026-0089** — Route deviation without notification: driver had
  traffic justification but failed to notify rider.  Partial refund of
  excess distance fare.  S$3.20.
- **DISP-2026-0142** — Surge pricing without rider notification: surge
  increased between quote and trip start, rider not notified.  Full refund
  of surge differential.  S$3.20.
- **DISP-2026-0211** — No-show cancellation without contact attempt:
  driver cancelled without sending any chat message or making any call.
  Cancellation fee voided.  S$5.00.

Ryde's precedents keep their IDs (`DISP-YYYY-NNNN`); our seed precedents
use `PC-YYYY-NNN`.  Both forms are valid citations.

---

## Interpretations

This section lists every reading this task states where v2.1 is ambiguous,
plus interpretations we add:

1. **"Excess distance fare" (§1.3)**: v2.1 says "partial refund of excess
   distance fare" but does not state the per-km rate explicitly.  We
   interpret the refund as `excess_km × per_km_distance_rate` using the
   fare's per-km rate from the trip's fare breakdown.  Tolls, time, and
   surge are not refunded (the sample's `recommended_ruling.breakdown`
   confirms this).

2. **Deviation tolerance (§1.3)**: the matrix says "Deviation > 1 km" but
   does not specify whether the boundary is strictly greater.  We treat
   exactly 1 km as within tolerance (no refund), consistent with the
   "≤ 1 km" wording in the "No" row.

3. **"Rider informed" (§1.2 / §1.3)**: v2.1 says the driver must inform
   the rider "before or at the point of deviation" but does not define
   "point of deviation" precisely.  We define it as the first GPS point
   that is off the recommended route.  The rider counts as informed only
   if the driver gave the route change **and a reason** in chat or by call
   before or at that point, or the rider asked for the change.  A rider's
   inquiry ("are we going the right way?") after the deviation does not
   count as being informed; a dismissive response without a reason does
   not count either.

4. **Traffic justification (§1.4)**: v2.1 says "traffic justification is
   verified via historical traffic API data" but does not define the
   severity threshold.  We interpret "justified" as a traffic incident of
   **moderate or high severity** that was active on the recommended route
   during the trip.  Low-severity or "traffic flow" incidents do not
   justify a deviation.

5. **Free wait time start (§2.1)**: v2.1 says "from the scheduled pickup
   time" but the sample data shows a driver arriving before the scheduled
   time.  We count the free wait from the **later of** the scheduled pickup
   time and the driver's arrival, so an early-arriving driver does not
   start the clock before the rider's scheduled time.

6. **No-show contact obligation (§2.3)**: v2.1 requires "at least 1
   in-app chat message AND at least 1 phone call".  We treat `type: "call"`
   chat entries as call attempts (the second chat-log shape).  The system
   message announcing cancellation is not a contact attempt.

7. **Driver no-show credit (§2.4)**: v2.1 says the S$5.00 goodwill credit
   applies when the driver "fails to arrive **and** fails to attempt
   contact".  We treat "fails to attempt contact" as zero chat messages and
   zero calls from the driver.  The credit is in addition to the fee
   refund.

8. **Surge differential (§3.4)**: v2.1's formula says
   `surge_differential = actual_fare − upfront_quote` but §3.5 clarifies
   the booking fee is exempt.  We compute the differential as
   `surgeable_subtotal × (trip_start_multiplier − quote_multiplier)`,
   where `surgeable_subtotal` excludes the booking fee.  This matches
   DISP-003: S$9.00 × (1.5 − 1.0) = S$4.50.

9. **Surge notification (§3.2)**: v2.1 says the rider "MUST be notified
   via an in-app notification before or at the point of pickup".  We
   interpret "notified" as `surge_pricing_log.rider_notified == true`.
   Notified increases are not refunded; un-notified increases are refunded
   in full.

10. **Property damage — "verifiable" (§4.2)**: v2.1 says photos must be
    "taken through the Ryde driver app's built-in camera, not external
    uploads".  We interpret this as: the photo source must be the in-app
    camera with an automatic timestamp, submitted within 24 hours, and
    linked to the trip via GPS telemetry.  External uploads or photos
    without a timestamp are not verifiable → claim rejected.

11. **Property damage — repair claims (§4.3)**: v2.1 says repair claims
    require a certified workshop quote within 7 days.  We route all repair
    claims to a human (they always exceed the AI threshold because the
    amount is not capped at S$50).

12. **Rider disputing a damage charge (§4.4)**: v2.1 says "the dispute is
    routed to human review".  We route any rider-filed dispute against a
    damage charge to a human, regardless of the evidence quality.

13. **Safety indicators in chat (§5.2)**: v2.1 says AI agents must flag any
    dispute that contains safety-related keywords or indicators "regardless
    of the dispute type selected by the filer".  We scan the claim text and
    the chat log for safety keywords; if found, the case is escalated even
    if it was filed under another category.

14. **§5.3 interim measure**: we record the account suspension as an
    action on the escalation, not as a ruling outcome.

15. **Dispute value (§6.1)**: v2.1 says "dispute value" but does not define
    it per category.  We define it as: for fare and route cases, the money
    at stake = actual fare − upfront quote or estimate; for no-show, the
    fee; for damage, the claimed amount.

16. **Confidence formula (§6.3)**: v2.1 lists four factors but does not give
    the formula.  We compute confidence as a weighted average:
    `0.30 × evidence_completeness + 0.25 × policy_clarity + 0.25 × precedent_alignment + 0.20 × party_credibility`,
    each factor scored 0–1.  A case where every fact is present and every
    condition holds with a clear margin scores high; missing evidence or
    ambiguity lowers the score.

17. **Fraud threshold (§6.3)**: v2.1 says "if any party has a fraud flag ≥
    1, the confidence threshold is raised to 95%".  We interpret this as
    the **routing** threshold — the case routes to human unless confidence
    ≥ 0.95.  §6 routing uses only the engine's computed confidence.  As a
    separate safeguard, a Judge that reports its own confidence below 0.7,
    or reaches a different outcome from the policy engine, also sends the
    case to a human.

18. **Strictest rule wins (§6)**: when multiple §6 rules apply (e.g. a route
    deviation over S$50 and over 2 km), we take the strictest — the one
    that requires human review.

19. **Uncovered disputes**: a dispute that no section of v2.1 covers (for
    example a promo code) goes to a human with the reason "not covered by
    policy v2.1".

20. **DISP-001 routing**: §6.2 says route deviation escalates when
    deviation > 2 km.  DISP-001 has a 4.3 km deviation, so it routes to a
    human even though Ryde's expected answer rules it automatically.  We
    follow v2.1's own rule and show the S$3.44 recommendation for one-click
    approval.
