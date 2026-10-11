# RideCourt Policy

The policy clauses below are the authoritative rules the Judge applies. Agents
cite clause IDs (e.g. `P-RD-2`) when arguing. All monetary amounts are in SGD.

---

## P-RD-1 Route deviation refund

If the actual trip distance exceeds the optimal route distance by more than 15 %
**and** by more than 1.5 km, and the deviation was not requested by the rider
nor caused by a documented road closure, the rider is refunded the fare for the
extra distance and extra time.

**Refund formula:** `(extra_km × per_km + extra_min × per_min) × surge_multiplier`,
rounded to the nearest cent. Both `extra_km` and `extra_min` are clamped to
`max(0, …)` — a detour that is shorter in distance or time than the optimal
route contributes zero, never a negative amount.

**Rationale:** Riders should pay only for the efficient route unless they
opted into a detour or the driver had no reasonable alternative.

---

## P-RD-2 Requested stops and route changes

A route change or stop requested by the rider — in chat or via the app — is not
a deviation. No refund is owed under P-RD-1 when this applies.

**Rationale:** The rider explicitly agreed to the longer route, so the extra
cost is fairly theirs.

---

## P-NS-1 Driver no-show

If the driver never came within 150 m of the pickup point, **or** cancelled
after the rider waited more than 5 minutes past the ETA, any cancellation fee
charged is refunded in full and the rider receives a S$3 goodwill credit.

**Rationale:** When the driver fails to show, charging the rider a cancellation
fee is unfair; the credit compensates for wasted time.

---

## P-NS-2 Rider no-show / cancellation fee stands

If the driver waited at least 5 minutes within 150 m of the pickup point and
made at least one contact attempt (call or chat), the cancellation fee stands.

**Rationale:** The driver fulfilled their obligation; the rider's no-show
justifies the fee.

---

## P-FA-1 Fare / surge / promo errors

If the charged total differs from the correctly recomputed fare by more than
S$0.50, the difference is refunded (or the undercharge is noted). Promo codes
valid at booking time must be applied.

**Refund = charged_total − recomputed_fare** (positive → rider is owed a refund;
negative → fare was undercharged, noted but not collected).

**Rationale:** Fares should match the deterministic fare calculation; promo
entitlements at booking time are binding.

---

## P-SF-1 Safety incidents

Safety incidents (threats, harassment, fear for personal safety, etc.) are
**never** resolved automatically. The case is escalated immediately to a human
reviewer with a neutral case summary and the full evidence pack.

**Rationale:** Safety demands human judgment; no automated refund or ruling
is appropriate.

---

## P-BF-1 Bad faith

Three or more upheld disputes of the same type filed by the claimant in the
preceding 90 days, or material evidence that contradicts the claim, lowers the
claimant's credibility. The Judge must state this explicitly in the ruling.

**Rationale:** Repeat complainants and contradictory evidence indicate
potential abuse; transparency requires the Judge to flag it.
