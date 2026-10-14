# RideCourt — Multi-Agent Autonomous Dispute Resolution (spec)

Challenge: Tencent Cloud AI CAN DO IT Hackathon Singapore 2026, Digital Native track (Ryde):
"Build an autonomous, multi-agent dispute resolution system that can handle complex, multi-party
conflicts between riders and drivers — gathering evidence, building cases, applying company policy,
and issuing fair rulings quickly, transparently, and without human intervention for the majority of
standard dispute categories."

Ryde's evaluation lens: multi-agent architecture, evidence gathering, fairness and transparency,
business impact and scalability. MVP (the three core agents on text and structured evidence) is what
we are judged on first; stretch goals earn bonus points only after the MVP works.

## 1. What must exist at submission

| Item | Requirement |
|:--|:--|
| Working prototype | End-to-end for at least two dispute categories (we do route deviation and no-show, plus safety escalation) |
| Three core agents | Rider Advocate, Driver Advocate, Judge |
| Observable inter-agent communication | A visible log/timeline of every message between agents |
| Architecture diagram | `docs/architecture.svg` (and PNG) |
| Demo walkthrough | One dispute processed live in the UI, start to ruling |
| Source code | GitHub repository, open source licence (Apache-2.0) |
| Live URL | Optional, earns bonus points: deploy to a public HTTPS URL |

## 2. Design principle: agents argue, code computes

Language models gather, argue and explain. Every number in a ruling (distances, minutes, dollars,
refund amounts) is computed by deterministic Python from the evidence, never typed by a model.
Every claim an agent makes must cite evidence IDs (`E3`) and policy section IDs (`§1.3`).
A guard rejects agent output that cites unknown IDs or states a number that is not in the
evidence pack, and the agent is asked to try again once; if it fails again the case escalates to a
human. This is our answer to "fairness and transparency" and to "inconsistent rulings".

### Guard

The guard (`ridecourt/guard.py`) performs three checks on every agent output:

1. **Citation check** — every ID mentioned in agent text must exist: `E#`
   in the evidence pack, `§N.M` in the valid v2.1 sections set or the
   applicable policy clauses, `PC-YYYY-NNN` (seed precedent) or
   `DISP-YYYY-NNNN` (Ryde precedent) in the precedent list. Legacy `P-..`
   IDs are accepted for backward compatibility. Unknown IDs are rejected.

2. **Number check** — every number with a unit (money `S$3.25`, distance
   `3.25 km`, metres `150 m`, minutes `5 min`, percent `15%`) must be
   backed by a value of the same unit from a valid source:

   - **Advocates**: numbers must come from evidence items the advocate
     cited (`E#`). The advocate cited E3, so only E3's data is valid.
   - **Judge**: the Judge sees the whole pack, so numbers may come from
     any evidence item or the policy computation.
   - **Precedent numbers**: the Judge is shown precedents with their IDs
     (`PC-2025-004` or `DISP-2026-0089`). A number from a precedent's
     summary is accepted **only** in a text block that cites that
     precedent's ID.
   - **Claim numbers**: the dispute's claim text (written by the
     claimant) is treated as an attributable source with citation ID
     `CLAIM`.

   Precision-aware: a number written with *k* decimal places matches an
   evidence value *v* when `round(v, k) == num.value`.

3. **Forbidden-words check** — no promise or certainty words
   (`"guaranteed"`, `"definitely fraud"`, etc.).

A machine-check sweep test (`tests/test_guard_number_sweep.py`) verifies
over all 51 cases that every number in every piece of text the prompt
builder shows an agent is either accepted by the guard for that agent or
deliberately excluded with a stated reason.

## 3. Pipeline

```
Dispute filed
   │
   ▼
Evidence Collection Agent (stretch) ── calls deterministic tools ──► Evidence pack E1..En
   │                                     (GPS analysis, arrival check, chat scan,
   │                                      fare recompute, history profile)
   ├──────────────► Rider Advocate  ──┐   (parallel, each sees the whole pack,
   ├──────────────► Driver Advocate ──┤    argues for its side, cites E# and §N.M)
   ├──────────────► Fraud & Bad-Faith Agent (parallel) ── risk signals (§6.3)
   └──────────────► Precedent Agent (stretch) ── similar past rulings (§7)
                                       ▼
                                Judge Agent
                    applies policy via the policy engine (code),
                    weighs both cases, issues ruling + confidence + explanation
                                       │
                       §6 routing: confidence < threshold, safety category,
                       fraud flag, or deviation > 2 km
                                       ▼
                         Escalation queue (human reviewer)
                                       │ human override
                                       ▼
                    Feedback stored as a new precedent (learning loop, stretch)
```

§6 routing takes precedence: after the Judge issues a ruling, the policy
engine's `routing` field (§6.1 monetary, §6.2 category, §6.3 fraud flag)
determines whether the ruling is final or a recommendation for human
review. Safety incidents (§5) always escalate before the Judge.

Every arrow above is a typed message on the case's event log:
`DISPUTE_FILED, EVIDENCE_REQUEST, EVIDENCE, CLAIM, REBUTTAL, RISK_SIGNAL, PRECEDENT, RULING,
ESCALATION, HUMAN_OVERRIDE`. Each message has `{id, case_id, ts, from_agent, to_agent, type,
content, cites: [E#, §N.M], tokens, latency_ms}`. The UI streams this log live.

## 4. Data

The organiser will send a sample dataset later. Until then we generate a synthetic one with a fixed
seed, and write a loader interface so the organiser's data can be plugged in with an adapter.

### Synthetic data

`data/synthetic/` (generated by `scripts/generate_data.py --seed 7`):

- `trips.json`: trip_id, rider_id, driver_id, pickup/dropoff (lat, lon, address),
  requested_at, accepted_at, driver_arrived_at, trip_started_at, trip_ended_at, status
  (completed | cancelled_by_rider | cancelled_by_driver | no_show_rider | no_show_driver),
  gps_trace `[{ts, lat, lon, speed_kmh}]`, optimal_route `{distance_km, duration_min, points}`,
  fare `{base, per_km, per_min, distance_km, duration_min, surge_multiplier,
  quote_surge_multiplier, trip_start_surge_multiplier, rider_notified_surge, surgeable_subtotal,
  upfront_quote, promo_code, discount, booking_fee, cancellation_fee, excess_distance_km,
  excess_distance_fare, damage_claim_amount, total_charged}`,
  traffic_conditions `[{incident_id, road, direction, stretch, incident_type, severity,
  timestamp, clear_time, description, on_recommended_route}]`,
  app_events `[{ts, event, surge_multiplier, rider_notified}]`,
  damage_evidence `{photos: [{id, source, timestamp, url}], damage_type, claim_amount,
  workshop_quote: {submitted, amount, workshop_name, submitted_within_7d}}`
- `chats.json`: trip_id → `messages: [{message_id, sender, message_type, text, ts}]`
  and call attempts `calls: [{ts, from, to, duration_s, answered}]`
- `profiles.json`: riders and drivers keyed by ID: account_age_days, rating, trips_completed,
  disputes_filed_90d, disputes_upheld_90d, cancellations_30d, no_show_cancellations_30d,
  prior_flags `["fraud_flag"]`, fraud_flags `int`
- `disputes.json`: dispute_id, trip_id, filed_by (rider|driver), category, claim_text, filed_at,
  and for evaluation only, `label: {outcome, amount}` computed from policy (hidden from agents)
- `policy.md`: v2.1 sections the agents must cite (see section 5)
- `precedents.json`: past rulings with short summaries, for the Precedent Agent.
  Ryde precedents use `DISP-YYYY-NNNN` IDs (source `ryde`); seed precedents use `PC-YYYY-NNN`
  (source `seed`)

### Ryde sample cases

`data/ryde_samples/` contains the three official Ryde case files
(`DISP-001_route_deviation.md`, `DISP-002_no_show.md`, `DISP-003_fare_surge.md`)
and the official policy text (`ryde_dispute_policy_v2.1.md`).
`ridecourt/ryde_adapter.py` reads these Markdown files, extracts the JSON
blocks, and maps them to `Dataset` objects, holding out answer-key fields
(`expected_*`, `ruling_rationale`, `recommended_ruling`, analysis fields)
so they never reach agents or tools.

Singapore geography: use real Singapore coordinates (e.g. Tampines, Jurong East, Changi Airport,
Orchard, Punggol). Routes do not need real road geometry; a polyline of plausible points is enough.
Optimal route = straight-ish path; deviated route = path with a detour loop.

Case mix (51 cases): route deviation 14 (within-tolerance and justified, un-informed, and
large-deviation >2 km), no-show by driver 2, no-show by rider / cancellation fee 10, fare/surge
error 11, property damage 10 (driver-filed cleaning and rider-filed disputes), safety 4 (always
escalate with a summary, never auto-rule). Include bad-faith patterns: a rider with fraud_flags=1
and excessive disputes, a driver whose GPS shows arrival 600 m away from pickup while claiming
no-show, and a surge-not-notified fraud case.

## 5. Policy v2.1 (excerpt; full text in `data/policy.md`)

The policy is our summary of Ryde's Dispute Resolution Policy v2.1, organised
into seven sections. Agents cite sections as `§1.3`, `§2.3`, `§6.2`, etc.

- **§1 Route Deviation**: if actual distance exceeds the optimal route by
  more than 1 km, the refund depends on whether the rider was informed
  (§1.2) and whether traffic justified the deviation (§1.4). Informed +
  traffic-justified → no refund; not-informed + traffic-justified → partial
  refund of excess distance fare; no traffic justification → full refund
  of excess distance fare. Refund = `excess_km × per_km_rate`; tolls, time
  and surge are not refunded.
- **§2 No-Show & Cancellation**: free wait time is 5 minutes (§2.1). A
  no-show cancellation fee of S$5.00 is charged to the rider (§2.2). The
  driver must attempt both a chat message and a phone call before cancelling
  (§2.3); missing either voids the fee. Driver no-show (§2.4) — driver never
  arrived and never contacted → refund + S$5.00 goodwill credit. Repeat
  no-show riders (3+ in 30 days) are flagged (§2.5).
- **§3 Fare & Surge**: the upfront quote is binding at the quote-time surge
  (§3.1). If surge increases between quote and trip start, the rider must
  be notified (§3.2); un-notified increases → full refund of the surge
  differential (§3.3). Differential = `surgeable_subtotal × (trip_surge −
  quote_surge)`, booking fee exempt (§3.4–§3.5).
- **§4 Property Damage**: drivers may claim cleaning (≤ S$50) or repair
  fees (§4.1). Photos must be in-app, timestamped, within 24 h, GPS-linked
  (§4.2). Rider disputing a damage charge → human review (§4.4).
- **§5 Safety Incident**: always escalated to human review (§5.1). AI agents
  gather evidence and summarise but cannot rule (§5.2).
- **§6 Ruling Thresholds**: §6.1 monetary — under S$10 AI-final, S$10–S$50
  needs 80%+ confidence, above S$50 human. §6.2 category — safety always
  human, route deviation >2 km human, repair >S$50 human, no-show AI-final.
  §6.3 confidence calibration — fraud flag ≥1 raises the threshold to 95%.
  Strictest rule wins.
- **§7 Precedent Rulings**: past rulings on similar cases are referenced for
  consistency. Ryde precedents use `DISP-YYYY-NNNN`; seed precedents use
  `PC-YYYY-NNN`.

The policy engine (`ridecourt/policy.py`) implements these sections as pure
functions and returns `{applicable_clauses, outcome, amount, computation, routing,
routing_reason}` where `computation` shows the arithmetic and `routing` is the
§6 determination (`ai_final`, `human`, or `escalate_safety`).

Confidence formula (§6.3 interpretation): the engine computes confidence as
`0.30 × evidence_completeness + 0.25 × policy_clarity + 0.25 × precedent_alignment +
0.20 × party_credibility`, each factor scored 0–1. Fraud flags lower
party_credibility and raise the routing threshold to 0.95.

## 6. Agents

All LLM agents return JSON validated against a schema (pydantic). The LLM layer is pluggable:

- `LLM_PROVIDER=gemini` (default): Google Gen AI SDK, Vertex AI mode
  (`GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION=asia-southeast1`)
- `LLM_PROVIDER=openai_compatible`: any OpenAI-compatible endpoint (`OPENAI_BASE_URL`,
  `OPENAI_API_KEY`, `MODEL`), so Hunyuan or DeepSeek can be swapped in
- `LLM_PROVIDER=mock`: deterministic canned outputs for tests and offline demos

| Agent | Input | Output |
|:--|:--|:--|
| Evidence Collection | dispute, trip | evidence pack: list of `{id: E#, kind, summary, data, source}` built by tools; the LLM only decides which tools to call |
| Rider Advocate | dispute, evidence pack, policy | `{position, arguments: [{text, cites}], requested_outcome}` |
| Driver Advocate | same | same shape |
| Fraud & Bad-Faith | profiles, history, evidence | `{risk: low/medium/high, signals: [{text, cites}]}` |
| Precedent | dispute summary, precedents | top 3 similar rulings with similarity reason |
| Judge | both cases, risk, precedents, policy engine result | `{outcome, amount, confidence 0-1, reasoning, cites, message_to_rider, message_to_driver}` |

Judge rules: the amount must equal the policy engine's amount; if the Judge disagrees with the
policy engine it must escalate, not override. Every claim must cite evidence IDs (`E#`) and
policy sections (`§N.M`). §6 routing takes precedence: after the Judge issues a ruling, the
policy engine's `routing` field (§6.1 monetary, §6.2 category, §6.3 fraud flag) determines
whether the ruling is final or a recommendation for human review. Confidence below 0.7
escalates. Fraud flag ≥1 raises the threshold to 0.95 (§6.3).

## 7. API (FastAPI)

- `GET /api/disputes` list with status and SLA priority (safety first, then by value and age)
- `POST /api/disputes` file a new dispute (category, trip_id, claim_text)
- `POST /api/disputes/{id}/resolve` run the pipeline; `GET /api/disputes/{id}/events` streams the
  event log (Server-Sent Events)
- `GET /api/disputes/{id}` case file: evidence, arguments, ruling
- `POST /api/disputes/{id}/override` human reviewer override, stored as a precedent
- `GET /api/metrics` evaluation and operating metrics
- `GET /healthz`

## 8. UI (single page, no build step: HTML + vanilla JS + CSS; Leaflet with OpenStreetMap tiles)

- Left: dispute queue with category, SLA badge, status (new, resolving, ruled, escalated)
- Centre: "courtroom" timeline. Rider Advocate messages on the left, Driver Advocate on the right,
  Judge in the middle, tool calls and evidence as small chips; streams live while resolving
- Right: evidence panel (map with actual GPS trace vs optimal route; fare table; chat excerpt) and
  the ruling card (outcome, amount, confidence, plain-English explanation to each party,
  cited clauses), plus Escalate / Override buttons
- Clean, readable, works at 1280 px and on a phone

## 9. Evaluation (business value with numbers)

`scripts/evaluate.py` runs every labelled synthetic case and reports:
- ruling accuracy vs the policy label, per category
- consistency: the same case run 3 times gives the same outcome
- escalation rate and whether every safety case escalated
- median time to ruling and LLM tokens per case
Write results to `eval/results-<date>.json` and a table in the README. The demo uses these numbers
for "Business Value" (for example: minutes instead of 24–72 hours; consistent outcomes).

## 10. Engineering rules

- Python 3.12 or newer (local machine has 3.14; Docker image uses python:3.12-slim), FastAPI, pydantic v2,
  uvicorn, pytest; `google-genai` for Gemini; use a virtualenv in `.venv/`
- Tests run offline with `LLM_PROVIDER=mock`; `pytest -q` must pass before any task is called done
- No secrets in the repo; config via environment variables; `.env.example` documents them
- Small modules: `ridecourt/data.py, tools.py, policy.py, guard.py, llm.py, agents/*.py,
  orchestrator.py, events.py, api.py`, `web/`
- Dockerfile for Cloud Run; `scripts/deploy.sh` (gcloud run deploy --source)
- README: problem, architecture diagram, how it works, how to run, evaluation table, limits

## 11. Build order (priority, not dates)

1. Scaffold, data generator, loader, policy engine with unit tests (no LLM)
2. Deterministic evidence tools with unit tests
3. LLM layer (gemini, openai_compatible, mock) and guard
4. Three core agents + Judge rules + orchestrator + event log
5. API + SSE + UI courtroom timeline
6. Evaluation script + README numbers
7. Stretch: Fraud & Bad-Faith, Precedent (with feedback loop from overrides), Escalation protocol,
   SLA routing
8. Deploy to Cloud Run; architecture diagram; cover image (16:9, 380×216)
