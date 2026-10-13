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
Every claim an agent makes must cite evidence IDs (`E3`) and policy clause IDs (`P-RD-2`).
A guard rejects agent output that cites unknown IDs or states a number that is not in the
evidence pack, and the agent is asked to try again once; if it fails again the case escalates to a
human. This is our answer to "fairness and transparency" and to "inconsistent rulings".

### Guard

The guard (`ridecourt/guard.py`) performs three checks on every agent output:

1. **Citation check** — every ID mentioned in agent text must exist: `E#`
   in the evidence pack, `P-..` in the applicable policy clauses, or
   `PC-YYYY-NNN` in the precedent list. Unknown IDs are rejected.

2. **Number check** — every number with a unit (money `S$3.25`, distance
   `3.25 km`, metres `150 m`, minutes `5 min`, percent `15%`) must be
   backed by a value of the same unit from a valid source:

   - **Advocates**: numbers must come from evidence items the advocate
     cited (`E#`). The advocate cited E3, so only E3's data is valid.
   - **Judge**: the Judge sees the whole pack, so numbers may come from
     any evidence item or the policy computation.
   - **Precedent numbers**: the Judge is shown precedents with their IDs
     (`PC-2025-004`). A number from a precedent's summary (e.g. `400 m`
     from `PC-2025-004`) is accepted **only** in a text block that cites
     that precedent's ID. A precedent number in a text block that does
     not cite the precedent is rejected, so the Judge cannot accidentally
     present a precedent fact as a fact of the current case. This is the
     same class of defect as P-BF-1 (the system asks an agent to use
     information the guard will not accept), but for numbers instead of
     clause IDs.
   - **Claim numbers**: the dispute's claim text (written by the
     claimant) is treated as an attributable source with citation ID
     `CLAIM`. A number from the claim (e.g. `S$10` or `5 km` in *"The
     driver overcharged me S$10 and drove 5 km out of the way"*) is
     accepted **only** in a text block that cites `CLAIM` and attributes
     the number to the claimant ("the rider says …", citing `CLAIM`). A
     claim number stated as a fact of the case without citing `CLAIM` is
     rejected, so an advocate cannot present the claimant's allegation as
     an established fact. The Judge's amount still comes only from the
     policy engine — `CLAIM` numbers never enter the ruling amount.

   Precision-aware: a number written with *k* decimal places matches an
   evidence value *v* when `round(v, k) == num.value`.

3. **Forbidden-words check** — no promise or certainty words
   (`"guaranteed"`, `"definitely fraud"`, etc.).

A machine-check sweep test (`tests/test_guard_number_sweep.py`) verifies
over all 48 cases that every number in every piece of text the prompt
builder shows an agent (claim text, evidence, policy computation,
precedents, risk signals, opponent's case) is either accepted by the guard
for that agent or deliberately excluded with a stated reason. A second
suite (`tests/test_guard_claim_numbers.py`) adds claim texts with numbers
in every unit the guard knows (`S$`, `km`, `m`, `min`, `%`) for each
category, and checks that every agent's prompt numbers are accepted when
attributed via `CLAIM` and rejected when stated as fact.

## 3. Pipeline

```
Dispute filed
   │
   ▼
Evidence Collection Agent (stretch) ── calls deterministic tools ──► Evidence pack E1..En
   │                                     (GPS analysis, arrival check, chat scan,
   │                                      fare recompute, history profile)
   ├──────────────► Rider Advocate  ──┐   (parallel, each sees the whole pack,
   ├──────────────► Driver Advocate ──┤    argues for its side, cites E# and P-#)
   ├──────────────► Fraud & Bad-Faith Agent (stretch, parallel) ── risk signals
   └──────────────► Precedent Agent (stretch) ── similar past rulings
                                       ▼
                                Judge Agent
                    applies policy via the policy engine (code),
                    weighs both cases, issues ruling + confidence + explanation
                                       │
                      confidence < threshold or safety category
                                       ▼
                         Escalation queue (human reviewer)
                                       │ human override
                                       ▼
                    Feedback stored as a new precedent (learning loop, stretch)
```

Every arrow above is a typed message on the case's event log:
`DISPUTE_FILED, EVIDENCE_REQUEST, EVIDENCE, CLAIM, REBUTTAL, RISK_SIGNAL, PRECEDENT, RULING,
ESCALATION, HUMAN_OVERRIDE`. Each message has `{id, case_id, ts, from_agent, to_agent, type,
content, cites: [E#, P-#], tokens, latency_ms}`. The UI streams this log live.

## 4. Data

The organiser will send a sample dataset later. Until then we generate a synthetic one with a fixed
seed, and write a loader interface so the organiser's data can be plugged in with an adapter.

`data/synthetic/` (generated by `scripts/generate_data.py --seed 7`):

- `trips.json`: trip_id, rider_id, driver_id, pickup/dropoff (lat, lon, address), requested_at,
  accepted_at, driver_arrived_at, trip_started_at, trip_ended_at, status
  (completed | cancelled_by_rider | cancelled_by_driver | no_show_rider | no_show_driver),
  gps_trace `[{ts, lat, lon, speed_kmh}]`, optimal_route `{distance_km, duration_min, points}`,
  fare `{base, per_km, per_min, distance_km, duration_min, surge_multiplier, promo_code, discount,
  booking_fee, cancellation_fee, total_charged}`
- `chats.json`: trip_id → `[{ts, from: rider|driver|system, text}]`, and call attempts
  `[{ts, from, to, duration_s, answered}]`
- `profiles.json`: riders and drivers: account_age_days, rating, trips_completed,
  disputes_filed_90d, disputes_upheld_90d, cancellations_30d, prior_flags
- `disputes.json`: dispute_id, trip_id, filed_by (rider|driver), category, claim_text, filed_at,
  and for evaluation only, `label: {outcome, amount}` computed from policy (hidden from agents)
- `policy.md`: numbered clauses the agents must cite (see section 5)
- `precedents.json`: past rulings with short summaries, for the Precedent Agent

Singapore geography: use real Singapore coordinates (e.g. Tampines, Jurong East, Changi Airport,
Orchard, Punggol). Routes do not need real road geometry; a polyline of plausible points is enough.
Optimal route = straight-ish path; deviated route = path with a detour loop.

Case mix (about 48 cases): route deviation 16 (half valid, half not: e.g. rider asked for a stop,
road closure noted in chat), no-show by driver 10, no-show by rider / cancellation fee 10,
fare/surge/promo error 6, safety incident 6 (always escalate with a summary, never auto-rule).
Include a few bad-faith patterns: a rider with 6 upheld fare disputes in 90 days, a driver whose
GPS shows arrival 600 m away from pickup while claiming no-show.

## 5. Policy (excerpt; full text in `data/policy.md`)

- P-RD-1 Route deviation: if actual distance exceeds the optimal route by more than 15% and more
  than 1.5 km, and the deviation was not requested by the rider or caused by a documented road
  closure, refund the fare for the extra distance and time.
- P-RD-2 Requested stops or route changes by the rider (in chat or app) are not deviations.
- P-NS-1 Driver no-show: if the driver never came within 150 m of the pickup point, or cancelled
  after the rider waited more than 5 minutes past the ETA, refund any cancellation fee in full and
  credit S$3.
- P-NS-2 Rider no-show: the cancellation fee stands if the driver waited at least 5 minutes within
  150 m of the pickup point and made at least one contact attempt (call or chat).
- P-FA-1 Fare errors: if the charged total differs from the recomputed fare by more than S$0.50,
  refund the difference; promo codes valid at booking time must be applied.
- P-SF-1 Safety incidents are never resolved automatically. Escalate immediately with a neutral
  case summary and the evidence pack.
- P-BF-1 Bad faith: three or more upheld disputes of the same type in 90 days, or evidence that
  contradicts the claim, lowers the claimant's credibility; the Judge must say so explicitly.

The policy engine (`ridecourt/policy.py`) implements these clauses as pure functions and returns
`{applicable_clauses, outcome, amount, computation}` where `computation` shows the arithmetic.

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
policy engine it must escalate, not override. Confidence below 0.7 escalates.

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
