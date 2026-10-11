# RideCourt

![Cover](docs/cover-380x216.png)

Multi-agent autonomous dispute resolution for ride-hailing. Built for the
Tencent Cloud AI CAN DO IT Hackathon Singapore 2026 (Ryde challenge).

RideCourt resolves rider–driver disputes — route deviation, no-show,
fare errors, and safety incidents — by gathering evidence, building both
sides of the case, applying company policy, and issuing a fair ruling with
a plain-English explanation, in seconds instead of 24–72 hours.

**Principle: agents argue, code computes.** Language models gather, argue,
and explain. Every number in a ruling (distances, minutes, dollars) is
computed by deterministic Python — never typed by a model. Every claim an
agent makes must cite evidence IDs (`E#`) and policy clause IDs (`P-..`).

---

## The problem

Ride-hailing platforms take 24–72 hours to resolve disputes, with
inconsistent outcomes and no transparency for riders or drivers. Human
agents handle routine disputes one at a time, costing money and goodwill.
Safety incidents get queued alongside fare complaints instead of being
prioritised. Riders and drivers never see *why* a decision was made.

## What it does

RideCourt automates the majority of standard dispute categories — route
deviation, no-show (rider or driver), fare errors — end-to-end:

1. **Gathers evidence** — deterministic tools analyse GPS traces, arrival
   distances, wait times, chat logs, fare breakdowns, and account history.
2. **Builds both sides** — a Rider Advocate and a Driver Advocate argue in
   parallel, each citing evidence and policy. A rebuttal round follows.
3. **Checks for bad faith** — a Fraud Agent computes deterministic risk
   signals (dispute rate, upheld rate per P-BF-1, GPS contradictions).
4. **Retrieves precedents** — the Precedent Agent finds the 3 most similar
   past rulings using TF-IDF keyword similarity.
5. **Applies policy** — the Policy Engine (pure Python functions) computes
   the outcome, refund amount, and a human-readable computation string.
6. **Issues a ruling** — the Judge weighs both cases, applies the policy
   engine result, considers risk and precedents, and issues a ruling with
   confidence, plain-English explanations to both parties, and cited
   clauses. Safety incidents always escalate to a human.

Every inter-agent message is a typed event on a case log, streamed live to
the UI via Server-Sent Events — so the entire reasoning chain is visible.

## Architecture

![Architecture Diagram](docs/architecture.png)

**Pipeline:** Evidence Collection (deterministic tools) → Evidence Pack →
Rider/Driver Advocates + Fraud Agent + Precedent Agent (parallel) →
Policy Engine (code) → Guard (validation) → Judge → Ruling or Escalation →
Feedback Loop (human overrides become precedents).

**Key components:**

| Component | Role |
|:--|:--|
| **Data sources** | `data/synthetic/`: trips, chats, profiles, disputes, precedents, policy. Adapters for the organiser's real dataset. |
| **Deterministic tools** (`ridecourt/tools.py`) | GPS analysis, arrival check (150 m), chat scan, fare recompute, history profile. No LLM calls. Pure Python + math. |
| **Evidence pack** | List of `E#` items, each with a summary, data, and source. Cited by every agent in every claim. |
| **Advocates** (`ridecourt/agents/advocate.py`) | Rider Advocate and Driver Advocate run in parallel, argue for their side, cite `E#` and `P-#`, then rebut each other. |
| **Fraud agent** (`ridecourt/agents/fraud.py`) | Deterministic features first (dispute rate, upheld rate P-BF-1, GPS contradiction, prior flags); the LLM only explains the signals. |
| **Precedent agent** (`ridecourt/agents/precedent.py`) | TF-IDF keyword similarity over category + claim text; top 3 past rulings; includes override precedents (learning loop). |
| **Judge** (`ridecourt/agents/judge.py`) | Weighs both cases + risk + precedents; applies the policy engine result; the amount must match; confidence < 0.7 escalates. |
| **Policy engine** (`ridecourt/policy.py`) | Pure functions, one per clause (P-RD-1, P-NS-1, P-FA-1, etc.). Returns outcome, amount, and computation string. |
| **Guard** (`ridecourt/guard.py`) | Validates every agent output: rejects unknown citation IDs or numbers not in the evidence pack. Re-asks once, then escalates. |
| **Escalation queue** (`ridecourt/escalation.py`) | Triggers: low confidence, guard failure, outcome mismatch, safety. Neutral case summary. `GET /api/escalations`. |
| **Feedback loop** | Human override stored as a precedent (`source="override"`); retrieved for similar future cases; "learned from override" in timeline. |
| **API / SSE / UI** (`ridecourt/api.py`, `web/`) | FastAPI endpoints, Server-Sent Events for live streaming, single-page courtroom timeline UI. |

## How the agents work: agents argue, code computes

This is the core design decision and the answer to "fairness and
transparency" and "inconsistent rulings":

- **Language models gather, argue, and explain.** The Rider Advocate argues
  for the rider, the Driver Advocate argues for the driver, the Fraud Agent
  explains risk signals, the Precedent Agent explains similar rulings, and
  the Judge writes the ruling and plain-English messages to both parties.
- **Code computes every number.** Distances (haversine), arrival radius,
  wait times, fare recomputation, refund amounts — all computed by
  deterministic Python in `ridecourt/tools.py` and `ridecourt/policy.py`.
  The Judge's amount must equal the policy engine's amount; if the Judge
  disagrees it must escalate, not override.
- **Every claim cites evidence and policy.** Agent outputs must reference
  evidence IDs (`E3`) and policy clause IDs (`P-RD-2`). The Guard validates
  every output: unknown IDs or numbers not in the evidence pack trigger a
  re-ask, then escalation. This prevents hallucinated facts and numbers.
- **The event log is the audit trail.** Every inter-agent message is a typed
  event (`DISPUTE_FILED`, `EVIDENCE`, `CLAIM`, `REBUTTAL`, `RISK_SIGNAL`,
  `PRECEDENT`, `RULING`, `ESCALATION`, `HUMAN_OVERRIDE`) with timestamp,
  tokens, and latency — streamed live via SSE.

## How to run

### Prerequisites

- Python 3.12+ (the project uses pydantic v2, FastAPI, uvicorn, pytest)
- A virtualenv in `.venv/`

### Local (mock LLM, offline)

```bash
# Set up
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests (offline, mock LLM)
LLM_PROVIDER=mock python -m pytest -q

# Start the API + UI
LLM_PROVIDER=mock uvicorn ridecourt.api:app --reload

# Open http://localhost:8000 — pick a dispute, click Resolve, watch the
# courtroom timeline stream live.

# Run the evaluation script
LLM_PROVIDER=mock python scripts/evaluate.py --provider mock
```

### With Gemini (Vertex AI)

```bash
# Authenticate
gcloud auth application-default login

# Set environment variables
export LLM_PROVIDER=gemini
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=global
export MODEL=gemini-3.8-flash

# Run
uvicorn ridecourt.api:app --reload
```

### Deploy to Google Cloud Run

```bash
export PROJECT_ID=your-project-id
# Optional: override the global daily resolve cap (default 300).
export DAILY_RESOLVE_CAP=300
./scripts/deploy.sh
```

The deploy script creates a dedicated service account with only
`roles/aiplatform.user`, deploys via `gcloud run deploy ridecourt --source .`,
sets the Gemini env vars, caps at 2 max instances, and waits for IAM
propagation with a retry loop. See `Dockerfile` for the container image
(python:3.12-slim, non-root user, uvicorn on `$PORT`).

#### Cost guard for the public demo

To keep the live demo within budget, every `/api/disputes/{id}/resolve` call
is counted against a global daily cap (`DAILY_RESOLVE_CAP`, default **300**).
Once the cap is reached the API returns `429 Too Many Requests` with a clear
message saying the limit is exhausted and when it resets (UTC midnight). The
remaining budget is exposed in `GET /api/metrics` as
`daily_resolve_cap`, `resolves_today`, and `remaining_resolves_today`. The
per-IP limit (20/hour) still applies — the daily cap is the outer one.

See `.env.example` for all configuration options.

## Evaluation

The evaluation script (`scripts/evaluate.py`) runs every labelled synthetic
dispute through the full pipeline and reports:

- **Ruling accuracy** vs the policy label, per category (outcome and amount,
  amount within S$0.01)
- **Consistency** — the same case run 3 times gives the same outcome
- **Escalation rate** and whether every safety case escalated
- **Median and p90 time to ruling** and **tokens per case**

Results are written to `eval/results-<date>-<provider>.json`.

### Mock provider results

Run with: `LLM_PROVIDER=mock python scripts/evaluate.py --provider mock`

| Category | N | Outcome acc. | Amount acc. | Both acc. | Escalation rate | Consistency |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| route_deviation | 16 | 87.5% | 100.0% | 87.5% | 12.5% | 100.0% |
| no_show_driver | 10 | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| no_show_rider | 10 | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| fare_error | 6 | 0.0% | 50.0% | 0.0% | 100.0% | 100.0% |
| safety | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| **Overall** | 48 | **83.3%** | **93.8%** | **83.3%** | **29.2%** | **100.0%** |

- Safety cases all escalated: **yes**
- Median time to ruling: 1.0 ms
- p90 time to ruling: 1.0 ms
- Median tokens per case: 500

> The Gemini run is reported separately — see
> `eval/results-<date>-gemini.json` after running
> `LLM_PROVIDER=gemini python scripts/evaluate.py --provider gemini`.

**Note on fare_error escalation:** the mock provider's canned advocate
output triggers the guard (it generates text containing "0.00 km" which
does not match any cited evidence value), causing all fare_error cases to
escalate. With a real LLM provider, advocates cite the correct evidence
and the guard passes.

## How CodeBuddy was used to build it

The entire project was built spec-first using **CodeBuddy Code** in
headless mode. The build was divided into spec-driven tasks (matching the
build order in `docs/SPEC.md`), each run as a separate CodeBuddy session:

| Task | Prompt | Transcript |
|:--|:--|:--|
| 1 — Foundation: scaffold, data, policy | `prompts/01-foundation.md` | `devlog/20261011-161337-01-foundation.jsonl` |
| 2 — Evidence tools | `prompts/02-evidence-tools.md` | `devlog/20261011-162623-02-evidence-tools.jsonl` |
| 3 — LLM layer + guard | `prompts/03-llm-and-guard.md` | `devlog/20261011-163333-03-llm-and-guard.jsonl` |
| 4 — Agents + orchestrator | `prompts/04-agents-orchestrator.md` | `devlog/20261011-164018-04-agents-orchestrator.jsonl` |
| 4b — Real-run fixes | `prompts/04b-real-run-fixes.md` | `devlog/20261011-165259-04b-real-run-fixes.jsonl` |
| 4c — Finish rebuttal | `prompts/04c-finish-rebuttal.md` | `devlog/20261011-185727-04c-finish-rebuttal.jsonl` |
| 5 — API + UI | `prompts/05-api-ui.md` | `devlog/20261011-190303-05-api-ui.jsonl` |
| 6 — Evaluation | `prompts/06-evaluation.md` | `devlog/20261011-191307-06-evaluation.jsonl` |
| 7 — Stretch agents | `prompts/07-stretch-agents.md` | `devlog/20261011-191736-07-stretch-agents.jsonl` |
| 8 — Deploy + docs | `prompts/08-deploy-docs.md` | `devlog/20261011-193355-08-deploy-docs.jsonl` |

Each prompt contains the full task specification (what to build, what
constraints to follow, what tests must pass). The `CODEBUDDY.md` file
enforces project rules across all sessions: agents argue / code computes,
Python 3.12 + FastAPI + pydantic v2, tests must pass with `LLM_PROVIDER=mock`,
English for all code and docs, commit after each task. The prompts are in
`prompts/` and the session transcripts are in `devlog/`.

## Stretch goals

| Feature | Status | Description |
|:--|:--|:--|
| Fraud & Bad-Faith Agent | Done | Runs in parallel with advocates; deterministic features (dispute rate, upheld rate P-BF-1, GPS contradiction, prior flags); Judge mentions high-risk signals explicitly |
| Policy & Precedent Agent | Done | TF-IDF keyword similarity over category, facts and claim text; top 3 past rulings; Judge states consistency |
| Escalation Protocol | Done | Triggers: low confidence, guard failure, outcome mismatch, safety; neutral case summary; `GET /api/escalations` queue |
| Learning Feedback Loop | Done | Override stored as precedent (`source="override"`); retrieved for similar cases; "learned from override" in timeline |
| SLA & Routing | Done | Priority = safety first, then amount at stake and age; queue sorts by it |
| Cloud Run deployment | Ready | `Dockerfile` + `scripts/deploy.sh`; dedicated SA with `roles/aiplatform.user`; max 2 instances; IAM retry loop |

## Limits

- 48 synthetic cases (16 route deviation, 10 no-show driver, 10 no-show
  rider, 6 fare error, 6 safety). The data loader supports an adapter for
  the organiser's real dataset.
- The mock provider is deterministic but limited — it does not produce
  real natural-language arguments; its purpose is offline testing.
- In-memory state (single process). The Cloud Run deployment caps at 2
  max instances; a production deployment would need external state
  (Redis or a database) for multi-instance consistency.
- Single rebuttal round. A production system might allow more rounds with
  a token budget.

## Licence

[Apache-2.0](LICENSE). Copyright 2026 RideCourt Contributors.
