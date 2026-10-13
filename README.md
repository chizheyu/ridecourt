# RideCourt

![Cover](docs/cover-380x216.png)

Multi-agent autonomous dispute resolution for ride-hailing. Built for the
Tencent Cloud AI CAN DO IT Hackathon Singapore 2026 (Ryde challenge).

RideCourt resolves rider–driver disputes — route deviation, no-show,
fare errors, and safety incidents — by gathering evidence, building both
sides of the case, applying company policy, and issuing a fair ruling with
a plain-English explanation, in about half a minute (median 26.5 s in the evaluation) instead of 24–72 hours.

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

The deploy script creates a dedicated service account with
`roles/aiplatform.user` and `roles/datastore.user`, creates a Firestore
database, deploys via `gcloud run deploy ridecourt --source .`, sets
`STATE_BACKEND=firestore` so case state survives instance restarts, allows
up to 3 max instances, and waits for IAM propagation with a retry loop.
See `Dockerfile` for the container image (python:3.12-slim, non-root user,
uvicorn on `$PORT`).

#### Cost guard for the public demo

To keep the live demo within budget, every `/api/disputes/{id}/resolve` call
is counted against a global daily cap (`DAILY_RESOLVE_CAP`, default **300**).
Once the cap is reached the API returns `429 Too Many Requests` with a clear
message saying the limit is exhausted and when it resets (UTC midnight). The
remaining budget is exposed in `GET /api/metrics` as
`daily_resolve_cap`, `resolves_today`, and `remaining_resolves_today`. The
per-IP limit (20/hour) still applies — the daily cap is the outer one.

See `.env.example` for all configuration options.

### Pre-resolve cases (warm the store)

`scripts/preresolve.py` resolves labelled cases in the current process
and writes results to the configured store. This is useful for populating
the store before a demo or for batch processing.

```bash
# Mock, offline (writes to the in-memory store — useful for testing)
LLM_PROVIDER=mock .venv/bin/python scripts/preresolve.py --provider mock

# Gemini (writes to Firestore when STATE_BACKEND=firestore)
LLM_PROVIDER=gemini \
GOOGLE_GENAI_USE_VERTEXAI=true \
GOOGLE_CLOUD_PROJECT=your-project \
GOOGLE_CLOUD_LOCATION=global \
MODEL=gemini-3.8-flash \
STATE_BACKEND=firestore \
.venv/bin/python scripts/preresolve.py --provider gemini
```

Options: `--limit N` (only first N cases), `--force` (re-resolve even if
already in the store), or pass specific dispute IDs as positional args.
The script prints one progress line per case to stderr and continues after
failures.

## Evaluation

The evaluation script (`scripts/evaluate.py`) runs every labelled synthetic
dispute through the full pipeline and reports:

- **Ruling accuracy** vs the policy label, per category (outcome and amount,
  amount within S$0.01)
- **Consistency** — the same case run 3 times gives the same outcome
- **Escalation rate** and whether every safety case escalated
- **Median and p90 time to ruling** and **tokens per case**

Results are written to `eval/results-<date>-<provider>.json`.

### Gemini 3.8 Flash results (13 October 2026)

Run with: `LLM_PROVIDER=gemini python scripts/evaluate.py --provider gemini --repeats 2`
(48 labelled cases, each run twice; full rows, including the guard problems of any escalated run, in `eval/results-2026-10-13-gemini.json`).

| Category | N | Outcome acc. | Amount acc. | Both acc. | Escalation rate | Errored runs | Consistency |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| route_deviation | 16 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| no_show_driver | 10 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| no_show_rider | 10 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| fare_error | 6 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 83.3% |
| safety | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| **Overall** | 48 | **100.0%** | **100.0%** | **100.0%** | **12.5%** | **0.0%** | **97.9%** |

- Every safety case escalated, as the policy requires, and no other case did in its first run.
- Both runs agreed on 47 of 48 cases. In one run of disp_fa_003, Vertex AI kept returning `429 RESOURCE_EXHAUSTED` through the retries, so the case went to a person as `model_unavailable`; its other run was correct. No run errored.
- Median time to ruling 26.5 s (p90 85.9 s); median 9,118 tokens per case. The times include waiting while Vertex AI was busy; when it was not, a case took 15 to 20 s.
- The machine running the evaluation restarted after 36 cases; the other 12 ran with `--resume` on the same code (the evaluation saves after every case).
- What this measures: the labels come from the same policy clauses the system applies, so the evaluation tests whether the multi-agent pipeline reaches the policy result without inventing numbers or escalating needlessly. It is not an independent human labelling. Reading every escalation of earlier runs found real defects, each fixed in the build log: the rider no-show clause read the minutes past the ETA instead of the minutes waited within 150 m (step 11); the Judge was told to cite the bad-faith clause P-BF-1 but the policy engine never applied it, so every high-risk claimant escalated (step 12); the Judge quoted a precedent's distance, which the guard accepted only from this case's evidence (step 15).
- An earlier full run on 11 October (before steps 11 to 16) gave 95.8% outcome accuracy; it is kept in `eval/results-2026-10-11-gemini.json`.

### Mock provider results

Run with: `LLM_PROVIDER=mock python scripts/evaluate.py --provider mock --repeats 2` (offline, a few seconds; it writes `eval/results-<date>-mock.json`).

| Category | N | Outcome acc. | Amount acc. | Both acc. | Escalation rate | Errored runs | Consistency |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| route_deviation | 16 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| no_show_driver | 10 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| no_show_rider | 10 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| fare_error | 6 | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% |
| safety | 6 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 100.0% |
| **Overall** | 48 | **100.0%** | **100.0%** | **100.0%** | **12.5%** | **0.0%** | **100.0%** |

The mock provider only checks that the pipeline, the guard and the policy engine fit together; it states numbers only from the evidence items it cites (step 14b).

## How CodeBuddy was used to build it

The whole project was built spec-first with **CodeBuddy Code** in headless mode. Each build step is one prompt, run as its own CodeBuddy session; CodeBuddy wrote the code and the tests. Between steps a reviewer ran the tests, real Gemini cases and the UI, and wrote what it found into the next prompt, which is why the later steps exist.

| Step | Task | Prompt |
|:--|:--|:--|
| 01 | Foundation: scaffold, data, policy | `prompts/01-foundation.md` |
| 02 | Evidence tools | `prompts/02-evidence-tools.md` |
| 03 | LLM layer + guard | `prompts/03-llm-and-guard.md` |
| 04 | Agents + orchestrator | `prompts/04-agents-orchestrator.md` |
| 04b | Real-run fixes | `prompts/04b-real-run-fixes.md` |
| 04c | Finish rebuttal | `prompts/04c-finish-rebuttal.md` |
| 05 | API + UI | `prompts/05-api-ui.md` |
| 06 | Evaluation | `prompts/06-evaluation.md` |
| 07 | Stretch agents | `prompts/07-stretch-agents.md` |
| 08 | Deploy + docs | `prompts/08-deploy-docs.md` |
| 09 | Polish from UI review | `prompts/09-polish.md` |
| 10 | Docs polish | `prompts/10-docs-polish.md` |
| 11 | Fixes from the real-Gemini evaluation | `prompts/11-eval-findings.md` |
| 12 | Fixes found while recording the demo (P-BF-1, SLA rank, negative extra, visitor IP) | `prompts/12-review-findings.md` |
| 13 | Retry transient model errors; evaluation saves after every case | `prompts/13-resilience.md` |
| 13b | Retry malformed model answers | `prompts/13b-validation-retry.md` |
| 14 | Show the Judge's reasoning and the bad-faith finding | `prompts/14-show-judge-reasoning.md` |
| 14b | Show learning from overrides; mock numbers only from evidence | `prompts/14b-learning-visible-and-mock.md` |
| 15 | Precedent numbers and the guard | `prompts/15-precedent-numbers.md` |
| 15b | Numbers in a claim a person types | `prompts/15b-claim-numbers.md` |
| 16 | Evidence panel layout; Firestore persistence | `prompts/16-layout-and-persistence.md` |

`docs/BUILD_LOG.md` lists every session with its duration, turns, result and test count (from 55 tests after step 1 to 337 after step 16), including the three times CodeBuddy credits ran out. The `CODEBUDDY.md` file enforces the project rules across all sessions: agents argue, code computes; Python 3.12, FastAPI and pydantic v2; tests must pass with `LLM_PROVIDER=mock`; English for all code and docs; commit after each task. The raw session transcripts are kept privately and were submitted with the entry as the CodeBuddy conversation history.

## Stretch goals

| Feature | Status | Description |
|:--|:--|:--|
| Fraud & Bad-Faith Agent | Done | Runs in parallel with advocates; deterministic features (dispute rate, upheld rate P-BF-1, GPS contradiction, prior flags); Judge mentions high-risk signals explicitly |
| Policy & Precedent Agent | Done | TF-IDF keyword similarity over category, facts and claim text; top 3 past rulings; Judge states consistency |
| Escalation Protocol | Done | Triggers: low confidence, guard failure, outcome mismatch, safety; neutral case summary; `GET /api/escalations` queue |
| Learning Feedback Loop | Done | Override stored as precedent (`source="override"`); retrieved for similar cases; "learned from override" in timeline |
| SLA & Routing | Done | Priority = safety first, then amount at stake and age; queue sorts by it |
| Cloud Run deployment | Done | `Dockerfile` + `scripts/deploy.sh`; dedicated SA with `roles/aiplatform.user` + `roles/datastore.user`; Firestore persistence via `STATE_BACKEND=firestore`; one instance, because the live event stream is served by the instance that runs the case |

## Limits

- 48 synthetic cases (16 route deviation, 10 no-show driver, 10 no-show
  rider, 6 fare error, 6 safety). The data loader supports an adapter for
  the organiser's real dataset.
- The mock provider is deterministic but limited — it does not produce
  real natural-language arguments; its purpose is offline testing.
- State is in-memory by default. Set `STATE_BACKEND=firestore` to persist
  case files, events, statuses, overrides, precedents, and the daily
  resolve counter to Firestore so they survive instance restarts and are
  visible across processes. The Cloud Run deploy script enables this
  automatically.
- Single rebuttal round. A production system might allow more rounds with
  a token budget.
- Gemini 3.8 Flash runs on Vertex AI's shared capacity. When it is busy, Vertex AI returns
  429 and RideCourt retries, so a ruling that normally takes 15 to 20 seconds can take a few
  minutes; if the model stays unavailable, the case goes to a person (`model_unavailable`).

## Licence

[Apache-2.0](LICENSE). Copyright 2026 RideCourt Contributors.
