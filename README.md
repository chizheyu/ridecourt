# RideCourt

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

## How it works

1. **Evidence Collection** — deterministic tools build an evidence pack
   (GPS analysis, arrival check, chat scan, fare recompute, history
   profile).
2. **Rider Advocate & Driver Advocate** — run in parallel, each argues for
   its side citing evidence and policy.
3. **Fraud & Bad-Faith Agent** (stretch) — runs in parallel with the
   advocates; computes deterministic fraud signals (dispute rate, upheld
   rate per P-BF-1, GPS contradiction, prior flags) and outputs a risk
   level with cited signals. The LLM only explains the signals.
4. **Precedent Agent** (stretch) — retrieves the 3 most similar past
   rulings from `precedents.json` using TF-IDF keyword similarity over
   category, facts and claim text (no external vector DB).
5. **Policy Engine** — pure functions compute the outcome and amount per
   the policy clauses.
6. **Judge** — weighs both cases, applies the policy engine result,
   considers risk signals and precedents, and issues a ruling (or
   escalates if confidence is low or the outcome disagrees with policy).
   The Judge states whether the ruling is consistent with the precedents
   and mentions high-risk bad-faith signals explicitly (P-BF-1).
7. **Safety incidents** always escalate to a human reviewer with a neutral
   case summary.
8. **Escalation Protocol** (stretch) — confidence below 0.7, guard failure
   after one retry, outcome mismatch, or safety category creates an
   escalation record with a neutral case summary in a queue visible via
   `GET /api/escalations`.
9. **Learning Feedback Loop** (stretch) — a human override is stored as a
   new precedent with `source="override"` and the reviewer's reason; it is
   retrieved for later similar cases and the timeline shows "learned from
   override" when such a precedent is used.
10. **SLA & Routing** (stretch) — priority score: safety first, then
    amount at stake and age; the dispute queue sorts by it.

## How to run

```bash
# Install dependencies
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests (offline, mock LLM)
LLM_PROVIDER=mock python -m pytest -q

# Start the API + UI
LLM_PROVIDER=mock uvicorn ridecourt.api:app --reload

# Run the evaluation script
LLM_PROVIDER=mock python scripts/evaluate.py --provider mock
```

See `.env.example` for configuration options. Set `LLM_PROVIDER=gemini` or
`openai_compatible` to use a real LLM.

## Evaluation

The evaluation script (`scripts/evaluate.py`) runs every labelled synthetic
dispute through the full pipeline and reports ruling accuracy against
`data/synthetic/labels.json` per category (outcome and amount, amount within
S$0.01), consistency (each case run 3 times), escalation rate, whether
every safety case escalated, median and p90 time to ruling, and tokens per
case.

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

- Safety cases all escalated: yes
- Median time to ruling: 1.0 ms
- p90 time to ruling: 1.0 ms
- Median tokens per case: 500

> The Gemini run is reported separately — see `eval/results-<date>-gemini.json`
> after running `LLM_PROVIDER=gemini python scripts/evaluate.py --provider gemini`.

**Note on fare_error escalation:** the mock provider's canned advocate
output triggers the guard (it generates text containing "0.00 km" which
does not match any cited evidence value), causing all fare_error cases to
escalate. With a real LLM provider, advocates cite the correct evidence
and the guard passes.

## Architecture

- **Pipeline:** Evidence Collection → Precedent Retrieval →
  Rider/Driver Advocates + Fraud Agent (parallel) → Policy Engine →
  Judge (with risk + precedents) → Ruling or Escalation
- **Guard:** every agent output is validated against the evidence pack;
  unknown citation IDs or numbers not in the evidence cause a re-ask, then
  escalation.
- **Event log:** every inter-agent message is a typed event
  (`DISPUTE_FILED`, `EVIDENCE`, `CLAIM`, `REBUTTAL`, `RISK_SIGNAL`,
  `PRECEDENT`, `RULING`, `ESCALATION`, `HUMAN_OVERRIDE`, …) with tokens
  and latency, streamed live via SSE.
- **LLM layer:** pluggable — `gemini`, `openai_compatible`, or `mock` for
  offline tests and demos.
- **Fraud detection:** deterministic features (dispute rate, upheld rate,
  GPS contradiction, prior flags) computed first; the LLM only explains.
- **Precedent retrieval:** in-memory TF-IDF over category + keyword
  similarity; human overrides become new precedents (learning loop).
- **Escalation queue:** `GET /api/escalations` shows all open escalations
  with neutral case summaries for human reviewers.

See `docs/architecture.svg` for the diagram.

## Stretch goals

| Feature | Status | Description |
|:--|:--|:--|
| Fraud & Bad-Faith Agent | Done | Runs in parallel with advocates; deterministic features (dispute rate, upheld rate P-BF-1, GPS contradiction, prior flags); Judge mentions high-risk signals explicitly |
| Policy & Precedent Agent | Done | TF-IDF keyword similarity over category, facts and claim text; top 3 past rulings; Judge states consistency |
| Escalation Protocol | Done | Triggers: low confidence, guard failure, outcome mismatch, safety; neutral case summary; `GET /api/escalations` queue |
| Learning Feedback Loop | Done | Override stored as precedent (`source="override"`); retrieved for similar cases; "learned from override" in timeline |
| SLA & Routing | Done | Priority = safety first, then amount at stake and age; queue sorts by it |

## Limits

- 48 synthetic cases (16 route deviation, 10 no-show driver, 10 no-show
  rider, 6 fare error, 6 safety). The data loader supports an adapter for
  the organiser's real dataset.
- The mock provider is deterministic but limited — it does not produce
  real natural-language arguments; its purpose is offline testing.

## Licence

Apache-2.0.
