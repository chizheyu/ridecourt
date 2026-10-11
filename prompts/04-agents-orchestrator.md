Read CODEBUDDY.md and docs/SPEC.md (sections 3, 6). Do build-order step 4: the three core agents, the Judge rules, the orchestrator and the event log.

1. `ridecourt/agents/advocate.py`: Rider Advocate and Driver Advocate share one class with a side parameter. Prompt: argue for your side using only the evidence pack and the policy; every argument cites E# and P- IDs; acknowledge evidence against your side briefly (judges reward candour). Output schema per SPEC section 6.
2. `ridecourt/agents/judge.py`: receives both cases, the policy engine result, risk signals and precedents when present. It must use the policy engine's outcome and amount; if it disagrees it escalates instead of overriding. Output: outcome, amount, confidence 0..1, reasoning, cites, message_to_rider, message_to_driver (plain English, polite, specific). Confidence below 0.7 escalates. Safety category always escalates with a neutral case summary.
3. `ridecourt/events.py`: typed event log per SPEC section 3 (DISPUTE_FILED, EVIDENCE_REQUEST, EVIDENCE, CLAIM, REBUTTAL, RULING, ESCALATION, ...), in memory plus append-only JSONL under `runs/` so a run can be replayed; an async subscriber API for streaming.
4. `ridecourt/orchestrator.py`: `resolve(dispute_id) -> CaseFile`: build evidence pack (EVIDENCE_REQUEST/EVIDENCE events), run both advocates in parallel (CLAIM events), give each the other's claim for one short rebuttal round (REBUTTAL events), run the guard on every agent output (re-ask once, then escalate), call the policy engine, then the Judge (RULING or ESCALATION). Record tokens and latency per step.
5. Tests with `LLM_PROVIDER=mock`: a route deviation case ends with the label's outcome and amount; a requested-stop case ends with no refund; a no-show case; a safety case escalates; a guard failure leads to a re-ask and then escalation; the event log has the expected sequence.

Run pytest until green, commit, summarize.

Fixes from review of step 3 (do these first):
- Local machines here export `all_proxy=socks5://...`; httpx then needs SOCKS support. Add `httpx[socks]` to requirements.txt and install it.
- Default `GOOGLE_CLOUD_LOCATION` to `global` (gemini-3.8-flash is served from the global endpoint, not asia-southeast1).
- Gemini thinking adds latency (a one-line call took about 13 s). Make the thinking setting configurable by env (`GEMINI_THINKING`, default low) using the google-genai thinking config, and record latency per call in meta.

After the mock tests pass, run one real end-to-end case with Gemini to prove the integration:
`LLM_PROVIDER=gemini GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=no-fine-print-2026 GOOGLE_CLOUD_LOCATION=global MODEL=gemini-3.8-flash .venv/bin/python scripts/smoke_case.py <a route-deviation dispute id>`
(write `scripts/smoke_case.py`: resolves one case and prints the event sequence, guard problems if any, the ruling, total tokens and seconds). gcloud credentials are already set up on this machine. Include its output in your final summary. Real calls must never run inside pytest.
