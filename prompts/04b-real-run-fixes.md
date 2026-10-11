Read CODEBUDDY.md and docs/SPEC.md. A real Gemini run of `scripts/smoke_case.py disp_rd_000` escalated a straightforward route-deviation case:

  [unverified_number] Number 2.9 minutes (2.9 min) in field 'arguments[0].text' does not match any cited evidence or policy computation value with the same unit.
  ESCALATED reason: guard_failure_rebuttal (policy outcome refund_route_deviation, S$7.96, which matches the label)

The guard did its job, but the system must resolve the majority of standard cases without a human (Ryde's brief). Fix the causes, not the symptom:

1. Evidence: add the derived quantities agents naturally talk about as explicit evidence values (extra_km, extra_min, deviation percent, wait minutes past ETA, fare difference, and the policy engine's computation line with its numbers), so an agent can cite them instead of doing arithmetic. Show values with the precision a person would use (two decimals for money, one for km and minutes) next to the raw value.
2. Guard: a number written with k decimals matches an evidence value v of the same unit when round(v, k) equals it (2.9 min matches 2.93 min; 3 min matches 2.93 min; 3.1 min does not). Keep units strict and keep the rest of the guard as is. Add tests for these cases.
3. Prompts: tell advocates and the Judge to copy numbers exactly as shown in the evidence and never compute new numbers; if they need a difference, cite the evidence item that already states it.
4. Orchestration: a rebuttal is optional. If a rebuttal still fails the guard after the one re-ask, drop that rebuttal (record a GUARD_REJECTED event that shows the problem in the timeline) and continue to the Judge with the original claims. Escalate on guard failure only for the initial claims or the Judge's ruling. Update tests.

Then run `scripts/smoke_case.py` with real Gemini (same env as before: LLM_PROVIDER=gemini GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=no-fine-print-2026 GOOGLE_CLOUD_LOCATION=global MODEL=gemini-3.8-flash) on disp_rd_000, disp_rd_008, disp_ns_r_000 and one fare_error case, and include each case's outcome, amount, label match and any guard problems in your summary. Run pytest until green, commit.
