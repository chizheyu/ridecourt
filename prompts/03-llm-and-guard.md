Read CODEBUDDY.md and docs/SPEC.md (sections 2 and 6). Do build-order step 3: the LLM layer and the guard.

1. `ridecourt/llm.py`: one interface `complete_json(system, user, schema: type[BaseModel], agent_name) -> (BaseModel, meta)` where meta has tokens and latency.
   - `LLM_PROVIDER=gemini` (default): `google-genai` SDK in Vertex AI mode (env `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, model from `MODEL`, default `gemini-3.8-flash` (the newest stable Flash on our Vertex project)), structured output with a JSON schema derived from the pydantic model, temperature 0.
   - `LLM_PROVIDER=openai_compatible`: plain httpx call to `${OPENAI_BASE_URL}/chat/completions` with `response_format={"type":"json_object"}`, so Hunyuan or DeepSeek can be swapped in.
   - `LLM_PROVIDER=mock`: deterministic outputs keyed by agent name and case, built from the evidence pack so they always cite real IDs; used by all tests.
   - Retries with backoff on 429/5xx; a per-case token budget (env `CASE_TOKEN_BUDGET`, default 60000) that stops the case and escalates when exceeded.
2. `ridecourt/guard.py`: `check(output, evidence_pack, policy_ids) -> list[Problem]`:
   - every cite must exist in the evidence pack or the policy clause list,
   - every number in agent text (money, km, minutes, metres, percent) must match a number in the cited evidence data or the policy engine computation (compare with units and a small tolerance; S$3.25 must not be backed by 3.25 km),
   - no promises or certainty words the policy does not allow ("guaranteed", "definitely fraud").
   The orchestrator will re-ask the agent once with the problems listed, then escalate.
3. Tests: `tests/test_llm_mock.py`, `tests/test_guard.py` with cases that must fail (invented number, unknown citation, number with the wrong unit) and pass.

Do not call real Gemini in tests. Run pytest until green, commit, summarize.
