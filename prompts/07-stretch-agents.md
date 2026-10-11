Read CODEBUDDY.md and docs/SPEC.md (sections 3, 6). The MVP works; now add the stretch goals that earn bonus points, keeping all tests green:

1. Fraud & Bad-Faith Detection Agent: runs in parallel with the advocates using history_profile and the evidence pack; outputs risk level and signals with citations; the Judge must mention high-risk signals explicitly (P-BF-1). Deterministic features first (dispute rate, upheld rate, GPS contradicting the claim), the LLM only explains them.
2. Policy & Precedent Agent: retrieve the 3 most similar past rulings from `precedents.json` (simple TF-IDF or keyword similarity over category, facts and claim text; no external vector DB) and give them to the Judge, which must say whether its ruling is consistent with them.
3. Escalation Protocol: confidence below 0.7, guard failure after one retry, agent disagreement with the policy engine, or safety category: create an escalation with a neutral case summary for a human reviewer; show an escalation queue in the UI.
4. Learning Feedback Loop: a human override is stored as a new precedent (with the reviewer's reason) and is used for later similar cases; show "learned from override" in the timeline when a precedent from an override is used.
5. SLA & Routing: priority score (safety first, then amount at stake and age); the queue sorts by it.

Add tests for each. Update README. Run pytest until green, commit, summarize.
