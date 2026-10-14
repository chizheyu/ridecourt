# Build log

RideCourt was built with CodeBuddy Code (Tencent) in headless mode, one task per prompt in `prompts/`. Each row is one CodeBuddy session, 11 to 14 October 2026 (Singapore time); the full session transcripts are kept privately as evidence and are summarised here.

| Start | Task (prompt) | Minutes | Turns | Result | Tests after the task |
|:--|:--|--:|--:|:--|:--|
| 16:13 | `prompts/01-foundation.md` | 12.1 | 107 | done | 55 passed |
| 16:26 | `prompts/02-evidence-tools.md` | 7.1 | 123 | done | 81 passed in 0.12s |
| 16:33 | `prompts/03-llm-and-guard.md` | 5.6 | 105 | done | 110 passed in 0.16s |
| 16:40 | `prompts/04-agents-orchestrator.md` | 11.3 | 253 | done | 133 passed in 0.19s |
| 16:52 | `prompts/04b-real-run-fixes.md` | 6.4 | 161 | stopped: 429 Credits exhausted | 140 passed (items 1-3 done, committed) |
| 16:59 | `prompts/05-api-ui.md` | 0.0 | 2 | stopped: 429 Credits exhausted | (no change) |
| 16:59 | `prompts/06-evaluation.md` | 0.0 | 2 | stopped: 429 Credits exhausted | (no change) |
| 18:57 | `prompts/04c-finish-rebuttal.md` | 5.5 | 126 | done | 145 passed in 0.31s |
| 19:03 | `prompts/05-api-ui.md` | 9.9 | 177 | done | 169 passed, 1 warning in 3.18s |
| 19:13 | `prompts/06-evaluation.md` | 3.8 | 111 | done | 180 passed, 1 warning in 3.19s |
| 19:17 | `prompts/07-stretch-agents.md` | 16.1 | 358 | done | 222 passed, 1 warning in 4.88s |
| 19:33 | `prompts/08-deploy-docs.md` | 9.2 | 154 | done | 222 passed, 1 warning in 4.98s |
| 19:43 | `prompts/09-polish.md` | 10.4 | 243 | done | 224 passed, 1 warning in 4.95s |
| 19:54 | `prompts/10-docs-polish.md` | 10.9 | 190 | done | 229 passed, 1 warning in 4.83s |
| 20:31 | `prompts/11-eval-findings.md` | 10.8 | 289 | stopped: 429 Credits exhausted (items 1-3 done; the reviewer ran the tests and committed) | 245 passed |
| 12 Oct 22:21 | `prompts/12-review-findings.md` | 3.7 | 75 | stopped: 429 Credits exhausted while reading the code | (no change) |
| 13 Oct 10:18 | `prompts/12-review-findings.md` | 12.1 | 209 | done | 256 passed, 1 warning in 5.13s |
| 13 Oct 11:06 | `prompts/13-resilience.md` | 26.4 | 287 | done | 280 passed |
| 13 Oct 11:48 | `prompts/13b-validation-retry.md` | 3.3 | 79 | stopped: CodeBuddy lost its own network connection | (uncommitted edits) |
| 13 Oct 11:53 | same session resumed (`codebuddy -r`) | 1.7 | 140 | done | 286 passed |
| 13 Oct 12:40 | `prompts/14-show-judge-reasoning.md` | 8.2 | 138 | done | 290 passed, 1 warning in 5.17s |
| 13 Oct 12:54 | `prompts/14b-learning-visible-and-mock.md` | 10.3 | 221 | done | 294 passed, 1 warning in 5.23s |
| 13 Oct 13:22 | `prompts/15-precedent-numbers.md` | 18.8 | 220 | done | 299 passed, 1 warning in 5.27s |
| 13 Oct 13:42 | `prompts/15b-claim-numbers.md` | 10.0 | 199 | done | 322 passed, 3 skipped, 1 warning in 6.00s |
| 13 Oct 17:00 | `prompts/16-layout-and-persistence.md` | 5.2 | 46 | stopped: entered plan mode and asked for approval (layout committed) | 2 failed, 335 passed |
| 13 Oct 17:34 | same session resumed: "proceed with the plan" | 1.8 | 91 | stopped: CodeBuddy lost its own network connection | (uncommitted edits) |
| 13 Oct 17:42 | same session resumed after the network error | 1.1 | 110 | done | 337 passed, 3 skipped, 1 warning in 6.06s |
| 13 Oct 18:17 | `prompts/17-events-from-store.md` | 6.6 | 133 | done | 352 passed, 3 skipped, 1 warning in 12.60s |
| 14 Oct 07:57 | `prompts/18-ryde-policy-v2.1.md` | 16.9 | 411 | stopped: 429 Credits exhausted (all but one test passing, uncommitted) | 1 failed, 428 passed |
| 14 Oct 09:05 | same session resumed after topping up credits | 5.0 | 134 | done | 446 passed, 3 skipped |
| 14 Oct 09:17 | `prompts/18b-v2.1-review-findings.md` | 1.1 | 37 | stopped: entered plan mode and could not leave it | (plan only) |
| 14 Oct 09:22 | same session resumed with the plan approved | 1.6 | 84 | done | 485 passed, 3 skipped |
| 14 Oct 09:55 | `prompts/18c-evidence-and-confidence.md` | 6.4 | 142 | stopped: 429 Credits exhausted (committed as it stood, tests passing) | 503 passed, 3 skipped |

Sessions stopped three times when CodeBuddy credits ran out ("429 Credits exhausted"): the free credits during step 4b (step 4c finished that work after the hackathon credits were redeemed), the hackathon credits during step 11, and the free daily credits on the first try of step 12 (rerun after upgrading to CodeBuddy Pro).

Between tasks a reviewer ran the tests, ran real Gemini cases and looked at the UI, and wrote the fixes it found into the next prompt (4b, 4c and 9 to 17 exist for that reason). Step 12 came from the demo video: recording it showed that every high-risk claimant escalated, because the Judge was told to cite the bad-faith clause P-BF-1 but the policy engine never applied it, so the guard rejected the citation. Step 13 came from the full evaluation: one dropped connection to the model ended the whole run and lost every result, because a model error in an optional step escaped the orchestrator and the evaluation saved only at the end. Step 14 came from the video as well: the Judge's bad-faith finding was in its reasoning, which the interface did not show. Step 15 came from reading the two remaining escalations of the full evaluation: the Judge quoted a precedent's distance (400 m), which the guard accepted only from this case's evidence; step 15b closed the same gap for numbers in a claim a person types in. Step 16 came from recording the demo on the live app: the ruling card crowded out the evidence list, and all state was lost whenever the Cloud Run instance restarted. Step 17 came from opening the pre-resolved cases on the live app: they were stored with their events, but the case file and the event stream did not read the events back. Step 18 came from Ryde's sample dataset, shared with participants on 13 October: Policy v2.1 and three cases with expected rulings, which our own policy got wrong on two of three. Steps 18b and 18c came from checking step 18 against those three cases: an evidence item said the detour was 0.25 km (straight lines between sparse GPS samples) while Ryde's trip record and the ruling said 4.3 km; fares were recomputed from rates the data does not contain; two of Ryde's precedents were restated with changed numbers; and the §6.3 confidence was built from constants, so any fraud flag sent a case to a person and DISP-002 missed Ryde's expected answer.

CodeBuddy credits ran out twice more on 14 October (steps 18 and 18c). The remaining work after step 18c was completed with other tools assisting.
