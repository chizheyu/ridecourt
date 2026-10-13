# Build log

RideCourt was built with CodeBuddy Code (Tencent) in headless mode, one task per prompt in `prompts/`. Each row is one CodeBuddy session, 11 to 13 October 2026 (Singapore time); the full session transcripts are kept privately as evidence and are summarised here.

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

Sessions stopped three times when CodeBuddy credits ran out ("429 Credits exhausted"): the free credits during step 4b (step 4c finished that work after the hackathon credits were redeemed), the hackathon credits during step 11, and the free daily credits on the first try of step 12 (rerun after upgrading to CodeBuddy Pro).

Between tasks a reviewer ran the tests, ran real Gemini cases and looked at the UI, and wrote the fixes it found into the next prompt (4b, 4c, 9, 10, 11, 12, 13, 13b and 14 exist for that reason). Step 12 came from the demo video: recording it showed that every high-risk claimant escalated, because the Judge was told to cite the bad-faith clause P-BF-1 but the policy engine never applied it, so the guard rejected the citation. Step 13 came from the full evaluation: one dropped connection to the model ended the whole run and lost every result, because a model error in an optional step escaped the orchestrator and the evaluation saved only at the end. Step 14 came from the video as well: the Judge's bad-faith finding was in its reasoning, which the interface did not show.
