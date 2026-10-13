Read CODEBUDDY.md. Step 13 made `PermanentLLMError` cover a model response that fails pydantic validation (in `_complete_gemini`, `_complete_openai` and `_complete_mock`). That is not what "errors a retry cannot fix" meant: a malformed or incomplete JSON answer from the model is often fixed by asking again, and before step 13 it was retried. Only request-side errors are permanent (HTTP 400 such as an invalid response schema, 401, 403).

1. Treat a response that fails JSON parsing or pydantic validation as retryable, but with a smaller budget than transport errors: at most 2 more attempts and no long backoff. Keep 400, 401 and 403 permanent. Keep the mock provider's validation failure permanent (it is a bug in the mock, not model variance).
2. Tests: a fake Gemini client that returns invalid JSON once and then a valid answer succeeds; one that always returns invalid JSON fails after 3 attempts in total; a 400 still fails on the first attempt. Make sure the tests fail on the current code.

Do not run real Gemini. Commit and summarize.
