# CODEBUDDY.md

Project: RideCourt, a multi-agent dispute resolution system for ride-hailing (Tencent Cloud AI CAN DO IT
Hackathon Singapore 2026, Ryde challenge). The full spec is `docs/SPEC.md`; read it before any task.

## Rules for every task

- Agents argue, code computes: numbers in rulings come from `ridecourt/policy.py` and `ridecourt/tools.py`,
  never from model text. Agent claims cite evidence IDs (`E#`) and policy clause IDs (`P-..`).
- Python 3.12, FastAPI, pydantic v2, pytest. Keep modules small and typed.
- Tests must run offline with `LLM_PROVIDER=mock`. Run `python -m pytest -q` before saying a task is done,
  and fix failures instead of skipping tests.
- No secrets in the repository. Configuration comes from environment variables (`.env.example`).
- English for code, comments, UI text and docs.
- Commit after each finished task with a short message saying what changed.
- Do not add features outside the spec without saying so in your final summary.
