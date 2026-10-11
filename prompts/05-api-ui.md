Read CODEBUDDY.md and docs/SPEC.md (sections 7 and 8). Do build-order step 5: the API and the web UI.

1. `ridecourt/api.py` (FastAPI) with the endpoints in SPEC section 7. `POST /api/disputes/{id}/resolve` starts the pipeline in the background and returns at once; `GET /api/disputes/{id}/events` streams the event log with Server-Sent Events (replays past events first, then live ones, then a final "done" event). Serve `web/` as static files at `/`. Health check at `/healthz`. Simple per-IP rate limit on resolve (for example 20 per hour) so a public demo cannot burn the model budget.
2. `web/index.html`, `web/app.js`, `web/style.css` (no build step):
   - Left: dispute queue (category, SLA priority badge, status). Safety cases at the top.
   - Centre: the "courtroom" timeline. Rider Advocate bubbles on the left, Driver Advocate on the right, Judge centred, evidence and tool calls as small chips with their E# IDs; clicking an E# chip opens that evidence on the right. It streams live while a case resolves.
   - Right: evidence panel with a Leaflet map (OpenStreetMap tiles) drawing the actual GPS trace and the optimal route in two colours, the fare table, chat excerpts with the quoted lines highlighted; below it the ruling card: outcome, amount, confidence bar, the computation string from the policy engine, the cited clauses, the two messages to rider and driver, and Escalate / Override buttons (override asks for outcome, amount and a reason).
   - A "File a dispute" form (pick a trip, category, claim text).
   - Clean, calm design: one accent colour, readable at 1280 px and on a phone. English only.
3. Tests: `tests/test_api.py` with FastAPI TestClient and `LLM_PROVIDER=mock`: list, resolve, events stream ends with RULING or ESCALATION, override stores a precedent.
4. Run the server locally with `LLM_PROVIDER=mock` and load the page once to check it renders (curl the HTML and the JS; make sure there are no missing files).

Run pytest until green, commit, summarize.
