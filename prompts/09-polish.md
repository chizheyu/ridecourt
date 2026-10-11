Read CODEBUDDY.md and docs/SPEC.md. Review fixes from running the UI and the data (do these, keep all tests green):

1. Negative extra time. A route-deviation ruling showed `extra_min=-2.9 × S$0.25/min = S$-0.72` inside the refund computation: the detour was 6.87 km longer yet 2.9 minutes shorter. Two fixes:
   - Policy (P-RD-1): extra distance and extra time each count only when positive (`max(0, …)`); state that in `data/policy.md` and in the computation string. Add tests.
   - Data generator: a detour must take longer than the optimal route (consistent speeds), so regenerate `data/synthetic/` with seed 7, recompute `labels.json` with the policy engine, and make sure tests and the evaluation still run. Do not hand-edit labels.
2. Leaflet error in the browser console: "Cannot read properties of undefined (reading '_leaflet_pos')". It appears when switching cases (the map is re-created or fitted while its container is hidden or removed). Create the map once, reuse it, call `invalidateSize()` after the panel is visible, and clear layers instead of removing the map. Check in a real browser that switching between three cases produces no console errors (use Playwright if available: `~/venv/automation/bin/python` has it).
3. In the ruling card and the Judge's messages, show amounts as plain money ("S$7.96"), never as negative or with internal field names; outcomes as human words ("Refund for route deviation") with the code underneath in small text.

Run pytest until green, commit, summarize.
