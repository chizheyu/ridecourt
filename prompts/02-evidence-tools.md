Read CODEBUDDY.md and docs/SPEC.md. Do build-order step 2: deterministic evidence tools in `ridecourt/tools.py` (no LLM).

Each tool returns evidence items `{id, kind, summary, data, source}` with stable IDs E1, E2, ... per case:
- `gps_analysis(trip)`: actual vs optimal distance and duration, deviation percent, detour segment (start/end timestamps), unexpected stops (speed below 3 km/h for over 2 minutes away from pickup/dropoff)
- `arrival_check(trip)`: driver's closest distance to pickup, time of arrival vs ETA, minutes waited within 150 m
- `chat_scan(chats)`: requested route changes or stops (quote the line), road closure mentions, agreements and disagreements, threatening or abusive language (quote the line), contact attempts (chat and calls)
- `fare_check(trip)`: recompute the fare from components and compare with the total charged; promo applied or not
- `history_profile(profiles, rider_id, driver_id)`: dispute counts, upheld rates, account age, ratings, prior flags
- `build_evidence_pack(dispute, dataset) -> EvidencePack` that runs the relevant tools for the category and numbers the items.

Then make `compute_facts` read from the evidence pack so the policy engine and the agents use the same numbers. Summaries must state numbers exactly as in `data` (one source of truth).

Tests in `tests/test_tools.py` using hand-built small trips (not only the generated data): a detour of known length, a driver who stopped 600 m from pickup, a chat with "can you stop at the 7-Eleven", a threatening message, a promo not applied. Run pytest until green, commit, and summarize.
