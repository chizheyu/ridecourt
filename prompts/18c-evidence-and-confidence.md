Read CODEBUDDY.md, docs/SPEC.md and data/policy.md. Step 18b (commit 4d53164) fixed the precedents, the dispute value and the labels, but two items are still not fixed on Ryde's cases. Keep this step to these two items.

1. Evidence that contradicts the source data. Built from the Ryde files today:
   - DISP-001 E4 says "Recomputed S$33.09 vs charged S$35.00 (diff S$1.91)".
   - DISP-003 E2 says "Recomputed S$12.38 vs charged S$16.50 (diff S$4.12)", next to a ruling of S$4.50.

   Ryde's data gives no base, per-km or per-min components for these trips, so these recomputations use rates the source does not contain. Other problems:
   - DISP-002 E1 still says "vs ETA +0.0 min" for a booking with a scheduled pickup time.
   - The chat-scan items give counts ("2 agreement(s); 2 disagreement(s)") instead of the messages the ruling depends on.

   Write the test first and run it to confirm it fails on the current code. For each of Ryde's three cases:
   - Collect every number in every evidence item's summary and data.
   - Assert that each one is either a number from the raw JSON (excluding the held-out fields), or a sum, difference or product of two such numbers, or the percentage one raw number is of another (within rounding).
   - Assert that no evidence item reports a value for a field the adapter had to default because the source lacks it.

   Then fix the tools so the test passes:
   - Recompute a fare only from components present in the source.
   - Otherwise show the source's fare lines: estimate or upfront quote, actual, delta, itemised lines, multipliers, surgeable subtotal.
   - Phrase arrival against the scheduled pickup time when the trip has one.
   - Make the chat item quote, with time and sender, the messages that matter for the category: route questions and answers before and after the deviation point; contact messages and calls before cancellation; any surge or fare messages.

   Run the same agreement check on the synthetic cases. There, components exist, so recomputation is allowed, but every evidence item must agree with the policy computation on shared quantities.

2. The §6.3 factors are still constants chosen per branch. Examples:
   - clarity is 0.90 for "both contact obligations met", 0.95 for "clear breach";
   - precedent alignment is 0.95 for any match with similarity of at least 0.4, whatever that precedent decided, so precedent 7.3 (fee voided) counts as support for upholding a fee;
   - credibility is 0.55 for DISP-003 because the rider's account is 45 days old, although the ruling rests on the surge log, not on the rider's word.

   DISP-002 reaches 0.9525, just over the 95% line, because of these constants. Replace them with these rules, each producing a reason line that names its inputs:
   - **Policy clarity** is the minimum over the conditions the applied rule depends on:
     - a numeric condition with value v and threshold t scores min(1.0, 0.6 + 2 × |v − t| / t);
     - a boolean that comes from a system record (app events, notification log, call log, GPS, traffic incident record, photo metadata) scores 1.0;
     - a boolean inferred from free text (keyword scan of chat or claim) scores 0.9.

     The reason lists each condition with its value, threshold and score.
   - **Precedent alignment**:
     - Give every precedent structured `deciding_facts` (for Ryde's three, taken from v2.1's text; for ours, from their scenario).
     - Among the top 3 retrieved precedents in the same category:
       - a precedent whose deciding facts match the case and whose outcome equals the ruling agrees;
       - matching facts with a different outcome is a conflict;
       - different deciding facts means the precedent is distinguished and neither helps nor hurts.
     - Score: 1.0 if at least one agrees and none conflict; 0.5 if any conflict; 0.85 if none agree and none conflict.
     - The reason names the precedents and why each agrees, conflicts or is distinguished.
   - **Party credibility** is 1.0 unless the ruling favours, or rests on the uncorroborated word of, a party with one of these signals; then 0.6, with the signal named:
     - a fraud flag;
     - a statement contradicted by system records;
     - three or more upheld disputes of the same type in 90 days.

     Account age and rating alone never lower it.
   - Evidence completeness stays as it is if every check reads a fact from the source. Confirm that none is always true.

   Expected on Ryde's cases:
   - DISP-001: about 0.975 (clarity 0.9 because "rider informed" is inferred from chat), human by §6.2;
   - DISP-002: 1.0 (cancellation six minutes after the free wait began, against a 5-minute threshold, both contacts in the call and chat records, precedent DISP-2026-0211 distinguished because no contact was made there, the ruling goes against the flagged rider and rests on records), AI final;
   - DISP-003: 1.0, AI final.

   Tests:
   - each rule on the three cases (within 0.02);
   - shrinking the wait margin lowers clarity;
   - a precedent with the same deciding facts and the opposite outcome lowers alignment;
   - a fraud-flagged party winning on their own word lowers credibility;
   - a new account with no other signal does not.

Update data/policy.md (Interpretations) and docs/SPEC.md for the confidence rules. Do not run real Gemini and do not touch Firestore. This is an unattended run: do not stop to ask for approval or to present a plan. Run the whole test suite, commit, and summarize with each official case's evidence items and confidence factors.
