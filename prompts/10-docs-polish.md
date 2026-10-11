Read CODEBUDDY.md. Polish the two images in docs/ (reviewed by eye at full size):

1. `docs/architecture.svg` / `.png`: every "→" arrow character renders as an empty box ("tofu") in the PNG because the rendering font lacks the glyph (e.g. "□ position + rebuttal round", "re-ask once □ escalate", "Confidence < 0.7 □ escalate", "□ stored as precedent", "□ Facts (numbers)"). Replace those characters with words or ASCII ("then", "->"), or embed a font that has them. In the Advocates box the "Fraud & Bad-Faith Agent" sub-box overlaps its own text; give each sub-box enough height. Reduce crossing lines where you can (route arrows around boxes). Re-render the PNG at 1600x1000 and check it by viewing it.
2. `docs/cover.png` and `docs/cover-380x216.png`: the gavel overlaps the "DISPUTE RESOLUTION SYSTEM" label. Rebalance the composition: courthouse columns and gavel above, then the label, title and tagline, with no overlaps and less empty space; make sure the 380x216 version is still readable (title large enough).

View both PNGs after rendering and fix anything that is cut, overlapping or unreadable. Commit and summarize.

3. Cost guard for the public demo: besides the per-IP limit, add a global daily cap on resolves (env `DAILY_RESOLVE_CAP`, default 300) that returns a clear 429 message when reached; show remaining resolves in `/api/metrics`. Add a test. Mention it in the README's deploy section.
