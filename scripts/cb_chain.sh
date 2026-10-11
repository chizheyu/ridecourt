#!/bin/sh
# Run several build tasks through CodeBuddy one after another; stop at the first task whose tests fail.
# Usage: scripts/cb_chain.sh 02-evidence-tools 03-llm-and-guard ...
set -u
for name in "$@"; do
  echo "== $(date +%H:%M:%S) start $name"
  log=$(./scripts/cb_task.sh "$name" "prompts/$name.md")
  # Stop on a CodeBuddy error (for example "429 Credits exhausted"); passing tests alone prove nothing
  # when the task did not run.
  err=$(python3 -c "import json,sys
r=[json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith('{') and '\"type\": \"result\"' in l or '\"type\":\"result\"' in l]
d=r[-1] if r else {}
print('' if d and not d.get('is_error') else (d.get('errors') or ['no result event'])[0][:200])" "$log" 2>/dev/null)
  if [ -n "$err" ]; then echo "== $(date +%H:%M:%S) CODEBUDDY ERROR in $name: $err"; exit 2; fi
  if LLM_PROVIDER=mock .venv/bin/python -m pytest -q > "devlog/pytest-$name.txt" 2>&1; then
    echo "== $(date +%H:%M:%S) done $name: $(tail -1 devlog/pytest-$name.txt)"
  else
    echo "== $(date +%H:%M:%S) TESTS FAILED after $name: $(tail -1 devlog/pytest-$name.txt)"; exit 1
  fi
  git add devlog/ >/dev/null 2>&1 && git commit -q -m "Devlog: CodeBuddy transcript for $name" >/dev/null 2>&1
done
echo "== chain finished"
