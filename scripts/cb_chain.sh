#!/bin/sh
# Run several build tasks through CodeBuddy one after another; stop at the first task whose tests fail.
# Usage: scripts/cb_chain.sh 02-evidence-tools 03-llm-and-guard ...
set -u
for name in "$@"; do
  echo "== $(date +%H:%M:%S) start $name"
  log=$(./scripts/cb_task.sh "$name" "prompts/$name.md")
  if LLM_PROVIDER=mock .venv/bin/python -m pytest -q > "devlog/pytest-$name.txt" 2>&1; then
    echo "== $(date +%H:%M:%S) done $name: $(tail -1 devlog/pytest-$name.txt)"
  else
    echo "== $(date +%H:%M:%S) TESTS FAILED after $name: $(tail -1 devlog/pytest-$name.txt)"; exit 1
  fi
  git add devlog/ >/dev/null 2>&1 && git commit -q -m "Devlog: CodeBuddy transcript for $name" >/dev/null 2>&1
done
echo "== chain finished"
