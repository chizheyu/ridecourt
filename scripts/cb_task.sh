#!/bin/sh
# Run one build task through CodeBuddy Code in headless mode and keep the full transcript.
# Usage: scripts/cb_task.sh <task-name> <prompt-file>
set -eu
name="$1"; prompt_file="$2"
ts=$(date +%Y%m%d-%H%M%S)
mkdir -p devlog
codebuddy -p "$(cat "$prompt_file")" -y --output-format stream-json --verbose \
  > "devlog/${ts}-${name}.jsonl" 2> "devlog/${ts}-${name}.stderr.log"
echo "devlog/${ts}-${name}.jsonl"
