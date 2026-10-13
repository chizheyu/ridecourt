#!/bin/sh
# Run one build task through CodeBuddy Code in headless mode and keep the full transcript.
# Usage: scripts/cb_task.sh <task-name> <prompt-file>
# If the session ends with a network error (not "Credits exhausted"), or stops to ask for
# approval of a plan (CodeBuddy sometimes enters plan mode on its own), it is resumed in the
# same CodeBuddy session up to two times, each into its own transcript file.
set -eu
name="$1"; prompt_file="$2"
ts=$(date +%Y%m%d-%H%M%S)
mkdir -p devlog
out="devlog/${ts}-${name}.jsonl"
codebuddy -p "$(cat "$prompt_file")" -y --output-format stream-json --verbose \
  > "$out" 2> "devlog/${ts}-${name}.stderr.log" || true
echo "$out"

# Prints "<session id> <network|asks|other|ok>" for the last result event of a transcript.
last_result() {
  python3 - "$1" <<'EOF'
import json, sys
res = None
for line in open(sys.argv[1], encoding="utf-8"):
    try:
        e = json.loads(line)
    except ValueError:
        continue
    if e.get("type") == "result":
        res = e
if res is None:
    print("- other")
elif not res.get("is_error"):
    text = (res.get("result") or "")[-600:].lower()
    asks = any(k in text for k in ("would you like me to proceed", "ready for your review", "shall i proceed", "should i proceed", "do you want me to"))
    print(res.get("session_id", "-"), "asks" if asks else "ok")
else:
    err = " ".join(res.get("errors") or [])
    network = "429" not in err and any(k in err for k in ("TLS", "socket", "ECONNRESET", "ETIMEDOUT", "502", "503", "network"))
    print(res.get("session_id", "-"), "network" if network else "other")
EOF
}

for attempt in 1 2; do
  set -- $(last_result "$out")
  case "$2" in
    network) sleep 30; msg="The session was cut by a network error before you finished. Check what is already changed (git diff), finish the task, run the whole test suite, commit and summarize." ;;
    asks) msg="Yes, proceed and implement everything, then run the whole test suite, commit and summarize. This is an unattended run: do not stop to ask for approval again." ;;
    *) break ;;
  esac
  out="devlog/${ts}-${name}-resume${attempt}.jsonl"
  codebuddy -r "$1" -p "$msg" \
    -y --output-format stream-json --verbose > "$out" 2> "devlog/${ts}-${name}-resume${attempt}.stderr.log" || true
  echo "$out"
done
