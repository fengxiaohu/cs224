# shared by quota-failover scripts
KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAILOVER_DIR="${FAILOVER_DIR:-$KIT_DIR}"
STATE_FILE="$FAILOVER_DIR/state.json"
DESIRED_FILE="$FAILOVER_DIR/desired-mode"
LOG_FILE="$FAILOVER_DIR/logs/watch.log"
CODEX_MODEL="${CODEX_MODEL:-gpt-6-luna}"
THRESHOLD_PCT="${THRESHOLD_PCT:-95}"
RESET_PCT="${RESET_PCT:-20}"
COOLDOWN_SEC="${COOLDOWN_SEC:-1800}"

export PATH="$HOME/grok-bot-setup:$HOME/.local/bin:/usr/local/bin:/opt/homebrew/bin:$PATH"

adapters_bin() {
  if command -v adapters >/dev/null 2>&1; then
    command -v adapters
  elif [[ -x "$HOME/grok-bot-setup/adapters" ]]; then
    echo "$HOME/grok-bot-setup/adapters"
  else
    return 1
  fi
}

in_sandbox() {
  [[ -d "$HOME/sand-host" || -d "$HOME/sand-data" ]]
}

log() {
  mkdir -p "$(dirname "$LOG_FILE")"
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

notify() {
  local msg="$1"
  log "NOTIFY $msg"
  mkdir -p "$FAILOVER_DIR"
  printf '%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" >>"$FAILOVER_DIR/notify.log"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"$(printf '%s' "$msg" | sed 's/"/\\"/g')\" with title \"Grok Bot quota failover\"" >/dev/null 2>&1 || true
  fi
}

current_provider() {
  local envf="$HOME/sand-data/xai-inference.env"
  if [[ -f "$envf" ]]; then
    grep -E '^SAND_INFERENCE_PROVIDER=' "$envf" 2>/dev/null | head -1 | cut -d= -f2-
  fi
}

write_state() {
  python3 - "$STATE_FILE" "$@" <<'PY'
import json, os, sys, time
path = sys.argv[1]
data = {}
if os.path.exists(path):
    try:
        data = json.load(open(path))
    except Exception:
        data = {}
for item in sys.argv[2:]:
    k, _, v = item.partition("=")
    data[k] = v
data["updated_at"] = int(time.time())
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(data, open(path, "w"), indent=2)
print(path)
PY
}

read_state() {
  python3 - "$STATE_FILE" "${1:-}" <<'PY'
import json, os, sys
path, key = sys.argv[1], sys.argv[2]
if not os.path.exists(path):
    sys.exit(0)
try:
    data = json.load(open(path))
except Exception:
    sys.exit(0)
if key:
    print(data.get(key, ""))
else:
    print(json.dumps(data))
PY
}
