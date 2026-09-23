#!/usr/bin/env bash
# Install official-Grok-first → Codex failover.
# Run this inside the Grok Bot sandbox for adapters switching.
# On the Mac it only installs the watcher + notifications.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib.sh
source "$ROOT/lib.sh"

chmod +x "$ROOT/bin/"* "$ROOT/fetch_usage.py" 2>/dev/null || true
mkdir -p "$ROOT/logs"

if in_sandbox; then
  log "sandbox detected; ensuring grok-bot-setup"
  if ! adapters_bin >/dev/null; then
    curl -fsSL https://raw.githubusercontent.com/BlockedPath/grok-bot-setup/main/scripts/bootstrap.sh | bash
  fi
  bin="$(adapters_bin)" || {
    log "ERROR: adapters still missing after bootstrap"
    exit 1
  }
  "$bin" install openai-oauth || true
  "$bin" install login-agents || true
  if [[ ! -f "$HOME/.codex/auth.json" ]]; then
    log "WARNING: no ~/.codex/auth.json — run: codex login"
  else
    "$bin" start openai-oauth || log "openai-oauth start failed (will retry on use-codex)"
  fi

  want="$(python3 - "$ROOT/fetch_usage.py" "$THRESHOLD_PCT" <<'PY' || true
import json, subprocess, sys
thr = float(sys.argv[2])
try:
    u = json.loads(subprocess.check_output([sys.executable, sys.argv[1]], text=True))
except Exception:
    u = {"ok": False}
if not u.get("ok"):
    print("stock")
    raise SystemExit(0)
auto = float(u.get("auto_percent") or 0)
if u.get("included_exhausted") or u.get("hit_limit") or auto >= thr:
    print("codex")
else:
    print("stock")
PY
)"
  log "initial want=$want"
  if [[ "$want" == "codex" ]]; then
    "$ROOT/bin/use-codex"
  else
    "$ROOT/bin/use-stock"
  fi

  cron_line="17 */2 * * * PATH=\"$HOME/grok-bot-setup:$HOME/.local/bin:/usr/bin:/bin\" $ROOT/bin/quota-watch >>$ROOT/logs/cron.log 2>&1"
  tmp="$(mktemp)"
  crontab -l 2>/dev/null | grep -v grok-bot-quota-failover/bin/quota-watch | grep -v "$ROOT/bin/quota-watch" >"$tmp" || true
  echo "$cron_line" >>"$tmp"
  crontab "$tmp"
  rm -f "$tmp"
  log "installed crontab every 2 hours"
  "$bin" status || true
else
  log "Mac control plane: watcher + notifications (adapters live in Grok Bot sandbox)"
  plist="$HOME/Library/LaunchAgents/com.xiaohu.grok-quota-watch.plist"
  mkdir -p "$HOME/Library/LaunchAgents"
  cat >"$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.xiaohu.grok-quota-watch</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$ROOT/bin/quota-watch</string>
  </array>
  <key>StartInterval</key><integer>7200</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$ROOT/logs/launchd.out</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/launchd.err</string>
</dict>
</plist>
EOF
  launchctl unload "$plist" 2>/dev/null || true
  launchctl load "$plist"
  log "loaded launchd $plist (every 2h)"
fi

"$ROOT/bin/quota-watch" || true
"$ROOT/bin/status" || true
log "install complete"
