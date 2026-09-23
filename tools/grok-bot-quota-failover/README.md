# Grok Bot quota failover

Use **official Grok Bot / Cursor included quota first**. When that bucket is exhausted, switch Grok Bot to **ChatGPT Codex** via [grok-bot-setup](https://github.com/BlockedPath/grok-bot-setup). After the weekly cycle resets, switch back to official Grok.

`grok-bot-setup` has no request-level failover. This kit only flips the provider config (`adapters use cursor` vs `adapters use openai-oauth`) and restarts `sand-host`. In-flight turns may drop.

## Strategy

1. **Default:** `adapters use cursor` (stock Cursor path, included Grok / Weekly usage).
2. **When included quota is gone** (included spend hits the plan limit, usage ≥ ~95%, or Cursor reports a usage-limit message): `adapters start openai-oauth` and `adapters use openai-oauth --model gpt-6-luna`.
3. **After a new billing cycle** (or usage drops back under ~20%): `adapters use cursor` again.

Do **not** use `adapters use grok-session`. That is an xAI `grok login` session, not the built-in Grok Bot allowance.

The Codex proxy on `:10531` can stay running in the background. Switching is still global, not per message.

## Layout

| Path | Role |
|------|------|
| `install.sh` | Mac: launchd watcher every 2h. Grok Bot sandbox: bootstrap adapters, pick stock vs Codex, install cron. |
| `bin/quota-watch` | Read Cursor period usage and decide `stock` vs `codex`. |
| `bin/use-stock` | Force official Grok (`adapters use cursor`). |
| `bin/use-codex` | Force Codex (`gpt-6-luna`). |
| `bin/status` | Print provider, usage JSON, and local state. |
| `fetch_usage.py` | Cursor `GetCurrentPeriodUsage` using the local Cursor login. Never prints the token. |

## Install

On the Mac (notifications only; this machine has no `sand-host`):

```bash
bash tools/grok-bot-quota-failover/install.sh
```

Inside the Grok Bot sandbox (this is what actually changes the brain):

```bash
bash tools/grok-bot-quota-failover/install.sh
```

Or paste `GROK_BOT_PROMPT.txt` into Grok Bot.

## Manual triggers

- “switch to Codex” / “quota is gone” → `bin/use-codex`
- “switch back to official” → `bin/use-stock`
