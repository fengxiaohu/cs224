#!/usr/bin/env python3
"""Read Cursor period usage without printing secrets. Exit 0 always; JSON on stdout."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import urllib.request

DB = os.path.expanduser(
    "~/Library/Application Support/Cursor/User/globalStorage/state.vscdb"
)
URL = "https://api2.cursor.sh/aiserver.v1.DashboardService/GetCurrentPeriodUsage"


def token() -> str | None:
    if not os.path.exists(DB):
        return None
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    row = con.execute(
        "select value from ItemTable where key='cursorAuth/accessToken'"
    ).fetchone()
    con.close()
    return row[0] if row else None


def fetch(tok: str) -> dict:
    req = urllib.request.Request(
        URL,
        data=b"{}",
        method="POST",
        headers={
            "Authorization": f"Bearer {tok}",
            "Content-Type": "application/json",
            "Connect-Protocol-Version": "1",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def summarize(raw: dict) -> dict:
    plan = raw.get("planUsage") or {}
    included = float(plan.get("includedSpend") or 0)
    limit = float(plan.get("limit") or 0)
    auto = float(plan.get("autoPercentUsed") or 0)
    total = float(plan.get("totalPercentUsed") or auto)
    msg = str(raw.get("displayMessage") or "")
    included_exhausted = bool(limit and included >= limit)
    hit_limit = included_exhausted or "usage limit" in msg.lower()
    return {
        "ok": True,
        "auto_percent": auto,
        "total_percent": total,
        "included_spend": included,
        "limit": limit,
        "included_exhausted": included_exhausted,
        "remaining_bonus": bool(plan.get("remainingBonus")),
        "display_message": msg,
        "hit_limit": hit_limit,
        "billing_cycle_start": str(raw.get("billingCycleStart") or ""),
        "billing_cycle_end": str(raw.get("billingCycleEnd") or ""),
        "auto_model_message": str(raw.get("autoModelSelectedDisplayMessage") or ""),
    }


def main() -> int:
    tok = token()
    if not tok:
        json.dump({"ok": False, "error": "no_cursor_token"}, sys.stdout)
        return 0
    try:
        json.dump(summarize(fetch(tok)), sys.stdout)
    except Exception as e:
        json.dump(
            {"ok": False, "error": type(e).__name__, "detail": str(e)[:200]},
            sys.stdout,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
