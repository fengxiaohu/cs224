"""Load latest Phase 1 run artifacts for the notebook."""

from __future__ import annotations

import json
from pathlib import Path

from config import PHASE1_RESULTS_DIR, RESEARCH_DIR


def latest_run_dir(module_prefix: str) -> Path | None:
    dirs = sorted(PHASE1_RESULTS_DIR.glob(f"{module_prefix}_*"))
    return dirs[-1] if dirs else None


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def discover_runs() -> dict[str, str | None]:
    return {
        "module_a": str(latest_run_dir("module_a")) if latest_run_dir("module_a") else None,
        "module_b": str(latest_run_dir("module_b")) if latest_run_dir("module_b") else None,
        "module_d": str(latest_run_dir("module_d")) if latest_run_dir("module_d") else None,
        "module_e": str(latest_run_dir("module_e")) if latest_run_dir("module_e") else None,
    }
