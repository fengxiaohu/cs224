"""Run directory management, hashing, and config serialization."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
import gensim
import datasets

from config import (
    HASH_ARCHIVE_PATH,
    PHASE1_RESULTS_DIR,
    RESEARCH_DIR,
    SEED,
)


def utc_run_id(module: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{module}_{ts}"


def make_run_dir(module: str) -> Path:
    run_dir = PHASE1_RESULTS_DIR / utc_run_id(module)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_token_stream(corpus: list[list[str]]) -> str:
    """Hash preprocessed token stream (all tokens in order, newline-separated docs)."""
    parts: list[str] = []
    for doc in corpus:
        parts.append(" ".join(doc))
    return sha256_text("\n".join(parts))


def package_versions() -> dict[str, str]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "gensim": gensim.__version__,
        "datasets": datasets.__version__,
    }


def save_config(run_dir: Path, config: dict[str, Any]) -> None:
    payload = {
        "seed": SEED,
        "packages": package_versions(),
        **config,
    }
    path = run_dir / "config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, slice):
        return {"start": obj.start, "stop": obj.stop, "step": obj.step}
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def load_or_assert_hashes(
    raw_text_hash: str,
    token_stream_hash: str,
) -> dict[str, str]:
    """First run writes hashes; subsequent runs assert equality."""
    hashes = {
        "raw_text_sha256": raw_text_hash,
        "preprocessed_token_sha256": token_stream_hash,
    }
    if HASH_ARCHIVE_PATH.exists():
        with HASH_ARCHIVE_PATH.open(encoding="utf-8") as f:
            archived = json.load(f)
        assert archived["raw_text_sha256"] == raw_text_hash, (
            f"raw text hash mismatch: {archived['raw_text_sha256']} != {raw_text_hash}"
        )
        assert archived["preprocessed_token_sha256"] == token_stream_hash, (
            f"token hash mismatch: {archived['preprocessed_token_sha256']} != {token_stream_hash}"
        )
    else:
        HASH_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with HASH_ARCHIVE_PATH.open("w", encoding="utf-8") as f:
            json.dump(hashes, f, indent=2)
    return hashes


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def add_research_to_path() -> None:
    if str(RESEARCH_DIR) not in sys.path:
        sys.path.insert(0, str(RESEARCH_DIR))
