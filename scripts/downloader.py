#!/usr/bin/env python3
"""Sequential, polite downloader for internal CS224N resources.

- Single-threaded, one request at a time.
- Streams to a temp file then atomically renames, so partial files never survive.
- Sleeps 2-4s (random) between requests; retries 3x with 5s/10s backoff.
- External URLs are never downloaded (is_internal guard).
- Existing non-empty local files are skipped unless force=True.
- Logs to _metadata/download.log; failures recorded in _metadata/failed.txt.
"""

import os
import random
import time
from urllib.parse import quote, urlparse, urlunparse

import requests

from parser import is_internal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_DIR = os.path.join(ROOT, "_metadata")
LOG_PATH = os.path.join(METADATA_DIR, "download.log")
FAILED_PATH = os.path.join(METADATA_DIR, "failed.txt")

USER_AGENT = "Mozilla/5.0 CS224N-personal-study-downloader"
TIMEOUT = 30
RETRIES = 3
CHUNK_SIZE = 65536


def _quoted_url(url: str) -> str:
    """Percent-encode the path segment (e.g. filenames containing spaces)."""
    parts = urlparse(url)
    path = quote(parts.path, safe="/%")
    return urlunparse((parts.scheme, parts.netloc, path, parts.params, parts.query, parts.fragment))


def _log(line: str) -> None:
    os.makedirs(METADATA_DIR, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _record_failure(url: str, local_path: str) -> None:
    os.makedirs(METADATA_DIR, exist_ok=True)
    with open(FAILED_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"{url}\t{local_path}\n")


def download_one(url: str, local_path: str, force: bool = False) -> bool:
    """Download one internal URL to local_path. Returns True on success or skip."""
    if not is_internal(url):
        return False
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0 and not force:
        return True  # incremental: already present, skip

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    tmp_path = local_path + ".part"

    for attempt in range(RETRIES):
        if attempt > 0:
            time.sleep(5 if attempt == 1 else 10)
        try:
            resp = requests.get(
                _quoted_url(url), stream=True, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT
            )
            resp.raise_for_status()
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        fh.write(chunk)
            resp.close()
            os.replace(tmp_path, local_path)
            _log(f"[OK] {local_path}")
            return True
        except Exception as exc:  # noqa: BLE001 - record and retry
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            _log(f"[RETRY {attempt + 1}] {url} ({exc.__class__.__name__}: {exc})")

    _record_failure(url, local_path)
    _log(f"[FAILED] {local_path}")
    return False


def download_all(resources, force: bool = False, on_progress=None) -> dict:
    """Download a list of {url, local_path, label} items sequentially.

    Returns {"ok": n, "failed": n, "skipped": n}.
    """
    ok = failed = skipped = 0
    total = len(resources)
    for i, item in enumerate(resources, 1):
        label = item.get("label") or item["local_path"]
        if on_progress:
            on_progress(i, total, label, item)
        if os.path.exists(item["local_path"]) and os.path.getsize(item["local_path"]) > 0 and not force:
            skipped += 1
            continue
        if download_one(item["url"], item["local_path"], force=force):
            ok += 1
        else:
            failed += 1
        time.sleep(random.uniform(2, 4))
    return {"ok": ok, "failed": failed, "skipped": skipped}


def download_from_failed_list() -> list:
    """Retry everything in _metadata/failed.txt; keep only still-failing entries."""
    if not os.path.exists(FAILED_PATH):
        return []
    with open(FAILED_PATH, "r", encoding="utf-8") as fh:
        entries = []
        for line in fh:
            line = line.strip()
            if not line:
                continue
            url, _, local = line.partition("\t")
            entries.append((url, local))

    remaining = []
    results = []
    for url, local in entries:
        if download_one(url, local, force=False):
            results.append((url, local, True))
        else:
            remaining.append((url, local))

    os.makedirs(METADATA_DIR, exist_ok=True)
    with open(FAILED_PATH, "w", encoding="utf-8") as fh:
        for url, local in remaining:
            fh.write(f"{url}\t{local}\n")
    return results