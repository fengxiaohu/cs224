#!/usr/bin/env python3
"""Unified entry point for the CS224N local study workspace.

Flow:
  1. Re-fetch the index HTML into _metadata/index.html (curl, else requests).
  2. Parse it -> full data model.
  3. Plan local paths, write _metadata/resources.json + _metadata/urls.txt.
  4. Create directory structure.
  5. Download internal files (skipped with --list).
  6. Generate all markdown / per-lecture metadata.
  7. Print summary.

Flags:
  --retry-failed   only retry entries in _metadata/failed.txt
  --list           parse + print manifest only, no downloads
  --force          re-download existing files
  --offline        reuse existing _metadata/index.html instead of re-fetching
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import requests

import downloader
import generator
from parser import INDEX_URL, parse_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_DIR = os.path.join(ROOT, "_metadata")
INDEX_HTML = os.path.join(METADATA_DIR, "index.html")

USER_AGENT = "Mozilla/5.0 CS224N-personal-study-downloader"


def fetch_index(offline: bool = False) -> None:
    """Fetch the schedule page into _metadata/index.html."""
    if offline:
        if not os.path.exists(INDEX_HTML):
            raise SystemExit("--offline requested but _metadata/index.html is missing")
        return
    os.makedirs(METADATA_DIR, exist_ok=True)
    if shutil.which("curl"):
        proc = subprocess.run(
            ["curl", "-sS", "-A", USER_AGENT, "--max-time", "60", "-o", INDEX_HTML, INDEX_URL],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and os.path.getsize(INDEX_HTML) > 0:
            return
    resp = requests.get(INDEX_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
    resp.raise_for_status()
    with open(INDEX_HTML, "w", encoding="utf-8") as fh:
        fh.write(resp.text)


def create_dirs(data: dict) -> None:
    for lec in data["lectures"]:
        os.makedirs(os.path.join(ROOT, "lectures", generator.lecture_dir_name(lec)), exist_ok=True)
    for a in data["assignments"]:
        os.makedirs(os.path.join(ROOT, "assignments", a["id"]), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "handouts"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "external-readings"), exist_ok=True)


def _progress(i: int, total: int, label: str, item: dict) -> None:
    ok = "✓" if os.path.exists(item["local_path"]) and os.path.getsize(item["local_path"]) > 0 else " "
    print(f"[{i:02d}/{total:02d}] {label} {ok}")


def _summary(data: dict, plan: list) -> None:
    n_slides = sum(1 for lec in data["lectures"] if "slides" in lec.get("resources", {}))
    n_notes = sum(1 for lec in data["lectures"] if "notes" in lec.get("resources", {}))
    n_code = sum(1 for lec in data["lectures"] if "code" in lec.get("resources", {}))
    n_files = len(plan)
    print()
    print(f"Lectures:            {len(data['lectures'])}")
    print(f"  with slides:       {n_slides}")
    print(f"  with notes:        {n_notes}")
    print(f"  with code:         {n_code}")
    print(f"Assignments:         {len(data['assignments'])} (a1..a{len(data['assignments'])})")
    print(f"Handouts:            {len(data['handouts'])}")
    print(f"External links:      {len(data['external_links'])}")
    print(f"Internal files planned ({'downloaded' if not args.list else 'listed, NOT downloaded'}): {n_files}")


def main(argv=None) -> int:
    global args
    parser = argparse.ArgumentParser(description="CS224N local study workspace sync")
    parser.add_argument("--retry-failed", action="store_true", help="retry _metadata/failed.txt only")
    parser.add_argument("--list", action="store_true", help="parse and print manifest, no downloads")
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    parser.add_argument("--offline", action="store_true", help="reuse existing _metadata/index.html")
    args = parser.parse_args(argv)

    if args.retry_failed:
        results = downloader.download_from_failed_list()
        print(f"Retried {len(results)} entries; {sum(1 for *_r, ok in results if ok)} succeeded.")
        return 0

    fetch_index(offline=args.offline)
    data = parse_index(INDEX_HTML)

    plan = generator.build_download_plan(data)
    create_dirs(data)
    generator.generate_all(data, plan)  # writes resources.json, urls.txt, markdown

    if args.list:
        _summary(data, plan)
        return 0

    t0 = time.time()
    result = downloader.download_all(plan, force=args.force, on_progress=_progress)
    elapsed = time.time() - t0

    _summary(data, plan)
    print(f"Downloaded {result['ok']}, skipped {result['skipped']}, failed {result['failed']} "
          f"({elapsed:.0f}s)")

    if result["failed"]:
        print("Some downloads failed — see _metadata/download.log and _metadata/failed.txt "
              "(`python3 scripts/sync.py --retry-failed`)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())