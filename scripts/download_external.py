#!/usr/bin/env python3
"""Download external readings listed in external-readings/links.md.

Saves papers (PDF) and page content (HTML / notebooks) under
external-readings/<lecture-slug>/, matching the course lecture folders.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINKS_MD = os.path.join(ROOT, "external-readings", "links.md")
OUT_ROOT = os.path.join(ROOT, "external-readings")
MANIFEST = os.path.join(OUT_ROOT, "manifest.json")

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
TIMEOUT = 90
WORKERS = 5

SKIP_URL_SUBSTR = (
    "rec.stanford.edu/visit/locations",  # campus map, not a reading
)

# Public author-posted / arxiv copies when the original URL is a paywall or blog post.
URL_FALLBACKS = {
    "https://ieeexplore.ieee.org/document/279181": [
        "https://www.iro.umontreal.ca/~lisa/pointeurs/ieeetrnn94.pdf",
        "https://www.iro.umontreal.ca/~bengioy/papers/ieeetrnn94.pdf",
    ],
    "https://openai.com/research/instruction-following": [
        "https://arxiv.org/pdf/2203.02155.pdf",
        "https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf",
    ],
    "https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b": [
        "https://web.archive.org/web/20201109022829id_/https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b",
        "https://web.archive.org/web/20170116024959id_/https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b",
    ],
    "https://openai.com/index/learning-to-reason-with-llms/": [
        "https://web.archive.org/web/20241205000000id_/https://openai.com/index/learning-to-reason-with-llms/",
        "https://web.archive.org/web/20240913000000id_/https://openai.com/index/learning-to-reason-with-llms/",
    ],
    "https://www.emnlp2014.org/papers/pdf/EMNLP2014082.pdf": [
        "https://aclanthology.org/D14-1082.pdf",
        "https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf",
    ],
    "http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf": [
        "https://proceedings.neurips.cc/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf",
    ],
    "https://papers.nips.cc/paper/7368-on-the-dimensionality-of-word-embedding.pdf": [
        "https://proceedings.neurips.cc/paper/2018/file/b3ba8f1bee1238a2f37603d90b58898d-Paper.pdf",
    ],
    "https://transacl.org/ojs/index.php/tacl/article/viewFile/1346/320": [
        "https://aclanthology.org/Q18-1002.pdf",
    ],
    "http://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf": [
        "https://www.iro.umontreal.ca/~vincentp/ift3395/lectures/backprop_old.pdf",
        "https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf",
    ],
}

PAYWALLED = {
    "https://link.springer.com/book/10.1007/978-3-031-02131-2",
    "https://ieeexplore.ieee.org/document/279181",
}


def slugify(text: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:maxlen].rstrip("-") or "untitled"


def lecture_folder(heading: str) -> str:
    heading = heading.strip()
    if heading.lower() == "other":
        return "other"
    m = re.match(r"Lecture\s+(\d+)\s+[—–-]\s+(.+)", heading, re.I)
    if not m:
        return slugify(heading)
    num = int(m.group(1))
    title = m.group(2).rstrip(" (")
    return f"{num:02d}-{slugify(title, 40)}"


def parse_links(path: str) -> list[dict]:
    items = []
    section = "other"
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith("## "):
                section = line[3:].strip()
                continue
            m = re.match(r"- \[(.+?)\]\((.+?)\)\s*$", line)
            if not m:
                continue
            title, url = m.group(1), m.group(2)
            items.append({"title": title, "url": url, "section": section})
    return items


def arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:pdf|abs|html)/([a-z\-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?", url, re.I)
    return m.group(1) if m else None


def anthology_id(url: str) -> str | None:
    m = re.search(
        r"(?:aclanthology\.org|aclweb\.org/anthology)/(?:[A-Z]/[A-Z]\d{2}/)?([A-Z]\d{2}-\d{4})",
        url,
        re.I,
    )
    return m.group(1).upper() if m else None


def colab_id(url: str) -> str | None:
    m = re.search(r"colab\.research\.google\.com/drive/([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


def gdoc_id(url: str) -> str | None:
    m = re.search(r"docs\.google\.com/document/d/([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else None


def resolve(item: dict) -> dict:
    """Return download plan: urls to try, extension, skip reason."""
    url = item["url"]
    title = item["title"]

    if any(s in url for s in SKIP_URL_SUBSTR):
        return {**item, "skip": "not a paper/reading", "urls": [], "ext": None}

    if url in PAYWALLED:
        return {**item, "skip": "paywalled", "urls": [], "ext": None}

    aid = arxiv_id(url)
    if aid:
        return {
            **item,
            "skip": None,
            "ext": "pdf",
            "urls": [
                f"https://arxiv.org/pdf/{aid}.pdf",
                f"https://export.arxiv.org/pdf/{aid}.pdf",
            ],
        }

    anth = anthology_id(url)
    if anth:
        return {
            **item,
            "skip": None,
            "ext": "pdf",
            "urls": [f"https://aclanthology.org/{anth}.pdf", url],
        }

    cid = colab_id(url)
    if cid:
        return {
            **item,
            "skip": None,
            "ext": "ipynb",
            "urls": [
                f"https://drive.google.com/uc?export=download&id={cid}",
                f"https://docs.google.com/uc?export=download&id={cid}",
            ],
        }

    gid = gdoc_id(url)
    if gid:
        return {
            **item,
            "skip": None,
            "ext": "pdf",
            "urls": [f"https://docs.google.com/document/d/{gid}/export?format=pdf"],
        }

    extra = URL_FALLBACKS.get(url, [])
    path = urllib.parse.urlparse(url).path.lower()
    if path.endswith(".pdf") or "pdf" in path:
        return {**item, "skip": None, "ext": "pdf", "urls": [url] + extra}

    # Remaining: HTML articles / sites / homepages. Fallbacks may be PDFs;
    # download_item detects type from the response bytes.
    return {**item, "skip": None, "ext": "html", "urls": [url] + extra}


def filename_for(item: dict, ext: str) -> str:
    aid = arxiv_id(item["url"])
    anth = anthology_id(item["url"])
    cid = colab_id(item["url"])
    gid = gdoc_id(item["url"])
    base = slugify(item["title"], 70)
    if aid:
        base = f"{aid.replace('/', '-')}-{base}"
    elif anth:
        base = f"{anth}-{base}"
    elif cid:
        base = f"{base}-{cid[:10]}"
    elif gid:
        base = f"{base}-{gid[:10]}"
    return f"{base}.{ext}"


def looks_like_pdf(data: bytes, content_type: str) -> bool:
    if data.startswith(b"%PDF"):
        return True
    return "pdf" in (content_type or "").lower() and len(data) > 10_000


def looks_like_html(data: bytes, content_type: str) -> bool:
    head = data[:4000].lower()
    if b"your request has been denied" in head:
        return False
    if b"just a moment" in head and b"cloudflare" in head:
        return False
    if b"making sure you" in head and b"not a bot" in head:
        return False
    if b"<html" in head or b"<!doctype html" in head:
        return len(data) > 1500
    return "html" in (content_type or "").lower() and len(data) > 1500


def looks_like_ipynb(data: bytes) -> bool:
    try:
        obj = json.loads(data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return isinstance(obj, dict) and "cells" in obj and "nbformat" in obj


def fetch(url: str) -> tuple[bytes, str, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    resp = requests.get(url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    return resp.content, ctype, resp.url


def save_bytes(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def download_item(plan: dict) -> dict:
    folder = lecture_folder(plan["section"])
    if plan.get("skip"):
        return {
            "title": plan["title"],
            "url": plan["url"],
            "folder": folder,
            "status": "skipped",
            "reason": plan["skip"],
        }

    dest_dir = os.path.join(OUT_ROOT, folder)
    dest = os.path.join(dest_dir, filename_for(plan, plan["ext"]))
    candidates = [dest]
    if plan["ext"] == "html":
        candidates.append(os.path.join(dest_dir, filename_for(plan, "pdf")))
    for existing in candidates:
        if os.path.exists(existing) and os.path.getsize(existing) > 0:
            return {
                "title": plan["title"],
                "url": plan["url"],
                "folder": folder,
                "path": os.path.relpath(existing, ROOT),
                "status": "exists",
                "bytes": os.path.getsize(existing),
            }

    errors = []
    for url in plan["urls"]:
        try:
            data, ctype, final = fetch(url)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc.__class__.__name__}: {exc}")
            continue

        ext = plan["ext"]
        ok = False
        if ext == "pdf":
            ok = looks_like_pdf(data, ctype)
        elif ext == "ipynb":
            ok = looks_like_ipynb(data)
        else:
            ok = looks_like_html(data, ctype) or len(data) > 400
            # Some "HTML" URLs actually return PDF.
            if looks_like_pdf(data, ctype):
                ext = "pdf"
                dest = os.path.join(dest_dir, filename_for(plan, "pdf"))
                ok = True

        if not ok:
            errors.append(f"{url}: unexpected type {ctype!r} ({len(data)} bytes)")
            continue

        dest = os.path.join(dest_dir, filename_for(plan, ext))
        save_bytes(dest, data)
        return {
            "title": plan["title"],
            "url": plan["url"],
            "source": final,
            "folder": folder,
            "path": os.path.relpath(dest, ROOT),
            "status": "downloaded",
            "bytes": len(data),
        }

    return {
        "title": plan["title"],
        "url": plan["url"],
        "folder": folder,
        "status": "failed",
        "reason": " | ".join(errors)[:800],
    }


def render_index(results: list[dict]) -> str:
    lines = ["# External Readings — Local Copies", "", "Downloaded from `links.md`.", ""]
    by_folder: dict[str, list[dict]] = {}
    for r in results:
        by_folder.setdefault(r["folder"], []).append(r)

    for folder in sorted(by_folder):
        lines.append(f"## {folder}")
        lines.append("")
        for r in by_folder[folder]:
            status = r["status"]
            if status in ("downloaded", "exists") and r.get("path"):
                size = r.get("bytes") or 0
                kb = f"{size/1024:.0f} KB" if size else ""
                tag = "cached" if status == "exists" else "ok"
                lines.append(f"- [{r['title']}]({os.path.relpath(os.path.join(ROOT, r['path']), OUT_ROOT)}) `({tag}, {kb})`")
            else:
                reason = r.get("reason", "")
                if r["status"] == "skipped":
                    short = reason
                elif "403" in reason and "429" in reason:
                    short = "source blocked (403); archive.org rate-limited"
                elif "403" in reason:
                    short = "source blocked (403)"
                elif "429" in reason:
                    short = "rate-limited"
                else:
                    short = reason[:160]
                lines.append(f"- {r['title']} — _{status}: {short}_")
                lines.append(f"  - original: <{r['url']}>")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    items = parse_links(LINKS_MD)
    plans = [resolve(it) for it in items]
    print(f"Found {len(plans)} links in {os.path.relpath(LINKS_MD, ROOT)}")

    results: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(download_item, p): p for p in plans}
        for fut in as_completed(futs):
            rec = fut.result()
            results.append(rec)
            done += 1
            mark = {"downloaded": "OK", "exists": "SKIP", "skipped": "—", "failed": "FAIL"}.get(
                rec["status"], rec["status"]
            )
            extra = rec.get("path") or rec.get("reason", "")
            print(f"[{done:02d}/{len(plans)}] {mark:4} {rec['title'][:60]}  {extra}")
            sys.stdout.flush()

    # Keep original order
    order = {(p["url"], p["title"]): i for i, p in enumerate(plans)}
    results.sort(key=lambda r: order.get((r["url"], r["title"]), 9999))

    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump({"counts": counts, "items": results}, fh, indent=2, ensure_ascii=False)

    index_path = os.path.join(OUT_ROOT, "INDEX.md")
    with open(index_path, "w", encoding="utf-8") as fh:
        fh.write(render_index(results))

    print()
    print("Summary:", counts)
    print(f"Index: {os.path.relpath(index_path, ROOT)}")
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print("Failed:")
        for r in failed:
            print(f"  - {r['title']}: {r.get('reason', '')[:200]}")
    return 1 if failed else 0


if __name__ == "__main__":
    # Be polite to arxiv/acl: slight stagger by not using too many workers.
    time.sleep(0)
    raise SystemExit(main())
