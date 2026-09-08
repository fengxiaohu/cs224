#!/usr/bin/env python3
"""Parse the CS224N Winter 2025 schedule index HTML into the canonical data model.

The parser reads the schedule table (div#schedule > table.table) and emits a dict:

{
  "course": "CS224N Winter 2025",
  "source": "<index URL>",
  "generated_at": "<ISO timestamp>",
  "lectures": [ ... ],
  "assignments": [ ... ],
  "handouts": [ ... ],
  "external_links": [ ... ]
}

Lecture rows are defined by the "robust rule": any schedule row whose Description
cell contains an internal slides/notes/code link counts as a lecture record,
numbered in row order (01, 02, ...). Titles are the first line of pure text in
the Description cell.
"""

import json
import re
from datetime import datetime, timezone
from urllib.parse import unquote, urljoin, urlparse

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

# Course archive root for Winter 2025.
BASE_URL = "https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/"
ALLOWED_PREFIX = BASE_URL  # internal links must start with this (trailing slash kept)

INDEX_URL = urljoin(BASE_URL, "index.html")

# Resource classification by URL path prefix (within the Description cell).
_PREFIX_TO_KIND = (
    ("slides_w25/", "slides"),
    ("slides/", "slides"),
    ("readings/", "notes"),
    ("materials/", "code"),
)

_ASSIGN_RE = re.compile(r"assignment\s+(\d+)", re.IGNORECASE)
_ASSIGN_ID_RE = re.compile(r"^a(\d+)")


def is_internal(url: str) -> bool:
    """True when url points inside the course archive."""
    return url.startswith(ALLOWED_PREFIX)


def _abs(href: str) -> str:
    return urljoin(BASE_URL, href)


def _filename(url: str) -> str:
    """Unquoted basename of the URL path."""
    path = unquote(urlparse(url).path)
    return path.rsplit("/", 1)[-1] if "/" in path else path


def _clean_title(s: str) -> str:
    """Collapse whitespace / newlines in a text snippet."""
    return re.sub(r"\s+", " ", s).strip()


def _first_text_line(cell: Tag) -> str:
    lines = [ln.strip() for ln in cell.get_text("\n").split("\n") if ln.strip()]
    return _clean_title(lines[0]) if lines else ""


def _date_value(cell: Tag) -> str:
    """Date column: 'Week 1\n\nTue Jan 7' -> 'Tue Jan 7', 'Thu Jan 9' -> 'Thu Jan 9'."""
    lines = [ln.strip() for ln in cell.get_text("\n").split("\n") if ln.strip()]
    return lines[-1] if lines else ""


def _linear(cell: Tag):
    """Flatten a cell into ordered ('text'|'link'|'br') tokens (comments ignored)."""
    tokens = []
    for node in cell.descendants:
        if isinstance(node, Comment):
            continue
        if isinstance(node, Tag):
            if node.name == "a":
                href = node.get("href")
                if href:
                    url = _abs(href)
                    title = _clean_title(node.get_text(" ", strip=True)) or _filename(url)
                    tokens.append(("link", url, title))
            elif node.name == "br":
                tokens.append(("br",))
        elif isinstance(node, NavigableString):
            if node.find_parent("a"):
                continue  # anchor text is captured in the ('link', ...) token
            if node.strip():
                tokens.append(("text", str(node)))
    return tokens


def _classify_resource(url: str):
    """Map an internal Description URL to slides/notes/code, or None."""
    path = urlparse(url).path
    for prefix, kind in _PREFIX_TO_KIND:
        if prefix in path:
            return kind
    return None


def _parse_readings(cell: Tag):
    """Course Materials cell -> suggested/additional reading records."""
    readings = []
    section = "suggested"
    for node in cell.children:
        if isinstance(node, NavigableString):
            text = str(node)
            if "additional" in text.lower():
                section = "additional"
            elif "suggested" in text.lower():
                section = "suggested"
        elif isinstance(node, Tag) and node.name == "ol":
            for li in node.find_all("li", recursive=False):
                a = li.find("a")
                if a and a.get("href"):
                    title = _clean_title(a.get_text(" ", strip=True)) or _filename(a["href"])
                    readings.append({"title": title, "url": _abs(a["href"]), "section": section})
    return readings


def _clean_event_title(segment: str) -> str:
    """'[Project Proposal out[' / ']Default Final Project out[' -> clean phrase."""
    s = segment.replace("[", " ").replace("]", " ")
    s = _clean_title(s)
    s = re.sub(r"\s*(out|due|returned)\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def _parse_events(cell, assignments, handouts, external):
    """Events cell -> assignment files, handouts, external links (mutates the dicts/lists)."""
    if cell is None:
        return
    segment_parts = []
    for tok in _linear(cell):
        if tok[0] == "br":
            continue
        if tok[0] == "text":
            segment_parts.append(tok[1])
            continue
        if tok[0] != "link":
            continue
        _, url, label = tok
        if is_internal(url):
            path = urlparse(url).path
            fname = _filename(url)
            if "assignments_w25/" in path or "assignments/" in path:
                m = _ASSIGN_ID_RE.match(fname)
                aid = ("a" + m.group(1)) if m else "a"
                if "tex" in fname or "latex" in label.lower() or "overleaf" in label.lower():
                    kind = "tex"
                elif label.lower().startswith("handout") or fname.lower().endswith(".pdf"):
                    kind = "handout"
                else:
                    kind = "code"
                assignments.setdefault(aid, {"id": aid, "title": f"Assignment {aid[1:]}", "files": []})
                assignments[aid]["files"].append(
                    {"url": url, "filename": fname, "kind": kind}
                )
            else:
                title = _clean_event_title("".join(segment_parts)) or label
                handouts.append({"url": url, "filename": fname, "title": title})
        else:
            external.append({"url": url, "title": label or _filename(url)})
        segment_parts = []  # next link starts a fresh context segment


def parse_index(html_path: str) -> dict:
    """Parse _metadata/index.html into the full data model dict."""
    with open(html_path, "r", encoding="utf-8") as fh:
        soup = BeautifulSoup(fh.read(), "html.parser")

    schedule = soup.find("div", id="schedule")
    if schedule is None:
        raise ValueError("schedule div not found in HTML")

    rows = schedule.select("table.table tbody tr")
    if not rows:
        raise ValueError("no schedule rows found")

    lectures = []
    assignments = {}
    handouts = []
    external = []
    current_week = None

    for row in rows:
        cells = row.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        date_cell, desc_cell = cells[0], cells[1]
        materials_cell = cells[2] if len(cells) > 2 else None
        events_cell = cells[3] if len(cells) > 3 else None
        deadline_cell = cells[4] if len(cells) > 4 else None

        # Track running week number.
        date_text = _date_value(date_cell)
        m = re.search(r"week\s+(\d+)", date_cell.get_text(" "), re.IGNORECASE)
        if m:
            current_week = int(m.group(1))

        # --- Description column: lecture resources, handouts, external links ---
        resources = {}
        desc_ext = []
        for tok in _linear(desc_cell):
            if tok[0] != "link":
                continue
            _, url, label = tok
            if not is_internal(url):
                desc_ext.append({"url": url, "title": label or _filename(url)})
                continue
            kind = _classify_resource(url)
            if kind in ("slides", "notes", "code"):
                resources[kind] = {"url": url, "filename": _filename(url)}
            else:
                # internal but not slides/notes/code (e.g. project/custom-final-project-tips.pdf)
                handouts.append(
                    {"url": url, "filename": _filename(url), "title": label or _filename(url)}
                )

        # --- Course Materials: external links from every row (readings for lecture rows) ---
        materials_ext = []
        if materials_cell is not None:
            for a in materials_cell.find_all("a"):
                href = a.get("href")
                if href:
                    url = _abs(href)
                    if not is_internal(url):
                        title = _clean_title(a.get_text(" ", strip=True)) or _filename(url)
                        materials_ext.append({"url": url, "title": title})

        # --- Events: assignments / handouts / external links ---
        events_ext = []
        if events_cell is not None:
            _parse_events(events_cell, assignments, handouts, events_ext)

        # --- Deadlines: external links ---
        deadline_ext = []
        if deadline_cell is not None:
            for a in deadline_cell.find_all("a"):
                href = a.get("href")
                if href:
                    url = _abs(href)
                    if not is_internal(url):
                        title = _clean_title(a.get_text(" ", strip=True)) or _filename(url)
                        deadline_ext.append({"url": url, "title": title})

        # All external links seen in this row.
        row_ext = desc_ext + materials_ext + events_ext + deadline_ext

        if resources:
            lectures.append(
                {
                    "number": len(lectures) + 1,
                    "title": _first_text_line(desc_cell),
                    "date": date_text,
                    "week": current_week if current_week is not None else 0,
                    "resources": resources,
                    "readings": _parse_readings(materials_cell) if materials_cell else [],
                    "external_links": row_ext,
                }
            )
        external.extend(row_ext)

    # Dedupe handouts and external links (keep first occurrence).
    seen_h, seen_e = set(), set()
    dedup_handouts, dedup_ext = [], []
    for h in handouts:
        if h["url"] not in seen_h:
            seen_h.add(h["url"])
            dedup_handouts.append(h)
    for e in external:
        if e["url"] not in seen_e:
            seen_e.add(e["url"])
            dedup_ext.append(e)

    return {
        "course": "CS224N Winter 2025",
        "source": INDEX_URL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "lectures": lectures,
        "assignments": [assignments[k] for k in sorted(assignments)],
        "handouts": dedup_handouts,
        "external_links": dedup_ext,
    }


def main():
    import sys

    html = sys.argv[1] if len(sys.argv) > 1 else "_metadata/index.html"
    data = parse_index(html)
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()