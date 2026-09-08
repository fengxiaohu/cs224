#!/usr/bin/env python3
"""Generate the local study workspace artifacts from the parsed data model.

Outputs:
  COURSE.md                          - course index with progress + per-lecture sections
  lectures/NN-slug/readings.md       - per-lecture reading link lists
  lectures/NN-slug/metadata.json     - full per-lecture record (original URLs + filenames)
  external-readings/links.md         - all external links grouped by lecture
  _metadata/resources.json           - complete parsed data model
  _metadata/urls.txt                 - "URL\\tLOCAL_PATH" download plan (internal only)
  README.md                          - workspace documentation
"""

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METADATA_DIR = os.path.join(ROOT, "_metadata")


def slugify(title: str, maxlen: int = 40) -> str:
    """Lowercase, spaces -> '-', drop other special chars, truncate ~maxlen."""
    s = re.sub(r"[^a-z0-9]+", "-", title.lower())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:maxlen].rstrip("-")


def lecture_dir_name(lecture: dict) -> str:
    return f"{lecture['number']:02d}-{slugify(lecture['title'])}"


def _local_link(path: str) -> str:
    """Relative markdown link from repo root to a local path."""
    return path.replace(os.sep, "/")


def build_download_plan(data: dict) -> list:
    """Map every internal, download-ready resource to a local path.

    Returns a list of {"url", "local_path", "label"} — the same items written to
    urls.txt and used by the downloader. Order: lectures, assignments, handouts.
    """
    plan = []
    for lec in data["lectures"]:
        d = os.path.join("lectures", lecture_dir_name(lec))
        for key, target in (("slides", "slides.pdf"), ("notes", "notes.pdf"), ("code", None)):
            res = lec.get("resources", {}).get(key)
            if not res:
                continue
            if key == "code":
                fname = f"code-{res['filename']}"
            else:
                fname = target or "slides.pdf"
            plan.append(
                {
                    "url": res["url"],
                    "local_path": os.path.join(d, fname),
                    "label": f"lecture {lec['number']:02d} {key}",
                }
            )
    for a in data["assignments"]:
        d = os.path.join("assignments", a["id"])
        for f in a["files"]:
            if f["kind"] == "code":
                fname = f"{a['id']}.zip"
            elif f["kind"] == "handout":
                fname = f"{a['id']}.pdf"
            else:  # tex
                fname = f"{a['id']}_tex.zip"
            plan.append(
                {
                    "url": f["url"],
                    "local_path": os.path.join(d, fname),
                    "label": f"assignment {a['id']} {f['kind']}",
                }
            )
    for h in data["handouts"]:
        plan.append(
            {
                "url": h["url"],
                "local_path": os.path.join("handouts", h["filename"]),
                "label": f"handout {h['filename']}",
            }
        )
    return plan


def _study_checklist() -> str:
    return "\n".join(f"- [ ] {item}" for item in (
        "Read slides", "Read notes", "Watch lecture",
        "Finish reading", "Produce notes", "Finish artifact",
    ))


def _resources_line(res: dict, local_path: str) -> str:
    if os.path.exists(os.path.join(ROOT, local_path)):
        return f"[{res['filename']}]({_local_link(local_path)})"
    return f"[{res['filename']}]({res['url']}) *(pending download)*"


def _render_course_md(data: dict, plan: list) -> str:
    lines = [
        "# Stanford CS224N — Winter 2025",
        "",
        "Natural Language Processing with Deep Learning — local study workspace.",
        f"Source: {data['source']}",
        "",
        "## Progress",
        "",
    ]
    for lec in data["lectures"]:
        lines.append(f"- [ ] Lecture {lec['number']:02d} — {lec['title']}")
    lines += ["", "## Lectures", ""]
    for lec in data["lectures"]:
        d = os.path.join("lectures", lecture_dir_name(lec))
        lines.append(f"### Lecture {lec['number']:02d} — {lec['title']}")
        lines.append("")
        lines.append(f"- **Date:** {lec['date']} (Week {lec['week']})")
        resources = lec.get("resources", {})
        for key, local_name in (("slides", "slides.pdf"), ("notes", "notes.pdf"), ("code", None)):
            res = resources.get(key)
            if not res:
                continue
            if key == "code":
                local_name = "code-" + res["filename"]
            else:
                local_name = local_name or "slides.pdf"
            label = key.title()
            local_path = os.path.join(d, local_name)
            lines.append(f"- **{label}:** {_resources_line(res, local_path)}")
        lines.append(f"- **Readings:** [readings.md]({_local_link(os.path.join(d, 'readings.md'))})")
        lines.append("")
        lines.append("**Study checklist:**")
        lines.append("")
        lines.append(_study_checklist())
        lines.append("")
    lines += ["## Assignments", ""]
    for a in data["assignments"]:
        d = os.path.join("assignments", a["id"])
        lines.append(f"### {a['title']} ({a['id']})")
        for f in a["files"]:
            local = os.path.join(d, {"code": "a.zip", "handout": "a.pdf", "tex": "a_tex.zip"}[f["kind"]].replace("a", a["id"]))
            if os.path.exists(os.path.join(ROOT, local)):
                lines.append(f"- [{f['kind'].title()}: {f['filename']}]({_local_link(local)})")
            else:
                lines.append(f"- [{f['kind'].title()}: {f['filename']}]({f['url']}) *(pending download)*")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_readings_md(lecture: dict) -> str:
    lines = [
        f"# Lecture {lecture['number']:02d} — {lecture['title']} — Readings",
        "",
        f"*Date:* {lecture['date']} (Week {lecture['week']})",
        "",
    ]
    by_section = {"suggested": [], "additional": []}
    for r in lecture.get("readings", []):
        by_section.setdefault(r.get("section", "suggested"), []).append(r)
    for section, label in (("suggested", "Suggested Readings"), ("additional", "Additional Readings")):
        items = by_section.get(section, [])
        if not items:
            continue
        lines.append(f"## {label}")
        lines.append("")
        for r in items:
            lines.append(f"- [{r['title']}]({r['url']})")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_links_md(data: dict) -> str:
    lines = ["# External Readings & Links", ""]
    lecture_groups = []
    for lec in data["lectures"]:
        ext = lec.get("external_links", [])
        if ext:
            lecture_groups.append((f"Lecture {lec['number']:02d} — {lec['title']}", ext))
    for heading, items in lecture_groups:
        lines.append(f"## {heading}")
        lines.append("")
        for it in items:
            lines.append(f"- [{it['title']}]({it['url']})")
        lines.append("")
    if data["external_links"]:
        seen = {it["url"] for _, items in lecture_groups for it in items}
        leftover = [it for it in data["external_links"] if it["url"] not in seen]
        if leftover:
            lines.append("## Other")
            lines.append("")
            for it in leftover:
                lines.append(f"- [{it['title']}]({it['url']})")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_readme_md(data: dict) -> str:
    n_slides = sum(1 for lec in data["lectures"] if "slides" in lec.get("resources", {}))
    n_notes = sum(1 for lec in data["lectures"] if "notes" in lec.get("resources", {}))
    n_code = sum(1 for lec in data["lectures"] if "code" in lec.get("resources", {}))
    n_assign = len(data["assignments"])
    n_handouts = len(data["handouts"])
    n_ext = len(data["external_links"])
    return f"""# CS224N Winter 2025 — Local Study Workspace

A personal, offline study workspace for the Stanford CS224N Winter 2025 course
(Natural Language Processing with Deep Learning). Source archive:
{data['source']}

> **Copyright notice:** All Stanford course materials (slides, notes,
> assignments, readings) are © Stanford University and the instructors. This
> workspace is for **personal study only**. Do **not** redistribute, publish,
> or publicly share any downloaded Stanford files.

## What's here

- `COURSE.md` — progress checklist + per-lecture study sections
- `lectures/NN-slug/` — per-lecture `slides.pdf`, `notes.pdf`, `code-*`,
  `readings.md`, `metadata.json`
- `assignments/a1..a4/` — starter code, handout PDFs, LaTeX templates
- `handouts/` — final-project handouts (project PDFs)
- `external-readings/links.md` — external papers/links grouped by lecture
- `_metadata/resources.json` — full parsed data model
- `_metadata/urls.txt` — download plan (internal URLs only)

## Usage

Sync (re-fetch index, parse, download missing files, regenerate markdown):

```sh
python3 scripts/sync.py
```

Retry previously failed downloads:

```sh
python3 scripts/sync.py --retry-failed
```

Parse and list without downloading:

```sh
python3 scripts/sync.py --list
```

Force re-download of everything (ignore existing files):

```sh
python3 scripts/sync.py --force
```

## Stats (as parsed)

- Lectures: {len(data['lectures'])} ({n_slides} with slides, {n_notes} with notes, {n_code} with code)
- Assignments: {n_assign}
- Handouts: {n_handouts}
- External links: {n_ext}
"""


def generate_all(data: dict, plan: list) -> None:
    """Write every artifact (course markdown, per-lecture files, metadata)."""
    os.makedirs(METADATA_DIR, exist_ok=True)

    # _metadata/resources.json + urls.txt
    with open(os.path.join(METADATA_DIR, "resources.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(METADATA_DIR, "urls.txt"), "w", encoding="utf-8") as fh:
        for p in plan:
            fh.write(f"{p['url']}\t{_local_link(p['local_path'])}\n")

    # COURSE.md
    with open(os.path.join(ROOT, "COURSE.md"), "w", encoding="utf-8") as fh:
        fh.write(_render_course_md(data, plan))

    # Per-lecture files
    for lec in data["lectures"]:
        d = os.path.join(ROOT, "lectures", lecture_dir_name(lec))
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "readings.md"), "w", encoding="utf-8") as fh:
            fh.write(_render_readings_md(lec))
        with open(os.path.join(d, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump(lec, fh, indent=2, ensure_ascii=False)

    # external-readings/links.md
    ext_dir = os.path.join(ROOT, "external-readings")
    os.makedirs(ext_dir, exist_ok=True)
    with open(os.path.join(ext_dir, "links.md"), "w", encoding="utf-8") as fh:
        fh.write(_render_links_md(data))

    # README.md
    with open(os.path.join(ROOT, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(_render_readme_md(data))


if __name__ == "__main__":
    import sys

    html = sys.argv[1] if len(sys.argv) > 1 else "_metadata/index.html"
    from parser import parse_index

    _data = parse_index(html)
    generate_all(_data, build_download_plan(_data))
    print(f"generated artifacts for {len(_data['lectures'])} lectures")