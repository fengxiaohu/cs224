# CS224N Winter 2025 — Local Study Workspace

A personal, offline study workspace for the Stanford CS224N Winter 2025 course
(Natural Language Processing with Deep Learning). Source archive:
https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/index.html

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

- Lectures: 15 (15 with slides, 7 with notes, 1 with code)
- Assignments: 4
- Handouts: 5
- External links: 86
