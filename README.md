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

## Environment setup (China mirrors)

Notes for setting up Assignment 1 on Apple Silicon (M1/M2/M3) from mainland China.
Use **Miniconda ARM64** and avoid mixing in Homebrew Python. Open a new terminal
after installing Miniconda before running the commands below.

### Conda mirrors (Tsinghua)

Speeds up `conda env create` when channels include `defaults`, `conda-forge`, and
`huggingface` (see `assignments/a1/student/env.yml`):

```sh
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes
```

### Assignment 1 conda environment

From the repo root (see also `assignments/a1/student/README.md`):

```sh
cd assignments/a1/student
conda env create -f env.yml   # if env exists: conda env update -f env.yml --prune
conda activate cs224n
python -m ipykernel install --user --name cs224n
jupyter notebook exploring_word_vectors.ipynb
```

In Jupyter: **Kernel → Change kernel → cs224n**. Do not use the system or
Homebrew Python kernel.

Verify the active interpreter (add a cell at the top of the notebook):

```python
import sys
print(sys.executable)  # path should contain envs/cs224n
```

### HuggingFace mirror (`load_dataset` / IMDB)

HuggingFace Hub downloads (e.g. IMDB in Part 1) can be routed through a mirror.
This does **not** affect gensim GloVe downloads in Part 2.

In the shell:

```sh
export HF_ENDPOINT=https://hf-mirror.com
```

In Cursor or Jupyter, set **before** `import datasets`:

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### GloVe download (Part 2)

`api.load("glove-wiki-gigaword-200")` uses gensim’s default source, not
HuggingFace. The first download may take several minutes; retries help if you
see connection errors (e.g. `reset by peer`). Later runs use the local cache.

### Optional: pip mirror (Tsinghua)

For extra packages outside `env.yml`:

```sh
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

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
