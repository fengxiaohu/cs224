"""Single configuration entry point for A1 embedding study Phase 1."""

from __future__ import annotations

import sys
from pathlib import Path

# Paths
RESEARCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = RESEARCH_DIR / "results"
PHASE1_RESULTS_DIR = RESULTS_DIR / "phase1"
BASELINE_DIR = RESULTS_DIR / "baseline"
SRC_DIR = RESEARCH_DIR / "src"
PAPERS_DIR = RESEARCH_DIR / "papers"
NOTEBOOKS_DIR = RESEARCH_DIR / "notebooks"

# Reproducibility
SEED = 0
SVD_N_ITER = 10

# IMDB research corpus
DATASET_NAME = "stanfordnlp/imdb"
DATASET_CONFIG = "plain_text"
DATASET_SPLIT = "train"
DATASET_SLICE = slice(11500, 13500)
EXPECTED_NEG_LABEL = 1000
EXPECTED_POS_LABEL = 1000
EXPECTED_TOKEN_COUNT = 470_407
DATASET_REVISION = "e6281661ce1c48d982bc483cf8a173c1bbeb5d31"

# Preprocessing (matches assignment)
START_TOKEN = "<START>"
END_TOKEN = "<END>"
PREPROCESS_PATTERN = r"[^\w]"

# Co-occurrence / PPMI
WINDOW_SIZE = 4
CDS_ALPHA = 0.75

# Vocabulary
EVAL_MIN_COUNT = 10
EXCLUDED_TOKENS = frozenset({"", START_TOKEN, END_TOKEN})

# Query words (assignment set)
QUERY_WORDS = [
    "movie",
    "book",
    "mysterious",
    "story",
    "fascinating",
    "good",
    "interesting",
    "large",
    "massive",
    "huge",
]

# Module A
MODULE_A_K = 50
MODULE_A_SEEDS = [0, 1, 2]

# Module B
MODULE_B_K = 100
MODULE_B_K_VALUES = [2, 10, 25, 50, 100]
MODULE_B_PIP_SUBSET_SIZE = 500
MODULE_B_ROTATION_COLS = 50

# GloVe
GLOVE_MODEL = "glove-wiki-gigaword-200"
GLOVE_SAMPLE_SIZE = 40_000
GLOVE_SAMPLE_SEED = 225  # matches assignment get_matrix_of_vectors

# Module D polysemy
POLYSEMY_WORDS = ["bank", "plant", "tie", "bark"]

# Module E
ANALOGY_SET_PATH = RESEARCH_DIR / "analogy_set.json"

# Archived hashes (written on first run, asserted on subsequent runs)
HASH_ARCHIVE_PATH = RESEARCH_DIR / "data_hashes.json"

# Python executable for runs
PYTHON = sys.executable
