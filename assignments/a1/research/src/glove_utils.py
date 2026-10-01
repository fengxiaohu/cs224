"""GloVe loading and neighbor utilities."""

from __future__ import annotations

import random
from typing import Any

import gensim.downloader as api
import numpy as np

from config import GLOVE_MODEL, GLOVE_SAMPLE_SEED, GLOVE_SAMPLE_SIZE, QUERY_WORDS


def load_glove() -> Any:
    return api.load(GLOVE_MODEL)


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec
    return vec / norm


def glove_most_similar(
    wv,
    word: str,
    topn: int = 10,
) -> list[tuple[str, float]]:
    if word not in wv:
        return []
    return wv.most_similar(word, topn=topn)


def sample_glove_indices(
    wv,
    sample_size: int = GLOVE_SAMPLE_SIZE,
    seed: int = GLOVE_SAMPLE_SEED,
    required_words: list[str] | None = None,
) -> list[str]:
    """Fixed sample of words for 2D visualization, plus required query words."""
    required_words = required_words or QUERY_WORDS
    vocab = list(wv.index_to_key)
    rng = random.Random(seed)
    rng.shuffle(vocab)
    selected = set(vocab[:sample_size])
    for w in required_words:
        if w in wv:
            selected.add(w)
    return sorted(selected)


def get_glove_vectors(wv, words: list[str]) -> tuple[np.ndarray, list[str]]:
    vecs = []
    present = []
    for w in words:
        if w in wv:
            vecs.append(wv[w])
            present.append(w)
    return np.array(vecs, dtype=np.float64), present
