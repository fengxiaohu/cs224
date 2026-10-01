"""Cosine neighbor search and overlap metrics."""

from __future__ import annotations

import numpy as np


def l2_normalize_rows(emb: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized matrix and boolean mask of zero-norm rows."""
    norms = np.linalg.norm(emb, axis=1)
    zero_mask = norms < eps
    normed = emb.copy()
    safe = ~zero_mask
    if safe.any():
        normed[safe] = emb[safe] / norms[safe, None]
    if zero_mask.any():
        normed[zero_mask] = 0.0
    return normed, zero_mask


def top_k_neighbors(
    emb: np.ndarray,
    words: list[str],
    word2ind: dict[str, int],
    query: str,
    candidate_words: list[str],
    k: int = 10,
) -> dict:
    """
    Cosine neighbors among candidate_words.
    Excludes query itself and zero-norm rows.
    """
    if query not in word2ind:
        return {"query": query, "evaluable": False, "reason": "missing_from_vocab", "neighbors": []}

    normed, zero_mask = l2_normalize_rows(emb)
    q_idx = word2ind[query]
    if zero_mask[q_idx]:
        return {"query": query, "evaluable": False, "reason": "zero_vector", "neighbors": []}

    q_vec = normed[q_idx]
    candidates = [
        w for w in candidate_words
        if w in word2ind and w != query and not zero_mask[word2ind[w]]
    ]
    if not candidates:
        return {"query": query, "evaluable": False, "reason": "no_candidates", "neighbors": []}

    idxs = [word2ind[w] for w in candidates]
    sims = normed[idxs] @ q_vec
    order = np.argsort(-sims)[:k]
    neighbors = [(candidates[i], float(sims[i])) for i in order]
    return {"query": query, "evaluable": True, "neighbors": neighbors}


def neighbor_overlap(neighbors_a: list[tuple[str, float]], neighbors_b: list[tuple[str, float]]) -> float:
    """Top-k overlap as |intersection| / k (plan: top-10 overlap)."""
    words_a = [w for w, _ in neighbors_a]
    words_b = [w for w, _ in neighbors_b]
    if not words_a or not words_b:
        return 0.0
    k = min(len(words_a), len(words_b))
    set_a = set(words_a[:k])
    set_b = set(words_b[:k])
    return len(set_a & set_b) / k


def jaccard_top_k(
    neighbors_a: list[tuple[str, float]],
    neighbors_b: list[tuple[str, float]],
    k: int = 10,
) -> float:
    set_a = {w for w, _ in neighbors_a[:k]}
    set_b = {w for w, _ in neighbors_b[:k]}
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def pip_loss(
    emb: np.ndarray,
    subset_indices: list[int],
    k: int,
    k_ref: int = 100,
) -> float:
    """
    L_PIP(k) = || Ê_{S,k} Ê_{S,k}^T - Ê_{S,100} Ê_{S,100}^T ||_F^2
    on row-L2-normalized vectors.
    """
    sub = emb[subset_indices, :k_ref]
    normed, _ = l2_normalize_rows(sub)
    e_k = normed[:, :k]
    e_ref = normed[:, :k_ref]
    gram_k = e_k @ e_k.T
    gram_ref = e_ref @ e_ref.T
    diff = gram_k - gram_ref
    return float(np.sum(diff ** 2))


def select_pip_subset(
    emb: np.ndarray,
    eval_vocab: list[str],
    word2ind: dict[str, int],
    counts: dict[str, int],
    max_size: int,
) -> tuple[list[str], list[int], int]:
    """Select up to max_size highest-frequency eval vocab words; exclude zero vectors."""
    _, zero_mask = l2_normalize_rows(emb)
    eligible = [
        w for w in eval_vocab
        if w in word2ind and not zero_mask[word2ind[w]]
    ]
    eligible.sort(key=lambda w: (-counts[w], w))
    selected = eligible[:max_size]
    indices = [word2ind[w] for w in selected]
    zero_dropped = sum(1 for w in eval_vocab if w in word2ind and zero_mask[word2ind[w]])
    return selected, indices, zero_dropped
