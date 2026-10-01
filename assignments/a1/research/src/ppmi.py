"""PMI, PPMI, and context distribution smoothing."""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse

from config import CDS_ALPHA
from src.sparse_counts import (
    col_marginals,
    compute_co_occurrence_matrix,
    row_marginals,
    total_mass,
)


def pmi_from_counts(
    count_ij: float,
    row_sum_i: float,
    col_sum_j: float,
    total: float,
) -> float:
    if count_ij <= 0 or row_sum_i <= 0 or col_sum_j <= 0 or total <= 0:
        return float("-inf")
    p_ij = count_ij / total
    p_i = row_sum_i / total
    p_j = col_sum_j / total
    return math.log(p_ij / (p_i * p_j))


def ppmi_from_pmi(pmi: float) -> float:
    return max(0.0, pmi)


def compute_ppmi_matrix(
    counts: sparse.csr_matrix,
    cds_alpha: float | None = None,
) -> sparse.csr_matrix:
    """
    Compute PPMI matrix from co-occurrence counts.

    If cds_alpha is not None, apply context distribution smoothing (CDS)
    to column marginals before PMI: p_j <- p_j^alpha / sum_k p_k^alpha.
    """
    total = total_mass(counts)
    row_sums = row_marginals(counts)
    col_sums = col_marginals(counts)

    if cds_alpha is not None:
        n = counts.shape[1]
        p_j = col_sums / total
        p_j_alpha = np.power(p_j, cds_alpha)
        denom = p_j_alpha.sum()
        if denom > 0:
            col_probs = p_j_alpha / denom
        else:
            col_probs = np.ones(n) / n
    else:
        col_probs = col_sums / total

    row_probs = row_sums / total
    coo = counts.tocoo()
    ppmi_data: list[float] = []
    for r, c, v in zip(coo.row, coo.col, coo.data):
        if v <= 0:
            ppmi_data.append(0.0)
            continue
        p_ij = v / total
        p_i = row_probs[r]
        p_j = col_probs[c]
        if p_i <= 0 or p_j <= 0:
            ppmi_data.append(0.0)
            continue
        pmi = math.log(p_ij / (p_i * p_j))
        ppmi_data.append(max(0.0, pmi))

    return sparse.coo_matrix(
        (ppmi_data, (coo.row, coo.col)),
        shape=counts.shape,
        dtype=np.float64,
    ).tocsr()


def toy_corpus_examples() -> dict:
    """
    Hand-checkable toy corpus with:
    - high-count function-word pair (the, a)
    - once-only pair (zebra, quasar)
    """
    corpus = [
        ["<START>", "the", "a", "the", "a", "the", "<END>"],
        ["<START>", "the", "a", "the", "<END>"],
        ["<START>", "zebra", "quasar", "<END>"],
    ]
    words = sorted(set(w for doc in corpus for w in doc))
    w2i = {w: i for i, w in enumerate(words)}
    counts = compute_co_occurrence_matrix(corpus, w2i, window_size=1)
    total = total_mass(counts)
    row_s = row_marginals(counts)
    col_s = col_marginals(counts)

    pairs = [
        ("the", "a", "high-count function words"),
        ("zebra", "quasar", "once-only pair"),
    ]
    examples = []
    for wi, wj, label in pairs:
        i, j = w2i[wi], w2i[wj]
        c_ij = counts[i, j]
        pmi = pmi_from_counts(c_ij, row_s[i], col_s[j], total)
        examples.append(
            {
                "label": label,
                "word_i": wi,
                "word_j": wj,
                "count_ij": float(c_ij),
                "row_marginal_i": float(row_s[i]),
                "col_marginal_j": float(col_s[j]),
                "total_mass": float(total),
                "pmi": pmi,
                "ppmi": ppmi_from_pmi(pmi),
            }
        )
    return {"corpus": corpus, "examples": examples}
