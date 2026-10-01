"""Sparse co-occurrence matrix construction."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from config import WINDOW_SIZE


def compute_co_occurrence_matrix(
    corpus: list[list[str]],
    word2ind: dict[str, int],
    window_size: int = WINDOW_SIZE,
) -> sparse.csr_matrix:
    """Symmetric word-word co-occurrence matrix in CSR format."""
    n = len(word2ind)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for doc in corpus:
        for i, center in enumerate(doc):
            if center not in word2ind:
                continue
            c_idx = word2ind[center]
            lo = max(0, i - window_size)
            hi = min(len(doc), i + window_size + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                context = doc[j]
                if context not in word2ind:
                    continue
                ctx_idx = word2ind[context]
                rows.append(c_idx)
                cols.append(ctx_idx)
                data.append(1.0)

    mat = sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)
    return mat.tocsr()


def row_marginals(mat: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(mat.sum(axis=1)).ravel()


def col_marginals(mat: sparse.csr_matrix) -> np.ndarray:
    return np.asarray(mat.sum(axis=0)).ravel()


def total_mass(mat: sparse.csr_matrix) -> float:
    return float(mat.sum())
