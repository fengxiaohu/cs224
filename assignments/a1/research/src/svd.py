"""TruncatedSVD wrappers with explicit random_state and n_iter."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from config import SVD_N_ITER


def fit_transform_svd(
    matrix: sparse.csr_matrix,
    k: int,
    random_state: int,
) -> tuple[np.ndarray, TruncatedSVD]:
    """Return fit_transform output E (n x k) and fitted model."""
    n_words = matrix.shape[0]
    k_eff = min(k, n_words - 1) if n_words > 1 else 1
    svd = TruncatedSVD(
        n_components=k_eff,
        random_state=random_state,
        n_iter=SVD_N_ITER,
    )
    emb = svd.fit_transform(matrix)
    return emb, svd


def reconstruction_squared_error(
    matrix: sparse.csr_matrix,
    emb_k: np.ndarray,
) -> float:
    """
    ||M||_F^2 - ||E_k||_F^2 where E_k is fit_transform output (first k cols).
    Does NOT materialize E V^T or use singular values directly.
    """
    m_fro_sq = float(sparse.linalg.norm(matrix) ** 2)
    e_fro_sq = float(np.sum(emb_k ** 2))
    return m_fro_sq - e_fro_sq
