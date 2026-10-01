"""Module B: dimension scan, reconstruction, PIP, rotation invariance."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RESEARCH_DIR = Path(__file__).resolve().parents[1]
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from config import (
    EVAL_MIN_COUNT,
    MODULE_B_K,
    MODULE_B_K_VALUES,
    MODULE_B_PIP_SUBSET_SIZE,
    MODULE_B_ROTATION_COLS,
    QUERY_WORDS,
    SEED,
)
from src.neighbors import (
    jaccard_top_k,
    l2_normalize_rows,
    pip_loss,
    select_pip_subset,
    top_k_neighbors,
)
from src.ppmi import compute_ppmi_matrix
from src.run_utils import make_run_dir, save_config, save_json
from src.shared_data import load_corpus_artifacts
from src.svd import fit_transform_svd, reconstruction_squared_error


def main() -> Path:
    run_dir = make_run_dir("module_b")
    artifacts = load_corpus_artifacts(run_dir)

    ppmi = compute_ppmi_matrix(artifacts.co_occurrence)
    k_max = MODULE_B_K
    n_words = len(artifacts.words)
    if n_words <= k_max:
        k_max = n_words - 1

    emb, _ = fit_transform_svd(ppmi, k_max, SEED)
    actual_k = emb.shape[1]

    pip_words, pip_indices, zero_dropped = select_pip_subset(
        emb,
        artifacts.eval_vocab,
        artifacts.word2ind,
        artifacts.counts,
        MODULE_B_PIP_SUBSET_SIZE,
    )
    save_json(run_dir / "pip_subset_S.json", pip_words)

    k_values = [k for k in MODULE_B_K_VALUES if k <= actual_k]
    metrics = {
        "reconstruction_squared_error": {},
        "pip_loss": {},
        "neighbor_jaccard_vs_k100": {},
    }
    neighbors_by_k = {}

    ref_neighbors = {}
    for q in QUERY_WORDS:
        if q in artifacts.counts and artifacts.counts[q] >= EVAL_MIN_COUNT:
            ref_neighbors[q] = top_k_neighbors(
                emb, artifacts.words, artifacts.word2ind, q, artifacts.eval_vocab, k=10
            )

    for k in k_values:
        emb_k = emb[:, :k]
        metrics["reconstruction_squared_error"][str(k)] = reconstruction_squared_error(
            ppmi, emb_k
        )
        metrics["pip_loss"][str(k)] = pip_loss(emb, pip_indices, k, k_ref=actual_k)

        n_at_k = {}
        jaccards = {}
        for q in QUERY_WORDS:
            if q not in ref_neighbors or not ref_neighbors[q].get("evaluable"):
                continue
            n_q = top_k_neighbors(
                emb_k, artifacts.words, artifacts.word2ind, q, artifacts.eval_vocab, k=10
            )
            n_at_k[q] = n_q
            if n_q.get("evaluable"):
                jaccards[q] = jaccard_top_k(
                    n_q["neighbors"], ref_neighbors[q]["neighbors"], k=10
                )
        neighbors_by_k[str(k)] = n_at_k
        if k != actual_k:
            metrics["neighbor_jaccard_vs_k100"][str(k)] = jaccards

    # 2D distortion: k=100 space vs first 2 columns, same eval vocab
    distortion = {}
    emb_2d = emb[:, :2]
    for q in QUERY_WORDS:
        if q not in artifacts.counts or artifacts.counts[q] < EVAL_MIN_COUNT:
            continue
        n_full = top_k_neighbors(
            emb, artifacts.words, artifacts.word2ind, q, artifacts.eval_vocab, k=10
        )
        n_2d = top_k_neighbors(
            emb_2d, artifacts.words, artifacts.word2ind, q, artifacts.eval_vocab, k=10
        )
        if n_full.get("evaluable") and n_2d.get("evaluable"):
            distortion[q] = {
                "jaccard": jaccard_top_k(n_2d["neighbors"], n_full["neighbors"], k=10),
                "neighbors_100d": n_full["neighbors"],
                "neighbors_2d": n_2d["neighbors"],
            }

    # Rotation invariance on first 50 columns
    rot_k = min(MODULE_B_ROTATION_COLS, actual_k)
    rng = np.random.default_rng(SEED)
    Q, _ = np.linalg.qr(rng.standard_normal((rot_k, rot_k)))
    emb_rot = emb.copy()
    emb_rot[:, :rot_k] = emb[:, :rot_k] @ Q

    rotation_tolerance = 1e-10
    rotation_check = {"cosine_neighbors_unchanged": {}, "pip_unchanged": None}
    for q in QUERY_WORDS:
        if q not in artifacts.counts or artifacts.counts[q] < EVAL_MIN_COUNT:
            continue
        n_orig = top_k_neighbors(
            emb, artifacts.words, artifacts.word2ind, q, artifacts.eval_vocab, k=10
        )
        n_rot = top_k_neighbors(
            emb_rot, artifacts.words, artifacts.word2ind, q, artifacts.eval_vocab, k=10
        )
        if n_orig.get("evaluable") and n_rot.get("evaluable"):
            orig_words = [w for w, _ in n_orig["neighbors"]]
            rot_words = [w for w, _ in n_rot["neighbors"]]
            rotation_check["cosine_neighbors_unchanged"][q] = orig_words == rot_words

    pip_before = pip_loss(emb, pip_indices, rot_k, k_ref=actual_k)
    pip_after = pip_loss(emb_rot, pip_indices, rot_k, k_ref=actual_k)
    rotation_check["pip_before"] = pip_before
    rotation_check["pip_after"] = pip_after
    rotation_check["pip_unchanged"] = abs(pip_before - pip_after) < rotation_tolerance
    rotation_check["tolerance"] = rotation_tolerance

    save_json(run_dir / "rotation_matrix_Q.json", Q.tolist())
    save_json(run_dir / "module_b_metrics.json", metrics)
    save_json(run_dir / "module_b_neighbors.json", neighbors_by_k)
    save_json(run_dir / "module_b_2d_distortion.json", distortion)
    save_json(run_dir / "module_b_rotation_check.json", rotation_check)

    save_config(
        run_dir,
        {
            "module": "B",
            "k_max": actual_k,
            "k_values": k_values,
            "pip_subset_size": len(pip_words),
            "zero_vectors_dropped_from_pip": zero_dropped,
            "eval_vocab_size": artifacts.eval_stats["eval_vocab_size"],
            "data_hashes": artifacts.hashes,
        },
    )
    print(f"Module B complete: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
