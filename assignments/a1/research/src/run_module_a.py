"""Module A: raw counts → PPMI → SVD neighbor comparison."""

from __future__ import annotations

import sys
from pathlib import Path

RESEARCH_DIR = Path(__file__).resolve().parents[1]
if str(RESEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(RESEARCH_DIR))

from config import (
    CDS_ALPHA,
    EVAL_MIN_COUNT,
    MODULE_A_K,
    MODULE_A_SEEDS,
    QUERY_WORDS,
    SEED,
)
from src.neighbors import neighbor_overlap, top_k_neighbors
from src.ppmi import compute_ppmi_matrix, toy_corpus_examples
from src.run_utils import make_run_dir, save_config, save_json
from src.shared_data import load_corpus_artifacts
from src.svd import fit_transform_svd


def _evaluable_queries(counts: dict[str, int]) -> dict[str, bool]:
    return {
        q: (q in counts and counts[q] >= EVAL_MIN_COUNT)
        for q in QUERY_WORDS
    }


def _run_pipeline(
    matrix,
    words,
    word2ind,
    eval_vocab,
    counts,
    label: str,
    k: int,
    random_state: int,
) -> dict:
    emb, _ = fit_transform_svd(matrix, k, random_state)
    results = {}
    for q in QUERY_WORDS:
        if not (q in counts and counts[q] >= EVAL_MIN_COUNT):
            results[q] = {"evaluable": False, "reason": "count_lt_10_or_missing"}
            continue
        results[q] = top_k_neighbors(
            emb, words, word2ind, q, eval_vocab, k=10
        )
    return {"label": label, "random_state": random_state, "queries": results}


def main() -> Path:
    run_dir = make_run_dir("module_a")
    artifacts = load_corpus_artifacts(run_dir)

    toy = toy_corpus_examples()
    save_json(run_dir / "toy_corpus_check.json", toy)

    pipelines = {
        "raw_svd": artifacts.co_occurrence,
        "ppmi_svd": compute_ppmi_matrix(artifacts.co_occurrence),
        "ppmi_cds_svd": compute_ppmi_matrix(artifacts.co_occurrence, cds_alpha=CDS_ALPHA),
    }

    all_results = {}
    overlap_report = {}

    for pname, matrix in pipelines.items():
        primary = _run_pipeline(
            matrix,
            artifacts.words,
            artifacts.word2ind,
            artifacts.eval_vocab,
            artifacts.counts,
            pname,
            MODULE_A_K,
            SEED,
        )
        all_results[pname] = {"seed_0": primary}

        overlaps = {}
        for alt_seed in MODULE_A_SEEDS[1:]:
            alt = _run_pipeline(
                matrix,
                artifacts.words,
                artifacts.word2ind,
                artifacts.eval_vocab,
                artifacts.counts,
                pname,
                MODULE_A_K,
                alt_seed,
            )
            all_results[pname][f"seed_{alt_seed}"] = alt
            per_query = {}
            for q in QUERY_WORDS:
                p0 = primary["queries"].get(q, {})
                pa = alt["queries"].get(q, {})
                if p0.get("evaluable") and pa.get("evaluable"):
                    per_query[q] = neighbor_overlap(
                        p0["neighbors"], pa["neighbors"]
                    )
                else:
                    per_query[q] = None
            overlaps[f"seed_{alt_seed}_vs_0"] = per_query
        overlap_report[pname] = overlaps

    query_evaluability = _evaluable_queries(artifacts.counts)
    save_json(run_dir / "module_a_neighbors.json", all_results)
    save_json(run_dir / "module_a_overlap.json", overlap_report)
    save_json(run_dir / "query_evaluability.json", query_evaluability)

    save_config(
        run_dir,
        {
            "module": "A",
            "k": MODULE_A_K,
            "seeds": MODULE_A_SEEDS,
            "pipelines": list(pipelines.keys()),
            "eval_vocab_size": artifacts.eval_stats["eval_vocab_size"],
            "count_ge_10_before_exclusion": artifacts.eval_stats["count_ge_10_before_exclusion"],
            "query_words": QUERY_WORDS,
            "data_hashes": artifacts.hashes,
            "dataset_revision": artifacts.meta["dataset_revision"],
        },
    )
    print(f"Module A complete: {run_dir}")
    return run_dir


if __name__ == "__main__":
    main()
