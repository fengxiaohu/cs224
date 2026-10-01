"""GloVe analogy evaluation with single and average offsets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.glove_utils import l2_normalize


@dataclass
class AnalogyResult:
    relation: str
    split: str
    a: str
    b: str
    c: str
    gold: str
    offset_type: str
    rank: int | None
    top5: list[str]
    status: str  # evaluable, oov, unevaluable


def _norm_vec(wv, word: str) -> np.ndarray | None:
    if word not in wv:
        return None
    return l2_normalize(wv[word].astype(np.float64))


def compute_offset(wv, a: str, b: str) -> np.ndarray | None:
    va = _norm_vec(wv, a)
    vb = _norm_vec(wv, b)
    if va is None or vb is None:
        return None
    return vb - va


def average_offset(wv, train_pairs: list[dict[str, str]]) -> np.ndarray | None:
    offsets = []
    for pair in train_pairs:
        off = compute_offset(wv, pair["a"], pair["b"])
        if off is None:
            return None
        offsets.append(off)
    return np.mean(offsets, axis=0)


def evaluate_analogy_pair(
    wv,
    a: str,
    b: str,
    c: str,
    gold: str,
    offset: np.ndarray,
    vocab_words: list[str] | None = None,
) -> tuple[int | None, list[str], str]:
    """
    Query: norm(c) + offset, rank gold by cosine among normalized candidates.
    Excludes input words a, b, c from candidates.
    """
    inputs = {a, b, c}
    if gold in inputs:
        return None, [], "unevaluable_gold_is_input"

    vc = _norm_vec(wv, c)
    vg = _norm_vec(wv, gold)
    if vc is None or vg is None:
        missing = [w for w in (a, b, c, gold) if w not in wv]
        return None, [], f"oov:{','.join(missing)}"

    query = l2_normalize(vc + offset)
    if vocab_words is None:
        vocab_words = list(wv.index_to_key)

    candidates = []
    for w in vocab_words:
        if w in inputs or w not in wv:
            continue
        vw = _norm_vec(wv, w)
        if vw is not None:
            sim = float(np.dot(query, vw))
            candidates.append((w, sim))

    candidates.sort(key=lambda x: -x[1])
    top5 = [w for w, _ in candidates[:5]]
    ranks = [i + 1 for i, (w, _) in enumerate(candidates) if w == gold]
    rank = ranks[0] if ranks else None
    return rank, top5, "evaluable"


def run_analogy_evaluation(
    wv,
    analogy_set: dict[str, Any],
) -> dict[str, Any]:
    results: list[dict] = []
    summary: dict[str, dict] = {}

    for relation in analogy_set["relations"]:
        rel_name = relation["name"]
        train_pairs = relation["train"]
        test_pairs = relation["test"]

        single_off = compute_offset(wv, train_pairs[0]["a"], train_pairs[0]["b"])
        avg_off = average_offset(wv, train_pairs)

        for offset_type, offset in [("single", single_off), ("average", avg_off)]:
            key = f"{rel_name}_{offset_type}"
            n_eval = n_oov = n_other = 0
            top1 = top5 = 0

            for pair in test_pairs:
                a, b, c, gold = pair["a"], pair["b"], pair["c"], pair["gold"]
                if offset is None:
                    rank, top5_preds, status = None, [], "oov_offset"
                    n_oov += 1
                else:
                    rank, top5_preds, status = evaluate_analogy_pair(
                        wv, a, b, c, gold, offset
                    )
                rec = {
                    "relation": rel_name,
                    "offset_type": offset_type,
                    "split": "test",
                    **pair,
                    "rank": rank,
                    "top5": top5_preds,
                    "status": status,
                }
                results.append(rec)

                if status.startswith("oov"):
                    n_oov += 1
                elif status != "evaluable":
                    n_other += 1
                else:
                    n_eval += 1
                    if rank == 1:
                        top1 += 1
                    if rank is not None and rank <= 5:
                        top5 += 1

            summary[key] = {
                "top1_accuracy": top1 / n_eval if n_eval else None,
                "top5_accuracy": top5 / n_eval if n_eval else None,
                "n_evaluable": n_eval,
                "n_oov": n_oov,
                "n_other_unevaluable": n_other,
                "top1_hits": top1,
                "top5_hits": top5,
            }

    # Known failure cases (not in accuracy denominator)
    failure_cases = []
    for case in analogy_set.get("failure_cases", []):
        a, b, c, gold = case["a"], case["b"], case["c"], case["gold"]
        off = compute_offset(wv, a, b)  # not used for failure demo; use train offset from relation if provided
        if "offset_from" in case:
            rel = next(r for r in analogy_set["relations"] if r["name"] == case["offset_from"])
            off = compute_offset(wv, rel["train"][0]["a"], rel["train"][0]["b"])
        if off is not None:
            rank, top5_preds, status = evaluate_analogy_pair(wv, a, b, c, gold, off)
        else:
            rank, top5_preds, status = None, [], "oov"
        failure_cases.append({**case, "rank": rank, "top5": top5_preds, "status": status})

    return {"results": results, "summary": summary, "failure_cases": failure_cases}
