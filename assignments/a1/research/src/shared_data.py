"""Shared corpus artifacts loaded once per pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from config import EVAL_MIN_COUNT, WINDOW_SIZE
from src.corpus import (
    build_word2ind,
    distinct_words,
    evaluation_vocabulary_with_stats,
    load_research_corpus,
    word_counts,
)
from src.run_utils import load_or_assert_hashes, save_json, sha256_token_stream
from src.sparse_counts import compute_co_occurrence_matrix


@dataclass
class CorpusArtifacts:
    corpus: list[list[str]]
    meta: dict
    words: list[str]
    word2ind: dict[str, int]
    counts: dict[str, int]
    eval_vocab: list[str]
    eval_stats: dict
    co_occurrence: sparse.csr_matrix
    hashes: dict[str, str] = field(default_factory=dict)


def load_corpus_artifacts(run_dir=None) -> CorpusArtifacts:
    corpus, meta = load_research_corpus()
    token_hash = sha256_token_stream(corpus)
    hashes = load_or_assert_hashes(meta["raw_text_sha256"], token_hash)
    meta["preprocessed_token_sha256"] = token_hash

    words = distinct_words(corpus)
    word2ind = build_word2ind(words)
    counts = dict(word_counts(corpus))
    eval_stats = evaluation_vocabulary_with_stats(counts, EVAL_MIN_COUNT)
    co_occ = compute_co_occurrence_matrix(corpus, word2ind, WINDOW_SIZE)

    if run_dir is not None:
        save_json(run_dir / "word2ind.json", word2ind)
        save_json(run_dir / "data_hashes.json", hashes)
        save_json(
            run_dir / "corpus_stats.json",
            {
                **meta,
                **eval_stats,
                "vocab_size": len(words),
                "co_occurrence_nnz": int(co_occ.nnz),
            },
        )

    return CorpusArtifacts(
        corpus=corpus,
        meta=meta,
        words=words,
        word2ind=word2ind,
        counts=counts,
        eval_vocab=eval_stats["eval_vocab"],
        eval_stats=eval_stats,
        co_occurrence=co_occ,
        hashes=hashes,
    )
