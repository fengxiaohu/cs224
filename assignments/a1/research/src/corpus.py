"""IMDB corpus loading and preprocessing."""

from __future__ import annotations

import re
from collections import Counter

from datasets import load_dataset

from config import (
    DATASET_CONFIG,
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SLICE,
    DATASET_SPLIT,
    END_TOKEN,
    EVAL_MIN_COUNT,
    EXCLUDED_TOKENS,
    EXPECTED_NEG_LABEL,
    EXPECTED_POS_LABEL,
    EXPECTED_TOKEN_COUNT,
    PREPROCESS_PATTERN,
    START_TOKEN,
)
from src.run_utils import sha256_text


def preprocess_text(text: str) -> list[str]:
    tokens = [re.sub(PREPROCESS_PATTERN, "", w.lower()) for w in text.split(" ")]
    return [START_TOKEN] + tokens + [END_TOKEN]


def load_research_corpus() -> tuple[list[list[str]], dict]:
    ds = load_dataset(
        DATASET_NAME,
        name=DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    subset = ds.select(range(DATASET_SLICE.start, DATASET_SLICE.stop))
    labels = subset["label"]
    neg = sum(1 for lb in labels if lb == 0)
    pos = sum(1 for lb in labels if lb == 1)
    assert neg == EXPECTED_NEG_LABEL, f"expected {EXPECTED_NEG_LABEL} neg, got {neg}"
    assert pos == EXPECTED_POS_LABEL, f"expected {EXPECTED_POS_LABEL} pos, got {pos}"

    raw_texts = subset["text"]
    raw_text_hash = sha256_text("\n".join(raw_texts))
    corpus = [preprocess_text(t) for t in raw_texts]
    token_count = sum(len(doc) for doc in corpus)
    assert token_count == EXPECTED_TOKEN_COUNT, (
        f"expected {EXPECTED_TOKEN_COUNT} tokens, got {token_count}"
    )

    meta = {
        "dataset_revision": DATASET_REVISION,
        "neg_labels": neg,
        "pos_labels": pos,
        "token_count": token_count,
        "doc_count": len(corpus),
        "raw_text_sha256": raw_text_hash,
    }
    return corpus, meta


def distinct_words(corpus: list[list[str]]) -> list[str]:
    return sorted(set(w for doc in corpus for w in doc))


def word_counts(corpus: list[list[str]]) -> Counter[str]:
    return Counter(w for doc in corpus for w in doc)


def build_word2ind(words: list[str]) -> dict[str, int]:
    return {w: i for i, w in enumerate(words)}


def evaluation_vocabulary(counts: Counter[str], min_count: int = EVAL_MIN_COUNT) -> list[str]:
    """Words with count >= min_count, excluding special tokens."""
    return sorted(
        w for w, c in counts.items()
        if c >= min_count and w not in EXCLUDED_TOKENS
    )


def evaluation_vocabulary_with_stats(
    counts: Counter[str],
    min_count: int = EVAL_MIN_COUNT,
) -> dict:
    before_exclusion = sum(1 for c in counts.values() if c >= min_count)
    eval_vocab = evaluation_vocabulary(counts, min_count)
    return {
        "eval_vocab": eval_vocab,
        "eval_vocab_size": len(eval_vocab),
        "count_ge_10_before_exclusion": before_exclusion,
    }
