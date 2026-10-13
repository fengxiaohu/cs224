#!/usr/bin/env python3
"""Decode a corpus with a checkpoint produced by ``mlx_train.py``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

try:
    from mlx_nmt_model import MLXNMT, MlxVocabulary
    from mlx_train import _set_default_device, read_corpus_mlx, resolve_path
except ModuleNotFoundError:  # Support ``python -m student.mlx_decode``.
    from .mlx_nmt_model import MLXNMT, MlxVocabulary
    from .mlx_train import _set_default_device, read_corpus_mlx, resolve_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "metal", "auto"], default="auto")
    parser.add_argument("--model", default="outputs/mlx/model.npz")
    parser.add_argument("--metrics", default="outputs/mlx/metrics.json")
    parser.add_argument("--vocab", default="vocab.json")
    parser.add_argument("--src-model", default="src.model")
    parser.add_argument("--tgt-model", default="tgt.model")
    parser.add_argument("--test-src", default="zh_en_data/test.zh")
    parser.add_argument("--test-tgt", default="zh_en_data/test.en")
    parser.add_argument("--output", default="outputs/mlx/test_outputs.txt")
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--max-decoding-time-step", type=int, default=70)
    return parser


def decode(args: argparse.Namespace) -> int:
    device = _set_default_device(args.device)
    print(f"MLX device: {device} (requested {args.device})", file=sys.stderr)

    metrics_path = resolve_path(args.metrics)
    metrics = json.loads(metrics_path.read_text(encoding="utf8"))
    vocab = MlxVocabulary.load(resolve_path(args.vocab))
    model = MLXNMT(
        embed_size=int(metrics["embed_size"]),
        hidden_size=int(metrics["hidden_size"]),
        vocab=vocab,
        dropout_rate=0.0,
    )
    model.load_weights_file(resolve_path(args.model))
    model.eval()

    src_model_path = resolve_path(args.src_model)
    source = read_corpus_mlx(
        resolve_path(args.test_src), source="src", model_path=src_model_path
    )
    if not source:
        raise ValueError("test source corpus is empty")

    hypotheses = []
    for index, sentence in enumerate(source, start=1):
        candidates = model.beam_search(
            sentence,
            beam_size=args.beam_size,
            max_decoding_time_step=args.max_decoding_time_step,
        )
        hypotheses.append(candidates[0])
        if index % 100 == 0 or index == len(source):
            print(f"decoded {index}/{len(source)} sentences", file=sys.stderr)

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decoded = ["".join(hypothesis.value).replace("▁", " ") for hypothesis in hypotheses]
    output_path.write_text("\n".join(decoded) + "\n", encoding="utf8")

    target_path = resolve_path(args.test_tgt)
    if target_path.exists():
        try:
            import sacrebleu
        except ImportError:
            print("sacrebleu is not installed; skipping BLEU calculation", file=sys.stderr)
        else:
            targets = read_corpus_mlx(
                target_path,
                source="tgt",
                model_path=resolve_path(args.tgt_model),
            )
            if len(targets) != len(decoded):
                raise ValueError("test source and target corpora have different line counts")
            references = [
                "".join(sentence[1:-1]).replace("▁", " ") for sentence in targets
            ]
            print(
                f"Corpus BLEU: {sacrebleu.corpus_bleu(decoded, [references]).score:.2f}",
                file=sys.stderr,
            )

    print(f"wrote translations: {output_path}", file=sys.stderr)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    return decode(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
