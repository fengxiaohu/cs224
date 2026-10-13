#!/usr/bin/env python3
"""Train the CS224N A3 NMT model with Apple's MLX.

The command is intentionally independent of ``run.py`` and PyTorch.  It
accepts the same corpus/vocabulary assets, writes model artifacts below
``outputs/mlx`` and writes JSONL training metrics below ``runs/mlx``.

Examples (run from ``student/`` or pass explicit paths)::

    python mlx_train.py --device cpu --max-iter 1 --embed-size 16 --hidden-size 16
    python mlx_train.py --device metal --max-iter 200

Validation runs every 200 iterations by default and records a beam-size-10
example in ``outputs/mlx/beam_search_diagnostics``.  ``--smoke`` selects a
small one-iteration CPU-friendly configuration for checking forward,
backward, checkpoint, and beam-search paths.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import json
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optimizers
    _MLX_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - host dependent
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optimizers = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR = exc

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - numpy is an MLX dependency
    np = None  # type: ignore[assignment]
    _NUMPY_IMPORT_ERROR: Optional[Exception] = exc
else:
    _NUMPY_IMPORT_ERROR = None

try:
    import sentencepiece as spm
    _SPM_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    spm = None  # type: ignore[assignment]
    _SPM_IMPORT_ERROR = exc

try:
    from tensorboardX import SummaryWriter
    _TENSORBOARD_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - logging is optional for smoke checks
    SummaryWriter = None  # type: ignore[assignment,misc]
    _TENSORBOARD_ERROR = exc

try:
    from mlx_nmt_model import (
        Hypothesis,
        MLXNMT,
        MlxVocabulary,
        _tree_leaves,
        _tree_map,
        clip_grad_norm,
    )
except ModuleNotFoundError:  # Support ``python -m student.mlx_train`` as well.
    from .mlx_nmt_model import (
        Hypothesis,
        MLXNMT,
        MlxVocabulary,
        _tree_leaves,
        _tree_map,
        clip_grad_norm,
    )


PROJECT_DIR = Path(__file__).resolve().parent


def require_runtime() -> None:
    if mx is None or nn is None or optimizers is None:
        detail = f" ({_MLX_IMPORT_ERROR})" if _MLX_IMPORT_ERROR else ""
        raise RuntimeError(
            "The MLX trainer requires the `mlx` package. Create the environment "
            "from student/env-mlx.yml, then rerun this command."
            + detail
        )
    if np is None:
        raise RuntimeError(f"The MLX trainer requires numpy ({_NUMPY_IMPORT_ERROR}).")
    if spm is None:
        raise RuntimeError(
            "The MLX trainer requires sentencepiece to read src.model/tgt.model."
            f" ({_SPM_IMPORT_ERROR})"
        )


def resolve_path(value: str | Path, *, base: Path = PROJECT_DIR) -> Path:
    """Resolve a CLI path from the caller's directory or this student folder."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return base / path


def read_corpus_mlx(
    file_path: str | Path,
    *,
    source: str,
    model_path: str | Path,
    max_sentences: Optional[int] = None,
) -> List[List[str]]:
    """SentencePiece-tokenize a corpus with the repository's conventions.

    This mirrors ``utils.read_corpus``: source lines are left unchanged while
    target lines receive ``<s>`` and ``</s>`` markers.  It is local so the MLX
    environment does not need to import the PyTorch-based ``utils.py``.
    """

    require_runtime()
    processor = spm.SentencePieceProcessor()
    loaded = processor.load(str(model_path))
    if not loaded:
        raise OSError(f"SentencePiece could not load model: {model_path}")
    corpus: List[List[str]] = []
    with open(file_path, "r", encoding="utf8") as handle:
        for line_number, line in enumerate(handle):
            if max_sentences is not None and line_number >= max_sentences:
                break
            pieces = processor.encode_as_pieces(line.rstrip("\n"))
            if source == "tgt":
                pieces = ["<s>"] + pieces + ["</s>"]
            corpus.append(pieces)
    return corpus


def batch_iter(
    data: Sequence[Tuple[List[str], List[str]]],
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Iterator[Tuple[List[List[str]], List[List[str]]]]:
    """Yield length-sorted mini-batches like ``utils.batch_iter``."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    indices = list(range(len(data)))
    if shuffle:
        random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        examples = [data[index] for index in indices[start : start + batch_size]]
        examples.sort(key=lambda example: len(example[0]), reverse=True)
        yield [example[0] for example in examples], [example[1] for example in examples]


def pad_ids(sentences: Sequence[Sequence[int]], pad_id: int) -> "mx.array":
    require_runtime()
    if not sentences:
        raise ValueError("cannot batch an empty sentence list")
    max_length = max(len(sentence) for sentence in sentences)
    if max_length == 0:
        raise ValueError("sentences must contain at least one token")
    padded = [list(sentence) + [pad_id] * (max_length - len(sentence)) for sentence in sentences]
    return mx.array(padded, dtype=mx.int32)


def numerical_batch(
    src_sents: Sequence[Sequence[str]],
    tgt_sents: Sequence[Sequence[str]],
    vocab: MlxVocabulary,
) -> Tuple["mx.array", "mx.array", "mx.array"]:
    source_ids = [vocab.src_ids(sentence) for sentence in src_sents]
    target_ids = [vocab.tgt_ids(sentence) for sentence in tgt_sents]
    source_lengths = mx.array([len(sentence) for sentence in source_ids], dtype=mx.int32)
    return (
        pad_ids(source_ids, vocab.src_pad_id),
        source_lengths,
        pad_ids(target_ids, vocab.tgt_pad_id),
    )


def _set_default_device(requested: str) -> str:
    """Select CPU or Metal where the installed MLX exposes the corresponding API."""

    require_runtime()
    requested = requested.lower()

    def metal_available() -> bool:
        metal = getattr(mx, "metal", None)
        checker = getattr(metal, "is_available", None)
        if checker is None:
            # Older MLX releases simply raise when a GPU is unavailable.
            try:
                mx.array([0.0], device=mx.gpu)
                return True
            except Exception:
                return False
        try:
            return bool(checker())
        except Exception:
            return False

    if requested == "cpu":
        mx.set_default_device(mx.cpu)
        return "cpu"
    if requested in ("metal", "gpu"):
        if not metal_available():
            raise RuntimeError(
                "Metal was requested, but MLX reports no Metal device. Use --device cpu "
                "for the portable smoke path."
            )
        mx.set_default_device(mx.gpu)
        return "metal"
    if requested == "auto":
        if metal_available():
            mx.set_default_device(mx.gpu)
            return "metal"
        mx.set_default_device(mx.cpu)
        return "cpu"
    raise ValueError("--device must be one of: cpu, metal, auto")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed * 13 // 7)
    if mx is not None:
        seed_fn = getattr(getattr(mx, "random", None), "seed", None)
        if seed_fn is not None:
            seed_fn(seed)


def _eval_tree(*trees: Any) -> None:
    leaves: List[Any] = []
    for tree in trees:
        leaves.extend(leaf for leaf in _tree_leaves(tree) if leaf is not None)
    if leaves:
        mx.eval(*leaves)


def evaluate_ppl(
    model: MLXNMT,
    data: Sequence[Tuple[List[str], List[str]]],
    vocab: MlxVocabulary,
    batch_size: int,
) -> float:
    """Compute token-level target perplexity over a corpus."""

    was_training = bool(getattr(model, "training", True))
    model.eval()
    total_negative_log_likelihood = 0.0
    total_target_words = 0
    for src_sents, tgt_sents in batch_iter(data, batch_size, shuffle=False):
        source_ids, source_lengths, target_ids = numerical_batch(src_sents, tgt_sents, vocab)
        scores = model(source_ids, source_lengths, target_ids)
        _eval_tree(scores)
        total_negative_log_likelihood += float(-mx.sum(scores))
        total_target_words += sum(max(0, len(sentence) - 1) for sentence in tgt_sents)
    if was_training:
        model.train()
    if total_target_words == 0:
        return float("inf")
    return math.exp(total_negative_log_likelihood / total_target_words)


def _diagnostic_payload(
    source: Sequence[str],
    target: Sequence[str],
    hypotheses: Sequence[Hypothesis],
    iteration: int,
) -> Dict[str, Any]:
    """Build the same JSON fields emitted by ``beam_search_diagnostics.py``."""

    return {
        "example_source": list(source),
        "example_target": list(target),
        "hypotheses": [
            {"hypothesis": list(hypothesis.value), "score": float(hypothesis.score)}
            for hypothesis in hypotheses
        ],
        "diagnostic_info": {
            "t": datetime.now(timezone.utc).isoformat(),
            "h": socket.gethostname(),
            "u": getpass.getuser(),
            "iteration": iteration,
        },
    }


def write_diagnostics(
    model: MLXNMT,
    source: Sequence[str],
    target: Sequence[str],
    *,
    iteration: int,
    output_dir: Path,
    max_decoding_time_step: int,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    was_training = bool(getattr(model, "training", True))
    model.eval()
    try:
        hypotheses = model.beam_search(
            source,
            beam_size=10,
            max_decoding_time_step=max_decoding_time_step,
        )
    finally:
        if was_training:
            model.train()
    path = output_dir / f"{iteration:06d}.json"
    path.write_text(
        json.dumps(
            _diagnostic_payload(source, target, hypotheses, iteration),
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf8",
    )
    return path


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["train"], default="train")
    parser.add_argument("--train-src", default="zh_en_data/train.zh")
    parser.add_argument("--train-tgt", default="zh_en_data/train.en")
    parser.add_argument("--dev-src", default="zh_en_data/dev.zh")
    parser.add_argument("--dev-tgt", default="zh_en_data/dev.en")
    parser.add_argument("--vocab", default="vocab.json")
    parser.add_argument("--src-model", default="src.model")
    parser.add_argument("--tgt-model", default="tgt.model")
    parser.add_argument("--device", choices=["cpu", "metal", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embed-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--uniform-init", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-grad", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--valid-niter", "--diagnostics-interval", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=0)
    parser.add_argument("--max-epoch", type=int, default=30)
    parser.add_argument("--max-decoding-time-step", type=int, default=70)
    parser.add_argument("--save-to", default="outputs/mlx/model.npz")
    parser.add_argument("--diagnostics-dir", default="outputs/mlx/beam_search_diagnostics")
    parser.add_argument("--metrics-out", default="outputs/mlx/metrics.json")
    parser.add_argument("--runs-dir", default="runs/mlx")
    parser.add_argument("--smoke", action="store_true", help="run one tiny CPU-friendly training iteration")
    return parser


def _load_data(args: argparse.Namespace) -> Tuple[
    MlxVocabulary,
    List[Tuple[List[str], List[str]]],
    List[Tuple[List[str], List[str]]],
]:
    vocab_path = resolve_path(args.vocab)
    src_model_path = resolve_path(args.src_model)
    tgt_model_path = resolve_path(args.tgt_model)
    vocab = MlxVocabulary.load(vocab_path)
    sentence_limit = 8 if getattr(args, "smoke", False) else None

    train_src = read_corpus_mlx(
        resolve_path(args.train_src),
        source="src",
        model_path=src_model_path,
        max_sentences=sentence_limit,
    )
    train_tgt = read_corpus_mlx(
        resolve_path(args.train_tgt),
        source="tgt",
        model_path=tgt_model_path,
        max_sentences=sentence_limit,
    )
    dev_src = read_corpus_mlx(
        resolve_path(args.dev_src),
        source="src",
        model_path=src_model_path,
        max_sentences=sentence_limit,
    )
    dev_tgt = read_corpus_mlx(
        resolve_path(args.dev_tgt),
        source="tgt",
        model_path=tgt_model_path,
        max_sentences=sentence_limit,
    )
    if len(train_src) != len(train_tgt):
        raise ValueError("training source and target corpora have different line counts")
    if len(dev_src) != len(dev_tgt):
        raise ValueError("development source and target corpora have different line counts")
    return vocab, list(zip(train_src, train_tgt)), list(zip(dev_src, dev_tgt))


def _validate_and_diagnose(
    model: MLXNMT,
    dev_data: Sequence[Tuple[List[str], List[str]]],
    vocab: MlxVocabulary,
    args: argparse.Namespace,
    iteration: int,
) -> Tuple[float, Path]:
    dev_ppl = evaluate_ppl(model, dev_data, vocab, batch_size=max(args.batch_size, 64))
    source, target = dev_data[min(3, len(dev_data) - 1)]
    diagnostics_path = write_diagnostics(
        model,
        source,
        target,
        iteration=iteration,
        output_dir=resolve_path(args.diagnostics_dir),
        max_decoding_time_step=args.max_decoding_time_step,
    )
    print(
        f"validation iter {iteration}: dev perplexity {dev_ppl:.4f}; "
        f"beam diagnostics {diagnostics_path}",
        file=sys.stderr,
    )
    return dev_ppl, diagnostics_path


def train(args: argparse.Namespace) -> Dict[str, Any]:
    require_runtime()
    device_name = _set_default_device(args.device)
    _seed_everything(args.seed)
    print(f"MLX device: {device_name} (requested {args.device})", file=sys.stderr)

    vocab, train_data, dev_data = _load_data(args)
    if not train_data:
        raise ValueError("training corpus is empty")
    if not dev_data:
        raise ValueError("development corpus is empty")
    if args.smoke:
        args.embed_size = min(args.embed_size, 16)
        args.hidden_size = min(args.hidden_size, 16)
        args.batch_size = min(args.batch_size, 2)
        args.max_iter = 1
        args.max_decoding_time_step = min(args.max_decoding_time_step, 5)
        args.dropout = 0.0
        print("MLX smoke mode: embed=16 hidden=16 batch=2 max_iter=1", file=sys.stderr)

    model = MLXNMT(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab=vocab,
        dropout_rate=args.dropout,
    )
    if args.uniform_init <= 0:
        raise ValueError("--uniform-init must be positive")
    model.update(
        _tree_map(
            lambda parameter: mx.random.uniform(
                low=-args.uniform_init,
                high=args.uniform_init,
                shape=parameter.shape,
                dtype=parameter.dtype,
            ),
            model.parameters(),
        )
    )
    _eval_tree(model.parameters())
    model.train()
    optimizer = optimizers.Adam(learning_rate=args.lr)

    def loss_fn(current_model: MLXNMT, source_ids: Any, source_lengths: Any, target_ids: Any) -> Any:
        return -mx.mean(current_model(source_ids, source_lengths, target_ids))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    runs_dir = resolve_path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = runs_dir / "metrics.jsonl"
    writer = SummaryWriter(logdir=str(runs_dir)) if SummaryWriter is not None else None
    if writer is None:
        print(
            f"TensorBoard event logging disabled: {_TENSORBOARD_ERROR}",
            file=sys.stderr,
        )
    iteration = 0
    epoch = 0
    total_nll = 0.0
    total_words = 0
    last_validation_iter = 0
    last_dev_ppl = float("inf")
    best_dev_ppl = float("inf")
    best_checkpoint_saved = False
    save_path = resolve_path(args.save_to)
    latest_save_path = save_path.with_name(
        f"{save_path.stem}.latest{save_path.suffix or '.npz'}"
    )
    train_start = time.time()

    while True:
        epoch += 1
        reached_max_iter = False
        for src_sents, tgt_sents in batch_iter(train_data, args.batch_size, shuffle=True):
            iteration += 1
            source_ids, source_lengths, target_ids = numerical_batch(src_sents, tgt_sents, vocab)
            loss, grads = loss_and_grad(model, source_ids, source_lengths, target_ids)
            grads, grad_norm = clip_grad_norm(grads, args.clip_grad)
            optimizer.update(model, grads)
            _eval_tree(loss, grads, model.parameters())

            batch_nll = float(loss) * len(src_sents)
            batch_words = sum(max(0, len(sentence) - 1) for sentence in tgt_sents)
            total_nll += batch_nll
            total_words += batch_words
            if iteration % max(1, args.log_every) == 0 or iteration == 1:
                token_loss = batch_nll / max(1, batch_words)
                payload = {
                    "iteration": iteration,
                    "epoch": epoch,
                    "train_loss": token_loss,
                    "train_perplexity": math.exp(min(50.0, token_loss)),
                    "grad_norm": grad_norm,
                    "device": device_name,
                }
                _write_jsonl(metrics_log, payload)
                if writer is not None:
                    writer.add_scalar("loss/train", token_loss, iteration)
                    writer.add_scalar(
                        "perplexity/train", payload["train_perplexity"], iteration
                    )
                    writer.add_scalar("gradient/norm", grad_norm, iteration)
                print(
                    f"epoch {epoch}, iter {iteration}, loss {token_loss:.4f}, "
                    f"ppl {payload['train_perplexity']:.2f}, grad_norm {grad_norm:.3f}",
                    file=sys.stderr,
                )

            if iteration % max(1, args.valid_niter) == 0:
                last_dev_ppl, _ = _validate_and_diagnose(
                    model, dev_data, vocab, args, iteration
                )
                last_validation_iter = iteration
                if writer is not None:
                    writer.add_scalar("perplexity/dev", last_dev_ppl, iteration)
                    writer.add_scalar("loss/dev", math.log(last_dev_ppl), iteration)
                # Keep both the latest state for recovery and the best dev-PPL
                # state for decoding, matching the reference trainer's policy.
                model.save_weights(latest_save_path)
                if not best_checkpoint_saved or last_dev_ppl < best_dev_ppl:
                    best_dev_ppl = last_dev_ppl
                    model.save_weights(save_path)
                    best_checkpoint_saved = True

            reached_max_iter = args.max_iter > 0 and iteration >= args.max_iter
            if reached_max_iter:
                break
        if reached_max_iter:
            break
        if args.max_epoch > 0 and epoch >= args.max_epoch:
            break

    # A short smoke run should still leave a useful validation artifact.  For
    # longer runs this only adds a final diagnostic when the stop point was not
    # already a 200-iteration validation boundary.
    if last_validation_iter != iteration:
        last_dev_ppl, _ = _validate_and_diagnose(model, dev_data, vocab, args, iteration)
        last_validation_iter = iteration
        if writer is not None:
            writer.add_scalar("perplexity/dev", last_dev_ppl, iteration)
            writer.add_scalar("loss/dev", math.log(last_dev_ppl), iteration)
        model.save_weights(latest_save_path)
        if not best_checkpoint_saved or last_dev_ppl < best_dev_ppl:
            best_dev_ppl = last_dev_ppl
            model.save_weights(save_path)
            best_checkpoint_saved = True

    # A validated corpus should always produce a best checkpoint; keep a
    # defensive fallback for malformed corpora whose perplexity is infinite.
    if not best_checkpoint_saved:
        model.save_weights(save_path)
    if writer is not None:
        writer.flush()
        writer.close()
    train_loss = total_nll / max(1, total_words)
    metrics = {
        "backend": "mlx",
        "device": device_name,
        "train_iter": iteration,
        "train_loss": train_loss,
        "last_dev_perplexity": last_dev_ppl,
        "best_dev_perplexity": best_dev_ppl,
        "embed_size": args.embed_size,
        "hidden_size": args.hidden_size,
        "uniform_init": args.uniform_init,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "checkpoint": str(save_path),
        "latest_checkpoint": str(latest_save_path),
        "diagnostics_dir": str(resolve_path(args.diagnostics_dir)),
        "elapsed_seconds": time.time() - train_start,
        "note": "MLX NPZ weights are not loadable by the PyTorch model.bin checkpoint path.",
    }
    metrics_path = resolve_path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf8")
    print(f"saved MLX checkpoint: {save_path}", file=sys.stderr)
    print(f"saved metrics: {metrics_path}", file=sys.stderr)
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.mode != "train":
        raise ValueError(f"unsupported mode: {args.mode}")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
