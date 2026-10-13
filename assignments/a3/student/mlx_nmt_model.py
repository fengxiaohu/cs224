#!/usr/bin/env python3
"""MLX implementation of the CS224N A3 neural machine translation model.

The coursework's PyTorch model uses tensors with shape ``(time, batch, dim)``
at its public boundary.  MLX code is written batch first (``(batch, time,
dim)``), which keeps the recurrent loops and the beam search easy to read.
The mathematical model is the same:

* source and target embeddings;
* a kernel-two, same-length convolution over source embeddings;
* a bidirectional LSTM encoder with length masks;
* an LSTM-cell decoder with multiplicative (Luong) attention; and
* a bias-free target vocabulary projection.

This module intentionally does not import PyTorch.  MLX is imported lazily so
that commands such as ``python -m py_compile`` can still inspect the source on
machines where MLX has not been installed yet.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Keep source inspection and documentation builds usable without MLX.
    import mlx.core as mx
    import mlx.nn as nn
    _MLX_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - depends on the host environment
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR = exc


_ModuleBase = nn.Module if nn is not None else object


def _require_mlx() -> None:
    """Raise an actionable error when this optional backend is unavailable."""

    if mx is None or nn is None:
        detail = f" ({_MLX_IMPORT_ERROR})" if _MLX_IMPORT_ERROR else ""
        raise RuntimeError(
            "The MLX backend requires the `mlx` package. Install the dependencies "
            "from student/env-mlx.yml or run the CPU syntax checks without importing "
            f"the model.{detail}"
        )


def _tree_leaves(tree: Any) -> List[Any]:
    """Return leaves from the small nested trees used by MLX modules."""

    if isinstance(tree, Mapping):
        leaves: List[Any] = []
        for value in tree.values():
            leaves.extend(_tree_leaves(value))
        return leaves
    if isinstance(tree, (list, tuple)):
        leaves = []
        for value in tree:
            leaves.extend(_tree_leaves(value))
        return leaves
    return [tree]


def _tree_map(fn, tree: Any) -> Any:
    """Map ``fn`` over a nested MLX parameter/gradient tree."""

    if isinstance(tree, dict):
        return {key: _tree_map(fn, value) for key, value in tree.items()}
    if isinstance(tree, list):
        return [_tree_map(fn, value) for value in tree]
    if isinstance(tree, tuple):
        return tuple(_tree_map(fn, value) for value in tree)
    return fn(tree)


@dataclass(frozen=True)
class Hypothesis:
    """A decoded target sentence and its log-likelihood score."""

    value: List[str]
    score: float


@dataclass(frozen=True)
class MlxVocabulary:
    """Small JSON-backed vocabulary adapter that has no PyTorch dependency."""

    src_word2id: Dict[str, int]
    tgt_word2id: Dict[str, int]

    @classmethod
    def load(cls, path: str | Path) -> "MlxVocabulary":
        with open(path, "r", encoding="utf8") as handle:
            raw = json.load(handle)
        return cls(
            src_word2id={str(k): int(v) for k, v in raw["src_word2id"].items()},
            tgt_word2id={str(k): int(v) for k, v in raw["tgt_word2id"].items()},
        )

    @property
    def src_id2word(self) -> Dict[int, str]:
        return {value: key for key, value in self.src_word2id.items()}

    @property
    def tgt_id2word(self) -> Dict[int, str]:
        return {value: key for key, value in self.tgt_word2id.items()}

    @property
    def src_pad_id(self) -> int:
        return self.src_word2id["<pad>"]

    @property
    def tgt_pad_id(self) -> int:
        return self.tgt_word2id["<pad>"]

    @property
    def start_id(self) -> int:
        return self.tgt_word2id["<s>"]

    @property
    def end_id(self) -> int:
        return self.tgt_word2id["</s>"]

    def src_ids(self, sentence: Sequence[str]) -> List[int]:
        unk = self.src_word2id["<unk>"]
        return [self.src_word2id.get(token, unk) for token in sentence]

    def tgt_ids(self, sentence: Sequence[str]) -> List[int]:
        unk = self.tgt_word2id["<unk>"]
        return [self.tgt_word2id.get(token, unk) for token in sentence]


if nn is not None:

    class _LSTMCell(nn.Module):
        """A compact LSTM cell written directly with MLX primitives."""

        def __init__(self, input_size: int, hidden_size: int):
            super().__init__()
            self.hidden_size = hidden_size
            self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

        def __call__(
            self,
            x: "mx.array",
            state: Tuple["mx.array", "mx.array"],
        ) -> Tuple["mx.array", "mx.array"]:
            h_prev, c_prev = state
            gates = self.gates(mx.concatenate([x, h_prev], axis=-1))
            i, f, g, o = mx.split(gates, 4, axis=-1)
            i = mx.sigmoid(i)
            f = mx.sigmoid(f)
            g = mx.tanh(g)
            o = mx.sigmoid(o)
            c = f * c_prev + i * g
            h = o * mx.tanh(c)
            return h, c


    class _SameLengthConv1d(nn.Module):
        """Kernel-two MLX convolution whose output has exactly input length.

        MLX's one-dimensional convolution uses batch/time/channel layout.  A
        valid convolution over one extra right-padded timestep gives the same
        output length as PyTorch's ``padding='same'`` for kernel size two.
        """

        def __init__(self, channels: int):
            super().__init__()
            self.conv = nn.Conv1d(channels, channels, kernel_size=2, padding=0)

        def __call__(self, x: "mx.array") -> "mx.array":
            x = mx.pad(x, [(0, 0), (0, 1), (0, 0)])
            return self.conv(x)


    class MLXNMT(_ModuleBase):
        """The MLX counterpart of ``student/nmt_model.py``.

        ``source_ids`` and ``target_ids`` are integer arrays with shape
        ``(batch, time)``.  ``source_lengths`` contains the unpadded source
        lengths.  The call returns one target log-likelihood per sentence.
        """

        def __init__(
            self,
            embed_size: int,
            hidden_size: int,
            vocab: MlxVocabulary,
            dropout_rate: float = 0.2,
            attention_type: str = "multiplicative",
        ):
            super().__init__()
            if attention_type != "multiplicative":
                raise ValueError(
                    "The MLX backend implements the assignment's multiplicative "
                    f"attention only, got {attention_type!r}."
                )
            self.embed_size = int(embed_size)
            self.hidden_size = int(hidden_size)
            self.dropout_rate = float(dropout_rate)
            self.vocab = vocab
            self.attention_type = attention_type

            self.source_embedding = nn.Embedding(len(vocab.src_word2id), embed_size)
            self.target_embedding = nn.Embedding(len(vocab.tgt_word2id), embed_size)
            self.post_embed_cnn = _SameLengthConv1d(embed_size)

            # MLX has a sequence-first LSTM convenience module, but explicit
            # cells make the length mask unambiguous in both directions.
            self.encoder_forward = _LSTMCell(embed_size, hidden_size)
            self.encoder_backward = _LSTMCell(embed_size, hidden_size)
            self.decoder = _LSTMCell(embed_size + hidden_size, hidden_size)

            self.h_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.c_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.att_projection = nn.Linear(2 * hidden_size, hidden_size, bias=False)
            self.combined_output_projection = nn.Linear(3 * hidden_size, hidden_size, bias=False)
            self.target_vocab_projection = nn.Linear(
                hidden_size, len(vocab.tgt_word2id), bias=False
            )
            self.dropout = nn.Dropout(dropout_rate)

        @staticmethod
        def _as_int_array(value: Any) -> "mx.array":
            if isinstance(value, (list, tuple)):
                return mx.array(value, dtype=mx.int32)
            return value.astype(mx.int32)

        def _masked_embedding(
            self,
            ids: "mx.array",
            embedding: Any,
            pad_id: int,
        ) -> "mx.array":
            values = embedding(ids)
            # MLX embeddings do not expose PyTorch's padding_idx parameter.
            # Zeroing pad rows here gives the same boundary behavior for both
            # the source convolution and target teacher forcing.
            valid = (ids != pad_id).astype(values.dtype)[..., None]
            return values * valid

        def encode(
            self,
            source_ids: "mx.array",
            source_lengths: Sequence[int] | "mx.array",
        ) -> Tuple["mx.array", Tuple["mx.array", "mx.array"]]:
            """Encode a padded source batch and return decoder initial state."""

            _require_mlx()
            source_ids = self._as_int_array(source_ids)
            source_lengths = self._as_int_array(source_lengths)
            if source_ids.ndim != 2:
                raise ValueError(f"source_ids must have shape (batch, time), got {source_ids.shape}")

            batch_size, source_time = source_ids.shape
            if source_time == 0:
                raise ValueError("source sentences must contain at least one token")

            x = self._masked_embedding(
                source_ids, self.source_embedding, self.vocab.src_pad_id
            )
            x = self.post_embed_cnn(x)

            zeros = mx.zeros((batch_size, self.hidden_size), dtype=x.dtype)
            h_f, c_f = zeros, zeros
            forward_outputs: List[mx.array] = []
            for time_index in range(source_time):
                h_new, c_new = self.encoder_forward(x[:, time_index, :], (h_f, c_f))
                valid = (source_lengths > time_index).astype(x.dtype)[:, None]
                h_f = valid * h_new + (1.0 - valid) * h_f
                c_f = valid * c_new + (1.0 - valid) * c_f
                forward_outputs.append(valid * h_f)

            h_b, c_b = zeros, zeros
            backward_outputs: List[Optional[mx.array]] = [None] * source_time
            for time_index in range(source_time - 1, -1, -1):
                h_new, c_new = self.encoder_backward(x[:, time_index, :], (h_b, c_b))
                valid = (source_lengths > time_index).astype(x.dtype)[:, None]
                h_b = valid * h_new + (1.0 - valid) * h_b
                c_b = valid * c_new + (1.0 - valid) * c_b
                backward_outputs[time_index] = valid * h_b

            enc_forward = mx.stack(forward_outputs, axis=1)
            enc_backward = mx.stack([item for item in backward_outputs if item is not None], axis=1)
            enc_hiddens = mx.concatenate([enc_forward, enc_backward], axis=-1)

            last_hidden = mx.concatenate([h_f, h_b], axis=-1)
            last_cell = mx.concatenate([c_f, c_b], axis=-1)
            dec_init_state = (self.h_projection(last_hidden), self.c_projection(last_cell))
            return enc_hiddens, dec_init_state

        def generate_sent_masks(
            self,
            enc_hiddens: "mx.array",
            source_lengths: Sequence[int] | "mx.array",
        ) -> "mx.array":
            """Return a boolean validity mask with shape ``(batch, source_time)``."""

            source_lengths = self._as_int_array(source_lengths)
            positions = mx.arange(enc_hiddens.shape[1])[None, :]
            return positions < source_lengths[:, None]

        def _attention_scores(
            self,
            dec_hidden: "mx.array",
            enc_hiddens_proj: "mx.array",
            enc_valid: Optional["mx.array"],
        ) -> "mx.array":
            scores = mx.sum(enc_hiddens_proj * dec_hidden[:, None, :], axis=-1)
            if enc_valid is not None:
                scores = mx.where(enc_valid, scores, mx.array(-1e9, dtype=scores.dtype))
            return scores

        def step(
            self,
            ybar_t: "mx.array",
            dec_state: Tuple["mx.array", "mx.array"],
            enc_hiddens: "mx.array",
            enc_hiddens_proj: "mx.array",
            enc_valid: Optional["mx.array"],
        ) -> Tuple[Tuple["mx.array", "mx.array"], "mx.array", "mx.array"]:
            """Run one decoder LSTM and multiplicative-attention step."""

            dec_hidden, dec_cell = self.decoder(ybar_t, dec_state)
            e_t = self._attention_scores(dec_hidden, enc_hiddens_proj, enc_valid)
            alpha_t = mx.softmax(e_t, axis=-1)
            a_t = mx.sum(alpha_t[..., None] * enc_hiddens, axis=1)
            combined = self.combined_output_projection(
                mx.concatenate([dec_hidden, a_t], axis=-1)
            )
            combined = self.dropout(mx.tanh(combined))
            return (dec_hidden, dec_cell), combined, e_t

        def decode(
            self,
            enc_hiddens: "mx.array",
            enc_valid: "mx.array",
            dec_init_state: Tuple["mx.array", "mx.array"],
            target_ids: "mx.array",
        ) -> "mx.array":
            """Teacher-force the target prefix and return ``(B, T - 1, H)``."""

            target_ids = self._as_int_array(target_ids)
            target_embeddings = self._masked_embedding(
                target_ids[:, :-1], self.target_embedding, self.vocab.tgt_pad_id
            )
            enc_hiddens_proj = self.att_projection(enc_hiddens)
            batch_size = enc_hiddens.shape[0]
            o_prev = mx.zeros((batch_size, self.hidden_size), dtype=enc_hiddens.dtype)
            dec_state = dec_init_state
            outputs: List[mx.array] = []
            for time_index in range(target_embeddings.shape[1]):
                ybar_t = mx.concatenate(
                    [target_embeddings[:, time_index, :], o_prev], axis=-1
                )
                dec_state, o_t, _ = self.step(
                    ybar_t, dec_state, enc_hiddens, enc_hiddens_proj, enc_valid
                )
                outputs.append(o_t)
                o_prev = o_t

            if not outputs:
                return mx.zeros((batch_size, 0, self.hidden_size), dtype=enc_hiddens.dtype)
            return mx.stack(outputs, axis=1)

        def __call__(
            self,
            source_ids: "mx.array",
            source_lengths: Sequence[int] | "mx.array",
            target_ids: "mx.array",
        ) -> "mx.array":
            """Return target log-likelihood for each sentence in a batch."""

            source_ids = self._as_int_array(source_ids)
            target_ids = self._as_int_array(target_ids)
            enc_hiddens, dec_init_state = self.encode(source_ids, source_lengths)
            enc_valid = self.generate_sent_masks(enc_hiddens, source_lengths)
            combined_outputs = self.decode(
                enc_hiddens, enc_valid, dec_init_state, target_ids
            )
            if target_ids.shape[1] < 2:
                return mx.zeros((target_ids.shape[0],), dtype=enc_hiddens.dtype)

            logits = self.target_vocab_projection(combined_outputs)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            gold_ids = target_ids[:, 1:]
            # take_along_axis expects an explicit vocabulary axis.
            gold_log_probs = mx.take_along_axis(
                log_probs, gold_ids[..., None], axis=-1
            ).squeeze(-1)
            target_mask = (gold_ids != self.vocab.tgt_pad_id).astype(log_probs.dtype)
            return mx.sum(gold_log_probs * target_mask, axis=1)

        # Alias used by callers that mirror the PyTorch implementation.
        forward = __call__

        def _decode_step_batch(
            self,
            previous_word_ids: "mx.array",
            dec_state: Tuple["mx.array", "mx.array"],
            o_prev: "mx.array",
            enc_hiddens: "mx.array",
            enc_hiddens_proj: "mx.array",
            enc_valid: "mx.array",
        ) -> Tuple[Tuple["mx.array", "mx.array"], "mx.array"]:
            y_embed = self._masked_embedding(
                previous_word_ids, self.target_embedding, self.vocab.tgt_pad_id
            )
            next_state, o_t, _ = self.step(
                mx.concatenate([y_embed, o_prev], axis=-1),
                dec_state,
                enc_hiddens,
                enc_hiddens_proj,
                enc_valid,
            )
            return next_state, o_t

        def beam_search(
            self,
            src_sent: Sequence[str] | Sequence[int],
            beam_size: int = 10,
            max_decoding_time_step: int = 70,
        ) -> List[Hypothesis]:
            """Decode one source sentence with score-sorted beam search."""

            _require_mlx()
            if beam_size < 1:
                raise ValueError("beam_size must be positive")

            if src_sent and isinstance(src_sent[0], str):
                source_ids = self.vocab.src_ids(src_sent)  # type: ignore[arg-type]
            else:
                source_ids = [int(token) for token in src_sent]  # type: ignore[arg-type]
            src_array = mx.array([source_ids], dtype=mx.int32)
            src_lengths = mx.array([len(source_ids)], dtype=mx.int32)
            enc_hiddens, dec_init_state = self.encode(src_array, src_lengths)
            enc_valid = self.generate_sent_masks(enc_hiddens, src_lengths)
            enc_hiddens_proj = self.att_projection(enc_hiddens)

            # A hypothesis stores ids including the initial <s> token.  State
            # arrays are kept as one row per live hypothesis.
            live: List[Tuple[List[int], float]] = [([self.vocab.start_id], 0.0)]
            completed: List[Hypothesis] = []
            target_id2word = self.vocab.tgt_id2word
            h0, c0 = dec_init_state
            state = (h0, c0)
            o_prev = mx.zeros((1, self.hidden_size), dtype=enc_hiddens.dtype)

            import numpy as np

            for _ in range(max_decoding_time_step):
                if not live or len(completed) >= beam_size:
                    break
                hypothesis_count = len(live)
                expanded_enc = mx.broadcast_to(
                    enc_hiddens, (hypothesis_count,) + enc_hiddens.shape[1:]
                )
                expanded_proj = mx.broadcast_to(
                    enc_hiddens_proj, (hypothesis_count,) + enc_hiddens_proj.shape[1:]
                )
                expanded_valid = mx.broadcast_to(
                    enc_valid, (hypothesis_count,) + enc_valid.shape[1:]
                )
                previous_ids = mx.array(
                    [hypothesis[0][-1] for hypothesis in live], dtype=mx.int32
                )
                next_state, o_t = self._decode_step_batch(
                    previous_ids,
                    state,
                    o_prev,
                    expanded_enc,
                    expanded_proj,
                    expanded_valid,
                )
                log_probs = self.target_vocab_projection(o_t)
                log_probs = log_probs - mx.logsumexp(
                    log_probs, axis=-1, keepdims=True
                )
                mx.eval(log_probs, next_state[0], next_state[1], o_t)
                scores = np.asarray(log_probs)
                candidates = []
                for previous_index, (_, previous_score) in enumerate(live):
                    for word_id, word_score in enumerate(scores[previous_index]):
                        candidates.append(
                            (previous_score + float(word_score), previous_index, word_id)
                        )
                candidates.sort(key=lambda item: item[0], reverse=True)

                # Match the assignment implementation: choose exactly the
                # best ``beam_size - completed`` candidates for this step,
                # then split those candidates into completed and live paths.
                # Counting only live paths here would keep taking candidates
                # past the top-k boundary whenever one of the selected paths
                # ends with </s>.
                live_limit = beam_size - len(completed)
                new_live: List[Tuple[List[int], float]] = []
                new_hypothesis_indices: List[int] = []
                for candidate_score, previous_index, word_id in candidates[:live_limit]:
                    previous_ids_list, _ = live[previous_index]
                    candidate_ids = previous_ids_list + [word_id]
                    if word_id == self.vocab.end_id:
                        words = [target_id2word[token] for token in candidate_ids[1:-1]]
                        completed.append(Hypothesis(value=words, score=float(candidate_score)))
                    else:
                        new_live.append((candidate_ids, float(candidate_score)))
                        new_hypothesis_indices.append(previous_index)

                if len(completed) >= beam_size:
                    break
                if not new_live:
                    break
                parent_indices = mx.array(new_hypothesis_indices, dtype=mx.int32)
                state = (
                    next_state[0][parent_indices],
                    next_state[1][parent_indices],
                )
                o_prev = o_t[parent_indices]
                live = new_live

            if not completed:
                best_ids, best_score = live[0]
                completed.append(
                    Hypothesis(
                        value=[target_id2word[token] for token in best_ids[1:]],
                        score=float(best_score),
                    )
                )
            completed.sort(key=lambda hypothesis: hypothesis.score, reverse=True)
            return completed[:beam_size]

        def save_weights(self, path: str | Path) -> None:
            """Save MLX weights in a portable NPZ container.

            This format is deliberately separate from the PyTorch ``model.bin``
            checkpoint.  The two backends have different parameter names and
            tensor layouts, so PyTorch checkpoints are not loadable here.
            """

            _require_mlx()
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            flattened: Dict[str, Any] = {}

            def collect(tree: Any, prefix: str = "") -> None:
                if isinstance(tree, Mapping):
                    for key, value in tree.items():
                        child = f"{prefix}.{key}" if prefix else str(key)
                        collect(value, child)
                    return
                # MLX's savez accepts arrays as keyword values.  Dots are
                # avoided in keys because some NumPy readers treat them as
                # structured names.
                flattened[prefix.replace(".", "__")] = tree

            collect(self.parameters())
            mx.savez(str(path), **flattened)

        def load_weights_file(self, path: str | Path) -> None:
            """Load a checkpoint produced by :meth:`save_weights`."""

            _require_mlx()
            arrays = mx.load(str(path))
            weights = [(key.replace("__", "."), value) for key, value in arrays.items()]
            self.load_weights(weights)

        # A PyTorch-shaped spelling is useful to scripts without making the
        # checkpoint formats look interchangeable.
        save = save_weights


else:

    class MLXNMT(_ModuleBase):  # pragma: no cover - exercised only without MLX
        """Placeholder that gives a clear error when MLX is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any):
            _require_mlx()


NMT = MLXNMT


def clip_grad_norm(grads: Any, max_norm: float) -> Tuple[Any, float]:
    """Clip an MLX gradient tree and return ``(clipped_tree, norm)``.

    MLX versions have moved the optimizer clipping helper over time.  Keeping
    this tiny implementation here makes the training loop work across those
    versions and also mirrors PyTorch's global norm clipping behavior.
    """

    _require_mlx()
    leaves = [leaf for leaf in _tree_leaves(grads) if leaf is not None]
    if not leaves:
        return grads, 0.0
    mx.eval(*leaves)
    norm_squared = 0.0
    for leaf in leaves:
        norm_squared += float(mx.sum(leaf * leaf))
    norm = norm_squared ** 0.5
    if max_norm < 0.0:
        raise ValueError("max_norm must be nonnegative")
    if norm <= max_norm:
        return grads, norm
    scale = max_norm / (norm + 1e-12)
    return _tree_map(lambda leaf: None if leaf is None else leaf * scale, grads), norm


def load_mlx_checkpoint(
    path: str | Path,
    *,
    embed_size: int,
    hidden_size: int,
    vocab: MlxVocabulary,
    dropout_rate: float = 0.0,
) -> MLXNMT:
    """Construct and load an MLX checkpoint produced by this module."""

    model = MLXNMT(
        embed_size=embed_size,
        hidden_size=hidden_size,
        vocab=vocab,
        dropout_rate=dropout_rate,
    )
    model.load_weights_file(path)
    return model
