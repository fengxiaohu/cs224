# MLX training backend

`mlx_train.py` is a self-contained MLX implementation of the Assignment 3
neural machine translation model. It reads the existing `vocab.json`,
`src.model`, `tgt.model`, and `zh_en_data` files. The source encoder uses a
kernel-two same-length convolution followed by a masked bidirectional LSTM;
the decoder is an LSTM cell with multiplicative attention and a target
vocabulary log-softmax loss.

The backend is separate from `run.py` so it does not import PyTorch. Its
checkpoint is an MLX/NPZ file and cannot be loaded by the PyTorch
`model.bin` code path. Start from the student directory, or provide explicit
paths on the command line:

```bash
conda env create -f env-mlx.yml
conda activate cs224n-mlx

# CPU forward/backward, validation, beam search, and checkpoint smoke path
python mlx_train.py --device cpu --smoke

# One-iteration Metal smoke test (forward/backward, validation, beam search, save)
sh run.sh mlx_check

# Assignment-length run on Apple Silicon
sh run.sh train_mlx

# Decode the test set and print corpus BLEU
sh run.sh test_mlx
```

`--device auto` selects Metal when MLX reports one and otherwise selects
CPU. `--device cpu` is useful on hosts where PyTorch MPS is unavailable or
where a deterministic smoke test is needed. The trainer prints the selected
device before loading data.

By default, model and summary artifacts are kept separate from the existing
PyTorch run:

* `outputs/mlx/model.npz` — best validated MLX weights (by dev perplexity);
* `outputs/mlx/model.latest.npz` — most recent validated weights;
* `outputs/mlx/metrics.json` — final metrics and checkpoint metadata;
* `outputs/mlx/beam_search_diagnostics/000200.json` — a validation example;
* `runs/mlx/metrics.jsonl` — per-logging-interval training metrics;
* `runs/mlx/events.out.tfevents...` — TensorBoard loss and perplexity scalars;
* `outputs/mlx/test_outputs.txt` — detokenized translations after `test_mlx`.

While training, start TensorBoard in another terminal with `sh run.sh tensorboard`
and open `http://localhost:6006/`.

Validation runs every 200 iterations (`--valid-niter` or
`--diagnostics-interval`) and writes a beam-size-10 JSON object with the same
`example_source`, `example_target`, and `hypotheses` fields as
`beam_search_diagnostics.py`. A run stopped by `--max-iter` also writes a
final diagnostic when its stop iteration is not already a validation
boundary, which makes short smoke runs inspectable.

The MLX backend intentionally does not try to translate PyTorch checkpoint
weights. Parameter names, convolution layout, recurrent implementation, and
serialization formats differ, so train a fresh MLX model when switching
backends. `model.npz` is for `mlx_decode.py`; the course's PyTorch `run.py`
cannot load it. The latest file stores model weights only and does not include
optimizer state for resuming an interrupted training run.

On the current M1 Pro host, MLX 0.32.2 successfully selected the Metal GPU and
completed a one-iteration smoke run. The installed PyTorch 2.14.0 build does
not expose MPS on macOS 27.0, so `train_mlx` is the working accelerated path on
this machine; see [MPS.md](MPS.md) for the PyTorch compatibility details.
