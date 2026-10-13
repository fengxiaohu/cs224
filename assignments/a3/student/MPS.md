# Apple Silicon (MPS)

This branch adds an explicit PyTorch MPS device path for Macs with Apple silicon. It keeps the existing CPU and CUDA paths available.

Create the environment from the `student/` directory:

```bash
conda env create -f env-mps.yml
conda activate cs224n-mps
```

Check that the installed PyTorch can see the MPS device, then exercise the NMT forward pass, backward pass, checkpoint save/load, and beam search:

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())"
sh run.sh mps_check
```

Train and decode with:

```bash
sh run.sh train_mps
sh run.sh test_mps
```

Training uses validation every 200 iterations, so TensorBoard and `outputs/mps/beam_search_diagnostics/` include the iteration 200 and 3000 examples as training reaches them. The MPS checkpoint and decoded test output go under `outputs/mps/`; TensorBoard scalars go under `runs/nmt_mps/`.

If `torch.backends.mps.is_available()` is false, fix the PyTorch/macOS environment before training. The MPS command exits with a clear error instead of silently moving the run to CPU.

## Current host limitation

The current host reports macOS 27.0 and has an M1 Pro with Metal, but its installed PyTorch 2.14.0 reports MPS unavailable. That PyTorch build's macOS version check only handles versions through macOS 26.0, so `train_mps` cannot run on this host until PyTorch adds macOS 27 support. The MPS smoke check should be rerun after upgrading to a compatible PyTorch build.

For this host, use the separate MLX backend, which has passed a Metal smoke run:
see [MLX.md](MLX.md), then run `sh run.sh train_mlx`. MLX writes a separate
`outputs/mlx/model.npz` checkpoint; it is not compatible with the PyTorch
`model.bin` path.
