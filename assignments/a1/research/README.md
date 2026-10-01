# A1 Embedding Study

Research extension for CS224N Assignment 1. Phase 1 covers sparse co-occurrence / PPMI / SVD (modules A–B), basic polysemy and analogy on GloVe (modules D–E), and baseline archival.

## Layout

- `config.py` — single configuration entry point
- `src/` — library code (corpus, sparse counts, PPMI, SVD, neighbors, analogy)
- `notebooks/phase1.ipynb` — readable Phase 1 report
- `papers/` — short paper notes
- `results/baseline/` — exported assignment notebook from `f97ea793`
- `results/phase1/<module>_<utc>/` — per-run artifacts

## Running

Use the `cs224n` conda environment:

```bash
cd assignments/a1/research
python -m src.run_module_a
python -m src.run_module_b
python -m src.run_module_d
python -m src.run_module_e
python -m src.run_baseline
```

Each module writes its own timestamped directory under `results/phase1/`.
