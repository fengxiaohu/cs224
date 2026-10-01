# Pennington, Socher, Manning 2014 — GloVe

## What the paper answers

How to combine global co-occurrence statistics with local context-window methods by factorizing a log co-occurrence matrix weighted toward frequent pairs.

## What Phase 1 uses

Pre-trained `glove-wiki-gigaword-200` for polysemy (Module D) and analogy (Module E). Distribution mismatch vs IMDB subset is noted in config.

## Deliberately not reproduced

Training GloVe on IMDB (Module C / Phase 3 placeholder). Weighted least-squares objective and adaptive learning rate.
