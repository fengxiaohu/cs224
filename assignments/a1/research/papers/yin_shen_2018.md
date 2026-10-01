# Yin & Shen 2018 — On the Dimensionality of Word Embedding (PIP / distortion)

## What the paper answers

Low-dimensional projections distort pairwise inner products (PIP loss); 2D visualizations can mislead about neighborhood structure even when reconstruction error is low.

## What Phase 1 uses

Module B: reconstruction error ||M||²_F − ||E_k||²_F, subset PIP loss vs k=100 reference, 2D neighbor Jaccard vs full k=100 space, rotation invariance check on normalized vectors.

## Deliberately not reproduced

Full PIP theory bounds, optimal dimension selection across many corpora, or comparison to non-linear dimensionality reduction methods.
