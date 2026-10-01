# Mikolov et al. 2013b — Linguistic Regularities and Vector Arithmetic

## What the paper answers

Word2vec vectors encode relational structure (e.g. gender, capital–country) such that offset vectors are approximately constant, enabling analogy queries via vector addition.

## What Phase 1 uses

Module E evaluates analogy on full GloVe with L2-normalized offsets aligned to gensim `most_similar`. We report top-1/top-5 on a fixed test set and document known failures (hand/glove/foot, bark polysemy).

## Deliberately not reproduced

Word2vec training; 3CosAdd vs 3CosMul comparison; large-scale analogy benchmarks beyond our fixed JSON set.
