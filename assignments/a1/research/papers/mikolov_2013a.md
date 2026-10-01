# Mikolov et al. 2013a — Efficient Estimation of Word Representations (CBOW / Skip-gram)

## What the paper answers

How to learn dense word vectors from large corpora with log-linear models (CBOW and Skip-gram) that scale to billions of tokens, using hierarchical softmax or negative sampling.

## What Phase 1 uses

Conceptual background only. Module C placeholder will train CBOW/SGNS on the same IMDB subset; Phase 1 does not run these models.

## Deliberately not reproduced

Full Wikipedia-scale training, hierarchical softmax implementation, subsampling and negative-sampling ablations (deferred to Module C).
