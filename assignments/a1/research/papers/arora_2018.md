# Arora et al. 2018 — Linear Structure in Contextual Word Embeddings (polysemy)

## What the paper answers

Polysemous words can be modeled as linear combinations of sense vectors; context clustering and sparse coding can recover multiple senses from static or contextual embeddings.

## What Phase 1 uses

Module D basic: GloVe top-10/top-30 sense annotation for bank, plant, tie, bark plus IMDB context snippets. Discusses whether missing senses in top-k imply the vector lacks them.

## Deliberately not reproduced

Context-vector clustering, sparse coding, or full sense recovery pipeline (Module D advanced placeholder).
