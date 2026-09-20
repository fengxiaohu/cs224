# A2 Auto Research Protocol

Human-maintained experiment protocol for CS224N Assignment 2. The controller executes only the runs listed here; it does not modify training code or expand the search space.

## Goal

Run a bounded learning-rate comparison while keeping the submission baseline fixed at the archived full run (**best dev 88.60%**, **test 89.03%**). New experiments are for learning how to tune hyperparameters, not for replacing the submission result automatically.

## Frozen scope

- Do not modify `parser_model.py`, transition system, data preprocessing, or evaluation logic.
- Keep the existing network structure; do not introduce `nn.Linear` or `nn.Embedding`.
- Do not auto-edit this protocol, widen the search, commit to Git, or restore working-tree files.

## Fixed hyperparameters (batch 1)

| Parameter | Value |
|---|---:|
| hidden_size | 200 |
| dropout | 0.5 |
| batch_size | 1024 |
| n_epochs | 10 |

Only learning rate varies in this batch.

## Candidate runs

```yaml
protocol:
  submission_baseline:
    note: "Archived pre-seed run; seed unknown"
    best_dev_uas_percent: 88.60
    test_uas_percent: 89.03
    final_train_loss: 0.06677
    best_epoch: 8
  budgets:
    max_debug_runs: 1
    max_full_runs: 5
    debug_timeout_seconds: 300
    full_timeout_seconds: 1800
    full_cumulative_timeout_seconds: 9000
  qualification:
    full_final_train_loss_lt: 0.08
    full_best_dev_uas_percent_gt: 87.0
    debug_final_train_loss_lt: 0.2
    debug_best_dev_uas_percent_gt: 65.0
  selection:
    primary_metric: best_dev_uas
    tie_breaker: keep_baseline_then_earlier_candidate
  phases:
    - id: A
      name: archive_and_debug
      runs:
        - role: debug_smoke
          mode: debug
          lr: 0.0005
          seed: 42
          skip_test: true
    - id: B
      name: lr_search_seed_42
      runs:
        - role: baseline
          mode: full
          lr: 0.0005
          seed: 42
          skip_test: true
        - role: candidate_low_lr
          mode: full
          lr: 0.00025
          seed: 42
          skip_test: true
        - role: candidate_high_lr
          mode: full
          lr: 0.001
          seed: 42
          skip_test: true
    - id: C
      name: seed_43_confirmation
      only_if: candidate_beats_baseline_on_seed_42
      runs:
        - role: baseline
          mode: full
          lr: 0.0005
          seed: 43
          skip_test: true
        - role: winner
          mode: full
          lr: winner_lr_from_phase_B
          seed: 43
          skip_test: true
```

## Selection rules

1. Primary metric: unrounded **best full dev UAS**.
2. A full run must finish all 10 epochs with finite loss and meet:
   - final train loss < 0.08
   - best dev UAS > 87%
3. Among qualified runs, rank by best dev UAS.
4. Ties keep the baseline; if two candidates tie, prefer the earlier candidate in Phase B order.
5. If the Phase B baseline misses the reference targets, pause the batch.
6. If no candidate beats the baseline, stop after Phase B.
7. Phase C runs only when a candidate beats the seed-42 baseline. Mark a config as batch-preferred only if the candidate beats the corresponding baseline on **both** seed 42 and seed 43 and both runs meet the reference targets.
8. Two-seed confirmation is a limited check, not a claim of statistical significance.

## Failure and timeout rules

- Timeout kills only the controller-launched process for that run; record incomplete and pause the batch.
- Non-zero exit, non-finite loss, or missing checkpoint: keep logs and pause; no automatic retry.
- A completed but weaker candidate is recorded as not selected and the next scheduled candidate may continue.

## Artifacts

Each run writes an independent directory under `results/` with config, metrics, status, checkpoint, parser bundle, source snapshot, and log. Summary table columns:

`run_id / code_hash / seed / mode / lr / epochs / batch / elapsed / final_train_loss / best_dev_UAS / best_epoch / test_UAS / checkpoint / status`

Test UAS stays empty unless a human triggers `--eval-test RUN_DIR`.
