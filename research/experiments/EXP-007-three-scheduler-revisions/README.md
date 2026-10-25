# EXP-007: Three scheduler revisions on the expanded step grid

**Run:** 2026-10-21, offline analysis of EXP-003's GPU data. Runner `code/colab/run_three_scheduler_revisions_dev.py` (reference copy of Shonget-DMOSpeech2 branch `early-handover-dev`, commit `5710d3f`). The run's metadata shows it ran without a GPU, which it does not need: it replays decisions on the stored grid.

**What:** can a stopping rule over the expanded candidate set {2, 3, 4, 5, 6, 8} beat simple SNR-gated fixed steps? Candidate A (drift trend) and three revisions (trend + drift, trend + conditioning disagreement, step-specific trend) were tuned by 5-fold speaker-grouped cross-validation on the 100 development speakers, with a causality check that blocks any use of future steps.

**Data:** [research/evidence/benchmarks/three_scheduler_revisions/](../../evidence/benchmarks/three_scheduler_revisions/THREE_SCHEDULER_REVISIONS_DEVELOPMENT_REPORT.md): 16 files identical to the run's archive (`three_scheduler_revisions_dev_artifacts.tar.gz`). The input grid hash in `experiment_metadata.json` is EXP-003's `development_expanded_grid.csv` with Windows line endings.

## Key results (development, out of fold)

| Policy | Mean NFE | Speaker failures (100) |
|---|---:|---:|
| Fixed Step 8 | 21.00 | 0 |
| SNR gate → Step 4 | 17.00 | 3 |
| SNR gate → Step 3 | 16.00 | 3 |
| SNR gate → Step 2 | 15.00 | 7 |
| Original SNR + drift (λ = 0.50, Steps 4–6) | 18.05 | 1 |
| Candidate A and Revisions 1–3 | 16.25–16.26 | 2 |
| Oracle, SNR-gated, Steps 2–8 (hindsight) | 15.07 | 0 |

- No revision beats SNR-gated Step 3 significantly (McNemar p = 1.00). All three revisions reproduce Candidate A's decisions.

## Caveats

- Development split only; no calibration or test evaluation.
- Ran offline, so no runtimes.
