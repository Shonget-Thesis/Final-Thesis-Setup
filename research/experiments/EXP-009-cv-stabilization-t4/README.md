# EXP-009: Cross-validated threshold for the stabilization policy (Appendix D.4, T4)

**Run:** 2026-10-25, offline (no GPU) on the T4 grid of EXP-002. Script `code/analysis/cv_stabilization_t4.py` (standard library); output [results.json](results.json).

**What:** the T4 re-run of the team's earlier CPU experiment. Four-fold speaker-grouped cross-validation on the 100 development speakers (clean and noisy15, 200 instances). In each fold, λ is the largest value in {0.20, …, 0.60} with zero added failures on the 75 training speakers, frozen, then scored on the 25 held-out speakers. Policy of Definition 3.4, candidates Steps 4/5/6, SNR gate 20 dB. Folds use a seeded shuffle (seed 2026).

## Key results

- λ selected per fold: 0.35, 0.35, 0.35, 0.50. None is at the edge of the search; widening the grid to 1.20 gives the same choices.
- Out of fold: 1 added failure in 200 instances; mean 18.88 NFE (−10.1% against 21). Fixed s = 0.15 (Step 6): 50/200 failures. All 100 noisy prompts go to the reference.
- With 75 training speakers and zero failures, the smallest risk certifiable at δ = 0.10 is about 3%.

## Caveats

- The development split also served other design decisions, so out-of-fold figures may be optimistic.
- The earlier CPU version (12 speakers, 48 instances, lighter evaluators) is superseded.
