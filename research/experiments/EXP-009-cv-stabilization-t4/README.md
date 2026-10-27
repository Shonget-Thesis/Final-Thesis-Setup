# EXP-009: Cross-validated threshold for the stabilization policy (Appendix D.4, T4)

**Run:** 2026-10-25, offline (no GPU) on the T4 grid of EXP-002. Script `code/analysis/cv_stabilization_t4.py` (standard library); output [results.json](results.json).

**What:** the T4 re-run of the team's earlier CPU experiment. Four-fold speaker-grouped cross-validation on the 100 development speakers (clean and noisy15, 200 instances). In each fold, λ is the largest value in {0.20, …, 0.60} with zero added failures on the 75 training speakers, frozen, then scored on the 25 held-out speakers. Policy of Definition 3.4, candidates Steps 4/5/6, SNR gate 20 dB. Folds use a seeded shuffle (seed 2026).

## Key results

- λ selected per fold: 0.35, 0.35, 0.35, 0.50. None is at the edge of the search; widening the grid to 1.20 gives the same choices.
- Out of fold: 1 added failure in 200 instances; mean 18.88 NFE (−10.1% against 21). Fixed s = 0.15 (Step 6): 50/200 failures. All 100 noisy prompts go to the reference.
- With 75 training speakers and zero failures, the smallest risk certifiable at δ = 0.10 is about 3%.

## Policy comparison (added 2026-10-27, `policy_comparison` in results.json)

Same 200 instances, all compared methods of Chapter 3 (Table 3.x `tab:methods`). Used for Appendix D.4, Tables `tab:cv-policies` and `tab:cv-quality`. Also in results.json: `cost_matched_random` and `tests` (exact McNemar against the full policy; Wilcoxon signed-rank on speaker-level differences from the reference, Holm across methods, for NFE, WER, SIM and UTMOS).

| Method | Fail clean | Fail noisy | CP95 upper | Mean NFE | McNemar vs full |
|---|---:|---:|---:|---:|---:|
| Fixed Step 4 / 5 / 6 | 3 / 2 / 0 | 63 / 49 / 50 | 40.0 / 32.1 / 31.6% | 13 / 15 / 17 | < 1e-13 |
| Reference Step 8 | 0 | 0 | 1.8% | 21.00 | – |
| SNR-only (Step 4) | 3 | 0 | 4.3% | 17.00 | 0.50 |
| SNR + fixed Step 5 / 6 | 2 / 0 | 0 / 0 | 3.6 / 1.8% | 18.00 / 19.00 | 1.00 / 1.00 |
| Stabilization only (no gate, out of fold) | 0 | 0 | 1.8% | 21.00 (no threshold in any fold) | 1.00 |
| Full policy (out of fold) | 1 | 0 | 2.8% | 18.88 | – |
| Cost-matched random (2,000 permutations) | 0.75 mean | 0 | – | 18.88 | – (85% of draws ≤ 1 failure) |
| Oracle | 0 | 0 | 1.8% | 14.70 | – |

Wilcoxon vs reference (Holm): NFE p < 1e-15 for every saving method; WER, SIM, UTMOS p_holm ≥ 0.10 for all (full policy 1.00). No TOST: Chapter 3 fixes equivalence margins before the held-out test. On clean prompts the full policy chose Step 4/5/6/8 for 6/28/52/14 instances (16.76 NFE, −20.2%). With ≤ 3 failures per method, the SNR-gated variants and the random control are not separable on these data.

## Caveats

- The development split also served other design decisions, so out-of-fold figures may be optimistic.
- The earlier CPU version (12 speakers, 48 instances, lighter evaluators) is superseded.
