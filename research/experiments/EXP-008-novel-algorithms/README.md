# EXP-008: Novel-algorithm comparison

**Run:** 2026-10-23 to 2026-10-24, Colab Tesla T4, fast mode (`RUN_C2 = False`). Notebook `code/colab/DMOSpeech2_Novel_Algorithm_Comparison.ipynb`, runner `code/colab/run_novel_algorithm_comparison.py` at Shonget-DMOSpeech2 `381ec71`. Protocol frozen in `novel_algorithms_preregistration.json` before the analysis.

**What:** candidate algorithms against the proposal, each calibrated with the same LTT procedure (α = 0.05, δ = 0.10) on the calibration split and evaluated once on the test split: C1 student-probe handover (gated and all prompts), C3 risk-budgeted SNR-band schedule (256 schedules), baselines B0–B2, and the proposal P0 (SNR + drift, λ = 0.50) and P1 (λ re-certified). C2 (joint handover and restart level) was not run.

**Data:** [research/evidence/benchmarks/novel_algorithms/](../../evidence/benchmarks/novel_algorithms/NOVEL_ALGORITHMS_REPORT.md): report, preregistration, summary JSON, `novel_features_all1000.csv`, RUN_INFO.

## Key results (test split, 200 speakers)

| Policy | Test failures | Mean NFE |
|---|---:|---:|
| B0 fixed Step 8 | 0/200 | 21.00 |
| B1 SNR → Step 4 (SNR-only) | 4/200 | 17.04 |
| P0 SNR + drift, λ = 0.50 | 3/200 | 18.15 |
| P1 SNR + drift, λ re-certified (1.20) | 4/200 | 17.04 |
| C1a student probe (gated) | 4/200 | 17.07 |
| C1b student probe (all prompts) | 0/200 | 23.97 |
| C3 SNR-band schedule | 4/200 | 17.04 |

No candidate is cheaper than SNR-only at the same risk. The student probe is a better development predictor than drift (AUC 0.711 vs 0.474) but does not save compute.

## Caveats

- The test split was inspected in earlier analyses (stated in the report).
- C2 was not run.
