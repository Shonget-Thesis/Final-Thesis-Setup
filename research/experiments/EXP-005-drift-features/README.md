# EXP-005: Richer zero-cost drift features

**Run:** 2026-10-23, Colab Tesla T4, exploration mode (`--explore_only`). Runner `code/colab/run_drift_features_benchmark.py`, a reference copy of Shonget-DMOSpeech2 branch `SNRgated` commit `e15cd55` (not on that repository's main branch).

**What:** a teacher-only pass for all 1,000 prompts (no student or scoring; failure labels come from EXP-002), extracting zero-cost features from teacher Steps 1–5: relative-L2 drift, per-frame drift (mean, 90th percentile, max), drift in low/mid/high mel bands, cosine change, CFG gap and velocity norm. The plan: explore on the 100 development speakers, freeze one score, and confirm on the 400 calibration and test speakers, aiming to rescue noisy prompts that the SNR gate sends to Step 8.

**Data:** [research/evidence/benchmarks/drift_features/drift_features_all1000.csv](../../evidence/benchmarks/drift_features/drift_features_all1000.csv) (1,000 rows). Its d₄ equals EXP-002's in all 1,000 cases.

## Key results (development exploration, as reported in the compiled reports, Part 1 §4.10)

- 67 features tested on the development speakers' low-SNR prompts. Cross-validated AUC 0.39–0.57 per feature family; 0.61 for all features with L1 logistic regression.
- Best single feature `fr_max_r54` (per-frame max drift ratio, Step 5 / Step 4): development AUC 0.69, against a chance maximum of 0.63 (p ≈ 0.02). Weak.

## Caveats

- Exploration only: the frozen-score confirmation on calibration and test was not run.
- The figures above are from the team's report, not recomputed here; recomputing them needs the script's analysis step on this CSV.
