# EXP-004: Signal-selection validation

**Run:** 2026-10-23, Colab Tesla T4. Notebook `code/colab/DMOSpeech2_Signal_Selection_Validation.ipynb`, runner `code/colab/run_signal_selection_validation.py` at Shonget-DMOSpeech2 `38ac387`. Preregistration written at 10:48 UTC, before the analysis (summary at 11:28 UTC).

**What:** three questions under one frozen protocol, with failure labels from EXP-002 and the teacher re-run to compute new signals: (Q1) which prompt signal should gate early handover; (Q2) which zero-cost trajectory signal should decide when to stop; (Q3) which calibration framework. Development → AUC and orientation; calibration → LTT (α = 0.05, δ = 0.10); test → risk and NFE.

**Data:** [research/evidence/benchmarks/signal_selection/](../../evidence/benchmarks/signal_selection/SIGNAL_SELECTION_REPORT.md): report, preregistration, summary JSON, `signal_features_all1000.csv`, RUN_INFO. The re-run reproduced the stored d₄–d₆ (max difference 0.00003).

## Key results

- **Q1 (gate):** the energy-percentile prompt SNR is as good as a perfect clean/noisy label (oracle) and is matched by spectral flatness; SNR → Step 4: 4/200 test failures, 17.06 NFE.
- **Q2 (stopping signal):** with the SNR gate fixed, drift is matched by simply handing over at a fixed Step 4 (4/200, 17.04 NFE); drift's development AUC for gated Step-4 failures is 0.526. No trajectory signal improves on the fixed step.
- **Q3 (calibration):** Hoeffding certifies nothing at n = 200; risk is not monotone in λ, which rules out CRC and RCPS; LTT is justified. On the T4 labels, LTT's modal λ in 200 re-splits is 1.20.

## Caveats

- The test split was inspected in earlier analyses (stated in the report).
- The report's Q3 line "LTT reproduces λ̂ = 0.50" holds for the CPU labels only.
