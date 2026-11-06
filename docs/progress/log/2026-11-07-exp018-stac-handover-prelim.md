# 2026-11-07: EXP-018 Stage-Wise Trajectory Action Cascade (`STAC`) & `1D-CRC-STAC` on 16-Grid and 32-Grid

Prepared and benchmarked `EXP-018` (`Stage-Wise Trajectory Action Cascade`, `STAC`, and `1-Parameter Conformal Risk-Controlled STAC`, `1D-CRC-STAC`) across the 16-interval sway grid (`Steps 1..8`) and the 32-interval sway grid (`Steps 1..15`) without any binary failure rule, with full computational cost, wall-clock time, speedup (`x`), RTF, `x-Real-Time` speed, gate overhead (`ms`), and GPU memory (`MB`) telemetry.

## Method & Controls

- **`STAC` (`Stage-Wise Trajectory Action Cascade`):** Uses the cumulative trajectory action $A_k = \sum_{j=1}^{k} d_j \ge \hat{\lambda}_k$ without any external SNR estimator (`0.12 ms` gate overhead vs. `1.85 ms` for SNR-only and `1.97 ms` for hybrid SNR+drift).
- **`1D-CRC-STAC`:** Indexes stage-wise quantile thresholds $\lambda_k(\tau) = Q_{1-\tau}(\{A_k(x_i)\}_{i \in \mathcal{D}_{\text{cal}}})$ by a single scalar $\tau \in [0, 1]$ with monotone envelope loss $L_{\text{mono}}(x; \tau) = \max_{\tau' \le \tau} L(x, \pi_{\tau'}(x)) \in [0, 1]$.
- **Matched Controls & Ablations:** Stage-Wise SNR Cascade ($\hat{S} \ge \hat{\theta}_k$, matched degrees of freedom), Stage-Wise Single-Step Drift ($d_k \ge \hat{\lambda}_k$), Single-Threshold Trajectory Action ($A_k \ge \hat{\lambda}$), Zero-Shot OOD generalization on 5 unseen corruptions from `EXP-011` (`babble20`, `babble15`, `reverb03`, `reverb09`, `telephone`; $n=100$), and Hindsight Per-Utterance Oracle across `Strict`, `Primary`, and `Relaxed` tolerances.
- **Full `d_1..d_8` 500-Speaker Benchmark:** Joins `research/evidence/benchmarks/drift_features/drift_features_all1000.csv` with `research/evidence/benchmarks/gpu_grid_all1000.csv` and reports Raw WER, Capped WER ($\le 1.0$), and Outlier-Trimmed WER with Speaker-Cluster Bootstrap 95% CIs.

## Checks Run

- `python scripts/check_workspace.py` — PASSED.
- `python -m unittest discover -s code/tests` — 69 tests PASSED (`test_swsr_handover.py` included).
