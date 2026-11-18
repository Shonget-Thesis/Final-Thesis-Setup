# EXP-036 (EVAL-031, Path B): Dual-Space Continuous Handover (DSGH) & Closed-Loop Wasserstein-Proximal Sampling (CWPS)

- **Status:** Prepared and preregistered before synthesis (`EVAL-031`, branch `rhnl-3`). Not yet run on T4.
- **Notebook:** [`DMOSpeech2_EXP036_PathB_Novel_Mechanisms.ipynb`](../../../code/colab/DMOSpeech2_EXP036_PathB_Novel_Mechanisms.ipynb)
- **Runner:** [`run_dual_space_mechanisms.py`](../../../code/colab/run_dual_space_mechanisms.py) + [`handover_mechanisms.py`](../../../code/colab/handover_mechanisms.py)

## Purpose
Evaluates two genuinely new inference-time mechanisms (0 training, fixed `13`-pass budget, **no binary `20 dB` / `0.340` gate**) targeting both open gaps remaining in `EXP-034` (`noisy15` UTMOS/WER lag behind `STUR2` and `CPQH`'s clean aligned diversity drop):
1. **`DSGH_MICRO` & `DSGH_DUAL` (Candidate 1 — Dual-Space Geodesic Handover, `13` passes):** Combines smooth 1D Wasserstein-2 quantile soft-shrinkage (continuous amplitude preservation on clean speech, full transport on spectral floors) with excess-roughness rank-coherence damping (preserving teacher macro rhythm while suppressing high-frequency frame-rank jitter).
2. **`CWPS` (Candidate 2 — Closed-Loop Wasserstein-Proximal Teacher Sampling, `13` passes):** Synthesizes the 1-pass student draft `s0` *before* teacher stepping and injects step-scaled proximal 1D `W2` endpoint projection inside Euler CFG steps `t_1..t_3` to prevent static amplification during guidance (`1 + 10 + 2 = 13` passes).

## Cohorts & Protocol
- **Quality (`240` cases):** `unseen_cal[60:90]` (disjoint from `dev[0:100]`, `EXP-028` `[0:30]`, and Path A `EXP-035` `[30:60]`) × `{clean, noisy15, gauss25, pink25, hp25, reverb09, telephone, babble15}`.
- **Diversity (`40` blocks):** `unseen_cal[70:90]` × `12` seeds × `{clean, gauss25}`.
- **Timing (`10` cases):** `unseen_cal[60:70]` (`clean`), per-arm isolated synchronization + `PassCounter`.
- **Baselines paired on identical seeds:** `STU` (`4`), `STUR2` (`3`), `CPH8` (`12`), `CPQH` (`13`), `SNR_R` (`13`), `SNR_ONLY`, and `P25` (`30`, diversity & timing).
