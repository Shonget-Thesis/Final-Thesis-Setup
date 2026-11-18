# EXP-035 + EXP-036 All-in-One (EVAL-030): Path A Ladder & Unseen Confirmation + Path B Dual-Space & Closed-Loop Mechanisms

- **Status:** Prepared and preregistered before synthesis (`EVAL-030`, branch `rhnl-3`). Not yet run on T4.
- **Single Colab Notebook (All-in-One):** [`DMOSpeech2_EXP035_EXP036_AllInOne.ipynb`](../../../code/colab/DMOSpeech2_EXP035_EXP036_AllInOne.ipynb)
- **Runner & Shared Operators:** [`run_exp035_036_combined.py`](../../../code/colab/run_exp035_036_combined.py) + [`handover_mechanisms.py`](../../../code/colab/handover_mechanisms.py)

## Purpose
Combines both **Path A** (`EXP-035`) and **Path B** (`EXP-036`) into one shared Colab Tesla T4 sweep on **`unseen_cal[30:60]`** (strictly disjoint from all `100` development speakers `EXP-001..034` and `EXP-028` `unseen_cal[0:30]`), avoiding duplicate baseline synthesis and yielding paired speaker-level comparisons across every arm on identical seeds:
1. **Path A — Condition-Invariance Ladder (`Base -> FCH -> G1 -> G2 -> G3`) & Unseen `SNR_R` Confirmation:**
   - Directly compares `CPH8` (`12`), `CPH8_DP` (`13`), `FCH_DP` (`13`), `G1_VSWAP` (`13`), `G2_AFFINE` (`13`), and `CPQH` (`13`) with clean draft prompt frames fixed, plus `SNR_R` (`theta=20 dB`, `lambda_r=0.340048`) vs. `SNR_ONLY`, `SNR_DRIFT`, `R_ONLY`, `W8`, and 5-fold speaker-grouped OOF calibration.
2. **Path B — Novel Inference-Time Mechanisms (Fixed `13` Passes, Zero Heuristic Gate):**
   - `DSGH_MICRO` (`macro_weight = 0.0`) & `DSGH_DUAL` (`macro_weight = 0.5`): Continuous 1D Wasserstein-2 quantile soft-shrinkage + excess-roughness rank-coherence damping toward `s0`.
   - `CWPS`: Closed-Loop Wasserstein-Proximal Teacher Sampling running `s0` first (`1` pass) and projecting Euler CFG endpoints inside steps `t_1..t_3` (`10` passes) before dual-space readout and `j=2` student finish (`2` passes).

## Outputs
Generates `results/PATH_A_INVARIANCE_LADDER_REPORT.md`, `results/PATH_B_DUAL_SPACE_REPORT.md`, `results/EXP035_036_COMBINED_REPORT.md`, `results/combined_results.json`, and `results/MANIFEST.json` into `MyDrive/exp035_036_all_in_one/` and `DMOSpeech2_EXP035_036_AllInOne_Results.zip`.
