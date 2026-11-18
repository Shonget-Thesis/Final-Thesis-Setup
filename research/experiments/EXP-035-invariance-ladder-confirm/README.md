# EXP-035 (EVAL-030, Path A): Condition-Invariance Ladder & Unseen Confirmation of SNR_R

- **Status:** Prepared and preregistered before synthesis (`EVAL-030`, branch `rhnl-3`). Not yet run on T4.
- **Notebook:** [`DMOSpeech2_EXP035_PathA_Ladder_Unseen.ipynb`](../../../code/colab/DMOSpeech2_EXP035_PathA_Ladder_Unseen.ipynb)
- **Runner:** [`run_invariance_ladder_confirm.py`](../../../code/colab/run_invariance_ladder_confirm.py) + [`handover_mechanisms.py`](../../../code/colab/handover_mechanisms.py)

## Purpose
Addresses two methodological confirmation questions for `SNR_R` (`EXP-034`):
1. **Invariance-ladder necessity (`Base -> FCH -> G1 -> G2 -> G3`):** Tests whether `CPQH` (`G3` monotone rank-quantile projection) is strictly needed over scalar power-floor clamping (`FCH_DP`), shift invariance (`G1_VSWAP`), positive-affine invariance (`G2_AFFINE`), and draft prompt-frame swapping (`CPH8_DP`) at the identical `(t = 0.2929, j = 2)` commit point.
2. **Disjoint calibration confirmation & 5-fold CV/LTT check:** Replicates `SNR_R` vs. `SNR_ONLY` and `SNR_DRIFT` on `30` unseen calibration speakers (`unseen_cal[30:60]`), strictly disjoint from all `100` development speakers (`EXP-001..034`) and `EXP-028` (`unseen_cal[0:30]`). *(Note: `EXP-002` previously ran baseline clean/`noisy15` fixed 16-grid steps across the 500-speaker splits; none of `EXP-011..034` or these 8 conditions/arms ran on `unseen_cal[30:60]`.)*

## Cohorts & Protocol
- **Quality (`240` cases):** `unseen_cal[30:60]` × `{clean, noisy15, gauss25, pink25, hp25, reverb09, telephone, babble15}`.
- **Diversity (`40` blocks):** `unseen_cal[40:60]` × `12` seeds × `{clean, gauss25}`.
- **Timing (`10` cases):** `unseen_cal[30:40]` (`clean`), per-arm isolated synchronization + `PassCounter`.
- **Frozen thresholds:** Same `EXP-034` CPU-screen constants (`theta = 20.0 dB`, `lambda_r = 0.340048`, `lambda_d = 0.225175`), plus out-of-fold 5-fold speaker calibration analysis.
