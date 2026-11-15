# 2026-11-15: EXP-029 Damped Slow-Band Transmissibility & Mixed-Condition `STAC` Setup

**Task ID:** `EVAL-022`  
**Branch:** `rhnl-3`

## Why This Follows Directly from `EXP-028`

Within the thesis progression **`STAC` (`EXP-018`) -> Transmissibility Theory (`EXP-026`/`EXP-027`) -> Upgraded `STAC` (`EXP-028`)**:
- `EXP-028` showed `TBRG` (`100%` slow-band swap at Step 4, `13` NFE) lost `-0.193` / `-0.160` UTMOS vs `W8`, whereas `BLND` (`0.5 S1 + 0.5 T4`, `13` NFE) kept UTMOS (`-0.069` / `-0.038`) and `W6R2` (`16` NFE) kept both `CVf0` (`0.1029` / `0.0840`) and `UTMOS` (`3.887` / `3.714`).
- Secondary `STAC-TBRG` (`3/40`, `19.48` NFE) only evaluated clean prompts with teacher-only `A_4` (near-zero correlation with clean `dUTMOS(TBRG)`), leaving the noisy/degraded stability of $\tau_2$ restarts and damped slow-band bridges untested.

## What Was Created & Verified Locally

1. **Runner:** `code/colab/run_damped_transmissibility_stac.py`
   - Reuses `sh.decompose`, `tt.grid16`, `gl.student`, `tc.boot_ci` / `sign_test` / `robust_stats`, `tb.student_step_tau1` / `_speaker_folds` / `paired_f0_corr`, and `rf.Pipeline`.
   - Implements energy-preserving `damped_bridge_handover(T4, S1, plen, alpha)` ($\alpha \in \{0.25, 0.50, 0.75, 1.00\}$) with exact boundary identities ($\alpha=0 \equiv S_1$, $\alpha=1 \equiv \text{TBRG}$), plus `W5R2` (`14` NFE) and `W6R2` (`16` NFE).
   - Evaluates `90` mixed-condition quality cases (`30` dev speakers `61-90` x `{clean, gauss18, noisy15}`) and `20 x 12` clean pitch-diversity trajectories while keeping all `170` remaining calibration speakers and all `200` test speakers untouched.
2. **Notebook:** `code/colab/DMOSpeech2_EXP029_Damped_Transmissibility_STAC.ipynb` (defaults to `STAGE='quality'` for a ~25–30 min T4 diagnostic before running `diversity`).
3. **Preregistration:** Frozen at [EXP-029/preregistration.json](../../../research/experiments/EXP-029-damped-transmissibility-stac/preregistration.json) (`sha256: e58f3856ad20...`).
4. **Unit Tests:** Added `code/tests/test_damped_transmissibility_stac.py` covering $\alpha \in \{0, 1\}$ identities, constant Frobenius norm across intermediate $\alpha$, post-`S1` discrepancy signals, cohort split disjointness, synthetic clean+noisy 5-fold `STAC` routing, and preregistration/notebook integrity.
