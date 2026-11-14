# 2026-11-14 — EXP-028 (EVAL-021): Transmissible Slow-Band Bridge (`TBRG`) Setup & Pre-Run Amendment

## Summary

Prepared (`8d6d68afa1d2...`) and subsequently amended prior to any GPU execution (`4caf6c951669...`) **EXP-028** ([README](../../../research/experiments/EXP-028-transmissibility-bridge/README.md), task `EVAL-021`, `status: todo`) implementing the **13-NFE Energy-Matched Transmissible Slow-Band Bridge (`TBRG`)**—framed as a **theory-guided, inference-time handover algorithm** with placement novelty at $\tau_2 = 0.2929$ and **no fitted parameters**—alongside all causal, intermediate-budget, and unseen-speaker controls requested by external novelty review. No GPU synthesis has been executed yet.

## Pre-Run Upgrades (Addressing All 6 Reviewer Critiques Before T4 Run)

1. **Explicit $\tau_2$ Re-Noising Equation:** Confirmed and documented that $X_{\tau_2}^{\text{TBRG}}$ is explicitly re-noised at $\tau_2 = 0.2929$ via `gl.student(pipe, ctx, x_tbrg, [0.2929, 0.6173])`, executing $\phi_2 = (1 - \tau_2)\varepsilon_2 + \tau_2 X_{\tau_2}^{\text{TBRG}}$ before the student generator forward pass.
2. **Added Cheap & Intermediate Baselines (`BLND`, `W6R2`):**
   - `BLND` (`13 NFE`, `blend_handover`): fixed 50/50 linear blend $0.5\,S_1 + 0.5\,T_4$ on generated frames at $\tau_2$ without band decomposition or Frobenius matching.
   - `W6R2` (`16 NFE`): Step-6 teacher ($T_6$, 14 teacher NFE) + $[\tau_2, \tau_3]$ restart (`2` student NFE) on the shared teacher trajectory (`teacher_steps(..., wanted=(4, 6))`).
3. **Causal Slow-Band Pitch-Tracking Control (`SHUF`, `slow_corr`, `f0_corr_w8r2`):**
   - `SHUF` (`13 NFE`, `shuf_handover`): injects a time-resampled (`resample_time`), Frobenius-matched slow band from a strictly disjoint donor speaker's prompt mel into $v_S + f_S$.
   - Added `slow_corr_<arm>` (cosine similarity between output mel's slow band and injected $s_{T_4}$) across all diversity runs, plus **matched-seed vs. shifted-seed $f_0(t)$ contour correlation** (`paired_f0_corr`) against `W8R2` in the joint unseen cohort.
4. **Tightened Preregistered Hypotheses (`H1`–`H5`):** Added direct `W8R2` (`20 NFE`) non-inferiority bounds (`CVf0 >= -0.008`, `UTMOS >= -0.05`), tightened `H3` `mean dUTMOS >= -0.08` (from `-0.12`), and added `H5` requiring `TBRG > SHUF` on `CVf0` and `TBRG > BLND` on `UTMOS` or `CVf0`.
5. **Prospective Unseen-Speaker Replication Cohort (`unseen_cal`):** Added `run_unseen_replication` evaluating all 8 arms jointly (`W4`, `W4R2`, `TBRG`, `BLND`, `SHUF`, `W6R2`, `W8R2`, `W8`) on disjoint speakers from [calibration_manifest.json](../../../research/protocols/splits/calibration_manifest.json) (`8 x 10` seeds diversity + `12` quality) never touched in `EXP-012..027`.
6. **Framing & `STAC-TBRG` Demotion:** Adopted the placement-novelty / no-fitted-parameters framing throughout, and kept 5-fold `STAC-TBRG` routing (`stac_routing_secondary`) strictly as an optional secondary exploratory diagnostic.

## Checks Run

- `python code/colab/run_transmissibility_bridge.py freeze --force` — froze amended `preregistration.json` (`sha256=4caf6c951669...`).
- `python scripts/render_progress.py` — updated `docs/progress/TASKS.md`.
- `python scripts/check_workspace.py` — `Workspace check PASSED`.
- `python -m unittest discover -s code/tests -p "test_transmissibility_bridge.py" -v` — 11 tests OK (`9.485s`).
- `python -m unittest discover -s code/tests` — 276 tests OK (`129.205s`).
- LaTeX build: not run (`paper/` unchanged; TeX build unverified for this setup).

## Remaining Blockers

- **Colab Tesla T4 execution:** Needs execution of `smoke` and `run --stage all` on Google Colab T4 using [DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb](../../../code/colab/DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb) (`~60–70 min` estimated T4 runtime; `EVAL-021` stays `todo` until GPU results are imported).
