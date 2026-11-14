# 2026-11-14 — EXP-028 (EVAL-021): Transmissible Slow-Band Bridge (`TBRG`) Setup & Pre-Run Amendment

## Summary

Prepared (`8d6d68afa1d27710d5677f43ac42927fb20c2a9205d5ab7f22fd5dfa7173f6cf`) and subsequently amended twice prior to any GPU execution (`4caf6c951669cdc54ad77897a9aaa77fbeb721b366b81dbdc4df494d8a39d713` $\rightarrow$ `5c6c66ab8feed937e9c47387a4d57cfe5a5bf484d7a21975519a2fe0084fe1fd`) **EXP-028** ([README](../../../research/experiments/EXP-028-transmissibility-bridge/README.md), task `EVAL-021`, `status: todo`) implementing the **13-NFE Energy-Matched Transmissible Slow-Band Bridge (`TBRG`)**—framed as a **theory-motivated, inference-time handover algorithm** (motivated post-EXP-027 by the $\tau_1$ vs $\tau_2$ transmissibility split, not predicted a priori) with placement novelty at $\tau_2 = 0.2929$ and **no fitted parameters**—alongside all causal, intermediate-budget, and unseen-speaker controls requested across two rounds of external review. No GPU synthesis has been executed yet.

## Pre-Run Upgrades (Addressing Two Rounds of Reviewer Critiques Before T4 Run)

1. **Explicit $\tau_2$ Re-Noising Equation:** Confirmed and documented that $X_{\tau_2}^{\text{TBRG}}$ is explicitly re-noised at $\tau_2 = 0.2929$ via `gl.student(pipe, ctx, x_tbrg, [0.2929, 0.6173])`, executing $\phi_2 = (1 - \tau_2)\varepsilon_2 + \tau_2 X_{\tau_2}^{\text{TBRG}}$ before the student generator forward pass.
2. **Added Cheap & Intermediate Baselines (`BLND`, `W6R2`):**
   - `BLND` (`13 NFE`, `blend_handover`): fixed 50/50 linear blend $0.5\,S_1 + 0.5\,T_4$ on generated frames at $\tau_2$ without band decomposition or Frobenius matching.
   - `W6R2` (`16 NFE`): Step-6 teacher ($T_6$, 14 teacher NFE) + $[\tau_2, \tau_3]$ restart (`2` student NFE) on the shared teacher trajectory (`teacher_steps(..., wanted=(4, 6))`).
3. **Causal Slow-Band Pitch-Tracking Control (`SHUF`, `slow_corr`, `f0_corr_w8r2`):**
   - `SHUF` (`13 NFE`, `shuf_handover`): injects a time-resampled (`resample_time`), Frobenius-matched slow band from a strictly disjoint donor speaker's prompt mel into $v_S + f_S$.
   - Added `slow_corr_<arm>` (cosine similarity between output mel's slow band and injected $s_{T_4}$) across all diversity runs, plus **matched-seed vs. shifted-seed $f_0(t)$ contour correlation** (`paired_f0_corr`) against `W8R2` in the joint unseen cohort.
4. **Co-Primary Multiplicity Rule & Tightened Hypotheses (`H1`–`H5`):** Designated **`H1` and `H2` as co-primary gatekeeper hypotheses** (intersection-union test at $\alpha = 0.05$; both must clear `W4`/`W4R2` and `W8R2` non-inferiority `CVf0 >= -0.008`, `UTMOS >= -0.05`) and **`H5` (`SHUF`/`BLND`), `H3` (`W8` quality with `mean dUTMOS >= -0.08`), and `H4` (`mel_spread`)** as hierarchical secondary/mechanistic hypotheses, with explicit pre-committed interpretation rules in `preregistration.json`.
5. **Expanded Primary Prospective Cohort (`unseen_cal`) vs. Development Cohort (`61–100`):** Expanded `run_unseen_replication` to **`20 x 15` seeds (`300` trajectories/arm) diversity + `30` quality** across all 8 arms jointly (`W4`, `W4R2`, `TBRG`, `BLND`, `SHUF`, `W6R2`, `W8R2`, `W8`) on disjoint speakers from [calibration_manifest.json](../../../research/protocols/splits/calibration_manifest.json) (`0` speaker overlap with `EXP-012..027`), designated `unseen_cal` as the **primary prospective cohort**, and labeled `speakers 61–100` (`20 x 20` + `40`) explicitly as development evidence.
6. **Theory-Motivated Framing, Amendment Provenance & `STAC-TBRG` Demotion:** Replaced `"theory-guided"` with `"theory-motivated"` throughout, recorded the complete 3-hash pre-synthesis freeze chain inside `preregistration.json`, and kept 5-fold `STAC-TBRG` routing (`stac_routing_secondary`) strictly as an optional secondary exploratory diagnostic.

## Checks Run

- `python code/colab/run_transmissibility_bridge.py freeze --force` — froze amended `preregistration.json` (`sha256=5c6c66ab8feed937e9c47387a4d57cfe5a5bf484d7a21975519a2fe0084fe1fd`).
- `python scripts/render_progress.py` — updated `docs/progress/TASKS.md`.
- `python scripts/check_workspace.py` — `Workspace check PASSED`.
- `python -m unittest discover -s code/tests -p "test_transmissibility_bridge.py" -v` — all tests OK.
- `python -m unittest discover -s code/tests` — all unit tests OK.
- LaTeX build: not run (`paper/` unchanged; TeX build unverified for this setup).

## Remaining Blockers

- **Colab Tesla T4 execution:** Needs execution of `smoke` and `run --stage all` (or `STAGE = 'unseen'` first) on Google Colab T4 using [DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb](../../../code/colab/DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb) (`~75–90 min` estimated T4 runtime for `all`; `EVAL-021` stays `todo` until GPU results are imported).
