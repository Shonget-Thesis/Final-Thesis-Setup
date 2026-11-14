# 2026-11-14 — EXP-028 (EVAL-021): Transmissible Slow-Band Bridge (`TBRG`) & `STAC-TBRG` Setup

## Summary

Prepared and froze **EXP-028** ([README](../../../research/experiments/EXP-028-transmissibility-bridge/README.md), task `EVAL-021`, `status: todo`) implementing the unified **13-NFE Energy-Matched Transmissible Slow-Band Bridge (`TBRG`)**, unnormalized ablation (`STS2`), and 5-fold cross-validated **`STAC-TBRG`** adaptive routing (`13–20 NFE`). The experiment builds directly on [EXP-027](2026-11-14-exp027-transmissibility-confirmation-results.md) on the exact same 40 fresh speakers (`61–100`) and locked T4 reference CSVs (`~25–30 min` estimated T4 runtime). No GPU synthesis has been executed yet.

## Changes Made

- **Runner:** Added [run_transmissibility_bridge.py](../../../code/colab/run_transmissibility_bridge.py) implementing:
  - `teacher_step4` (`i = 0..4`, 10 NFE) logging $T_4$, step drifts $d_1 \dots d_4$, cumulative STAC action $A_4$, and contraction $\rho_4$.
  - `student_step_tau1` (1 NFE) + zero-cost `bridge_handover` injecting Frobenius-normalized (`TBRG`) or raw (`STS2`) slow prosody $s_T$ from $T_4$ into $S_1$'s denoised `voice + fine` bands before $[\tau_2, \tau_3]$ (`2 NFE` -> `13 NFE` total).
  - Resumable `run_diversity` (`20` fresh speakers $\times$ `20` seeds) and `run_quality` (`40` fresh speakers), plus 2-speaker `smoke` checking split-student identity (`max_abs_split_diff <= 1e-5`) and stored `EXP-027` `W4` reproduction (`<= 0.01`).
  - `evaluate_stac_routing` (5-fold speaker-CV routing between 13-NFE `TBRG` and 20-NFE `W8R2` using Step-4 relative action velocity $r_4 = d_4 / (A_4 + 10^{-8})$ and $A_4$) and frozen `analyze` (`H1`–`H4`, `TBRG_vs_STS2`, `TBRG_vs_W8R2`, `MANIFEST.json`).
- **Preregistration:** Froze [preregistration.json](../../../research/experiments/EXP-028-transmissibility-bridge/preregistration.json) (`sha256: 8d6d68afa1d2...`), hashing all 7 dependency scripts, fresh speaker manifest, and locked `EXP-027` `diversity.csv` / `quality.csv` inputs, and excluded `research/experiments/EXP-028-transmissibility-bridge/results/**` text normalization in `.gitattributes`.
- **Colab Notebook:** Created the 17-cell T4 notebook [DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb](../../../code/colab/DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb) targeting branch `rhnl-3` and `Shonget-DMOSpeech2` commit `8a7868fc...` with Drive cache support (`dmo_cache/`) and 60s Drive backup (`exp028_transmissibility_bridge/`).
- **Unit Tests:** Added [test_transmissibility_bridge.py](../../../code/tests/test_transmissibility_bridge.py) (8 tests covering `bridge_handover` prompt/norm/direction invariants, costs & fresh speaker alignment with EXP-027, mocked `smoke`/resumable loops/error guards, 5-fold `STAC-TBRG` routing, all four `analyze` verdicts including `--allow_partial`, preregistration/code/input-hash tampering detection, and notebook AST syntax).
- **Progress Tracking:** Registered `EVAL-021` in `docs/progress/tasks.json`, re-rendered `docs/progress/TASKS.md`, updated `research/experiments/README.md` and `docs/progress/STATUS.md`.

## Checks Run

- `python code/colab/run_transmissibility_bridge.py freeze` — froze `preregistration.json` (`sha256=8d6d68afa1d2...`).
- `python scripts/render_progress.py` — updated `docs/progress/TASKS.md`.
- `python scripts/check_workspace.py` — `Workspace check PASSED: structure, task evidence, source hashes, links, and manuscript references.`
- `python -m unittest discover -s code/tests -p "test_transmissibility_bridge.py" -v` — 8 tests OK (`5.661s`).
- `python -m unittest discover -s code/tests` — 273 tests OK (`115.495s`).
- LaTeX build: not run (`paper/` unchanged; TeX build unverified for this experimental setup).

## Remaining Blockers

- **Colab Tesla T4 execution:** Needs execution of `smoke` and `run` on Google Colab T4 using `DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb` (`EVAL-021` stays `todo` until GPU results are imported).
