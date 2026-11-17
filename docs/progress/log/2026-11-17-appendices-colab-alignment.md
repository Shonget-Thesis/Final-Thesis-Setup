# 2026-11-17: Ground all Appendices A–D on Colab Tesla T4 experiments

**Task:** `PAPER-012` / `PAPER-017` (`review`)  
**Target:** [paper/chapters/appendices.tex](../../../paper/chapters/appendices.tex) (Appendices A–D only; Chapters 1–3 untouched on `origin/main` PR #19)

## Changes Applied

1. **Appendix A (`Sample Calculations`, pp. 66–68):**
   - Replaced all synthetic toy utterance values with real Google Colab Tesla T4 recordings from `EXP-011` ([worked_example.json](../../../research/experiments/EXP-011-followup-t4/results/pilot/worked_example.json)) and [gpu_grid_all1000.csv](../../../research/evidence/benchmarks/gpu_grid_all1000.csv) on development utterance `common_voice_en_13900-common_voice_en_13901` (459 prompt frames: `30.68` dB clean / `17.65` dB `noisy15`; Step-5 tensor cell `x_t = -0.6905`, `v = -0.5198` -> `-1.1489`; generated-frame norms `d_4 = 116.17/218.01 = 0.533`, `d_5 = 85.92/204.42 = 0.420`, stopping at Step 5 at `15` passes; noisy15 `d_4 = 42.93/295.32 = 0.145` held by `\theta = 20` dB) and failure check against Step-8 (`common_voice_en_15734837-common_voice_en_15734838` `noisy15` Step 4 `37.50%` WER, `0.889` SIM).
   - Updated `Table A.2` (`tab:app-ltt`) to real Colab T4 development-split fixed-sequence LTT counts (`n = 100`, `\lambda = 0.20..0.35 -> 0` failures `p = 0.0059`, `0.40..0.50 -> 1` failure `p = 0.0371`, stopping at `0.55` with `2` failures `p = 0.1183`, returning `\hat\lambda = 0.50`), while explicitly linking to Appendix D.4 for the 200-speaker calibration T4 run (`\hat\lambda = 1.20` on 2 conditions) and multi-condition T4 runs (`EXP-012`/`EXP-017`/`EXP-018`).

2. **Appendix B (`Sample Simulation`, pp. 69–71):**
   - Fixed the companion script path (`code/analysis/ltt_sim.py`), clarified that the `Figure B.1`/`Table B.1` stress test tests `k \in \{4,5,6\}` when unsafe thresholds exist, and added the **Colab Tesla T4-fitted simulation** (`python code/analysis/ltt_sim.py --fitted`, fitted directly to the 100 development speakers' `gpu_grid_all1000.csv` runs: SNR `37.9 +- 7.5` dB clean vs `17.8 +- 0.3` dB noisy15, `\ln d_4 = -0.542 +- 0.217` vs `-1.739 +- 0.198`, fitted failure logit `-2.640 - 2.504 d_k + 3.583 noisy - 0.350(k-4)`, max true risk `0.0089`, `100%` of repetitions `<= 0.05`, median `\hat\lambda = 0.40` at `n = 50` to `1.20` at `n = 200`).

3. **Appendix C (`Prototype`, pp. 72–73, `Figure C.1`):**
   - Grounded Inputs A–C in `Figure C.1` on three real Colab T4 development utterances from `gpu_grid_all1000.csv` (`common_voice_en_15734837-common_voice_en_15734838` stopping at Step 4, `common_voice_en_10933823-common_voice_en_10933825` stopping at Step 6, and `common_voice_en_13900-common_voice_en_13901` `noisy15` held at Step 8).
   - Added Step 7's grid tick (`0.227` at `16` passes) to the handover timeline axis, included `\hat S \ge \theta` in Box 1, and scaled the landscape diagram cleanly (`0.88\linewidth`) so there is no float overflow.

4. **Appendix D (`Preliminary Experiments`, pp. 74–85):**
   - Removed all remaining non-Colab CPU experiment descriptions (`12-speaker / 48-instance Whisper-base CPU` restart check in Intro/D.2 and `12-speaker CPU \lambda=0.60` note / CPU-label framing in D.4 / `Table D.8`) so Appendix D rests exclusively on Google Colab Tesla T4 evidence.
   - Clarified the difference between pre-LTT 4-fold cross-validation (`Tables D.6–D.7`), live T4 timing (`Table D.8`, `EXP-010`), 2-condition T4 LTT (`AUDIT-001`, `\hat\lambda = 1.20` -> `SNR-only`), and 6-condition T4 LTT (`EXP-012`/`EXP-017`/`EXP-018`).
   - Added `Table D.9` (`tab:t4-steps1to8-policies`) from `EXP-018` ([SWSR_HANDOVER_REPORT.md](../../../research/experiments/EXP-018-swsr-handover-prelim/results/SWSR_HANDOVER_REPORT.md), `n = 360` Colab T4 cases) comparing all fixed checkpoints (`Steps 1–8`) directly alongside Chapter 3's adaptive policies (`SNR-only 20 dB -> Step 1`, `SNR-only calibrated`, `SNR + Step 5/6`, `Stabilization only`, `Full policy Def. 3.4`), without re-introducing STAC (`PAPER-015`).
   - Added concise Colab T4 summaries for `EXP-030`, `EXP-031`, and `EXP-033` in `D.6` and `Table D.10` (`tab:prelim`), clearly flagged as unapproved development evidence.

## Checks Run

- `python code/analysis/ltt_sim.py --fitted`: reproduced all fitted T4 parameters and repetition metrics quoted in Appendix B.
- `python research/audits/2026-10-25/recompute_gpu_grid.py` + development split LTT check: verified all Table A.2 and AUDIT-001 counts against `gpu_grid_all1000.csv`.
- `latexmk -interaction=nonstopmode -halt-on-error -Werror main.tex` (inside `paper/`): exited `0`, built `paper/build/main.pdf` (`93 pages`, `1,213,460 bytes`) with zero undefined references, zero missing citations, zero `Float too large` warnings, and zero overfull boxes in `appendices.tex`.
- Rendered Appendix C (`Figure C.1`, PDF p. 81 / printed p. 73) and Appendix D (`Tables D.9–D.10`, PDF pp. 90, 93 / printed pp. 82, 85) with `pdftoppm` and visually verified layout.
- `python scripts/render_progress.py`, `python scripts/check_workspace.py`, and `python -m unittest discover -s code/tests`: run after final edits (see session output).

## Remaining Blockers

- Chapters 1–3 remain byte-identical to `origin/main` (`d169bef`); synchronizing Chapter 3 (`Table 3.9`) with post-proposal T4 evidence and adviser review of all `review` manuscript tasks (`PAPER-001`..`PAPER-017`) remain pending.
