# Appendices A–E reconciled with the candidate-path payload router (2026-11-28)

Task: `PAPER-022` (`review`, owner: Rhenel Jhon R. Sajol).

## Context

After merging `origin/main` (`531a26a`, which ported Chapter 1 and updated Chapter 3 around candidate teacher paths $\mathcal{K}=\{2,4,6\}$ and the two-signal payload router `SNR_R`), `paper/chapters/appendices.tex` still computed, simulated, and diagrammed the superseded 16-grid stopping-time scheduler ($d_i \le \lambda$ at $j=1$), stated in Appendices D and E that Chapter 3 was unchanged, and duplicated 4.5 pages of Chapter 3's invariance propositions and router architecture figure inside Appendix E.

## What changed

- **Branch sync:** Fast-forwarded local `main` to `origin/main` (`531a26a`) and merged `origin/main` into `rhnl-3` with zero diff against `origin/main` prior to appendix edits.
- **Appendix A (`app:calc`, Sample Calculations):** Replaced the 16-grid stopping-step table, $d_4/d_5$ calculation, and old two-metric margin check with the 8-interval candidate paths $\mathcal{K}=\{2,4,6\}$ (`Table A.1`, `10/13/16` passes vs.\ `21`), prompt SNR examples from `EXP-011` and `EXP-034`, step-by-step quantile projection $Q(\hat x_1,s_0)$ and 1D 2-Wasserstein residual $r(\hat x_1,s_0)$, the absolute 3-metric failure rule (`s_min=0.8482`, `u_min=3.4031` on `EXP-034` clean `W8`) and excess-failure indicator $e(x,k,\lambda)$ on real development speakers (`common_voice_en_203357-common_voice_en_203358` and `common_voice_en_19833804-common_voice_en_19833805`), and the $\delta/3=0.0333$ binomial p-value sequence (`Table A.2`) and $n\ge 67$ zero-failure bound.
- **Appendix B (`app:sim`, Sample Simulation) & `code/analysis/ltt_sim.py`:** Updated `ltt_sim.py` and Appendix B (`Figure B.1`, `Table B.1`) to simulate Algorithm 3.4's three-path Learn-Then-Test selection ($\mathcal{K}=\{2,4,6\}$ at $\delta/3$, nonzero baseline failure rate $F_{\text{ref}}$) against naive selection across $n\in\{50,75,100,200,400\}$ ($1{,}000$ repetitions each), plus a bootstrap check (`--fitted`) on the $240$ recorded `EXP-034` Tesla T4 rows.
- **Appendix C (`app:prototype`, Prototype) & `paper/revision-matrix.tex`:** Rewrote Appendix C, `Table C.1` (`tab:samples`), landscape `Figure C.1` (`fig:glance`), and `revision-matrix.tex` item 2 to describe the `SNR_R` prototype in `code/prototype/` (`snr_invariant.py`, `k=4`, `13` fixed passes) on development speaker `common_voice_en_203357-common_voice_en_203358`.
- **Appendix D (`app:prelim`, Preliminary Experiments):** Preserved the complete `Steps 1–8` fixed baseline benchmark table (`Table D.3`, `tab:prelim-steps1to7`: Clean & Noisy WER/SIM/UTMOS, latency, `EXP-003` failure rates, and TOST) right in `D.1` while removing the duplicated `Steps 1–8` rows from `D.4`'s policy table (`tab:t4-steps1to8-policies`), embedded `fig_baseline_handover` in `D.2`, condensed the superseded 2-condition CV stopping tables, renamed old stopping rules to "earlier stopping-scheduler policy", and updated `Table D.8` (`tab:prelim`) to show how each preliminary finding shaped Chapter 3.
- **Appendix E (`app:snr-r`, Development Pilot of the Payload Router):** Removed duplicated Section `E.1` (`G_1\subset G_2\subset G_3` ladder and Propositions `E.1–E.3`, already in Chapter 3 Section `3.6`) and duplicated `Figure E.2`, removed stale `"Chapter 3 remains unchanged"` statements, and retained the complete `EXP-034` quality, cost, per-condition, and multi-seed diversity tables (`Tables E.1–E.4`) and diagnostic limits as exploratory $k=4$ development evidence.
- **Revision matrix, panel request to list all fixed values:** Items 4 (candidates / best value), 14 (is 0.25 best?) and 16 (pre-experiment validates the problem) now also cite `Table D.3` (`tab:prelim-steps1to7`), which lists all eight fixed switching values of the released 16-interval grid (Steps 1–8) with their clean/noisy WER, SIM, UTMOS, time and TOST results. Item 14 states the development findings already in `Table D.3` (clean settles by Step 2, noisy fails at Steps 1–4, only Step 7 is WER-equivalent to Step 8).

## Checks run

- `python code/analysis/ltt_sim.py` and `python code/analysis/ltt_sim.py --fitted`: reproduced `Figure B.1` and `Table B.1` deterministically.
- `latexmk -g -pdf -interaction=nonstopmode -halt-on-error -Werror main.tex` and `latexmk -g -pdf -interaction=nonstopmode -halt-on-error revision-matrix.tex` in `paper/`: both exited `0` with zero undefined references/citations and zero `Overfull \hbox` warnings in `appendices.tex`.
- `python scripts/render_progress.py`, `python scripts/check_workspace.py`, and `python -m unittest discover -s code/tests`: all passed.

## Remaining blockers

- Confirmatory Colab Tesla T4 runs across candidate paths $\mathcal{K}=\{2,4,6\}$, calibration on the 200-speaker calibration split, and held-out evaluation on the 200-speaker Seed-TTS test split and LibriSpeech-PC remain pending (`EVAL-002` / `EVAL-030`).
- `L. Zhang et al. (2025)` citation placeholder in Chapter 1 (`01-introduction.tex`) from `origin/main` still needs resolving before `--submission` freeze.
- Manuscript changes remain unapproved draft revisions pending team and adviser review.
