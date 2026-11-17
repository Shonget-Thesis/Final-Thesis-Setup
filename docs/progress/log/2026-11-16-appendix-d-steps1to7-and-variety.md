# 2026-11-16: Appendix D update — Steps 1–7 quality and prosodic variety findings

**Task:** `PAPER-014` (now `review`)  
**Target:** [paper/chapters/appendices.tex](../../../paper/chapters/appendices.tex) (Appendix D, keeping Chapters 1–3 unchanged)

## Changes Applied to Appendix D

1. **Added full Step 1–7 quality benchmark in Appendix D.1** (`\label{app:prelim-steps1to7}`, `Table~\ref{tab:prelim-steps1to7}`):
   - Incorporates `EXP-018` (`360` utterances across `60` development speakers $\times$ `6` prompt conditions: `clean` and 15–25 dB Gaussian noise across `Steps 1..7` vs `Step 8`) alongside `EXP-003` (`100` development speakers at `Steps 2..6` vs `8`) and `EXP-032` (`n=30` `noisy15` runaway counts and `STU` comparison).
   - Summarizes clean convergence by `Step 2–3` (`9–11` NFE; Step 1 `UTMOS` drops to `3.06` / `2.962`) versus degraded-prompt breakdown across `Steps 1–4` (`280.17%` down to `24.69%` raw WER) and earliest pooled TOST equivalence at `Step 7` (`t_7 = 0.2270`, `19` NFE, `p_WER = 0.0047`), while noting student-only (`4` NFE) avoids copying prompt white noise.
   - Cross-references Appendix E (`Table~\ref{tab:fixed-steps1to8}`) for the companion STAC multi-grid benchmarks.

2. **Added prosodic variety subsection in Appendix D** (`\subsection{Prosodic Variety Across Switching Steps and Restart Levels}`, `\label{app:prelim-variety}`):
   - Synthesizes `EXP-020`, `EXP-022`, `EXP-026`–`EXP-029`, and `EXP-032` using `pYIN` unaligned `CVf0`, time-aligned `CVf0`, inter-seed timing spread, common-frame `CVf0`, and slow-band decomposition.
   - Adds `Table~\ref{tab:prelim-variety-steps}` (`EXP-032`, `20` $\times$ `12` clean diversity, `30` clean `UTMOS`): shows standard `j=1` (`tau = 0.0761`) plateaus by `Step 2–3` through `Step 8` (`W8` keeps `60%` of `P25`'s raw `CVf0` gain and `85%–88%` of timing-spread gain, but only `35% [9%, 53%]` of aligned pitch-contour gain; notes `W5`–`W7` were not separately scored for `CVf0`), plus the `EXP-032` tensor-bank mechanism (`82%` Step-8 melody agreement vs `53%` final retention under `j=1` vs `79%–86%` under `P25R`/`P25`).
   - Adds `Table~\ref{tab:prelim-variety-r2}` (`EXP-027`, `EXP-028` `unseen_cal`, `EXP-029`): documents `j=2` (`tau = 0.2929`) restarts (`W4R2`, `W5R2`, `W6R2`, `W8R2`) and slow-band bridges (`TBRG`) — `W8R2` (`20` NFE) recovers pure-teacher `T16` / `P25` variety at `-0.105` to `-0.169` UTMOS below `W8`, while earlier `j=2` steps (`Steps 1–3, 7` not run) follow a monotone variety–UTMOS trade-off (`Spearman rho = -0.93` in `EXP-029`) and incur `3–6` `noisy15` WER runaways vs `0` for `W8`.
   - Documents the `noisy15` `pYIN` tracking limit (`0%` voiced frames in `>= half` the seeds across teacher arms in `EXP-022`/`EXP-032`) and preserves Chapter 3's scope boundary (all figures in Appendix D remain development evidence, awaiting adviser/team review).

3. **Updated `Table~\ref{tab:prelim}`** summary rows to reflect `EXP-003`/`EXP-018` `Steps 1–7` coverage and `EXP-026`–`EXP-029`/`EXP-032` variety and restart-level findings.

## Checks Run
- `latexmk -interaction=nonstopmode -halt-on-error -Werror main.tex` (inside `paper/`): exited `0`, wrote `paper/build/main.pdf` (`86 pages`, `1,129,461` bytes) with `0` undefined references, `0` missing citations, and `0` overfull boxes.
- `python scripts/render_progress.py` and `python scripts/check_workspace.py`: executed after these edits.
- `python -m unittest discover -s code/tests`: executed after these edits.

## Remaining Blockers
- Adviser and team review of `PAPER-014` (`paper/chapters/appendices.tex` updates), alongside open decisions on Chapter~3's fixed `j = 1` vs `j = 2` (`EVAL-009`), listening evaluation (`EVAL-002`), and held-out unseen corpus confirmation.
