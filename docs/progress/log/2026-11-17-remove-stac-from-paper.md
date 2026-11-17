# 2026-11-17: Remove STAC and Appendix E from the LaTeX manuscript

**Task:** `PAPER-015` (`review`)  
**Targets:** [paper/chapters/appendices.tex](../../../paper/chapters/appendices.tex), [paper/frontmatter.tex](../../../paper/frontmatter.tex), [paper/main.tex](../../../paper/main.tex)

## Changes Applied
1. **Removed Appendix E (`\label{app:stac}`) and all `STAC` / `1D-CRC-STAC` sections from `paper/chapters/appendices.tex`:**
   - Deleted `\section[APPENDIX E.\ STAGE-WISE TRAJECTORY ACTION CASCADE]{...}` (method, `fig_stac` include, multi-grid `STAC`/`1D-CRC-STAC` tables, ablation table, OOD table) and removed `E` from the header outline.
   - Removed the `Appendix~\ref{app:stac}` / `CRC` sentence from Appendix B.
   - Removed `\subsection{Live Online Early-Stopping Verification on the Tesla T4}` (`\label{app:proto-live}`, `Table~\ref{tab:live-t4}`) from Appendix C (which was the live GPU run of `STAC`; non-STAC Chapter 3 live timing remains in `Table~\ref{tab:cv-runtime}`).
   - Cleaned dangling `\ref{app:stac}` / `\ref{app:stac-results}` / `\ref{tab:fixed-steps1to8}` links and STAC prose in Appendix D.1, Appendix D.4, Appendix D.6, and `Table~\ref{tab:prelim}` while keeping all Appendix D Steps~1--7 quality benchmarks (`Table~\ref{tab:prelim-steps1to7}`) and prosodic variety findings (`\label{app:prelim-variety}`).
2. **Cleaned frontmatter & header metadata:**
   - Removed unused `CRC` and `STAC` entries from `\citcAcronyms` in `paper/frontmatter.tex` (retaining `OOD` and `TOST`, still active in Appendix D).
   - Updated the comment in `paper/main.tex` to `Appendices A-D`.
   - Kept unreferenced `paper/figures/fig_stac.tex` on disk without importing it into `main.pdf`.

## Checks Run
- Verified `0` remaining `STAC|app:stac|fig_stac|fig:stac|1D-CRC|Appendix E` matches across `paper/main.tex`, `paper/frontmatter.tex`, `paper/chapters/0*.tex`, and `paper/chapters/appendices.tex`.
- `latexmk -interaction=nonstopmode -halt-on-error -Werror main.tex` (inside `paper/`): exited `0`, wrote `paper/build/main.pdf` (`78 pages`, `1,054,567` bytes) with `0` undefined references, `0` missing citations, and `0` overfull boxes.
- `python scripts/render_progress.py`, `python scripts/check_workspace.py`, and `python -m unittest discover -s code/tests`: run after these edits.

## Remaining Blockers
- Adviser and team review of updated Appendices A–D, `PAPER-002` formatting confirmation, listening evaluation (`EVAL-002`), and held-out unseen evaluation.
