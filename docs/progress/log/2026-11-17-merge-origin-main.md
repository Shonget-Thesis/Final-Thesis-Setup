# 2026-11-17: Merge origin/main (PR #19) into rhnl-3 and reconcile manuscript & workspace

**Task:** `PAPER-016` (`review`)  
**Merge commit:** `e0bb5c4` (`Merge remote-tracking branch 'origin/main' into rhnl-3`, fast-forwarded local `main` to `d169bef`)

## Changes Applied
1. **Merged `origin/main` (`d169bef` / PR #19 `03a8af2`, *Post-concept revisions*) into `rhnl-3`:**
   - Brought in all `12` non-conflicting PR #19 files byte-for-byte (`git diff origin/main` is empty on all of them): [01-introduction.tex](../../../paper/chapters/01-introduction.tex), [02-related-work.tex](../../../paper/chapters/02-related-work.tex), [03-methodology.tex](../../../paper/chapters/03-methodology.tex), [fig_conceptual.tex](../../../paper/figures/fig_conceptual.tex), [fig_grid.tex](../../../paper/figures/fig_grid.tex), [fig_method.tex](../../../paper/figures/fig_method.tex), [fig_theoretical.tex](../../../paper/figures/fig_theoretical.tex), [fig_workflow.tex](../../../paper/figures/fig_workflow.tex), [generate_sway_grid_check.py](../../../paper/figures/scripts/generate_sway_grid_check.py), [sway_grid_tswitch.png](../../../paper/figures/sway_grid_tswitch.png), [references.bib](../../../paper/references.bib), and [revision-matrix.tex](../../../paper/revision-matrix.tex).
   - Fast-forwarded local `main` from `a849432` to `d169bef` (`origin/main`).
2. **Reconciled `paper/chapters/appendices.tex` and preserved all `rhnl-3` contributions:**
   - Kept all PR #19 edits inside [appendices.tex](../../../paper/chapters/appendices.tex) (`Table A.1` Step~7 candidate row, `Objective~2`/`Objective~3` renumbering across Appendices B–D, and the preliminary Steps 4–6 note).
   - Re-applied and retained all `rhnl-3` working-tree additions on top of the merge commit: `PAPER-014` Appendix D Steps~1--7 quality benchmarks (`Table~\ref{tab:prelim-steps1to7}`) & Prosodic Variety (`\label{app:prelim-variety}`), `PAPER-015` `STAC` / Appendix E removal ([appendices.tex](../../../paper/chapters/appendices.tex), [frontmatter.tex](../../../paper/frontmatter.tex), [main.tex](../../../paper/main.tex)), `PROTO-001` (`code/prototype/`), `EXP-016`–`EXP-033` + offline SNR/adaptive screens & tests (`code/analysis/`, `code/tests/`), and `PRES-002` (`19`-slide [presentation/index.html](../../../presentation/index.html), [SCRIPT.md](../../../presentation/SCRIPT.md), [README.md](../../../presentation/README.md), and [proposal-defense-slides.pdf](../../../presentation/proposal-defense-slides.pdf)).

## Checks Run
- Verified `git diff origin/main` across all `12` non-appendix PR #19 files returns empty (exact match).
- `latexmk -interaction=nonstopmode -halt-on-error -Werror main.tex` (in `paper/`): exited `0`, built `paper/build/main.pdf` (`90 pages`, `1,182,187` bytes).
- `python scripts/render_progress.py`, `python scripts/check_workspace.py`, and `python -m unittest discover -s code/tests`: run cleanly after final edits.

## Remaining Blockers
- Main presentation slides (`06-objectives.html` still showing 4 objectives and `11-where-the-switch-lives.html` omitting Step 7) reflect the pre-PR #19 proposal until updated to match PR #19; `rhnl-3`'s uncommitted working-tree files (`EXP-033`, `PAPER-014/015`, `PRES-002`, analysis/tests) remain unstaged/untracked on top of merge commit `e0bb5c4` ready for commit/push when desired.
- Adviser and panel confirmation pending (`revision-matrix.tex`, `PAPER-002`, `EVAL-001`/`EVAL-002`).
