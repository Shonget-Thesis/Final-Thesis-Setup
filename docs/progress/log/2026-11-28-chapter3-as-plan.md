# Chapter 3 rewritten as a plan; paths calibrated and compared separately (2026-11-28)

Requested by Mark Limpahan.

## Framing

The candidate teacher paths (2, 4, 6 steps) are no longer "selected" by calibration. Learn-Then-Test calibrates the router's threshold for each path separately (delta/3 per path, so the guarantee holds for all three at once); the evaluation compares the calibrated paths, and the least costly path that stays within the limit is a finding for the conclusions, not part of the architecture.

## Chapter 3 (`paper/chapters/03-methodology.tex`)

- Section 3.1 shortened to three paragraphs (explanatory phrases and the success-criterion paragraph removed).
- Figure 3.2 (workflow diagram) removed as redundant with Table 3.2; `figures/fig_workflow.tex` deleted; step references now point to Table 3.2.
- The negative-results passage at the end of Section 3.3 removed.
- Remark 3.1 (paper configuration) removed; its point kept as a phrase in Table 3.15. The remaining remark in the calibration section turned into two plain sentences.
- Approximate commit points (0.0761, 0.2929, 0.6173) removed; commit points and the student schedule given by formula only.
- Computed values removed: router pass counts (now C(k)), pass counts in algorithm comments, the sample-size table and the minimum-calibration-size numbers. Planned settings (Table 3.11, now "Planned Settings") kept, since the guarantee needs them fixed in advance.
- Problem formulation (`eq:objective`) is now per path; Algorithm 3.4 returns one threshold per path; Proposition 3.10 is a simultaneous guarantee over the three paths; verification, evaluation, robustness and the compared-systems table restated per path.
- `figures/fig_method.tex`: calibrator outputs one threshold per path; no path arrow to the teacher; cost shown as C(k).

## Chapter 1 and revision matrix (kept consistent)

- Background, conceptual-framework text, RQ2, RQ3, Objectives 2 and 3, Scope, Significance (future researchers) and the definitions of Calibration, Candidate teacher path, Learn-Then-Test and Reliability guarantee now say that each path is calibrated and the paths are compared. Figure 1.1: calibration "sets the threshold for each teacher path once".
- Revision matrix items 4, 5, 13 and 20 reworded without computed values.

## Checks

- `latexmk -pdf -Werror` clean (106 pages, no overfull boxes); revision matrix rebuilt; Sections 3.1-3.2, Table 3.3 and Figures 1.1 and 3.1 checked visually.
