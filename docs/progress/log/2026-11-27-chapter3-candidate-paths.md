# Chapter 3 aligned with the candidate-path framing (2026-11-27)

Task: PAPER-020 (review). Requested by Mark Limpahan after the Chapter 1 port of the same day.

## What changed

- `paper/chapters/03-methodology.tex`:
  - The teacher path is no longer fixed at four steps. Three candidate paths on the 8-interval grid, k in {2, 4, 6} (commit points 0.0761, 0.2929, 0.6173, the only grid points that coincide with the student's schedule), cost C(k) = 1.5k + 7 = 10, 13 or 16 passes (new Eq. `eq:paths`; Proposition 3.1 generalized; Table 3.3 rows per path).
  - Problem formulation (`eq:objective`): choose the least costly path with a safe threshold, then the highest safe threshold on that path.
  - Calibration (Algorithm 3.4, Proposition 3.10): fixed-sequence Learn-Then-Test over the threshold grid separately for each path at level delta/3; select the cheapest path with an accepted threshold; union bound gives P(R(k_hat, lambda_hat) <= alpha) >= 1 - delta. Minimum zero-failure calibration size becomes 67 (was 45); Table 3.9 recomputed at delta/3 (scipy binomial); smaller-sample robustness uses n = 75, 100, 200 (was 50).
  - Routing rule, Algorithm 3.3, Propositions on cost, monotone routing, fallback and hindsight bound restated per path; notation table updated.
  - Characterization (Algorithm 3.2): one six-step teacher run records the endpoint estimates at t2, t4 and t6; both payloads of every path are finished and scored. A pipeline check confirms these equal separate runs.
  - Compared systems: the 2- and 6-step paths are no longer ablations; the unselected paths are reported with their own accepted thresholds; ablations use the selected path.
  - Research design, architecture, workflow table and reporting paragraph: the selected path and threshold are a calibration result, not part of the design.
- `paper/figures/fig_method.tex` (calibrator now feeds k_hat to the teacher; costs per path) and `paper/figures/fig_workflow.tex` (steps 4, 8, 9, 10).

## Design choice to confirm with the team

- The error budget is split equally across the three paths (delta/3 each) rather than testing all candidates in one chain, so an unsafe threshold on one path cannot stop the test of another. This makes each test slightly stricter (4 instead of 5 tolerated excess failures at n = 200).

## Checks

- `latexmk -g -pdf -Werror` in `paper/`: clean, 108 pages, no undefined references, no overfull boxes; Figures 3.1 and 3.2 and Algorithm 3.4 visually checked with `pdftoppm`.

## Remaining

- Appendices C-E and the revision matrix still describe the four-step path as fixed in places.
- Not adviser-approved; EXP-034 remains development evidence.
