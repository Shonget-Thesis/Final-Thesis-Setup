# 2026-10-27: Problem-validation tables in Appendix D

Requested by a team member: show the 500-speaker fixed-switching results as tables, not only in prose.

## Changed

- `paper/chapters/appendices.tex`, Appendix D "Fixed Switching Times": added Table `tab:fixed-fail` (added failures per fixed switching time, clean/noisy/all) and Table `tab:fixed-oracle` (hindsight cheapest safe step per instance and mean passes). The paragraph now cites the tables instead of listing the numbers. All figures copied from `research/experiments/EXP-001-problem-validation-t4/README.md` and `results.json`; no new results, meaning unchanged.

- Follow-up (same day): added a Saving column to both tables, the share of the reference's 21 forward passes avoided: (21 − passes)/21, i.e. 38.1%, 28.6%, 19.0% per fixed step and 37.8%, 20.8%, 29.3% for the hindsight choice. Derived from the existing numbers; no new results.
- Follow-up: table captions now state the takeaway of each table and how they differ (D.1: one switching time for every input, last three columns are added failures; D.2: per-input hindsight choice, no failures by construction, bounds what an adaptive rule can save).

- Follow-up: Appendix D.4 gets Table `tab:cv-policies`, the SNR + stabilization policy (out of fold) beside fixed Steps 4/5/6, the SNR gate at Step 4, the reference and the hindsight oracle on the same 200 T4 development instances, plus two sentences on it. `code/analysis/cv_stabilization_t4.py` now also writes `policy_comparison` to EXP-009's results.json (additive; earlier fields unchanged on rerun). Development evidence; 1 vs 3 failures against SNR-only is not a significant difference.
- Follow-up: D.4 now reports every compared method of Chapter 3 with its measures and statistics (the "Compared methods / Measures / Statistics" boxes of the defense slide): Table `tab:cv-policies` (failures, Clopper–Pearson upper bound, passes, saving, exact McNemar against the full policy) adds SNR + fixed Step 5/6, stabilization only (no gate, out of fold) and cost-matched random (2,000 permutations of the full policy's steps among gated inputs); new Table `tab:cv-quality` (mean WER, SIM, UTMOS, clean and all); new Table `tab:cv-runtime` (T4 timing with real early stopping, from EXP-010's committed summaries, with its cluster-bootstrap interval). The D.4 text states both the significant results (fewer failures than every fixed early step, McNemar p < 1e-13; fewer passes than the reference, Wilcoxon-Holm p < 1e-15, with no significant WER/SIM/UTMOS difference) and that the SNR-gated controls and the random control are not separable from the full policy on these data (McNemar p ≥ 0.50). No TOST: Chapter 3 fixes its margins before the held-out test. The `tab:prelim` summary row mentions the 27% clean-prompt time saving. `cv_stabilization_t4.py` gains the controls and tests; earlier results.json fields are unchanged on rerun.
- Follow-up: D.4 ends with a "Verdict and trade-off" paragraph summarizing the above: per-input switching removes almost all fixed-switch failures while saving 10.1% of passes (20.2% and 27% of time on clean prompts) with no significant quality change; SNR-only saves more with more failures; the stabilization score adds caution but is not yet separable from the SNR-gated controls; noisy prompts keep the reference cost; Objectives 3 and 4 decide.
- Follow-up (Appendix E only): Table `tab:candidates` now has the same columns as D.4's compared-methods table (failures clean / noisy over the five noisy conditions, Clopper–Pearson upper 95%, passes, saving, exact McNemar against SNR-only), adds the calibrated SNR gate row already discussed in the text, and moves the fixed/new counts into the caption. All values from EXP-015's committed `snng_results.json` (`cross_validation.pooled`); the upper bounds are computed from those counts. One sentence in E.3 notes that TAS and SWC-heavy have pooled rates under 5% but upper 95% bounds of 5.8% and 5.4%.

## Checks run

- python scripts/check_workspace.py and python -m unittest discover -s code/tests: see handover message.
- latexmk -Werror in paper/ (MiKTeX 25.12, installed today): passes, no undefined references; Tables D.1 and D.2 render on manuscript page 61.

## Open

- The added-failure constants (c = 12.5%, m_W = 1 point, m_S = 0.02) have no cited source in Chapter 3.
