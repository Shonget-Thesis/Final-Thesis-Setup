# Appendices rewritten; Table 3.11, paper configuration and most notes removed (2026-11-28)

Requested by Mark Limpahan.

## Chapter 3 and figures
- Table 3.11 (planned settings) and its section removed; the text says the settings will be fixed in the study protocol before calibration and points to Appendix A.8 for one worked choice of values. References to the table removed.
- The DMOSpeech 2 paper configuration removed from the compared systems (Tables 3.3 and 3.9, Section 3.5, Section 3.8); the released system is the only DMOSpeech 2 reference.
- Notes removed under Figure 3.1, Table 3.3 and the baseline-handover figure; Chapter 2 figure notes cut to panel labels and attribution.

## Appendices (`paper/chapters/appendices.tex`, now two appendices)
- Appendix A, Sample Calculations: one input followed through the planned method in eight steps (costs and commit points, prompt SNR, quantile projection, residual, routing decision, failure and excess failure, fixed-sequence calibration, minimum calibration size). Input values are labelled illustrative; costs, commit points, p-values (scipy binomial) and the sample-size limit follow from the definitions. The released-baseline figure from the laptop merge is used in Step 1.
- Appendix B, Preliminary Experiments: only the results that led to the router (fixed early handover fails on noisy prompts; adapting the time could not be certified, noise makes the teacher look settled, the SNR gate misses moderate damage; router development run with a condensed results table and three findings). Numbers copied from the previous appendix text.
- Removed: sample simulation, prototype appendix, stopping-time cross-validation/runtime/restart/variety details, investigation plan, router theory and protocol sections, per-condition and diversity tables, next steps. All remain in git history.
- Revision matrix items 2, 6 and 8 updated.

## Checks
- `latexmk -pdf -Werror` clean (79 pages, no overfull boxes); revision matrix rebuilt without undefined references; Appendix A pages checked visually; `scripts/check_workspace.py` run.
- Same day: Tables 3.7 and 3.10 now name the variety row "Prosodic variety, CV_f0" (it was listed only by its symbol).
- Same day: Table 3.10 (comparison-matrix columns) merged into Table 3.7, now "Measures of quality, variety, risk and cost" (adds failure rate, excess-failure rate, routing rate and saving); the comparison-matrix references point to Table 3.7.
- Same day: Figure A.1 (released-baseline image) removed from Appendix A; Step 1 now refers to Figure 2.2 for the released sampler and Figure 3.1 for the proposed router. `fig_baseline_handover.tex` and its image are kept in the repository but no longer included.
- Same day: the bulleted list of pre-run checks in Section 3.4.5 replaced by one sentence.
