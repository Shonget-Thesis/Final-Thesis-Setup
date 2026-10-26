# 2026-10-26: Appendix E, Candidate Solutions (TAS and SWC)

Branch `rhenelAudit`. Requested by a team member: add only a new appendix section for the two solution candidates.

## Changed

- `paper/chapters/appendices.tex`: new Appendix E, "Candidate Solutions" (E.1 Trajectory Anomaly Score, E.2 SNR-Wiener Conditioning, E.3 Results with Table E.1, E.4 Limitations and Next Steps), appended after Appendix D; the header comment lists it. No other section changed.
- `paper/references.bib`: two entries, `heng2024diffpath` (DiffPath, NeurIPS 2024) and `boll1979spectral` (spectral subtraction, IEEE Trans. ASSP 27(2), 1979).
- Content follows `research/reports/Ref_reports/PROPOSED_CANDIDATES_TAS_SWC_REPORT.md` and the committed EXP-012 to EXP-015 results: SNR-only 86/360 failures at 14.40 forward passes; SNR + TAS 12/360 at 18.20; SNR + SWC 20/360 at 13.00 or 11/360 at 16.98 routed; SWC's noisy-prompt similarity +0.126 over the reference. Past tense for completed studies, future tense for the planned confirmation; development evidence and the A100-versus-T4 caveat are stated.

## Checks run

- latexmk -Werror in paper/: exit 0; no undefined or multiply defined references in the final log; no overfull boxes. Appendix E is on printed pages 64-66.
- python scripts/check_workspace.py and python -m unittest discover -s code/tests: see below.

## Open

- Whether to keep Appendix E in the proposal is a team and adviser decision; it is not adviser-approved.

## Follow-up: candidate figures (same day)

- `paper/figures/tas_handover_gate.jpg` (SNR + TAS handover gate) and `paper/figures/swc_conditioning.jpg` (SNR-Wiener Conditioning with SNR routing), drawn in the style of `handover_scheduling_architecture.jpg` (five panels, highlighted method panel, dashed offline-calibration box, caption) by `paper/figures/scripts/generate_candidate_figures.py`; formulas follow Appendix E. Generated only; not yet included in the manuscript.

## Follow-up: figure and appendix formulas made identical (same day)

The first figures used shortened notation, a different font, and two SWC thresholds (θ_hi, θ_e) that Appendix E did not define.

- `paper/chapters/appendices.tex`: E.1 states that θ in π_TAS was the 20 dB gate of SNR-only in these studies (the EXP-015 TAS policy). E.2 adds the SWC routing rule π_SWC, which defines θ_e ≤ θ_hi. E.3 gives the certified values from the EXP-015 report: θ_hi = 26 dB in all five folds; θ_e = θ_hi for `swc_noisy`; θ_e = 21, 18, 20, 20 and 21 dB for `swc_heavy`. The two SWC rows of Table E.1 are now labelled by these thresholds. No result changed.
- `paper/figures/scripts/generate_candidate_figures.py`: text and formulas are now typeset by LaTeX (matplotlib `text.usetex`; Palatino math to match mathpazo; Helvetica labels) with the exact Appendix E notation. The G_{f,b} formula is split over two lines to fit its panel. Both JPGs were regenerated. Regenerating them now requires a TeX installation with dvipng.

### Checks run

- latexmk -Werror in paper/: exit 0; no undefined or multiply defined references; no overfull boxes. There are 9 underfull warnings, none of them in Appendix E. Appendix E is on printed pages 64–66. Pages 65–66 and both figures were inspected as rendered images.
- python scripts/check_workspace.py: PASSED.
- python -m unittest discover -s code/tests: 55 tests, OK.
