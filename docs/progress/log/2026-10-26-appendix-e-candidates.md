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
