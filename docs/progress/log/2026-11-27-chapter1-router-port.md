# Chapter 1 ported for the payload router (2026-11-27)

Task: PAPER-020 (review). Requested by Mark Limpahan: port the team's latest Chapter 1 draft (pasted 27 Nov) into the manuscript.

## What changed

- `paper/chapters/01-introduction.tex`: Sections 1.1-1.5 replaced with the team draft (candidate-path framing: Learn-Then-Test selects the least costly teacher path and router threshold; "excess failures" as the single term). Corrections made while porting:
  - Lu et al. (2026) lead-in now says "zero-shot models that learn the speaker's voice from the audio prompt" (the paper says this of in-context methods, not of infilling models such as F5-TTS).
  - The draft is made "from noise" (it does not depend on the teacher's handover).
  - The signals need "no network evaluation beyond the draft"; "if no candidate passes".
  - RQ1 says "shortened teacher paths"; Objective 3 no longer mixes a header with unheaded Objectives 1-2; Future Researchers names the selected teacher path and threshold.
  - L. Zhang et al. (2025) is still unidentified; marked with `\citationneeded`.
- Section 1.6 (conceptual framework) text and `figures/fig_conceptual.tex` labels: calibration selects the teacher path and threshold; calibration row moved down to clear the dashed band.
- Section 1.7 (definitions): scheduler terms removed (adaptive handover policy, cap, candidate switching time, endpoint-stabilization score, oracle, reference switching time, restart level); router terms added (candidate teacher path, corrected payload, draft, excess failure, payload, payload router, prosodic variety, quality failure, released system, residual).
- `paper/references.bib`: added `lu2026environment` (arXiv 2608.03011) and `xie2026lowlight` (arXiv 2610.21335); author lists taken from the arXiv PDFs.

## Checks

- `latexmk -pdf -Werror` in `paper/`: clean build, no undefined citations or references. Background is about 2.5 pages; Figure 1.1 visually checked with `pdftoppm`.

## Remaining

- Chapter 3 still treats the 4-step path as fixed and the 2- and 6-step paths as ablations; it must adopt the candidate-path calibration (and decide how the error budget is split across paths).
- Supply or drop L. Zhang et al. (2025).
- Not adviser-approved.
