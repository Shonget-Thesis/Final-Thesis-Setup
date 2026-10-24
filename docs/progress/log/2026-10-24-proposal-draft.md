# 2026-10-24: Proposal draft promoted to the final manuscript

Task: PAPER-008. User request (Mark Vincent D. Limpahan): clean the paper folder and put the latest draft of the paper, in the CITC format, in this repository.

## Source

The team's LaTeX proposal "A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching Text-to-Speech Inference" (Final_Paper.zip, AIT-style template, supplied 2026-10-24), with the settled edits from the Shonget-Thesis/Shonget-DMOSpeech2 report `Handover Scheduling/reports/ALL_REPORTS_COMPILED.md` (commit 8a7868f, Part 5 tasks T49, T50, T52-T55) already applied:

- Remark 3.1 corrected: the paper's 14 teacher steps end at t14 = 1 - cos(14 pi/64), about 0.227 (equal to t7 on the 16-interval grid), not 0.236.
- Section 2.3 filled from T52.1-T52.6, including the point that the student always restarts at tau = 0.0761 = t4; Section 2.6 filled from T53.1 without this study's own results.
- Figures 2.2 and 2.4 replaced with `sway_grid_tswitch.png` and `teacher_guided_timeline_t55.png` (generated in Shonget-DMOSpeech2 from the sampler formulas only).
- Data section: 1,088 utterances, 666 prompt speakers, 100/200/200 speaker-disjoint splits; speaker = prompt clip ID.
- SIM verifier named as actually run (`microsoft/wavlm-base-plus-sv`); epsilon added to the drift-score denominator; robust (per-speaker) failure counting; statistics list matches the tests run.
- "Published default j = 1" corrected to "the released code's default" (the DMOSpeech 2 paper's main configuration uses two student steps, j = 2).

## Changes in this repository

- `paper/main.tex`: kept the PAPER-007 CITC setup (headheight fix, prelim list numbering, csquotes) and added the packages, macros, theorem environments and figure settings the proposal needs. Metadata filled from the proposal draft's title page; submission month, middle names and panel/dean names stay TBD. A `\citcproposal` switch (default true) shows only the proposal's preliminary pages and Chapters 1-3; Chapters 4-5 remain placeholders for the final thesis.
- `paper/frontmatter.tex`: approval, acknowledgement, dedication, acronym and abstract pages wrapped in the proposal switch; acronym list filled.
- `paper/chapters/01-introduction.tex`, `02-related-work.tex`, `03-methodology.tex`, new `appendices.tex`: proposal content converted to CITC conventions (chapter = `\section`; `\parencite`/`\textcite`; equation references through `\Cref`; `\eqentry` after each equation; `whereterms`). The draft-note macro `\citeneeded` is renamed `\citationneeded` so the workspace check does not read it as a citation command.
- `paper/figures/`: four TikZ figures (wrapped to fit the 14.7 cm text block) and the two PNGs.
- `paper/references.bib`: the proposal's bibliography.
- `code/analysis/ltt_sim.py`: the simulation behind Appendix B.
- Not removed: `paper/drafts/`, `paper/sources/` (AGENTS.md: preserve alternate drafts and provenance) and the Chapter 4-5 placeholders (required by the workspace check).

## Checks run

- `latexmk -Werror` in `paper/` (TeX Live, biber 2.19): exit 0; built main.pdf (66 pages), adaptive-handover.pdf and tanh-velocity-blending.pdf. No overfull lines above 3 pt in main.log; remaining warnings are font-size substitutions inside TikZ figures.
- `python scripts/check_workspace.py`: passed.
- `python -m unittest discover -s code/tests`: 15 tests passed.
- Rendered main.pdf pages at low resolution and inspected the title page, contents, chapter openings, equations with where-lists, tables, figures and appendix openings.

## Not verified or still open

- Content is the team's working proposal, not adviser-approved. Drift's role as the decision signal still awaits the adviser's decision after the signal-selection results.
- Runtime scope (Ryzen laptop vs. Tesla T4) and the student restart level remain open decisions.
- Middle names in full (3.2.7), submission month and panel/dean names are TBD; `check_workspace.py --submission` is expected to fail.
- The wide TikZ figures (workflow, architecture) were drawn for a wider page and render with small text.
- CI has not run remotely for this branch.
