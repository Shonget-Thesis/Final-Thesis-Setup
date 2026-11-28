# Revision matrix rewritten with locations instead of status (2026-11-28)

Requested by Mark Limpahan. `paper/revision-matrix.tex` now describes how each of the 27 concept-defense comments is addressed in the current manuscript (payload router, candidate teacher paths) and where (section and page, read from `build/main.aux`). The status column and the Pending/To-confirm notes were removed; anything not yet in the manuscript is marked in the location column.

At the team's request, the claims/theoretical-framework item and the prototype item were removed and so was the defense-date item (24 items remain, renumbered); the not-yet notes on sample inputs and the preliminary experiments were dropped.

Same day: Figure 1.1 now has the synthesized speech as its only output, with the calibration result as a note (`fig_conceptual.tex`, Section 1.6 text).

Checks: `latexmk -pdf revision-matrix.tex` after a `-Werror` build of `main.tex`; no undefined references; 3 pages, visually checked.

## Update later the same day: matrix rebuilt from the minutes and the transcript

- Source: signed minutes (items 21-47) and the concept-presentation transcript in the team's Application Forms document.
- Clarifying questions (what is handover, sample inputs, what the teacher passes), the theoretical-framework/claims item, the prototype item and the defense date are left out at the team's request. 22 items remain.
- Added from the transcript only: Dr. Landicho's remark that the experiment did not seem complex enough (Recording 26, 02:24), marked with a footnote.
- Every row describes the current paper (payload router, per-path calibration, Table 3.7 measures, Appendix A worked example, Appendix B preliminary results).
- Checks: compiled after a `-Werror` build of `main.tex`; no undefined references; 3 pages.
