# Revision matrix rewritten with locations instead of status (2026-11-28)

Requested by Mark Limpahan. `paper/revision-matrix.tex` now describes how each of the 27 concept-defense comments is addressed in the current manuscript (payload router, candidate teacher paths) and where (section and page, read from `build/main.aux`). The status column and the Pending/To-confirm notes were removed; anything not yet in the manuscript is marked in the location column.

At the team's request, the claims/theoretical-framework item and the prototype item were removed and so was the defense-date item (24 items remain, renumbered); the not-yet notes on sample inputs and the preliminary experiments were dropped.

Same day: Figure 1.1 now has the synthesized speech as its only output, with the calibration result as a note (`fig_conceptual.tex`, Section 1.6 text).

Checks: `latexmk -pdf revision-matrix.tex` after a `-Werror` build of `main.tex`; no undefined references; 3 pages, visually checked.
