# 2026-10-27: Compact presentation (Chapter I, Chapter III, appendix)

Task: PRES-001. User request: make the deck more compact, focused on Chapter I (validating the scope and problem) and Chapter III (the plan and contribution), with the appendix for preliminary tests and supplementary data. Decisions from the user: keep Chapter II's comparative table and gap map; leave the theoretical and conceptual frameworks out for now; put the appendix slides before the end; merge the summary; no separate thank-you slide, the deck returns to the title; bring back a back button.

## Changed (22 slides to 20: 16 main, 4 appendix)

- Removed: the chain-of-ideas slide, the research-gap requirements slide, the separate summary/thank-you slide; the "promise you can check" slide was folded into calibration, and the flow-matching slide into the background slide.
- Merged or moved: background + cost (now four panels); candidate switching times moved from Chapter II to Chapter III (Section 3.5); novelty and summary merged into "Contribution and Summary" at the end of Chapter III; the simulation slide moved to the appendix.
- Edited: the problem slide now shows the T4 evidence of Appendix D.1 (fixed early switches fail on 25-33% of inputs, noisy 49-64%, clean 1.0-2.4%; per-input choice in hindsight 14.85 of 21 passes); scope states restart j = 1 as a controlled setting; the comparative table now has all seven rows of Table 2.1 (adds T-Stitch, Adaptive Guidance, adaptive ASR correction); calibration states the guarantee and the 45-input minimum; the workflow slide marks the exploratory and confirmatory stages (Section 3.1); the outline follows the new structure.
- Added: appendix slides for preliminary experiments (D.1-D.3, with the restart result worded as a departure from the j = 1 reference), the signal diagnostic (D.4-D.5) and candidate solutions (E, Table E.1). The numbers are copied from the manuscript tables.
- presentation/template.html: back and next buttons at the bottom right (hidden in print), an "Appendix" chapter label, and a prev() function. The left-arrow key had been calling an undefined prev() since the modular rewrite, so going back did not work at all. build.py: the "Appendix" label for SCRIPT.md.

## Checks run

- python presentation/build.py --sync-md --pdf: 20 slides, 20-page PDF.
- Rendered every changed or new slide in headless Edge (print state) and fixed the overflow found: the comparative table ran into the footer, appendix tables were too small, an axis label wrapped.
- Navigation tested through the DevTools protocol: the back button and left key each go back one slide, back from the title goes to the last slide, and forward from the last slide returns to the title.
- python scripts/check_workspace.py and python -m unittest discover -s code/tests: see the session.

## Open

- The team has not reviewed the compact deck. Slides 1, 4, 6, 9, 10, 12 and 13 were not re-rendered after the restructure; they changed only in position.
- Slides 10-15 still use the Appendix A illustrations (lambda-hat 0.50), not the T4 data, and Appendix E reports that the stopping rule certified no threshold on its development instances.
- The Figma copy reflects the original 19-slide version.
