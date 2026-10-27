# 2026-10-27: Appendix E removed from the paper and the presentation

User request: remove Appendix E (Candidate Solutions: TAS and SWC) from both the manuscript and the presentation.

## Changed

- paper/chapters/appendices.tex: deleted Appendix E (E.1 TAS, E.2 SWC, E.3 results with Table E.1, E.4 limitations) and its entry in the header comment. Nothing else in the manuscript referred to it. The bibliography entries it used (heng2024diffpath, boll1979spectral) stay in references.bib but are no longer cited, so they do not print.
- presentation: deleted the candidate-solutions appendix slide; the simulation slide is now 19; slide 12 (two signals) no longer mentions TAS and SWC in its text, script or sources. The deck is 19 slides (16 main, 3 appendix). README and STATUS updated.
- The EXP-015 results and the earlier logs that describe TAS and SWC are unchanged; this removes the material from the manuscript, not from the research record.

## Checks run

- latexmk -g -Werror -interaction=nonstopmode in paper/: exit 0, 71 pages, no undefined references.
- python presentation/build.py --sync-md --pdf: 19 slides, 19-page PDF.
- python scripts/check_workspace.py and python -m unittest discover -s code/tests: see the session.
