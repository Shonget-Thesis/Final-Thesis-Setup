# 2026-10-27: Merge main and setup/workspace into rhenelAudit; Appendix E kept hidden

Requested by a team member: bring in the latest main and the teammate's removal of Appendix E, keep Appendix E as a hidden (not printed) part of the paper without deleting its results, and drop the Saving columns from the Appendix D and E tables.

## Changed

- Merged `origin/main` (presentation rebuild) and `origin/setup/workspace` (commit 32cd196 and two later presentation commits: Appendix D citations, restart-level reasoning, endpoint cushioning and Appendix E removed from the manuscript).
- `paper/chapters/appendices.tex`, conflict resolution: Appendix E is not printed, as on setup/workspace. D.4 keeps this branch's compared-methods tables, runtime table and verdict, plus the teammate's two new citations in its limitations (hanley1983zero, cawley2010overfitting); its one reference to Appendix E was removed. The summary table takes the teammate's restart-level row and this branch's stabilization-policy row. Endpoint cushioning stays removed.
- `paper/chapters/appendix-e-hidden.tex` (new, committed before the merge): this branch's Appendix E (TAS, SWC, Table E.1 with the D.4 columns) as a file that `main.tex` does not `\input`. Restoring it is one `\input` line. The EXP-015 data and earlier logs are unchanged.
- Saving columns removed from Tables `tab:fixed-fail`, `tab:fixed-oracle`, `tab:cv-policies` and the hidden `tab:candidates`, with the caption wording that described them. Text percentages are unchanged.

## Checks run

- See the merge commit: `python scripts/check_workspace.py`, `python -m unittest discover -s code/tests`, `latexmk -Werror` in paper/.

## Open

- `presentation/PANEL_QA.md` still points to Appendix E and to the old Appendix D numbering (D.2–D.6 before endpoint cushioning was removed).
