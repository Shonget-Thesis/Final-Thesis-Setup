# Short figure and table titles (2026-11-28)

Requested by Mark Limpahan: figure and table titles of one or two lines.

- Every caption longer than about two lines was cut to a short title; the explanatory text moved into a "Note." paragraph under the figure or table, so no information was removed. Affected: Figures 2.1, 2.2, 3.1, E.1, E.2 and Tables 3.3, A.2, B.1, C.1, D.1-D.9, E.1-E.5 (and Figure B.1). Tables 2.1-2.3, 3.9 and 3.10 also shortened to plain titles.
- Trailing periods removed from all caption titles for consistency.
- The laptop's `paper/chapters/appendices.tex`, `paper/figures/fig_snr_r.tex`, `paper/figures/fig_baseline_handover.tex`, `baseline_handover_architectures.png` and `exp034_architecture.png` (from the rhnl-3 merge, commit 9379b45 on the laptop) were brought into this branch before editing, so the team's appendix additions are kept.
- Checks: `latexmk -pdf -Werror` clean (105 pages); lists of tables and figures checked.
