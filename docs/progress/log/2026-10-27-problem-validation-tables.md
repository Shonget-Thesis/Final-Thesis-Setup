# 2026-10-27: Problem-validation tables in Appendix D

Requested by a team member: show the 500-speaker fixed-switching results as tables, not only in prose.

## Changed

- `paper/chapters/appendices.tex`, Appendix D "Fixed Switching Times": added Table `tab:fixed-fail` (added failures per fixed switching time, clean/noisy/all) and Table `tab:fixed-oracle` (hindsight cheapest safe step per instance and mean passes). The paragraph now cites the tables instead of listing the numbers. All figures copied from `research/experiments/EXP-001-problem-validation-t4/README.md` and `results.json`; no new results, meaning unchanged.

## Checks run

- python scripts/check_workspace.py and python -m unittest discover -s code/tests: see handover message.
- latexmk -Werror in paper/ (MiKTeX 25.12, installed today): passes, no undefined references; Tables D.1 and D.2 render on manuscript page 61.

## Open

- The added-failure constants (c = 12.5%, m_W = 1 point, m_S = 0.02) have no cited source in Chapter 3.
