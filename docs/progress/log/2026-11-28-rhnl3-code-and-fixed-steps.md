# PR #27 code changes taken over; Steps 1–8 table added to Appendix B (2026-11-28)

Task: `PAPER-022` (`review`, owners: Rhenel Jhon R. Sajol and Mark Vincent D. Limpahan).

## Context

PR #27 (`rhnl-3`, commits `bfc5c3e` and `adbd47a`) was opened against `main` before PR #26 merged and conflicts with it in `paper/chapters/appendices.tex`, `paper/revision-matrix.tex` and the progress files. Main's two-appendix version (sample calculations and preliminary experiments) and its 22-item revision matrix are kept. Rhenel's code changes and his log ([appendices reconciliation log](2026-11-28-appendices-router-reconciliation.md)) are taken over; the log describes his appendix version, which is not used.

## What changed

- **From PR #27, unchanged:** the prototype moves from `code/prototype/` to [code/scheduling/prototype/](../../../code/scheduling/prototype/README.md) with the new [SNR_INVARIANT_MATH_GUIDE.md](../../../code/scheduling/prototype/SNR_INVARIANT_MATH_GUIDE.md); updated `cli.py`, `engine.py`, `server.py`, `start_prototype.bat`, `README.md`; both prototype tests; and the three-path [ltt_sim.py](../../../code/analysis/ltt_sim.py). `ltt_sim.py` simulates selecting one path, which the paper no longer claims; it is kept as code only and no appendix cites it.
- **Follow-up fixes for the move:** the 15 offline screens in `code/analysis/` that put `code/prototype` on `sys.path` now use `code/scheduling/prototype`; links in `STATUS.md` and `tasks.json` updated.
- **Appendix B.1:** one paragraph and Table B.2 (`tab:fixed-steps`), all eight fixed switching steps of the released grid on 60 development speakers x 6 conditions (passes, clean UTMOS, capped noisy WER, noisy UTMOS, TOST WER equivalence with Step 8), from Rhenel's Table D.3. Every step copies the prompt's noise (noisy UTMOS 1.28–1.48 against 2.69 for the student alone); only Step 7 (19 passes) is WER-equivalent to Step 8.
- **Revision matrix:** item 1 (identify the candidates) and item 11 (is 0.25 best?) cite Table B.2.

## Checks run

- `latexmk -pdf -Werror` on `main.tex` and `revision-matrix.tex`: exit 0, no undefined references, no overfull boxes in the appendices.
- `python scripts/render_progress.py`, `python scripts/check_workspace.py`, `python -m unittest discover -s code/tests`.
