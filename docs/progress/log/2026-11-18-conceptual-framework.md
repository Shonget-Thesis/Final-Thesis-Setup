# 2026-11-18: Conceptual framework redrawn abstractly

**Task:** `PAPER-020` (`review`)

- `paper/figures/fig_conceptual.tex`: replaced the architecture-style figure (a near-duplicate of Figure 3.1) with an input--process--output view. The released teacher--student system is one dashed band; the contribution (handover decision and one-time calibration) is highlighted; outputs are speech at a fixed lower cost, a stated limit on extra failures, and a reusable threshold.
- `paper/chapters/01-introduction.tex`, Section 1.7: description shortened to one paragraph that points to Section 3.2 for detail; Table 1.1 (variables) rewritten for the payload router.
- Theoretical framework (Section 1.6, Figure 1.1) not yet changed; Dr. Landicho's comments (item 15; audit note "ground the theoretical framework in TTS; use existing theory, not a new framework"; orientation note on CS principles) are being discussed with Vincent first.

## Checks Run

- `latexmk -Werror main.tex`: exit 0, zero Overfull boxes, zero undefined references; Figure 1.2 rendered and inspected.

## Addendum (Vincent's request, 22:18)

- Theoretical framework (Section 1.6, claims, Figure 1.1, `fig_theoretical.tex`) removed until the Chapter 1 rewrite; the conceptual framework is now Section 1.6 and Figure 1.1.
- Figure: INPUT/PROCESS/OUTPUT headers and the legend removed; caption is "Conceptual Framework of the Study".
- Variables table (Table 1.1) removed.
- Revision matrix item 15 now points to the conceptual framework and marks the claims and theoretical framework as pending.
- `latexmk -Werror main.tex` and `revision-matrix.tex`: exit 0, zero Overfull boxes, zero undefined references.
