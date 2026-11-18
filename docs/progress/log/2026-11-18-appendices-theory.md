# 2026-11-18: Add conceptual and mathematical foundations to Appendices D and E

**Task:** `PAPER-019` (`review`)  
**Target:** [paper/chapters/appendices.tex](../../../paper/chapters/appendices.tex) (Chapters 1–3 and `frontmatter.tex` remain untouched)

## Changes Applied

1. **Appendix D.3 ([appendices.tex](../../../paper/chapters/appendices.tex)) — Handover Transmissibility Theory (`EXP-026`):**
   - Added the one-dimensional binary-mode re-noising model (`A = 2m^2 \tau^2 / (1-\tau)^2`, `B = 2m\tau / (1-\tau)`, retention `R_k = \Phi((A |\tanh(\ell_k/2)| - \beta)/B)` and ceiling `R^* = \Phi((A-\beta)/B)` from [transmissibility log](2026-11-14-exp026-transmissibility-test.md)).
   - Explains mathematically why changing the student restart from `j=1` (`\tau = 0.0761`) to `j=2` (`\tau = 0.2929`) increases `A/B` by `~5x` (`25x` signal, `5x` noise scaling), explaining both the Step 8 retention gap (`W8` vs `W8R2`/`P25`) and the Step 3–8 plateau at fixed `j`, while noting `EXP-029`'s monotone `CV_f0`–UTMOS trade-off.

2. **Appendix D.5 & D.6 ([appendices.tex](../../../paper/chapters/appendices.tex)) — Drift LTE Identity, Spread Principle & Invariance Synthesis:**
   - Added the exact Euler estimate-change identity `\hat x_1(t_i) - \hat x_1(t_{i-1}) = (1-t_i)(v_i - v_{i-1})` linking Chapter 3's `d_i` to Euler's local truncation error and explaining why weakened CFG contrast under noise causes premature stabilization ([adaptive teacher screen log](2026-11-16-adaptive-teacher-screen.md)).
   - Added the *spread principle* (`EVAL-027`, `\mathrm{SNR}_{\mathrm{sw}} = 10\log_{10}(P_{\text{speech}}/\mathrm{mean}_f N(f)) - 10\log_{10}\mathrm{SFM}_N`, [spread SNR log](2026-11-17-spread-weighted-snr.md)) and linked `EXP-020` (`Voice-Swap`), `EXP-030` (`TOH`), and `EXP-031` (`QH`) to the condition-invariance ladder.

3. **Appendix E.1 ([appendices.tex](../../../paper/chapters/appendices.tex)) — Conceptual Framework & Mathematical Theory (`\label{app:snr-r-theory}`):**
   - Formalized the log-mel stationary condition map `g_b(x) = 0.5 log(h_b e^{2x} + \nu_b)` and nested condition-invariance ladder `G_1 \subset G_2 \subset G_3` (`Table E.1`).
   - Added **Propositions E.1–E.3** with complete proofs matching [handover invariance theory](2026-11-15-handover-invariance-theory.md), [invariant residual](2026-11-17-invariant-residual.md), and [test_invariant_residual.py](../../../code/tests/test_invariant_residual.py):
     - **Proposition E.1:** `G_3`-invariance and rank-vector maximality of `Q(\hat x_1, s_0)`.
     - **Proposition E.2:** Time-rearrangement nullity (`r = 0` iff identical per-band value multisets).
     - **Proposition E.3:** Exact summed 1D 2-Wasserstein energy `\|\hat x_1 - Q\|_M^2 = n_{\mathrm{gen}} \sum_b W_2^2` and sensitivity to positive stationary noise floors (`\|g(\hat x_1)-Q\|_M` strictly grows on the clean matched range while `Q` stays invariant).
   - Added **Remark E.1** bounding theoretical scope (`gauss25`/`pink25`/`hp25` broadband texture vs. non-`G_3` `reverb09` / `telephone` / `babble15`).

## Checks Run

- `latexmk -interaction=nonstopmode -halt-on-error -Werror main.tex` (inside `paper/`): exited `0`, built `paper/build/main.pdf` (`104 pages`, `2,178,976 bytes`), zero new Overfull `\hbox` warnings, zero undefined refs/citations.
- Rendered updated pages via `pdftoppm` (`dpage-088..095`, `page-096..100`, `lot-006`, `te1-098`) and verified typography and List of Tables cleanly.
- `python scripts/render_progress.py`, `python scripts/check_workspace.py`, and full `python -m unittest discover -s code/tests` run after final edits.
