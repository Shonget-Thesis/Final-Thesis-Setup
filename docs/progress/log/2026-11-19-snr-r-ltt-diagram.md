# 2026-11-19: Update `SNR_R` architecture figure with Learn-Then-Test (LTT) risk calibration

Task: `PAPER-021` (`review`). Requested by user (`"updae this too with ltt"`).

## Changes

- Rebuilt the HTML/SVG schematic (`3200 x 1800` RGB PNG, `906,106` bytes, SHA-256 `0782DE05C0D0ABE310039D8BB707ED2E489970B5BBFE4EE4B58062C57C45874C`) and synced it to [paper/figures/exp034_architecture.png](../../../paper/figures/exp034_architecture.png) and [presentation/exp034_architecture.png](../../../presentation/exp034_architecture.png):
  - Added an **Offline LTT Risk Calibration** block feeding $\lambda_r^*$ directly into the `SNR_R` router diamond (`\hat S >= 20 dB and r <= lambda_r^*`, `0` extra runtime passes), noting `EXP-034` binomial LTT calibration ($\alpha \le 15\%$, $\delta = 0.10$, $\lambda_r^* = 0.342$, $p = 0.048$; strict $\alpha \le 10\%$: $\lambda_r^* = 0.382$, $p = 0.042$).
  - Updated the middle summary card to state `"What"` payload adaptation at `8`-grid `Step 4` ($t = 0.2929$, `13` passes) under bounded LTT clean-intervention risk.
  - Updated the bottom-right evidence card and footer to summarize both `EXP-034` (`dev 31–60`, `AUC 0.864` / `0.897`, `SNR_R` `2.376 -> 2.950`, `snr_r_better`) and `EXP-035` (`cal 30–60` disjoint unseen speakers from `DMOSpeech2_EXP035_036_AllInOne_Results.zip`: `AUC 0.901` / `0.957`, `SNR_R` `2.201 -> 2.889`, `90/90` spectral caught, `10.0%` clean flags / `3.3%` at `0.382`).
- Updated [`paper/figures/fig_snr_r.tex`](../../../paper/figures/fig_snr_r.tex) caption to match the LTT block and completed `EXP-034` / `EXP-035` summary in the graphic. Chapters 1–3 and Appendix text remain unchanged.

## Checks Run

- Rendered and visually inspected `3200 x 1800` PNG via Chromium headless + image inspection.
- Built `paper/build/main.pdf` via `latexmk -interaction=nonstopmode -halt-on-error -Werror main.tex` inside `paper/` (`108` pages, `2,213,689` bytes, exit `0`, Figure E.1 embedded cleanly against rebased `origin/rhnl-3`).
- Ran `python scripts/render_progress.py`, `python scripts/check_workspace.py`, `python scripts/check_workspace.py --base origin/main`, and `python -m unittest discover -s code/tests`.
