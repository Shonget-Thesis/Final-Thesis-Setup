# 2026-11-18: Add Appendix E (`SNR + Invariant Residual Handover`) to the manuscript

**Task:** `PAPER-018` (`review`)  
**Targets:** [paper/chapters/appendices.tex](../../../paper/chapters/appendices.tex), [paper/figures/fig_snr_r.tex](../../../paper/figures/fig_snr_r.tex), [paper/figures/exp034_architecture.png](../../../paper/figures/exp034_architecture.png), [paper/main.tex](../../../paper/main.tex) (Chapters 1–3 remain untouched)

## Changes Applied

1. **Architecture Figure (`Figure E.1`, [fig_snr_r.tex](../../../paper/figures/fig_snr_r.tex) + [exp034_architecture.png](../../../paper/figures/exp034_architecture.png)):**
   - Placed the `EXP-034` architecture graphic (`3200x1800` PNG, `888,975 bytes`) into `paper/figures/exp034_architecture.png`, updating only the bottom-right `EVIDENCE & STATUS (EVAL-029)` card text from `"T4 run pending"` (CPU screen) to the completed Colab T4 outcome (`AUC 0.864`, `SNR_R 2.950` vs. `SNR-only 2.376` and `SNR+drift 2.519`, `Verdict snr_r_better`, `H1–H5` pass, clean variety `105%/96%`, Chapter 3 unchanged).
   - Embedded via [fig_snr_r.tex](../../../paper/figures/fig_snr_r.tex) as `Figure E.1`, contrasting Chapter 3's variable-step `j=1` loop (`13`–`21` passes) with `EXP-034`'s fixed commit point (`t_4 = 0.2929` on the 8-interval grid, `10` teacher passes), 1-pass student draft `s_0`, per-band quantile projector `Q(\hat x_1, s_0)` and invariant residual `r` (`0` extra passes beyond `s_0`), routing decision `\hat S >= \theta and r <= \lambda_r` between `CPH8` and `CPQH`, and 2-pass `j=2` student completion (`13` passes total, `0.44x P25`).

2. **Appendix E (`SNR and Invariant Residual Handover`, `\label{app:snr-r}`, [appendices.tex](../../../paper/chapters/appendices.tex)):**
   - Added epistemic notice stating `EXP-034` (`EVAL-029`, Amendment 1 freeze `eb1a38d3225a...`) is Colab Tesla T4 development evidence on `positions 31–60` (`41–60` fresh to T4 quality runs; Chapters 1–3 unchanged; pending held-out LTT confirmation, listening checks, and team/adviser approval).
   - **E.1 Architecture and Invariant Residual Formulation:** Formalized `Q(\hat x_1, s_0)` and `r = \|\hat x_1 - Q(\hat x_1, s_0)\|_M / (\|\hat x_1\|_M + \epsilon)` (Wasserstein-2 connection and invariance properties), compared rules (`SNR_ONLY`, `SNR_DRIFT`, `SNR_R`, `R_ONLY`, `SW_SNR`, `ORACLE`), and integrated Figure E.1.
   - **E.2 Protocol, Pre-Run Checks, and Preregistered Verdict (`Table E.1`):** Summarized the three T4 stages (`240/240` quality, `40/40` diversity, `10/10` timing), `0.0` smoke identities/reproductions, DEMAND SHA-256 check, paired bootstrap contrasts (`SNR_R - SNR_ONLY +0.574` UTMOS / `+0.0156` SIM; `SNR_R - SNR_DRIFT +0.431` UTMOS / `+0.0091` SIM; fresh `41–60` `+0.668` / `+0.501` UTMOS), detection AUCs (`r` `0.864` / `0.897` vs. `drift8` `0.697` / `0.645` and `SNR` `0.538` / `0.751`), wrong clean flags (`0.067 <= 0.15`), clean variety (`105%` / `96%`), and `31–45 -> 46–60` split recalibration (`2.947` vs. `2.723` vs. `2.290` UTMOS). All hypotheses `H1–H5` pass (`verdict = snr_r_better`).
   - **E.3 Complete Quality and Cost Comparison (`Tables E.2`–`E.3`):** Tabulated overall (`n = 240`), fresh unseen (`n = 160`), `noisy15` runaways, pass budgets, and T4 runtime (`n = 10` clean) across all policies and reference arms (`STU`, `STUR2`, `CPH8`, `CPQH`, `W8`, `P25`), plus all `8` conditions (`clean`, `noisy15`, `gauss25`, `pink25`, `hp25`, `reverb09`, `telephone`, `babble15`), highlighting the `25 dB` hiss/pink/hp mechanism.
   - **E.4 Multi-Seed Prosodic Diversity (`Table E.4`):** Compared all `9` diversity arms and policies (`STU`, `STUR2`, `CPH8`, `CPQH`, `P25`, `SNR_ONLY`, `SNR_DRIFT`, `SNR_R`, `R_ONLY`) across `clean` and `gauss25` (`20` speakers × `12` seeds), showing `SNR_R` keeps `105%` (`CV_f0`) / `96%` (aligned) of `CPH8`'s clean gain vs. `87%` for unconditional `CPQH` while restoring `0.63` voiced frames (`19/20` speakers `>= 0.30`) on `gauss25` where `CPH8`, `P25`, and `SNR_ONLY` collapse to `0.11–0.12` (`3/20` speakers).
   - **E.5 Alternative Baselines, Scope Limits, and Next Steps:** Reported post-hoc caveats (`SNR_R` vs. `R_ONLY`, `SNR_R` vs. unconditional `CPQH`, teacher-free `STUR2` leading UTMOS `3.114` / WER `0.0310` / `0` runaways when clean pitch variety is not required, and unsolved `reverb09`/`telephone`).
   - Updated Appendix D.6 (`Table D.10`) and `paper/main.tex` header comment to cross-reference Appendix E.

## Checks Run

- `latexmk -interaction=nonstopmode -halt-on-error -Werror main.tex` (inside `paper/`): exited `0`, built `paper/build/main.pdf` (`100 pages`, `2,154,767 bytes`) with zero undefined references, zero missing citations, and zero overfull boxes in Appendix E / `fig_snr_r.tex`.
- Rendered Appendix E pages with `pdftoppm` and visually verified `Figure E.1` and `Tables E.1–E.4`.
- `python scripts/render_progress.py`, `python scripts/check_workspace.py`, and `python -m unittest discover -s code/tests` run after final edits.
