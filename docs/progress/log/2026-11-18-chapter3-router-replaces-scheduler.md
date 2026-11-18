# 2026-11-18: Chapter 3 router draft replaces the original chapter; plain-language pass

**Task:** `PAPER-020` (`review`)

## Changes

- `paper/chapters/03-methodology.tex` now holds the payload-router chapter (formerly `03-methodology-v2.tex`); `fig_method.tex` and `fig_workflow.tex` now hold the router architecture and workflow figures. The `-v2` files and `main-ch3v2.tex` were removed. The concept-defense chapter remains in git history (commit 9bfa0af and earlier).
- Revisions requested by Vincent: change tags and the methodology summary removed; frozen-sampler equations (guided velocity, Euler step, handover, student step) folded into prose and Algorithm 3.1; the empty comparison matrix replaced by Table 3.11 (comparison matrix: measures reported for every system of Table 3.10).
- Plain-language pass (Ms. Eder's comment that unfamiliar readers should follow the paper): teacher, student, prompt, payload, mel-spectrogram, forward pass, flow steps, guidance, SNR, WER/SIM/UTMOS, embedding, splits, Learn-Then-Test, p-value, Holm, McNemar, Wilcoxon, equivalence test, bootstrap and Wasserstein distance are explained on first use; each proposition group has a short plain restatement. Formal definitions, equations and proofs are unchanged.
- References outside Chapter 3 updated: Chapter 2 and Appendix D point to Algorithm 3.1 instead of the removed equations; the endpoint-stabilization score (`eq:stab`) is now defined at the start of Appendix D; `sec:drift-plan` references point to Appendix `app:drift-plan`.

## Still open

- Chapter 1 objectives/scope/conceptual figure/terms; Chapter 2 additions; Appendix D/E wording that still calls the scheduler "the Chapter 3 method"; revision-matrix wording; five citations marked \citationneeded (DEMAND, pYIN, DTW, bootstrap, quantile mapping / histogram matching).

## Checks Run

- `latexmk -Werror main.tex` in `paper/`: exit 0, 106 pages, no undefined references or citations, no Overfull boxes in Chapter 3.
