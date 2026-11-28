# Chapter 3: symbols in the text, values in one settings table; plain-language pass (2026-11-28)

Requested by Mark Limpahan: keep Chapter 3 on the theory, with variables rather than values where possible, and recheck the plain-language style (Ms. Eder).

## What changed (`paper/chapters/03-methodology.tex`)

- Chapter 3 contained no experimental results; the remaining numbers were design settings written inline. These are now symbols in the text, equations and tables: split sizes (n_dev, n, n_test, n_LS), seeds and variety speakers (K, n_var), SNR gate theta, threshold grid Lambda, WER ceiling c, floor quantile p, alpha and delta, repeated divisions R, significance level a, equivalence margins, bootstrap resamples B.
- New Section 3.8.5 "Settings of the Study" (Table 3.11, `tab:levels`) gathers every value with its reason, and states why they stay in Chapter 3: the Learn-Then-Test guarantee requires them to be fixed before calibration. The sample-size table (now Table 3.12) moved there. Data-derived values (s_min, u_min, k_hat, lambda_hat) are left to Chapter 4.
- Kept as they are: quantities that follow from the design or the released model (pass counts 10/13/16/21/30/64, commit points, mel size), dataset facts (1,088 utterances, 666 speakers) and the definition of the prompt conditions (Table 3.6).
- Plain language: long sentences in Sections 3.1-3.2 split; negative outcomes and pre-run checks turned into lists; the router's four properties stated as short sentences; "stochastically larger" and "union bound" in the risk-control proof explained in words.

## Checks

- `latexmk -pdf -Werror` clean, no undefined references, no overfull boxes; Table 3.11 checked visually.
