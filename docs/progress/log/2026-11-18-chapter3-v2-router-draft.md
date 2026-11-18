# 2026-11-18: Draft Chapter 3 v2 (risk-calibrated payload router)

**Task:** `PAPER-020` (`review`)  
**Target:** `paper/chapters/03-methodology-v2.tex` with figures `fig_method_v2.tex` and `fig_workflow_v2.tex` and preview root `main-ch3v2.tex` (since merged into `03-methodology.tex`, `fig_method.tex` and `fig_workflow.tex`). The current `chapters/03-methodology.tex` and `main.tex` are untouched.

## Decisions taken with Vincent before drafting

- Direction: adopt the `SNR_R` payload router of Appendix E as the method; the stopping-time scheduler moves to Appendix D only (one motivating sentence in Ch3).
- Failure rule (option A): absolute rule applied to every system, DMOSpeech 2 included (WER > 12.5%, SIM < s_min, UTMOS < u_min; floors = 5th percentile of reproduced released DMOSpeech 2 clean-prompt scores on the development split, fixed before calibration).
- Risk: excess failure e = F(router) * (1 - F(released DMOSpeech 2)); LTT bounds R <= alpha, which implies router failure rate <= DMOSpeech 2 failure rate + alpha (Proposition "Failure rate relative to the baseline").
- LTT objective "keep variety, bound failures": largest lambda_r (fewest projections) with certified risk; grid 0.00..1.00 step 0.05 tested from most to least projecting; fallback = released DMOSpeech 2 (risk zero).
- alpha = 0.05, delta = 0.10, sensitivity alpha = 0.10; theta = 20 dB fixed (Appendix D); commit t4 = 0.2929 on the 8-interval grid; j = 2; 13 passes.
- Comparison set: released DMOSpeech 2 (21), paper configuration (30), F5-TTS teacher alone (64), student alone (4); router (13); ablations always/never project, SNR-only, STUR2; hindsight router descriptive.
- Conditions: clean, white 15 dB, white/pink/high-pass 25 dB, reverb (0.9 s), telephone, babble 15 dB; one balanced condition per calibration/test speaker for LTT, all eight synthesized for per-condition reporting.
- TOST margins: WER +1 pt, SIM -0.02, UTMOS -0.10.
- Writing rules: future tense; no result values; every display equation has a "where:" list and \eqentry; objectives appear only in headings and the workflow table.

## Open items (not done here)

- Ch1 §1.3 objectives, scope, conceptual figure, terms; Ch2 additions (quantile mapping / histogram matching, optimal transport, invariance) and citations for DEMAND, pYIN, DTW, bootstrap (marked \citationneeded).
- Appendices still reference `eq:stab` and `sec:drift-plan` (undefined under v2) and describe the scheduler as "the Chapter 3 method".
- Revision matrix items 4, 5, 7, 8, 13, 17, 18, 25 need new wording once v2 replaces the current chapter.

## Checks Run

- `latexmk -pdf main-ch3v2.tex`: exit 0; only undefined references are `eq:stab` and `sec:drift-plan` from the appendices; zero Overfull boxes in Chapter 3 v2; no undefined citations.
- 28 display equations, 28 "where:" lists, 28 List of Equations entries.
- Figures rendered with pdftoppm and inspected.
