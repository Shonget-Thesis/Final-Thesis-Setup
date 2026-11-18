# 2026-11-18: Chapter 1 rewritten for the payload router

**Task:** `PAPER-020` (`review`)

## Changes

- `paper/chapters/01-introduction.tex`, Sections 1.1-1.5: rebuilt from the team's latest Chapter 1 draft (pasted by Roxanne) and edited to follow the risk-calibrated payload router of Chapter 3 instead of the per-utterance switching-time scheduler.
  - Background: the gap is now that the handover is fixed in both cost (18 of 21 passes on the 16-step grid) and content (always the teacher's estimate, regardless of prompt condition); related work is framed as adapting *how much* / *when*, not *what* is handed over; Cheng et al. (2025) kept (`cheng2025srdiffusion`); method paragraph describes the commit point (13 passes), the quantile projection, the invariant residual, the SNR gate and LTT calibration of one threshold.
  - Problem statement, three research questions, general and specific objectives now mirror Chapter 3's stages (characterize baseline and payloads; develop and calibrate the router; evaluate and test robustness) and its compared systems/ablations.
  - Significance updated (fixed known cost, robustness to noisy prompts, fallback to the released system).
  - Scope and Limitations: router scope (content-adaptive, fixed cost; t2/t6 commit-point ablations; LibriSpeech-PC test-only; pitch variety measured); keeps the draft's statement that step-pruning methods such as EPSS are not compared.
- Definition of Terms: scheduler terms (adaptive handover policy, candidate switching time, cap, endpoint-stabilization score, oracle) replaced by commit point, payload, payload router, quantile projection, invariant residual, student draft, routing rate, released system, quality failure, prompt condition, pitch variety, mel-spectrogram, vocoder.
- Conceptual Framework (Section 1.6, Figure 1.1) unchanged; theoretical framework still removed.

## Checks Run

- `latexmk -g -Werror main.tex` in `paper/`: exit 0, no Overfull boxes, no unresolved references (`??`) in Chapter 1; Chapter 1 is pages 1-12. A truncated `build/main.out` left by an interrupted build had to be deleted first.
- Chapter 1 extracted to a standalone PDF outside the repo (`D:\Roxanne\Thesis\Chapter1_Introduction.pdf`, 12 pages).

## Still open

- Team and adviser review of the rewritten Chapter 1; Google Doc draft must be synced with this text.

## Addendum (2026-11-19)

- Problem statement, research questions and objectives shortened at Roxanne's request (same content and stage order as Chapter 3); Chapter 1 is now pages 1-10.
- `latexmk -Werror main.tex`: exit 0, no undefined references, no Overfull boxes.
- Tense pass: statements about what this study will do (objectives, problem, significance, scope, conceptual-framework paragraph, definitions) put in future tense; facts about published work, released code and past team experiments left in present/past. `latexmk -Werror`: exit 0, no undefined references, no Overfull boxes.
