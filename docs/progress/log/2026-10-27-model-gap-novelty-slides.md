# 2026-10-27: Model outline, literature gap map and novelty slides

Task: PRES-001. User request (Mark Vincent D. Limpahan): an abstract teacher-guided model diagram early in the deck that zooms into inference to show where the contribution sits; a Venn or web diagram of the unaddressed gaps in the related literature; and a separate slide that highlights the novelty and contribution.

## Changes

- `presentation/slides/05-the-model-at-a-glance.html` (new, scene 5, after the sampling-cost scene): an outline of teacher-guided TTS (training by others: teacher F5-TTS, distilled student; inference: teacher steps, handover, student steps, speech). It zooms into the inference box, then into the handover point, where two callouts contrast DMOSpeech 2's one fixed switch (t = 0.25) with this study's per-utterance choice under a calibrated bound. The zoom is a CSS transform driven by two invisible beats (`:has()`); in print the zoom is off and the callouts sit beside the handover.
- `presentation/slides/13-the-gap-map.html` (new, scene 13, before the research-gap scene): a three-set Venn diagram (per-input adaptive decision, calibrated risk bound, teacher-to-student handover). Placement follows the Chapter 2 comparison table: ACT, BranchyNet and Adaptive Guidance are adaptive only; adaptive teacher-student collaboration is adaptive plus handover; CALM and adaptive ASR correction are adaptive plus bound; DMOSpeech 2 and T-Stitch are handover only; Learn-Then-Test is the bound-only tool; the intersections without a method are labeled "none found", and the centre is this study. The slide says the gap holds within the sources examined.
- `presentation/slides/15-novelty-and-contribution.html` (new, scene 15, after the research-gap scene): prior work against this study on when to hand over, how sure we can be and what it costs, with a "Not claimed" line (a proven advantage of the hybrid over its parts, any change to duration prediction, any new model training).
- Renumbered slides 06-22 (`git mv`); `presentation/template.html` chapter keys 2-4 now find the first scene of the chapter instead of fixed indices; `presentation/build.py` computes the script's total minutes (now about 13) instead of hard-coding 11.
- Regenerated `presentation/index.html`, `presentation/SCRIPT.md` and `presentation/proposal-defense-slides.pdf` (22 pages) with `python presentation/build.py --sync-md --pdf`. Updated README scene numbers (the illustrative-number scenes are now 17, 19 and 21; update scenes 16-21 if the manuscript changes).

## Checks run

- Each new slide was rendered in headless Edge at 1600x900 in its animated state and in print state: found and fixed a clipped label and pill at the first zoom stage, a print callout covering the "Inference" label, and flex-split text in the novelty cells.
- `python presentation/build.py --sync-md --pdf`: 22 slides, 22 PDF pages.
- `python scripts/check_workspace.py` and `python -m unittest discover -s code/tests`: results in the commit or pull request description.

## Open

- The team has not reviewed the new slides. Placement on the Venn rests on the Chapter 2 comparison table; the citations are the proposal's and were not re-verified against the primary sources here.
- The Figma copy does not have the three new scenes.
- The slides still follow the proposal text from before the October 25 audit and the EXP-001 pilot (scenes 16-21).
- Only scenes 5, 13 and 15 were inspected after the renumbering; the other 19 were not re-rendered.

## Alignment check against the manuscript (same day)

Compared every slide's numbers, section labels and citations with the compiled paper (TOC, aux labels, Chapters 1-3, Appendices A-E). Matches: speaker splits 100/200/200; candidate costs 13/15/17/21 and savings 38/29/19%; tau 0.0761 and t_h 0.293; the failure rule (12.5%, 1 point, 0.02); alpha 0.05 and delta 0.10; the 15 workflow steps; compared methods; WER and SIM models; Propositions 3.2-3.5 and Equations 3-11, 3-12, 3-15; Appendix A SNR (30.5 and 17.8 dB) and p-values (0.0059, 0.0371, 0.2578); sample sizes 45 (exact) and 461 (Hoeffding); Table B.1 ranges; Table 2.2, Figure 3.2, Figure C.1.

Fixed:
- Scene 17 was labeled Section 3.5.2; the decision signals are now 3.6.2.
- The outline's Chapter III numbers were one off since Research Design became 3.1 (now 3.1-3.2, 3.6, 3.7, 3.8); it now lists 1.6-1.7 (frameworks), research gap and novelty, and the candidates appendix.
- Scene 15 cited Proposition 3.1 (cost) for "never costs more"; now Propositions 3.2 and 3.4.
- Scene 17 and its script now mention Appendix E's two development-stage candidates (TAS and SWC), and say neither is adopted until tested on new speakers.

Not changed, still open:
- No slide covers the theoretical framework (1.6) or conceptual framework (1.7), the Chapter 3 Research Design and its exploratory/confirmatory split, or Appendix E's results. Appendix E reports that the stabilization stopping rule certified no threshold on its 360 development instances; the slides' calibration walk-through (lambda-hat 0.50) stays labeled illustrative.
- Scenes 16-21 still show the Appendix A illustrations, not the T4 development data.
