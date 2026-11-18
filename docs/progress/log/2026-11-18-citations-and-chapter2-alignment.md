# 2026-11-18: Chapter 3 citations filled; Chapter 2 aligned with the router chapter

**Task:** `PAPER-020` (`review`)

## Citations (all records checked online on 2026-11-18)

Added to `paper/references.bib`: `thiemann2013demand`, `mauch2014pyin`, `morrison2023pitch` (arXiv, first posted 28 Jan 2023), `yang2025prosody` (arXiv), `sakoe1978dtw`, `efron1979bootstrap`, `liu2020blockwise`, `hall1974histogram` (volume number not shown in the record checked, so omitted), `dharanipragada2000nonlinear` (ICSLP 2000; pages not shown, omitted), `delatorre2005histogram`, `hilger2006quantile`, `peyre2019computational`, `fujita2024noiserobust` (arXiv; ICASSP publication not confirmed), `lu2025noise` (arXiv). `wang2024noise` already existed.

Chapter 3: the five `\citationneeded` markers are replaced (DEMAND, pYIN, DTW, bootstrap, histogram/quantile matching); CVf0 limits stated with `yang2025prosody` and `morrison2023pitch`; the 1-D optimal-transport step of the residual-as-distance proposition cites `peyre2019computational`; speaker-level bootstrap justified with `liu2020blockwise`.

## Chapter 2 alignment

- Switching time is now fixed background; the variable is the payload (2.1, 2.3, Figures 2.1--2.2 captions, 2.5 framing, Table 2.1 "This study" row, synthesis and gap paragraphs).
- Diversity is now measured (2.2); restart at tau = 0.2929 justified (2.3); UTMOS now in the failure rule with a floor (2.7); absolute failure rule applied to DMOSpeech 2 too (2.7); "added failures" replaced by excess failures against the released system.
- New Section 2.4 "Noisy Prompts and Robustness to Recording Conditions": noise-robust zero-shot TTS (Wang et al. 2024; Fujita et al. 2024; Lu et al. 2025) and histogram/quantile equalization (Hall 1974; Dharanipragada & Padmanabhan 2000; de la Torre et al. 2005; Hilger & Ney 2006; Peyré & Cuturi 2019); two new rows in Table 2.2 (now 17 rows: 8 TTS, 3 other speech, 5 other domains, 1 AAC).
- Old stabilization-signal comparisons (Adaptive Guidance, CALM) now refer to the earlier proposal (Appendix D).

## Open

- Team decision: add "DMOSpeech 2 + prompt speech enhancement" as a compared system (suggested by Wang et al. 2024).
- Chapter 1, Appendices D--E wording and the revision matrix still describe the old method.

## Checks Run

- `latexmk -Werror main.tex`: exit 0, 108 pages, zero undefined references/citations, zero Overfull boxes. New Section 2.4 and Table 2.2 rows rendered and inspected.

## Addendum: Table 2.1 categorized update

Vincent supplied a categorized Table 2.1 with four new methods. All four were checked online: SRDiffusion (Cheng et al., arXiv 2505.19151, 2025), TRACK (Munir et al., arXiv 2609.30096, 2026), HybridStitch (Sun et al., arXiv 2603.07815, 2026; current v3 title used), Distillation++ (Park, Lee & Ye, ICCV 2025, pp. 4049--4058; guidance applied in the first one or two steps, Sections 3 and 4.1). Fixes applied: SRDiffusion signal is the change in the rate of change of the denoised sample, and its threshold is no longer labelled delta (clash with the LTT confidence parameter); Adaptive Guidance moved to "Adaptive computation within one model"; "This study" row matches the payload router. Section 2.5 text now discusses the four methods (TRACK and HybridStitch use large--small disagreement, like the residual, without a bound; SRDiffusion resembles the earlier scheduler), and Table 2.2's image row lists them. `latexmk -Werror` exit 0, zero Overfull boxes.

## Addendum 2: Table 2.2 regrouped, new Table 2.3, Chapter 2 trimmed

- Table 2.2 now uses the Table 2.1 grouping format (italic group rows kept with their first row, rules between groups) and is regrouped by the part of the argument each row supports: baseline and cost; cheaper teacher path and restart (new SDEdit/restart row); prompt noise and its removal; per-input computation (Adaptive Guidance moved here); large and small models, teacher and student (split into two rows); calibrated adaptive computation; motivation. ZipVoice row dropped. 18 rows: 7 TTS, 3 other speech, 7 other domains, 1 AAC.
- New Table 2.3 "Sources of the components added by the proposed method" (answers panel item 9, "Where does the method come from?"): coarse teacher path, student draft, restart at the commit point, prompt SNR estimate, quantile projection, invariant residual, routing rule, calibrator, failure rule and excess risk; each with the studies it builds on, what is taken, and what this study changes.
- Removed passages tied only to the old stopping-time scheduler and not requested by the panel: learned-path curvature (Dedek et al.), the early-step guidance and coarse-to-fine quotes in 2.1.2, FlashSpeech/ZipVoice, the step-budget quotes in 2.3, the early-handover restart case, the second UTMOS-proxy quotes in 2.7; the two LTT-assumption paragraphs in 2.6 merged.
- `latexmk -Werror`: exit 0, 110 pages, zero undefined references or citations, zero Overfull boxes; Tables 2.2 and 2.3 rendered and inspected.
