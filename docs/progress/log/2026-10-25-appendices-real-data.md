# 2026-10-25: Appendices A–C based on the T4 development data

Task: PAPER-012. Requested by a team member: replace the illustrative values in the appendices with the team's real results, without changing the design of any figure. Chapters 1 and 2 are not touched; Chapter 4 is not written.

## Changes

All values come from `research/evidence/benchmarks/gpu_grid_all1000.csv`, development split only.

- **Appendix A.** Follows `common_voice_en_13900-common_voice_en_13901` (the first development utterance, in ID order, that the policy at λ = 0.50, θ = 20 dB keeps past Step 4): prompt SNR 30.7 dB (noisy15: 17.7 dB), d₄ = 0.533, d₅ = 0.420, d₆ = 0.335, stop at Step 5, 15 passes. The failure check uses its Step-5 output (no failure) and the noisy15 Step-4 output of `common_voice_en_15734837-common_voice_en_15734838` (WER 37.5%, failure). Table A.2 is fixed-sequence LTT on the 100 development speakers: λ = 0.20–0.50 certified, stop at 0.55 (2 failures, p = 0.1183), λ̂ = 0.50. The frame-energy and endpoint-vector arithmetic stays as labelled worked examples because the T4 grid does not store those inputs; the development pilot (EXP-001 Step 6b) now records them (`worked_example.json`) for a later update.
- **Algorithm 3.3** gained invisible line labels; Appendix A cites them with `\ref`, which corrects "line 7" to line 6 for continuing the teacher.
- **Appendix B.** The stress-test simulation, Figure B.1 and Table B.1 are unchanged (the script reproduces them exactly). Added a paragraph on the same model fitted to the T4 development data (`python code/analysis/ltt_sim.py --fitted`): every threshold has true risk below 1% (at most 0.89%), so no unsafe threshold exists for calibration to avoid; the fitted failure model does not rise with d_k (coefficient −2.5). Script path corrected to `code/analysis/ltt_sim.py`.
- **Appendix C.** Figure C.1 is unchanged in design and in its bars: three real development utterances stop at Steps 4, 6 and 8 under λ = 0.50, θ = 20 dB, exactly the illustrative steps. Sub-labels and caption name them; the figure got a short List-of-Figures title, as other figures have, so the long IDs do not enter the list.

## Checks run

- latexmk -Werror in paper/: exit 0; Overfull/Underfull count 13, equal to the previous commit; no undefined references.
- Rendered and inspected the pages with Appendix A (two pages), Appendix B's new paragraph and Figure C.1.
- python code/analysis/ltt_sim.py: output identical to Table B.1; `--fitted` gives the numbers quoted.
- python scripts/check_workspace.py and the unit tests: see the commit.
