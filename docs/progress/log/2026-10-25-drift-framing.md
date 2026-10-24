# 2026-10-25: Stabilization signal framed as a hypothesis to be tested

Task: PAPER-010. Team decisions relayed by Mark Vincent D. Limpahan:

1. Drift (the endpoint-stabilization score) is not removed or hidden. The proposal presents it as a signal still to be tested, states openly that it may not perform on par with the prompt SNR, and describes the post-proposal investigation.
2. Hardware: all synthesis, scoring and timing run on an NVIDIA Tesla T4 GPU through Google Colab, as in the latest Shonget-DMOSpeech2 report; the Ryzen 7 laptop is no longer used (an earlier instruction in the same conversation was reversed).
3. The student restart level stays fixed at the released code's default (j = 1), as the manuscript already states.

## Changes

- Hardware (Chapters 1 and 3): the Ryzen 7 laptop is replaced by an NVIDIA Tesla T4 GPU through Google Colab in the background, scope, measures table and tools section; the tools section now states the timing protocol (full-pipeline boundary, GPU synchronization, five warm-up utterances, balanced policy order) and that each Colab session's GPU and software versions are recorded.

- Chapter 1: Objective 4 now compares policy variants that use the prompt SNR, the stabilization score or both, all calibrated the same way, and asks whether the score adds value beyond the SNR. The "adaptive handover policy" definition names either or both signals.
- Chapter 2: the endpoint paragraph and the adaptive-methods table describe the stabilization score as one of two signals to be compared.
- Chapter 3: Decision Signals explains what each signal can and cannot see and calls the score's added value an open question; the policy definition notes that the SNR-only and stabilization-only rules are special cases (lambda = infinity, theta = minus infinity); Objective 4 adds the "SNR gate, fixed later step" control; new Section 3.7.5 sets out the post-proposal plan (discrimination on development data, controls including a cost-matched random rule, pre-specified tests on unused speakers under reverberant and telephone-band prompts, richer zero-cost variants explored on development data only) and states that the SNR-gated variant is recommended if the score adds no value. "Published level" corrected to "the released code's default level".
- Appendix D: new subsection with the development-split diagnostic from Shonget-DMOSpeech2 (`Handover Scheduling/reports/Ref_reports/DEVELOPMENT_SET_DRIFT_VALIDATION_REPORT.md`, commit 8a7868f): SNR separation of the two conditions, clean and noisy early-handover failure rates, d4 AUC about 0.47, 75-77 of 78 waits unnecessary. Only development-split figures are used, because the test split must stay locked for the proposal's held-out claims.

## Checks run

- latexmk -Werror in paper/: exit 0; main.pdf 68 pages, no overfull lines above 3 pt.
- python scripts/check_workspace.py: passed.
- Rendered the page holding Section 3.7.5 and inspected it.

## Open

- Adviser review of the new framing.
- Valid runtimes come only from the full-pipeline T4 timing run; the grid CSVs' runtime columns exclude the teacher and must not be cited.

## CI fix (same branch)

The paper job had failed on every run since the proposal content landed (main at 97ec687 and this branch): the algorithm environments need `algorithm.sty` and `algpseudocode.sty` from texlive-science, which the workflow did not install. Mapping every file in build/main.fls to its Debian package showed texlive-science as the only missing package. The workflow now installs texlive-science and cm-super (outline versions of the T1 sans and typewriter fonts used by draft notes and code names); docs/SETUP.md lists both.
