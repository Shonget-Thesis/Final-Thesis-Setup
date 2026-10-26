# 2026-10-25: Notebook for the proposed gates (EXP-012)

Task: EVAL-004. Requested by a team member: a Colab notebook, in the same style as the others, to test the proposed solutions against SNR-only and SNR + drift before choosing one for the proposed-solution appendix.

## Added

- `code/colab/run_consistency_gate_prelim.py`: reuses the EXP-011 pipeline (same sampler, metrics, seeds and degradations); synthesizes only the missing development labels; records prompt SNR, drift and guidance contrast with a teacher-only pass; compares seven policies with pooled and stratified (intersection-union) Learn-Then-Test under 5-fold speaker cross-validation; every reported number is computed.
- `code/colab/DMOSpeech2_Consistency_Gate_Prelim.ipynb`: Steps 1-5 copied from the follow-up notebook; smoke test, background run with resume on Google Drive, progress check, report and zip.
- `code/tests/test_consistency_gate.py`: gate definitions, the SDC fit, stratified vs pooled certification, and an end-to-end analysis on the committed labels with synthetic guidance values.
- research/experiments/EXP-012-consistency-gate-prelim/README.md and an index row.

## Quick mode (added the same day)

Requested by a team member because the full run takes 3-4 hours. `--quick` runs the first 60 development speakers by utterance ID (including the 20 EXP-011 pilot speakers) under clean, noisy15 and Gaussian 18/20/22/25 dB: 160 new synthesized cases and 360 teacher-only cases, about 75 minutes on a T4 (estimate). The notebook's Step 6 sets `QUICK = True` (default) or `False` for the full run; the choice is saved on Drive and each mode has its own results folder. `preregistration.json` now records the speaker count and conditions, and the analysis is restricted to them. A new test checks that a quick analysis expects 360 cases and that a restart with different settings is refused.

## Checks run

- python -m unittest discover -s code/tests and python scripts/check_workspace.py: see the final run in this session.
- The notebook was not run on a GPU. Development speakers only; no held-out data is used.
