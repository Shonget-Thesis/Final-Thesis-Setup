# 2026-10-25: EXP-012 results and the Drift-Extrapolated Handover notebook (EXP-013)

Tasks: EVAL-004, EVAL-005.

## EXP-012 Quick results imported

The team ran the EXP-012 notebook in Quick mode on a Colab Tesla T4 (thesis `3d0b3ef`, Shonget-DMOSpeech2 `8a7868f`). The report, CSVs, preregistration and results JSON are copied byte-for-byte from `DMOSpeech2_Consistency_Gate_Results.zip` into `research/experiments/EXP-012-consistency-gate-prelim/results/` with a SHA-256 manifest (checked by `code/tests/test_evidence_manifest.py`; `.gitattributes` keeps the bytes). The runner copy in the zip is identical to `3d0b3ef`.

Main results are in the EXP-012 README: 360 of 360 cases; drift reproduced exactly; the 20 dB gate fails 86/360; Definition 3.4 certifies nothing because noise lowers drift; no gate beats an SNR gate with a calibrated threshold. The calibrated-threshold baseline and the SNR + d₄ ≥ λ_low combination were computed offline after the run and are labelled exploratory.

## EXP-013 designed

Requested by a team member: a genuinely new method rather than another gate, with a preliminary Colab test. Drift-Extrapolated Handover (DEH) predicts the teacher's Step-8 endpoint from the contraction rate of its endpoint changes and hands it to the student at Step 4, at no extra network cost. Basis: the student restarts at τ = t₄ = 0.0761, so Step 4 and Step 8 differ only in the endpoint handed over; EXP-012's drift ratios d₅/d₄ are about 0.8–0.98.

- `code/colab/run_endpoint_extrapolation_prelim.py`: smoke (with a reproduction check of the stored Step-4 and Step-8 scores), run, analyze; preregistered criteria C1–C3.
- `code/colab/DMOSpeech2_Endpoint_Extrapolation_Prelim.ipynb`: Steps 1–4 copied from the follow-up notebook; smoke test, background run with resume on Google Drive, progress check, report, zip.
- `code/tests/test_endpoint_extrapolation.py`: the operator is exact on a geometric trajectory and leaves prompt frames unchanged; end-to-end analysis on the committed labels with synthetic outcomes.
- A web search found related extrapolation methods in diffusion sampling (RX-DPM, multistep solvers, feature forecasting) but none that extrapolates the endpoint at a teacher-to-student handover; recorded in the EXP-013 README as "not found", not as proof of absence.

## Checks run

- python -m unittest discover -s code/tests and python scripts/check_workspace.py: see the final run in this session.
- The EXP-013 notebook was not run on a GPU.
