# 2026-10-25: EXP-013 results and the handover operators notebook (EXP-014)

Tasks: EVAL-005 (done), EVAL-006.

## EXP-013 results imported

The team ran the EXP-013 notebook on a Colab Tesla T4 (thesis `07639ea`, Shonget-DMOSpeech2 `8a7868f`). The report, CSVs, preregistration, results JSON and smoke reproduction are copied byte-for-byte from `DMOSpeech2_Endpoint_Extrapolation_Results.zip` into `research/experiments/EXP-013-endpoint-extrapolation-prelim/results/` with a SHA-256 manifest (added to `code/tests/test_evidence_manifest.py`). The runner in the zip is identical to `07639ea`.

As computed: C1 pass (noisy Step-4 failures 126 → 90, p 0.0005), C2 fail (clean 2 → 25), C3 fail. The extrapolated endpoint was farther from the Step-8 endpoint in every condition, because the endpoint changes direction from step to step. DEH is not adopted.

Exploratory checks after the run (in the EXP-013 README, labelled as such): DEH gated by direction coherence (cos ≥ 0.2) keeps clean failures at 3 and lowers total Step-4 failures from 128 to 95; 96 of 126 noisy Step-4 failures are speaker-similarity failures; plain Steps 5 and 6 still fail 14–31 of 60 noisy prompts at 18–22 dB; any extrapolation along Δ₄ is capped near 0% gain.

## EXP-014 designed

Requested by a team member: novel candidate solutions that keep the title, tested together so the best can be picked. Ten candidates were listed and screened against the existing data; three zero-cost Step-4 operators were kept (Teacher-Coupled Restart, coherence-gated DEH, Speech-Envelope Anchoring).

- `code/colab/run_handover_operators_prelim.py`: smoke (with a reproduction check through the runner's own student loop), run, analyze; preregistered criteria C1–C3 and a selection rule.
- `code/colab/DMOSpeech2_Handover_Operators_Prelim.ipynb`: same structure as EXP-013.
- `code/tests/test_handover_operators.py`: exact recovery of the teacher state from the endpoints, the coherence gate, SEA on a worked example, and an end-to-end analysis on the committed labels with synthetic outcomes.

## Checks run

- python -m unittest discover -s code/tests and python scripts/check_workspace.py: see the final run in this session.
- torch is not installed locally, so the operators were tested on numpy arrays only; the Colab smoke test exercises the torch path before the main run. The EXP-014 notebook was not run on a GPU.
