# 2026-10-26: EXP-014 results and the noise guidance notebook (EXP-015)

Tasks: EVAL-006 (done), EVAL-007.

## EXP-014 results imported

The team ran the EXP-014 notebook on a Colab Tesla T4 (thesis `8614bf7`, Shonget-DMOSpeech2 `8a7868f`). The report, CSVs, preregistration, results JSON and smoke reproduction are copied byte-for-byte from `DMOSpeech2_Handover_Operators_Results.zip` into `research/experiments/EXP-014-handover-operators-prelim/results/` with a SHA-256 manifest (added to `code/tests/test_evidence_manifest.py`). The runner in the zip matches `8614bf7`.

As computed: no operator passes C3. cDEH passes C1 and C2 (noisy failures 126 → 92, clean 2 → 3), exactly as predicted from EXP-013 before the run; SEA passes C2 only; TCR fails all three. An exploratory comparison after the run (in the EXP-014 README, labelled as such) found cDEH's average speaker similarity on noisy prompts higher than the Step-8 reference (Wilcoxon p 0.0007) at 13 instead of 21 NFE.

## EXP-015 designed

A team member asked for new solutions, allowing a change of title and objectives. Self-Noise Negative Guidance (SNNG; η = 0.5 and 1.0) and an antithetic student pair are tested together. Whether to change the title and objectives is a team and adviser decision, not made here.

- `code/colab/run_noise_guidance_prelim.py`: smoke (SNNG with η = 0 must reproduce the plain endpoint; stored Step-4 and Step-8 scores re-scored), run, analyze; preregistered criteria C1–C3, a secondary average-quality comparison with Step 8, and a selection rule.
- `code/colab/DMOSpeech2_Noise_Guidance_Prelim.ipynb`: same structure as EXP-014.
- `code/tests/test_noise_guidance.py`: noise-only prompt construction, the antithetic choice, SNR routing, and an end-to-end analysis on the committed labels with synthetic outcomes.
- Web search: the nearest prior work is a trained triple guidance with a noise condition (arXiv 2608.03011); no training-free version using the prompt's own quiet frames was found.

## Checks run

- python -m unittest discover -s code/tests and python scripts/check_workspace.py: see the final run in this session.
- torch is not installed locally; the operators were tested on numpy arrays. The Colab smoke test exercises the torch path. The EXP-015 notebook was not run on a GPU.
