# 2026-11-08: EXP-020 Voice-Swap Handover, prepared for Colab

Task: EVAL-013 (todo). Requested by Rhenel Jhon R. Sajol after a review of whether STAC and GPC are novel: build a candidate handover algorithm, with no new training, and have it ready for Colab.

## Why

- Every part of the Chapter 3 method already exists. SRDiffusion (Cheng et al., 2025, video) already switches models per generation when the estimate settles, using almost the same policy structure as Definition 3.4.
- A read-only replay of SRDiffusion's switch rule on the 360 committed EXP-018 development cases performed worse than giving the same step mix to random inputs (59 failures against 46.5 on average at 18.53 passes, δ = 0.05), and far worse than the calibrated SNR gate (7 failures at 18.71 passes from EXP-014). The replay used this repository's L2 drift, not SRDiffusion's L1, and is not saved in the repository.
- In the EXP-018 ablation, STAC's cumulative sum adds almost nothing on the 16-interval grid (16.82 passes with it, 16.91 without). On the EXP-011 degraded prompts, re-scoring with the pilot's own failure labels gives STAC 8 of 100 failures against 3 for the SNR cascade; the appendix table reports WER only, and that replay had drift only for Steps 4 to 6.
- Existing handover and switching methods found so far pass the whole state across, so what crosses the handover is the part left open.

## What was added

- `code/colab/run_voice_swap_prelim.py`: the swap (`voice_swap`), the reversed control (`reverse_swap`), resumable quality and pitch-diversity runs, the three-way SNR route with two calibrated cut-offs, analysis, and `freeze`.
- `research/experiments/EXP-020-voice-swap-handover-prelim/`: README and `preregistration.json` (sha256 `8fb1600f2355604e...`), frozen before any synthesis.
- `code/colab/DMOSpeech2_EXP020_Voice_Swap_Handover.ipynb`: smoke test, background full run, progress check, report, zip.
- `code/tests/test_voice_swap.py`: the swap's invariants, routing and calibration, CVf0, preregistration tamper checks, the reproduction gate, the run loops on a fake model, and the analysis on committed labels with synthetic outcomes.

## Decisions in this change

- No model is trained; the released teacher and student are used unchanged.
- An earlier candidate from the same day, Student-Completed Handover, was prepared but never run or committed, and was removed. The student's restart is mostly fresh noise, so filled-in detail is largely washed out, and it kept the teacher's strongest components, where a noisy prompt's damage can stay.
- The two SNR cut-offs use one fixed-sequence Learn-Then-Test chain per prespecified theta_hi, with delta split across chains. Choosing the second cut-off after seeing the first on the same data would not preserve the guarantee.
- The run uses only the 60 development speakers. Speakers 61-100 stay unused, so EXP-019's confirmation set is not spent.
- The runner refuses to start if it or the preregistration changed since the freeze, if the GPU is not a T4, or if the smoke test did not reproduce the stored labels.

Not changed: the manuscript, the presentation, and the thesis method. Adopting Voice-Swap Handover would need a dated decision record and adviser review (docs/RESEARCH_DIRECTION.md).

## Checks run

- `python -m unittest discover -s code/tests`: OK, 94 tests, including 21 new tests in `test_voice_swap.py`. The run loops were exercised on numpy with a fake model.
- `python scripts/check_workspace.py`: PASSED.
- `latexmk -Werror` in `paper/`: exit 0.
- Every code cell of the notebook compiles with its notebook magics removed. The notebook has not been run.
- `check_prereg()` confirms that the frozen preregistration matches the final runner.

## Remaining blockers

- No GPU run has happened, so there are no results. The torch path through `student_sample` and the operators has not executed anywhere yet; the notebook's smoke test is its first run.
- Novelty is not established. A full literature review is outstanding, and SRDiffusion, Gandikota and Bau, and the other handover papers are not yet cited in the manuscript.
- About 2.5 hours on a T4 is an estimate, not a measurement.
