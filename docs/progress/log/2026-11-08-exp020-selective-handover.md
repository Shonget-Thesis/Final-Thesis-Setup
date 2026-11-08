# 2026-11-08: EXP-020 upgraded to Selective Handover

Task: EVAL-013 (todo). Requested by Rhenel Jhon R. Sajol: replace the prepared Voice-Swap test with Selective Handover and have the notebook ready for Colab.

## Why

- Voice-Swap alone is one small change, and the whole contribution would depend on it working.
- Selective Handover asks which parts of the teacher's estimate the student needs. Voice-Swap becomes one of eight combinations, and the map of which part causes what remains a finding even if the swap fails.
- The per-input step choice now covers Steps 4 to 7, which the Step-4-only version did not.
- The team confirmed that training is allowed but chose inference-only at the concept stage, so a trained component was not pursued.

## What changed

- Replaced, before any synthesis: the Voice-Swap runner, notebook, tests and preregistration (`8fb1600f2355604e...`, commit `9554165`). They remain in git history.
- `code/colab/run_selective_handover_prelim.py`: the three-part split, all combinations, resumable quality and pitch-diversity runs, the SNR ladder with calibrated cut-offs, the effect map, analysis, and `freeze`.
- `research/experiments/EXP-020-selective-handover-prelim/`: README and `preregistration.json` (sha256 `7afeef06fabefaa8...`), frozen before any synthesis.
- `code/colab/DMOSpeech2_EXP020_Selective_Handover.ipynb`: smoke test, background full run, progress check, report, zip.
- `code/tests/test_selective_handover.py`: exact reconstruction of the three parts, every combination, routing and calibration, CVf0, preregistration tamper checks, the reproduction gate, the run loops on a fake model, and the analysis on committed labels with synthetic outcomes.

## Decisions in this change

- STT is fixed in advance as the primary combination, so that eight combinations on 60 speakers cannot become a search for the best one.
- The ladder uses one fixed-sequence Learn-Then-Test chain per prespecified rung spacing, with delta split across chains.
- The run uses only the 60 development speakers. Speakers 61-100 stay unused.
- The output check (Verified Voice-Swap) was left out. It adds speech-recognition and speaker-model cost and conflicts most with the objectives, and it can be added later if it is still needed.

Not changed: the manuscript, the presentation, and the thesis method. Adopting Selective Handover would need a dated decision record and adviser review (docs/RESEARCH_DIRECTION.md).

## Checks run

- `python -m unittest discover -s code/tests`: OK, 99 tests, including 26 new tests in `test_selective_handover.py`. The run loops were exercised on numpy with a fake model.
- `python scripts/check_workspace.py`: PASSED.
- `latexmk -Werror` in `paper/`: exit 0. The manuscript was not changed.
- Every code cell of the notebook compiles with its notebook magics removed. The notebook has not been run.
- `check_prereg()` confirms that the frozen preregistration matches the final runner.

## Remaining blockers

- No GPU run has happened, so there are no results. The torch path through `student_sample` and the split has not executed anywhere yet; the notebook's smoke test is its first run.
- Novelty is not established. A full literature review is outstanding, and SRDiffusion, Gandikota and Bau, and the other handover papers are not yet cited in the manuscript.
- About 5 to 6 hours on a T4 is an estimate, not a measurement.
- Appendix E describes 1D-CRC-STAC as using one shared parameter, but `fit_crc_stac` in `code/colab/run_swsr_handover_prelim.py` tunes each step separately on the same data, so the guarantee it states is not established. Not fixed in this change.
