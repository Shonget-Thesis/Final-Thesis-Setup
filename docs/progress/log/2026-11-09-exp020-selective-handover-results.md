# 2026-11-09: EXP-020 Selective Handover results

Task: EVAL-013 (review). Requested by Rhenel Jhon R. Sajol, who ran the notebook on Colab and supplied the results zip (`4d9f9378e8fb90dc...`).

## Outcome

**Selective Handover does not advance under its preregistration.** C1 to C4 pass; C5 and C6 fail. As preregistered, it is reported as a negative result with no retuning on these data, and speakers 61-100 stay unused.

| Criterion | Measured | Result |
|---|---|---|
| C1 noisy failures | `STT_4` 74 of 300 against `plain_4` 126; p = 2.3e-6 | Pass |
| C2 clean failures | 2 against 2 | Pass |
| C3 voice effect on noisy similarity | +0.0752 [+0.0635, +0.0874] | Pass |
| C4 diversity | CVf0 against student +0.0383 [+0.0240, +0.0555]; against `plain_4` -0.0044 [-0.0144, +0.0018] | Pass |
| C5 beats the calibrated SNR rule | `ladder` 19.68 passes, 9 failures; `snr_theta` 18.71 passes, 7 failures | Fail |
| C6 beats the ladder without selection | `ladder` 19.68 passes, 9 failures; `ladder_plain` 19.27 passes, 8 failures | Fail |

The effect map is reported either way. The voice part carries the prompt's noise, the slow part carries the pitch diversity, and the fine part has little effect.

## What changed

- `research/experiments/EXP-020-selective-handover-prelim/results/`: the report, CSV files, results JSON, run info, preregistration copy and smoke checks, byte-identical to the zip, with `MANIFEST.json`. The zip's `code/` snapshot was not copied because its six modules match `code/colab/` once line endings are normalised.
- `.gitattributes`: the results folder keeps its original bytes, as for earlier experiments.
- `code/analysis/exp020_diagnostics.py`: recomputes failure counts without importing the runner and prints the follow-up comparisons.
- EXP-020 README, `research/experiments/README.md`, `docs/progress/STATUS.md`, `tasks.json` and the rendered `TASKS.md`.

Not changed: the manuscript, the presentation, the runner, the preregistration, and the thesis method.

## Diagnostics, computed after the results

These explain the verdict and cannot change it. Details are in the EXP-020 README.

- At 25 dB or more the swap has no failure advantage over plain handover. Below 25 dB it reduces failures, but the rate stays near 20% or above, so no noisy rung can be certified at 5%.
- The failure rule's floor on noisy prompts is about a quarter (78 and 81 of 300 between two equally good references, EXP-017). This was known before the design, so C5 and C6 had little chance on noisy prompts. That is a weakness of the preregistration, not a reason to change its verdict.
- The swap removes voice failures (96 to 34 on the voice arm) and few word failures (47 to 43).
- Student only is still better on noisy prompts: 51 failures against 74 (p = 0.004) at 4 passes against 14.
- Steps 5 to 7 add little once the voice is swapped (noisy failures 74, 73, 61, 61; no detectable change in similarity or WER).

## Decisions in this change

- The verdict was not revised, and no threshold, rung or criterion was changed after the results were seen.
- Whether to preregister a follow-up, and whether Selective Handover has any place in the thesis, is left to the team and the adviser.

## Checks run

- Results provenance: runner identical to the committed file (`b61ef06b9ed0a741`), all six modules in the zip identical to `code/colab/` once line endings are normalised, committed preregistration blob identical to the run's copy, `RUN_INFO.txt` shows Tesla T4 and thesis commit `07fe593`, and all 360 rows report Tesla T4.
- Smoke test: 8 of 8 stored labels reproduced, largest difference about 1e-7.
- Re-analysis: `analyze` on a copy of the CSV files gave a results JSON and report identical to the Colab output.
- Independent recomputation: `code/analysis/exp020_diagnostics.py`, which does not import the runner, gives the same failure counts for every output and condition.
- Data checks: 360 unique cases, 60 speakers, 6 conditions, 20 diversity pairs at 20 seeds, no invalid metric values and no NaN CVf0 cells.
- `python -m unittest discover -s code/tests`: OK, 99 tests.
- `python scripts/check_workspace.py`: PASSED.
- `latexmk -Werror` in `paper/`: exit 0. The manuscript was not changed.

## Remaining blockers

- Development evidence on inspected speakers with white-noise prompts only. No fresh-speaker or real-degradation test exists.
- CVf0 was measured only on clean prompts, so pitch diversity on noisy prompts, where the swap matters, is unknown.
- Costs are forward-pass counts. The wall-clock time of each combination was not measured separately.
- Novelty is not established. A full literature review is outstanding, and adoption would need a dated decision record and adviser review (docs/RESEARCH_DIRECTION.md).
- Appendix E's 1D-CRC-STAC mismatch with `fit_crc_stac` remains unfixed.
