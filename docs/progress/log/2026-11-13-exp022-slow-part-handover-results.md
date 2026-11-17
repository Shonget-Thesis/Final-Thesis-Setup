# 2026-11-13: EXP-022 Slow-Part Handover results

Task: EVAL-015 (review). Requested by Rhenel Jhon R. Sajol, who ran the notebook on Colab and supplied the results zip (`290a0f077b63c8c7...`).

## Outcome

**Slow-Part Handover does not advance under its preregistration.** C2 and C4 pass; C1 and C3 fail. As preregistered, it is reported as a negative result with no retuning on these data.

| Criterion | Measured (out of fold, 95% speaker interval) | Result |
|---|---|---|
| C1 cheaper than Voice-Swap | 14.19 mean passes (clean 15.13, noisy 14.00) | Fail |
| C2 quality | WER +0.0031 [-0.0129, +0.0177]; similarity +0.0523 [+0.0404, +0.0651] against Step 8 | Pass |
| C3 pitch variety at Step 3 | clean -0.0025 [-0.0152, +0.0072]; noisy15 not computable | Fail |
| C4 the slow part beats the whole estimate | 14.189 against 14.194 mean passes | Pass |

**How C3 and C1 failed.**
- **The reference cannot be tracked:** on noisy15 prompts, the reference's pitch had no frame voiced in at least 10 of 20 seeds, for any of the 20 speakers.
- **The frozen rule:** it treats the missing lower bound as a failure, so Steps 2 and 3 were dropped before calibration.
- **The consequence:** the cheapest stop cost 14 passes, so C1 (below 14) could not pass.
- **Why this is a design weakness:** EXP-020 measured pitch variety on clean prompts only, and noisy15 was added without checking that the reference's pitch could be tracked there. It is not a reason to change the verdict.

**C4 and the rule's behaviour.**
- **C4:** the pass is 2 passes across 360 cases (5,108 against 5,110); the two rules routed only 9 clean inputs differently.
- **The quality limit never bound** in any fold, so the rule stopped every noisy input at Step 4.

## What changed

- **`research/experiments/EXP-022-slow-part-handover-prelim/results/`:**
  - **Contents:** the report, CSV files, results JSON, run info, preregistration copy and smoke checks, byte-identical to the zip, with `MANIFEST.json`.
  - **Not copied:** the zip's `code/` snapshot, because its seven modules match `code/colab/` once line endings are normalised.
  - **Bytes:** `.gitattributes` already keeps this folder's bytes unchanged.
- **`code/analysis/exp022_diagnostics.py`:** recounts failures and pitch tracking from the raw CSV files without the runner, and reuses the runner's calibration unchanged for the exploratory analyses below. It only reads committed files.
- **Documentation:**
  - EXP-022 README: Results and Diagnostics sections.
  - EXP-021 README: a warning not to run it as frozen, since replaced by the amendment note (see the [amendment log](2026-11-13-exp021-p3-amendment.md)).
  - `research/experiments/README.md`, `docs/progress/STATUS.md`, `tasks.json` and the rendered `TASKS.md`.

Not changed: the manuscript, the presentation, the runners, the preregistrations, the notebooks, and the thesis method.

## Diagnostics, computed after the results

These explain the verdict and cannot change it, and they cannot support a claim. Details are in the EXP-022 README.

- **The reference carries the noise:** its outputs on white-noise prompts score UTMOS 1.25 to 2.08 (3.92 on clean prompts). This fits EXP-020's finding that the teacher's voice part carries the prompt's noise. The handovers' pitch could be tracked on 14 or 15 of the 20 noisy15 pairs, the reference's on none.
- **With Step 3 allowed** (it passes the clean pitch check; Step 2 does not), the slow-part rule would cost 13.08 passes within the C2 margins. Its WER is higher than Voice-Swap's: +0.0084 [+0.0013, +0.0159].
- **The signal does not choose the right inputs:**
  - **Against random routing:** routing at random while keeping the rule's stops per condition matches the rule (WER +0.0086 [+0.0051, +0.0121] against +0.0080; failures 84.3 against 84).
  - **Why noisy inputs stop early:** their slow part changes less between steps, which is the "stalled, not settled" caveat in the README.
- **Fixed Step 3 against Voice-Swap:** WER +0.0110 [+0.0020, +0.0199], at 12 passes against 14.

## Consequence for EXP-021

- **The same weakness:** EXP-021's P3 also compares pitch variety with the reference on noisy15. When that comparison has no value, its runner reports P3 as n/a and the confirmation as failed.
- **Not run:** EXP-021 has not been run, and nothing has been synthesized for it.
- **Amended the same day:** at the requester's direction, P3 now judges pitch variety on clean prompts only, and the preregistration was frozen again before any synthesis. See the [amendment log](2026-11-13-exp021-p3-amendment.md).

## Decisions in this change

- **The verdict stands:** it was not revised, and no threshold, step set or criterion was changed after the results were seen.
- **Left to the team and the adviser:** whether Slow-Part Handover has any place in the thesis.

## Checks run

- **Results provenance:**
  - All seven modules in the zip are identical to `code/colab/` once line endings are normalised (runner `692e2345bf2653d6`).
  - The committed preregistration blob at `a068a0d` is byte-identical to the run's copy.
  - `RUN_INFO.txt` shows a Tesla T4, thesis commit `a068a0d` and Shonget-DMOSpeech2 `8a7868f`, and all 360 rows report a Tesla T4.
- **Smoke test:** 4 of 4 stored Step-4 handover scores reproduced (largest difference about 6e-8), and the stored teacher drift reproduced at every step (largest difference 7e-8).
- **Re-analysis:** `analyze` on a copy of the CSV files gave a results JSON and report identical to the Colab output once line endings are normalised.
- **Independent recount:** `code/analysis/exp022_diagnostics.py` gives the report's Definition 3.1 failure count for every fixed output without the runner (`STT_4` 76; Steps 2 to 7: 94, 88, 77, 70, 61, 71).
- **Data checks:**
  - **Counts:** 360 unique cases in 6 conditions, and 40 diversity pairs at 20 seeds.
  - **Timestamps:** quality rows from 00:43 to 03:10 UTC, diversity rows from 03:13 to 04:58 UTC.
- **`python -m unittest discover -s code/tests`:** OK, 140 tests.
- **`python scripts/check_workspace.py`:** PASSED.
- **`latexmk -Werror` in `paper/`:** exit 0. The manuscript was not changed.

## Remaining blockers

- **Evidence type:** development evidence on inspected speakers, with white-noise prompts only.
- **EXP-021:** P3 was amended (see above). It is ready to run once pushed.
- **Pitch variety on noisy prompts** is still unmeasured against the reference, which has no trackable pitch there.
- **Novelty is not established.** A full literature review is outstanding, and adoption would need a dated decision record and adviser review (docs/RESEARCH_DIRECTION.md).
- **Appendix E:** its STAC 19.9% figure (34 Definition 3.1 failures, 9.4%, above the 5% limit) and its 1D-CRC-STAC mismatch with `fit_crc_stac` remain unfixed.
