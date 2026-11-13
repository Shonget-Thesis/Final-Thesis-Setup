# 2026-11-13: EXP-022 Slow-Part Handover prepared

Task: EVAL-015 (todo). Requested by Rhenel Jhon R. Sajol: a handover method focused on saving computation, tested in a new notebook. The requester will run it on Colab and send the results zip.

## Why

EXP-020's map shows the student uses only the teacher's slow part (timing and pitch movement). The voice part can come from a student draft, and the fine detail barely matters. A handover that passes only the slow part could stop the teacher as soon as that part has settled, for each input, possibly before Step 4. Voice-Swap always waits until Step 4.

## What changed

- **`code/colab/run_slow_part_handover_prelim.py`:**
  - **Per case:** the handover at every Step 2 to 7 (code STS, from the EXP-020 functions), and the change of the voice, slow and fine parts and of the whole estimate at every teacher step.
  - **Pitch variety:** CVf0 at Steps 2, 3, 4 and 8.
  - **Analysis:** an offline stop-rule analysis with 5-fold speaker cross-validation.
  - **Smoke test:** it must reproduce EXP-020's stored Step-4 handover scores and EXP-018's stored teacher drift.
  - **Freeze and safety checks:** `freeze` pins seven code files and four input files by SHA-256.
- **`research/experiments/EXP-022-slow-part-handover-prelim/`:** README and `preregistration.json` (sha256 `c311aeb52d424344...`, created 2026-11-13T00:17 UTC), frozen before GPU execution.
- **`code/colab/DMOSpeech2_EXP022_Slow_Part_Handover.ipynb`:** 17 cells, the same layout as the EXP-020 notebook. No progress sync was added, at the requester's choice.
- **`code/tests/test_slow_part_handover.py`:** 19 tests, covering:
  - the pass accounting;
  - the signals: the whole-estimate change equals the sampler's drift, and a change only in the band averages moves only the voice part;
  - the stop rule and the calibration on a synthetic table;
  - the run loops on a fake numpy model;
  - the analysis on synthetic data (a pass, early steps dropped for lost pitch variety, no gain over the whole estimate, partial data);
  - preregistration tampering and code changes.
- Rows in `research/experiments/README.md`, `tasks.json`, the rendered `TASKS.md` and `STATUS.md`; `.gitattributes` keeps future EXP-022 result bytes unchanged.

## Decisions in this change

- **Re-frozen before any synthesis (same day).** The first freeze hashed text inputs byte for byte. `code/scheduling/failure_config.json` and `research/protocols/splits/development_manifest.json` have CRLF line endings in the Windows checkout but LF on Colab, so the runner would have refused to start there. Both runners now hash text inputs with line endings normalized (`input_hash`, with a regression test). Because that changed the runner code, both preregistrations were deleted, unused, and frozen again.
- **Calibration controls the mean:** it limits the mean change against Step 8 (WER +0.01, similarity -0.01 on the training folds). The thesis's 5% Definition 3.1 limit is not used, because its floor on noisy prompts is about a quarter. Definition 3.1 failures are reported descriptively.
- **Data mix behind C1:** C1 (cheaper than Voice-Swap's 14 passes) is judged on the Quick set, where five of six conditions are noisy. It therefore needs noisy inputs to settle by about Step 4 as well.
- **Candidate steps:** Steps 2 and 3 are used only if their pitch variety holds, which is decided before the stop rules are calibrated.
- **Process note:** the preregistration was frozen before the last analysis test was corrected. Only the test's synthetic scenario changed (noisy inputs settling at Step 4 instead of 5); the runner did not change, and the frozen hashes still match.

## Checks run

- `python -m unittest discover -s code/tests`: OK, 138 tests at the first freeze; after the re-freeze, OK, 140 tests, including the 19 tests in `test_slow_part_handover.py`.
- `python scripts/check_workspace.py`: PASSED.
- `latexmk -Werror` in `paper/`: exit 0. The manuscript was not changed.
- Every code cell of the notebook compiles with its notebook magics removed; `analyze --allow_partial` on an empty folder writes a report without errors.

## Remaining blockers

- **Not run yet, so no results.** The torch path runs first in the notebook's smoke test.
- **Push needed:** the branch must include the EXP-020 results (not yet committed), EXP-021 and EXP-022.
- **Novelty not established:** it rests on targeted searches only.
- **Adoption:** it would need a dated decision record and adviser review.
