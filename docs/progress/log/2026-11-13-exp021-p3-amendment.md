# 2026-11-13: EXP-021 P3 amended before any synthesis

Task: EVAL-014 (todo). Requested by Rhenel Jhon R. Sajol after EXP-022 showed that EXP-021's P3 could not be computed as frozen.

## Why

- **The defect:** P3 required, on clean and on noisy15 prompts separately, that Voice-Swap's CVf0 minus the reference's have a lower bound of at least -0.02.
  - **What EXP-022 found:** on noisy15 prompts (development speakers 1-20, 20 seeds), the reference had no frame voiced in at least half of the seeds, for any speaker. Its CVf0 does not exist there.
  - **The consequence:** the runner reports P3 as n/a, and the confirmation as failed, whatever Voice-Swap does.
- **The cause:** the reference reproduces the prompt's white noise (UTMOS 1.25 on noisy15 against 3.92 on clean prompts, EXP-022), which defeats the pitch tracker.
- **Why it was missed:** EXP-020 measured pitch variety on clean prompts only, so this was not known when EXP-021 was frozen.

## What changed

- **`code/colab/run_voice_swap_confirm.py`:**
  - **P3** is judged on clean prompts only (`P3_CONDS = ("clean",)`), with the same margins: Voice-Swap minus the reference at least -0.02, and Voice-Swap minus the student alone above 0.
  - **Noisy15 pitch variety** is still measured (`DIV_CONDS` is unchanged). It is reported without a pass or fail, together with the number of speakers whose pitch could be tracked (`cvf0_pairs`, and n in the report).
  - **Preregistration text:** the claim and P3 now say "on clean prompts", and a new `amendment` field records the date, the replaced preregistration, the change and the reason.
- **`research/experiments/EXP-021-voice-swap-confirmation/preregistration.json`:**
  - **Old:** the first preregistration (`1082c66869fd2f33...`, created 2026-11-13T00:17 UTC, committed in `8b8bc54`) was deleted unused.
  - **New:** frozen again as `33f37d54ea30cdca...`, created 2026-11-13T05:50 UTC. Its `thesis_commit` (`a068a0d`) is the commit the amendment was made on.
- **`code/tests/test_voice_swap_confirm.py`:** 27 tests (22 before). New tests:
  - an untrackable reference on noisy15, as in EXP-022, leaves P3 to be decided by clean prompts;
  - lost pitch variety on noisy15 alone does not fail P3;
  - Voice-Swap no better than the student alone on clean prompts fails P3;
  - the committed preregistration records the amendment.
  
  The clean-prompt failure test now keeps the noisy15 values good, so it tests the clean rule alone.
- **`code/analysis/exp022_diagnostics.py`:** Part 3 applies the amended P3 to EXP-020's clean-prompt pitch data (below).
- **Documentation:** the notebook's description text (its code cells are unchanged), the EXP-021 README, the EXP-022 README and results log, `research/experiments/README.md`, `STATUS.md`, `tasks.json` and the rendered `TASKS.md`.

Not changed: margins, speakers, conditions, outputs, seeds, timing, P1, P2, P4, the decision rule, the shared runner modules, the manuscript and the thesis method.

## Why the amendment is legitimate

- **No EXP-021 data exist.** Nothing has been synthesized on speakers 61-100 for EXP-021. The requester ran EXP-022 instead, and no EXP-021 output is in the repository.
- **The reason comes from other data.** The defect was found on EXP-022's development speakers 1-20, not on EXP-021's speakers.
- **Nothing was loosened.** The clean-prompt requirement and every margin are unchanged; only a comparison that cannot exist was removed from the pass rule.
- **Same procedure as before.** The line-ending fix on the same day replaced both preregistrations in the same way before any synthesis.

## Check on development data

On EXP-020's clean-prompt pitch data, the amended P3 can be computed (`code/analysis/exp022_diagnostics.py`, Part 3). The data are speakers 1-20 with the same three outputs: Step 8, Voice-Swap (`STT_4`) and student only.

| Comparison | Result | Requirement |
|---|---|---|
| Voice-Swap minus the reference | -0.0050 [-0.0157, +0.0009] | at least -0.02 |
| Voice-Swap minus the student alone | +0.0383 [+0.0240, +0.0555] | above 0 |

These speakers designed Voice-Swap, so this shows only that the rule is computable, not that EXP-021 will pass.

## Checks run

- **`python -m unittest discover -s code/tests -p test_voice_swap_confirm.py`:** OK, 27 tests.
- **`python -m unittest discover -s code/tests`:** OK, 145 tests.
- **`python scripts/check_workspace.py`:** PASSED.
- **`latexmk -Werror` in `paper/`:** exit 0. The manuscript was not changed.
- **The notebook** still parses, with 19 cells; its code cells are byte-identical to before.

## Remaining blockers

- **Push needed:** the notebook clones `rhnl-3`, so the amended runner and preregistration must be pushed before the run.
- **Shared speakers:** EXP-019 names the same 40 speakers as its fresh set. Whether running EXP-021 first is acceptable for EXP-019 is still a team decision.
- **Pitch variety on noisy prompts** stays unjudged, because no pitch measure of the reference exists there.
- **Evidence type:** a confirmation on development speakers not used to design Voice-Swap. It is not the thesis's held-out evaluation.
