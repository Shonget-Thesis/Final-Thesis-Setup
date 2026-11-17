# 2026-11-16: EXP-032 tensor bank, when the teacher's melody is decided and what each restart keeps (offline screen)

Task: EVAL-025 (review); a post hoc follow-up to its results.
- **Requested by:** Rhenel Jhon R. Sajol, who supplied the tensor bank.
- **Exploratory:** five development speakers. Nothing here is adopted, and no adviser has reviewed it.

## Input

- **The bank:** `DMOSpeech2_EXP032_Tensor_Bank.zip`, received 2026-11-16 (sha256 `7da7385a6c4f...`, 97,746,282 bytes).
  - **Verification:** extracted to a scratch folder, not into the repository. All 120 files match the SHA-256 index committed with the results (`results/bank_index.json`).
  - **Not committed** (`.npz` and `.zip` are ignored).
- **Contents:** development speakers 1–5 × 12 seeds × clean and noisy15. For each seed:
  - the teacher's endpoint estimates at Steps 1, 2, 3, 4 and 8 of the 16-interval grid and at the paper's Step 14;
  - the student's one-step draft;
  - the outputs of `STU`, `W4`, `W8` and `P25`;
  - the T4 pitch tracks of every arm.

## Method

**Script:** `code/analysis/exp032_bank_pitch_screen.py`. Output: `research/experiments/EXP-032-variety-per-pass/offline/bank_pitch_screen.json`.
- **No TTS network runs.** Each stored mel is multiplied by the checkpoint's mel scale (2.9019), as the pipeline does, then vocoded with Vocos on this computer's CPU.
- **Pitch tracking:** the experiments' pYIN settings (`run_selective_handover_prelim.f0_track`).
- **Check of this pipeline:** re-tracking the four stored outputs on the CPU agrees with their T4 tracks:
  - 100% of frames within a semitone, on clean prompts;
  - voiced frames overlap 0.99.
- **Melody agreement within a seed:**
  - **What it compares:** two pitch tracks of the same seed, frame by frame, on frames voiced in both.
  - **Reported:** the share of frames within one semitone, and the median difference in cents.
  - **The "no shared melody" level:** two different seeds of the paper's setting agree on 36% of frames.

## Results: clean prompts (5 speakers × 12 seeds)

**When the paper's final melody is decided.** How close each teacher estimate's melody is to `P25`'s final melody, same seed:

| Estimate | Teacher time t | Frames within a semitone of `P25`'s final melody | Median difference | Frames voiced in both |
|---|---:|---:|---:|---:|
| Student draft (`s0`) | | 40% | 358 cents | 0.57 |
| Step 1 | 0.0048 | 18% | 765 cents | 0.31 |
| Step 2 | 0.0192 | 25% | 767 cents | 0.38 |
| Step 3 | 0.0431 | 45% | 396 cents | 0.38 |
| Step 4 | 0.0761 | 71% | 127 cents | 0.41 |
| Step 8 | 0.2929 | 82% | 39 cents | 0.76 |
| Step 14 (32-interval grid) | 0.2270 | 86% | 31 cents | 0.76 |
| *Different seed (`P25`, next seed)* | | *36%* | *304 cents* | *0.59* |

- **Steps 1–2:** no more agreement than a different seed.
- **Between Steps 3 and 8:** the melody forms.
- **By Step 8:** the teacher's estimate already carries nearly the final melody of the paper's setting.
- **Caution about Step 4:** the 71% is measured on the few frames pYIN voices in that blurry estimate (0.41 overlap), so it is weaker evidence than the Step 8 figure.

**What each restart keeps of the melody it was handed:**

| Handover | Restart | Output's frames within a semitone of the handed-over estimate | Median difference |
|---|---|---:|---:|
| Step 4 → `W4` | re-noised at 0.0761 (j = 1) | 55% | 243 cents |
| Step 8 → `W8` | re-noised at 0.0761 (j = 1) | 53% | 197 cents |
| Step 14 → `P25R` | re-noised at 0.2929 | 79% | 43 cents |
| Step 14 → `P25` | continued at 0.2270 | 86% | 31 cents |

**The thesis's restart loses the melody.** `W8`'s teacher had nearly the final melody at Step 8, but `W8`'s output keeps only 53% of it, which is close to the different-seed level (36%). The paper's restart keeps 79–86%. Two comparisons point the same way:
- **Against the paper's output:** `W8`'s output agrees with it on 57% of frames, while the Step 8 estimate that `W8` was handed agrees on 82%.
- **Distance from the paper's output:** the Step 8 estimate is closer to it (relative mel distance 0.31) than `W8`'s output is (0.44).

**Variety present in the estimates:**
- **From Step 2 on,** the across-seed CVf0 of the teacher's estimates (0.17–0.21) is higher than any output's.
- **Steps 2–4 are unreliable:** pYIN voices only 28–31% of frames in those estimates, so their CVf0 is not comparable.
- **The four outputs' CVf0** matches EXP-032's ordering on these five speakers: `P25` 0.196, `W8` 0.119, `W4` 0.120, `STU` 0.062. `W8` keeps 35% of `P25`'s aligned gain, as on 20 speakers.

## Results: noisy15

- **The estimates:** pYIN voices under 10% of frames in the teacher's estimates from Step 2 on (0.7–9%). The teacher's estimates already carry the prompt's noise by Step 2.
- **The outputs:** at most 0.1% of frames voiced in any teacher arm's output, against 30% for the student alone.
- **So:** the melody cannot be measured on noisy15 by this method. The agreement figures there rest on 3–12 seeds and are not interpreted.

## What this means

**Correction (same day):** an earlier version of this section said that `W8` with j = 2 and the cheaper j = 2 stops had not been run. They had been: the requester pointed out that the restart test was already done.
- **The arms:** EXP-026–029 ran `W4R2`, `W5R2`, `W6R2` and `W8R2`.
- **EXP-026's transmissibility result had already shown** that under j = 1 the restart level, not the number of teacher steps, limits the variety.
- **Where this is recorded:** point 8 of the [EXP-032 results log](2026-11-16-exp032-results.md). Below, the screen is read against those experiments.

1. **Where the variety goes:**
   - **The teacher's part:** it decides the melody between Steps 3 and 8, and by Step 8 the melody is nearly final.
   - **The thesis's handover:** with j = 1, it restarts the student at 0.0761, where most of that melody is lost.
   - **What is new here:** this is the melody-level mechanism behind EXP-026's result and the "E3" finding of the 2026-11-15 synthesis. What the screen adds is when, along the teacher's steps, the melody forms, and how much of it each restart keeps.
2. **The screen explains the known j = 2 curve.**
   - **The curve:** `W4R2` recovers variety only partly (EXP-026: not reliably; EXP-027: +0.0089), and `W8R2` recovers it fully. That matches a melody that is only partly formed at Step 4 (71% on few frames) and nearly final by Step 8 (82%).
   - **`W8R2` against the paper's setting:** paired on 30 clean cases, `W8R2` (20 passes) matches `P25` (30 passes) in UTMOS (+0.003 [−0.126, +0.112]). Its CVf0 is about the same on the same speakers (0.1171 against 0.1173, different seed sets).
   - **The decision:** changing Chapter 3's fixed j = 1 is a team decision; EVAL-009 is unassigned.
3. **What a new handover would have to solve:**
   - **Under j = 2,** EXP-029 found only a monotone trade-off between variety and UTMOS from Step 4 to Step 8, and more runaway outputs on noisy15.
   - **The target:** the paper's variety for fewer than `W8R2`'s 20 passes, or at higher quality, and without the noise.
   - **What this screen suggests:** the melody forms during teacher Steps 4–8, and those are the passes such a handover would have to save.

## Checks run

- **The bank:** all 120 files checked against the committed hash index; the script re-checks them on every run.
- **CPU pipeline against the T4 tracks:** agreement as above.
- **The script:** `.venv/Scripts/python -I code/analysis/exp032_bank_pitch_screen.py <bank folder>`, about 2 minutes. A first run omitted the Step 14 → `P25` comparison because of a key bug. It was fixed and rerun, and only the second run's output is kept.
- **Tests:** `code/tests/test_exp032_bank_screen.py`, 6 tests of the agreement measure and the grid times. It caught a rounded constant (Step 3's time), which is now computed from the grid; the constant does not enter the output.
- **Workspace check:** `python scripts/check_workspace.py` PASSED.
- **Full suite:** `python -m unittest discover -s code/tests` ran 347 tests, OK (1 skipped).
