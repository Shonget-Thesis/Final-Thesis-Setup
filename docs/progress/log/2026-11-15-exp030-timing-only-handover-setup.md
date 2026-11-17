# 2026-11-15: EXP-030 Timing-Only Handover pilot prepared

Task: EVAL-023 (todo). Requested by Rhenel Jhon R. Sajol: "exp 30 for TOH only first, then see results, and if it works combine it with SNR so it is adaptive". The design and offline checks are in the [design log](2026-11-15-timing-only-handover-design.md).

## What was created

- **Runner:** `code/colab/run_timing_only_handover.py`.
  - **Time map:** features, a constrained DTW, the map, its mirror, and resampling.
  - **TOH handover:** built from the time map.
  - **Synthesis:** eight arms from one teacher trajectory and one student draft per case.
  - **Stage loops:** quality and diversity, both resumable.
  - **Smoke test:** with exact identity checks.
  - **Preregistration:** freeze and check.
  - **Analysis:** with the frozen hypotheses and verdict.
  - **Reuse:** EXP-029's teacher loop and the released student loop, so W4 and W8 repeat EXP-029's code path.
- **Experiment folder:** `research/experiments/EXP-030-timing-only-handover/`:
  - the README;
  - `preregistration.json`, frozen before any synthesis (first freeze: payload sha256 `575df8686329...`, file `5d3ddb0561e5...`, 08:17 UTC; re-frozen under Amendment 1 below).
  - **What it locks:**
    - the runner and its nine dependencies;
    - EXP-027's `quality.csv`;
    - EXP-029's `quality.csv` and `diversity.csv`;
    - the cohorts, bars, hypotheses and verdict rule.
- **Notebook:** `code/colab/DMOSpeech2_EXP030_Timing_Only_Handover.ipynb`.
  - **Built from EXP-029's notebook:** RAM-safe launch, Drive restore and sync, and resume.
  - **Stage default:** `STAGE = "all"`, so Run all needs no edits.
  - **Zip:** saved to Drive before the browser download.
- **Tests:** `code/tests/test_timing_only_handover.py`, 12 tests, covering:
  - **The time map:** it ignores per-band offsets and scales, gives the identity on identical inputs, is monotone, slope-limited and inside the band, and recovers a known re-timing within 1 frame RMS.
  - **Condition-blindness:** a stationary condition added to the teacher leaves the payload unchanged, and the payload stays inside the range of the student's own frames.
  - **Identities:** the identity map is exact, and mirroring keeps the displacement statistics.
  - **Design:** costs and cohorts (identical to EXP-029's, and disjoint from the calibration and test splits).
  - **Verdict rule:** every branch, plus the partial case.
  - **Tampering detection.**
  - **The notebook.**
- **Tracking:**
  - EVAL-023 in `tasks.json` and `TASKS.md`;
  - STATUS;
  - the experiments index;
  - `.gitattributes`, so result bytes stay unchanged.

## Changes from the design log

**The other-seed warp control was replaced by MIR3, TOH3's map mirrored in time:**
- **It couldn't test variety.** Giving each seed another seed's map only permutes the maps across seeds, so the set of timings, and so CVf0, stays the same.
- **It would cost more.** On quality cases it would need a second teacher run.

MIR3 keeps the amount and kind of re-timing, but not where the teacher put it.

**Other changes:**
- **TOH4 was dropped** to keep the run near 2 hours. TOH3 against W3 isolates timing-only against the whole estimate at the same teacher step.
- **Telephone prompts were added** as an exploratory condition. That is where SNR routing cannot help.

## Checks run

- `python -m unittest discover -s code/tests -p "test_timing_only_handover.py"`: 12 tests, OK.
- **On synthetic data:**
  - the time map recovered a known re-timing (RMS error 0.32 frames);
  - it gave the identity exactly on identical inputs;
  - it took 0.04 s for a 900-frame utterance.
- `check_prereg()`: accepts `575df8686329...`.
- `python scripts/check_workspace.py` and `python -m unittest discover -s code/tests`: results are in the handoff.
- LaTeX build: not run. The manuscript did not change.

**Not testable here:** the torch path of the runner (this PC has no PyTorch). The smoke test checks it on the T4 before any synthesis.

## Remaining blockers

- **Push before running:** the notebook clones `rhnl-3`, so the runner, the preregistration and EXP-029's final results must be pushed first.
- **The GPU run:** about 2 hours on a T4. Nothing has been synthesized.

## Amendment 1 (2026-11-15, before any main-run synthesis)

**What happened:** the first Colab smoke test (Step 4) stopped with exit code 1. The runner's own message was not sent, but a local search reproduces the most likely cause.
- **The bug:** `path_to_map` pinned the map's last frame to the student's last frame.
- **When it bites:** when the warping path ends with two (1,2) steps, the last teacher frame's mean student frame is m − 1.5, and pinning it to m − 1 makes a final step of 2.5. That fails the smoke test's "steps between 0 and 2" check.
- **How often:** in a local search of 400 maps, 18 of the 200 structured, re-timed inputs failed this way. None of the random-noise inputs the unit tests used did.

**Change:**
- **The fix:** the last frame is no longer pinned. The map is the mean student frame per teacher frame, made non-decreasing, so w[0] = 0 and w[n − 1] is between m − 1.5 and m − 1. Nothing else in the method, the arms, the bars or the verdict rule changed.
- **Smoke test:** its map check now accepts that end range, and each smoke record is printed before any check raises.
- **Unit test:** a new regression test runs the smoke map check on 60 structured, re-timed inputs.
- **Result:** the same local search now gives 0 failures of 400, and the largest step is 2.0.

**Re-freeze:** the runner's hash changed (`817e2bd600cd...` → `d04f27fc0bef...`), so the preregistration was re-frozen.
- **New payload hash:** `ebe2b773c02f...`.
- **New file hash:** `e2e93cc5f5a9...`.
- **Created:** 08:47 UTC.

The first freeze, `575df8686329...`, never governed any result. The failed smoke test wrote no main-run rows.

**Checks:**
- `python -m unittest discover -s code/tests -p "test_timing_only_handover.py"`: 13 tests, OK.
- `check_prereg()`: accepts `ebe2b773c02f...`.
