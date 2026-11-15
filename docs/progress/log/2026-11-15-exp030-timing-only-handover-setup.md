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
  - `preregistration.json`, frozen before any synthesis (payload sha256 `575df8686329...`, file `5d3ddb0561e5...`, created 2026-11-15T08:17 UTC).
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
