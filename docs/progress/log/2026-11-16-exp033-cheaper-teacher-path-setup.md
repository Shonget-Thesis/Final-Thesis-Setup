# 2026-11-16: EXP-033 prepared (cheaper teacher paths to the paper's 0.25), with a local screen of the candidates

Task: EVAL-026 (todo). Requested by Rhenel Jhon R. Sajol.
- **The request:** study every experiment, propose solutions ("brand new or modified"), test each locally with an easy test, test the survivors on the GPU, and say how each would save compute while keeping the rest.
- **Then, while the local test was still running:** "do the new ipynb now and push".
- **Status:** development work. Nothing is adopted, and no adviser has reviewed it.

## Scope decision recorded

On 2026-11-16 the requester said the method can still change as long as the proposal defense is not scheduled. So "SNR + drift stays until the defense" is no longer treated as a constraint when candidates are designed. Any change to the manuscript still goes through Chapter 3 and the appendices and is the team's decision.

## What the experiments say (EXP-001 to EXP-032), and what was left to try

| | Finding | Experiments |
|---|---|---|
| 1 | Every "when to stop" signal reduces to the prompt SNR; drift adds nothing beyond it | EXP-004 to 009, 012, 017, 018, 023, 029 |
| 2 | Every handover copies a noisy prompt's noise; the student alone is better on noisy prompts | EXP-015, 020, 029 to 032 |
| 3 | The teacher's job is pitch variety. Under the thesis's restart at 0.0761 its steps after Step 3 add none | EXP-026, 027, 029, 032 |
| 4 | The teacher decides the melody between Steps 3 and 8; the restart at 0.2929 keeps it, and `W8R2` (20 passes) matches the paper's 30-pass setting | EXP-026 to 029, EXP-032 and its bank screen |
| 5 | Changing what crosses at an early step (extrapolation, operators, band swaps, bridges, timing maps) either loses variety or loses quality | EXP-013, 014, 020, 022, 023, 028 to 031 |
| 6 | Stopping the teacher earlier under the 0.2929 restart trades variety for passes, and drift cannot say who may stop early | EXP-029, the re-screen |

**What was never tried:** making the teacher's own path to t = 0.2929 cheaper, instead of shortening it or changing what crosses.

## The candidates

| Candidate | Passes | Saved vs the paper's 30 | How it saves teacher passes | New or known |
|---|---:|---:|---|---|
| `CPH12`, `CPH8`, `CPH6` | 16, 12, 10 | 47%, 60%, 67% | The teacher reaches the same time (t = 0.2929) in 6, 4 or 3 larger steps instead of 8 | Known lever (fewer solver steps); it is the released sampler with fewer teacher steps |
| `GI` | 15 | 50% | Teacher Steps 4–8 skip the unconditional pass (one pass each instead of two) | Known (guidance interval, Kynkäänniemi et al., 2024) |
| `SST` | 13 | 57% | One student pass replaces teacher Steps 0–3 (8 passes) | Modified; close to student-then-teacher refinement, listed as known on 2026-11-14 |
| `CWH4` | 13 | 57% | The teacher stops at Step 4; only the cells it has stopped changing cross, and the student's draft fills the rest | The weighting by the teacher's own per-cell change was not found in the project's earlier audits; no literature audit run. Per-region strength maps exist (for example Differential Diffusion) |

All of them finish with the student's two steps at 0.2929 and 0.6173, as `W8R2` does. The student's passes are 2 or 3 in every arm; every saving is in the teacher.

## Local screen (screening only, not evidence)

**Script:** `code/analysis/local_candidate_screen.py`. Output: `research/experiments/EXP-032-variety-per-pass/offline/local_candidate_screen.json`.
- **What ran:** DMOSpeech 2 on this computer's CPU, with three development speakers (the three shortest in EXP-032's tensor bank) × six diversity seeds, clean prompts. That is 18 cases in 24 minutes.
- **Check of the local teacher:** its Step 8 estimate differs from the T4 bank's by 0.02% (the bank's float16 rounding).
- **The student's noise** is drawn on the GPU in the experiments and on the CPU here, so outputs are not identical to the T4's.
- **`CPH12` was not in the screen;** it was added to the T4 run afterwards.
- **Pass rule, fixed before the run** (against `W8R2` on the same cases):
  - melody agreement with the paper's stored melody ≥ `W8R2`'s − 0.10;
  - CVf0 ≥ `W8` + 0.75 × (`W8R2` − `W8`);
  - UTMOS ≥ `W8R2`'s − 0.10.

| Arm | Passes | CVf0 (bar ≥ 0.199) | Melody vs the paper (bar ≥ 0.616) | UTMOS (bar ≥ 2.752) | Result |
|---|---:|---:|---:|---:|---|
| `STU` | 4 | 0.054 | 0.352 | 3.612 | known setting |
| `W8` | 21 | 0.158 | 0.500 | 3.394 | known setting |
| `W4R2` | 12 | 0.153 | 0.686 | 2.421 | known setting (fails variety and quality) |
| `W6R2` | 16 | 0.181 | 0.738 | 2.840 | known setting (fails variety) |
| `W8R2` | 20 | 0.212 | 0.716 | 2.852 | reference |
| `CPH8` | 12 | 0.195 | 0.632 | 2.756 | **fails variety only, narrowly** |
| `CPH6` | 10 | 0.183 | 0.587 | 2.461 | fails all three |
| `GI` | 15 | 0.298 | 0.677 | 2.488 | fails quality (−0.36 UTMOS) |
| `SST` | 13 | 0.070 | 0.395 | 3.741 | fails melody and variety: it behaves like the student alone |
| `CWH4` | 13 | 0.134 | 0.613 | 2.736 | fails all three, two of them narrowly |

**Reading:**
- **No candidate passed all three bars.** `CPH8` is the closest: within the margins on melody and UTMOS, just under the variety bar, and 8 passes cheaper than `W8R2`.
- **`SST` shows the teacher's first steps cannot be replaced by the student's draft.** The output then has the student's melody and variety.
- **`GI` keeps the variety but loses quality** when guidance is dropped on Steps 4–8.
- **`CPH6` is too coarse.**
- **`CWH4` lands between `W4R2` and the student:** better UTMOS than `W4R2`, less variety.

**Why this is not a verdict:**
- **Three speakers, and unusual ones.** On the T4, the paper's own setting scores far lower UTMOS than the student alone for two of them:

  | Speaker (utterance) | `P25` UTMOS (T4, EXP-032) | `STU` UTMOS (T4) | `P25` CVf0 (T4) |
  |---|---:|---:|---:|
  | `common_voice_en_15734837` | 2.44 | 4.31 | 0.441 |
  | `common_voice_en_15735839` | 2.61 | 3.94 | 0.244 |
  | `common_voice_en_13900` | 2.84 | 2.93 | 0.064 |

  The first speaker's CVf0 of 0.44 dominates every mean above.
- **Consistent with the T4 where they overlap:** for that speaker the local `W8R2` averages UTMOS 2.28 over its six seeds, and EXP-026's T4 `W8R2` scored 2.29 on its one seed.
- **Six seeds per speaker.**

## EXP-033 (the T4 run)

**Files:**
- runner `code/colab/run_cheaper_teacher_path.py`;
- notebook `code/colab/DMOSpeech2_EXP033_Cheaper_Teacher_Path.ipynb`;
- tests `code/tests/test_cheaper_teacher_path.py` (21 tests);
- [README](../../../research/experiments/EXP-033-cheaper-teacher-path/README.md);
- preregistration `12aec5f7f9fd...`, frozen 2026-11-16 05:04 UTC.

**Design:**
- **Arms:** `P25` (the 100% reference), `W8R2`, `W8`, `W6R2`, `W4R2`, `STU`, and the six candidates above.
- **Quality:** development speakers 1–30 × clean and noisy15.
- **Variety:** speakers 1–20 × 12 seeds, clean prompts (EXP-032 showed pitch cannot be tracked in noisy teacher outputs).
- **Cost:** each arm alone on 10 clean cases, with forward passes counted by hooks on both networks, and wall-clock time.
- **Rules:** EXP-032's variety share (90% bar) and quality bars against `P25`. One label per arm, plus `cheapest_match_point`, `cheapest_match_confirmed` and `w8r2_matches_paper`.
- **All five candidates are run although the screen passed none.** The screen is too small to rule a candidate out, and a preregistered result on 30 speakers settles each one. The four screen failures (`SST`, `GI`, `CWH4`, `CPH6`) add roughly 40 minutes to a run of about 3 hours.

**Order of events (for the preregistration's honesty):**
1. The local screen was started.
2. Its two-seed trial output on one speaker was seen.
3. The runner was written, with EXP-032's rules.
4. The runner was dry-run on this computer (below).
5. The preregistration was frozen.
6. Only then were the full screen's results opened.

The preregistration states this in its `local_screen` field.

**Dry run of the runner (this computer's CPU, one short development speaker, scoring stubbed; no result kept):**
- **Identities:** the teacher loops against the released loop, the continuation from Step 4, guidance on every step, the split student and κ = 1 were all exactly 0.
- **Each arm alone** equals the shared run (difference 0).
- **Counted passes** equal the cost table for all twelve arms.
- **The three stages** wrote complete rows, and the partial analysis ran.

## Checks run

- `python -m unittest discover -s code/tests -p test_cheaper_teacher_path.py`: 21 tests, OK. They include the real teacher loops and the arm wiring on a NumPy teacher and student: each arm alone equals the shared run and costs its listed passes.
- **Dry run:** as above (9 minutes).
- **Workspace check:** `python scripts/check_workspace.py` PASSED.
- **Full suite:** `python -m unittest discover -s code/tests` ran 373 tests, OK (1 skipped).
- **Manuscript:** not rebuilt, because no manuscript file changed.

## Remaining

- **Run EXP-033 on a Colab T4** (about 3 hours), then import and check the results zip.
- **Literature audit** before any candidate is called a contribution.
- **Noisy prompts:** no candidate is expected to help there. The descriptive SNR-routing table shows what routing noisy prompts to the student alone would cost and give.
