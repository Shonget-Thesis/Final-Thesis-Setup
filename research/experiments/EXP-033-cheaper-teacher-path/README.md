# EXP-033: cheaper teacher paths to the paper's t_switch = 0.25

**Status:** prepared and preregistered on 2026-11-16 (Asia/Manila), before any GPU synthesis; freeze `12aec5f7f9fd...`. Not yet run.

Task EVAL-026 (todo). The run needs a Colab Tesla T4, about 3 hours. Files: [notebook](../../../code/colab/DMOSpeech2_EXP033_Cheaper_Teacher_Path.ipynb), [runner](../../../code/colab/run_cheaper_teacher_path.py), [tests](../../../code/tests/test_cheaper_teacher_path.py).

- **A development test:** the speakers are development speakers. It is not held-out evidence.
- **Nothing is adopted:** the thesis method is unchanged. Adopting any arm is a team decision.
- **No novelty claim:** none of the candidates has been through a literature audit.

## Why this test

**The target** (requested by Rhenel Jhon R. Sajol on 2026-11-16): keep the variety and quality of the paper's 0.25 setting while spending fewer teacher passes.

**What is already known:**
- **The paper's setting** (`P25`, 30 passes) has far more pitch variety than the thesis's Step 8 with the student restarted at 0.0761 (`W8`, 21 passes): `W8` keeps about a third of its aligned gain over the student alone ([EXP-032](../EXP-032-variety-per-pass/README.md)).
- **Why:** the teacher decides the melody between Steps 3 and 8, and the restart at 0.0761 discards most of it ([bank screen](../../../docs/progress/log/2026-11-16-exp032-bank-pitch-screen.md)).
- **The restart at 0.2929 keeps it.** `W8R2` (20 passes, EXP-026) matches `P25` in quality on 30 shared cases and roughly in variety. That comparison joins two runs and is post hoc.
- **Stopping the teacher earlier costs variety:** `W6R2` (16 passes) keeps 69–86% of `W8R2`'s gain over `W4`, and `W4R2` (12) loses 0.27 UTMOS ([re-screen](../../../docs/progress/log/2026-11-16-paper-anchor-rescreen.md)).
- **Drift does not say who can stop early** under that restart.

**What has never been run:** making the teacher's own path to t = 0.2929 cheaper. The melody forms on that path, so it cannot be skipped, but it might be walked in fewer or cheaper steps.

## What runs

| Arm | What it is | Teacher + student passes | Total | Saved vs `P25` |
|---|---|---|---:|---:|
| `P25` | **The 100% reference.** The paper's Algorithm 2 at 0.25: 14 guided teacher steps on the 32-interval grid, then 2 student steps (as in EXP-032) | 28 + 2 | 30 | |
| `W8R2` | 16-interval grid to Step 8 (t = 0.2929), student restarted at 0.2929 and 0.6173 | 18 + 2 | 20 | 33% |
| `W8` | The same teacher, student restarted at 0.0761 (the thesis's fixed j = 1) | 18 + 3 | 21 | 30% |
| `W6R2`, `W4R2` | The teacher stopped earlier (t = 0.1685, 0.0761), same restart as `W8R2` | 14 + 2, 10 + 2 | 16, 12 | 47%, 60% |
| `STU` | The student alone | 0 + 4 | 4 | 87% |
| `CPH12`, `CPH8`, `CPH6` | **Commit-Point Handover:** the guided teacher reaches the same time, t = 0.2929, on a 12-, 8- or 6-interval grid (6, 4 or 3 Euler steps instead of 8), then the `W8R2` restart. This is the released sampler with fewer teacher steps | N + 2, then 2 | 16, 12, 10 | 47%, 60%, 67% |
| `GI` | **Guidance on the early steps only:** 16-interval grid, the released guidance on Steps 0–3, the conditional branch alone on Steps 4–8, then the `W8R2` restart | 13 + 2 | 15 | 50% |
| `SST` | **Student-seeded teacher:** the student's one-pass draft S₀ stands in for teacher Steps 0–3. The teacher starts at Step 4 from (1 − t₄)·y₀ + t₄·S₀ and runs Steps 4–8, then the `W8R2` restart | 10 + 3 | 13 | 57% |
| `CWH4` | **Commitment-Weighted Handover** at Step 4. Per time–frequency cell, κ = 1 / (1 + (D / rms D)²) with D = T₄ − T₃, the teacher's own last change. The student's first-level draft S₁ fills what the teacher has not decided: x = κ·T₄ + (1 − κ)·S₁, then the student's levels 0.2929 and 0.6173. No fitted parameter | 10 + 3 | 13 | 57% |

| Stage | Cases | Arms |
|---|---|---|
| Quality (UTMOS, WER, speaker similarity) | Development speakers 1–30 × clean and noisy15 = 60 | all twelve |
| Prosodic variety (CVf0, aligned CVf0, timing spread; melody agreement with `P25` per seed) | Speakers 1–20 × 12 seeds, clean prompts | all twelve |
| Cost (forward passes counted on the networks; wall-clock seconds) | Speakers 1–10, clean, each arm run on its own | all twelve |

- **Variety on clean prompts only:** EXP-032 showed that pitch cannot be tracked in the teacher arms' noisy15 outputs.
- **The speakers and seeds** are EXP-032's. `STU`, `W8` and `P25` must repeat its scores.
- **Measures:** EXP-032's. Melody agreement is the share of frames within one semitone of `P25`'s pitch track for the same seed.

## Smoke test

A run stops before synthesis if any of these fails:
- **Teacher identities (≤ 1e-4):** the new teacher loop against the released one, from noise and continued from the state at Step 4; guidance on every step equals the released teacher.
- **Student identities (≤ 1e-5):** the split student; `CWH4` with κ = 1 hands over the teacher's estimate unchanged.
- **Each arm on its own** equals the shared run (≤ 1e-4), and its counted forward passes equal the cost table.
- **Reproductions (≤ 0.01):**
  - EXP-027's `W4`;
  - development speaker 1 under both conditions, `STU`, `W8` and `P25` against EXP-032's stored scores.
  - `W8R2` and `W4R2` are compared with EXP-026's stored scores and recorded, not gated.

## Estimands and label rules (frozen)

**Variety share** (EXP-032's), with V = CVf0 or aligned CVf0 averaged over the 20 speakers:

share_A = (V_A − V_STU) / (V_P25 − V_STU)

- **Interval:** a 95% speaker-level bootstrap interval (B = 2000, seed 2026).
- **"At least 90%" confirmed:** when L_A = V_A − 0.9·V_P25 − 0.1·V_STU has a bootstrap lower bound above 0.

**Quality bars** (EXP-032's; all three must hold, on clean prompts against `P25`): mean ΔUTMOS ≥ −0.08, ΔWER upper bound ≤ +0.01, ΔSIM lower bound ≥ −0.01.

| Label | Rule |
|---|---|
| `arm_labels` | For each of `W8`, `W8R2`, `W6R2`, `W4R2` and the six candidates: `matches_paper` if the arm keeps ≥ 90% of `P25`'s gain on both measures (point estimates) and passes the quality bars; `matches_paper_confirmed` if the 90% is also confirmed on both; otherwise `quality_short`, `variety_short` or `both_short` |
| `cheapest_match_point` | The arm with the fewest passes labelled `matches_paper` or `matches_paper_confirmed`; `none` if no arm qualifies |
| `cheapest_match_confirmed` | The same over `matches_paper_confirmed` |
| `w8r2_matches_paper` | `W8R2` is labelled `matches_paper` or `matches_paper_confirmed` (the post hoc reading above, tested in one run) |
| `noisy_quality_safe_arms` | noisy15: the arms that pass the quality bars against the student alone |

**Reported descriptively:**
- the counted passes and the wall-clock time of every arm;
- melody agreement with `P25`;
- the equal-cost contrasts `CPH12` − `W6R2` and `CPH8` − `W4R2` (reaching t = 0.2929 in larger steps against stopping earlier);
- how far each candidate's handed-over estimate is from the 16-interval Step 8 estimate;
- SNR routing with Chapter 3's 20 dB gate (below it the student alone, 4 passes; above it the arm): mean passes and quality over the 60 cases, computed without new synthesis.

## What a result would mean

| Result | Meaning |
|---|---|
| `w8r2_matches_paper` true | The paper's variety and quality cost 20 passes, not 30, on the released grid. `W8R2` is then the natural reference for a cheaper handover |
| `cheapest_match_point` is `CPH8` or `CPH6` | Fewer, larger teacher steps to the same point keep the paper's variety and quality: 60–67% fewer passes than the paper's setting, with no new operator |
| A `CPH` arm matches where the equal-cost early stop (`W6R2`, `W4R2`) does not | What matters is reaching t = 0.2929, not the number of steps taken to get there |
| `GI`, `SST` or `CWH4` matches | That arm's saving is real, and it becomes a candidate for a literature audit |
| `cheapest_match_point` is `W8R2` or `none` | The cheaper paths lose variety or quality. The paper's variety then costs about 20 passes, and saving more needs a different idea |
| `noisy_quality_safe_arms` empty | As in EXP-032: on noisy prompts no teacher handover keeps the student alone's quality, which supports routing noisy prompts to the student |

## The local screen

A quick CPU screen of the five candidates was run on this computer the same day (`code/analysis/local_candidate_screen.py`: three development speakers × six seeds, clean prompts).
- **What had been seen at the freeze:** only its two-seed trial output on one speaker.
- **The rules were not adjusted to it:** they are EXP-032's.
- **All five candidates are run here whatever the screen shows.** Its results are in the [setup log](../../../docs/progress/log/2026-11-16-exp033-cheaper-teacher-path-setup.md).

## Known limits

- **No listening test, and no held-out speakers.**
- **UTMOS is a predictor:** it has been lower for arms with more teacher variety (EXP-029, 031, 032).
- **The paper's second student time** (0.6173) is assumed, as in EXP-017 and EXP-032.
- **Timing** covers the teacher, the student and the vocoder for each arm run alone, on 10 clean cases. The preparation shared by all arms is left out.

## Data and evidence status

- **Development evidence only.**
- **Audio** is not committed.
- **Results** will be stored under `results/` with a SHA-256 manifest.
