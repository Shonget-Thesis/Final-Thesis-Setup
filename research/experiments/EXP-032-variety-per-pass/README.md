# EXP-032: how much variety each teacher pass buys, against the paper's t_switch = 0.25

**Status:** prepared and preregistered on 2026-11-16 (Asia/Manila), before any synthesis; freeze `c952fe555d49...`. Ran on a Colab Tesla T4 on 2026-11-15 (UTC); results imported on 2026-11-16 ([results](#results-2026-11-16), [results log](../../../docs/progress/log/2026-11-16-exp032-results.md)).

Task EVAL-025 (review). The run took about 3 hours on a Colab Tesla T4. Files: [notebook](../../../code/colab/DMOSpeech2_EXP032_Variety_Per_Pass.ipynb), [runner](../../../code/colab/run_variety_per_pass.py), [tests](../../../code/tests/test_variety_per_pass.py).

- **A development study:** the speakers are development speakers. It is not held-out evidence.
- **Not a method test:** no handover method is tested, and nothing here changes the thesis method.

## Why this study

**The teacher's job is variety:**
- **What the paper says:** DMOSpeech 2 runs its teacher early because its fast student loses prosodic variety (the paper's "mode shrinkage", [arXiv 2507.14988](https://arxiv.org/html/2507.14988v1), Sections 3.3 and 4.3.3). The teacher's early steps "establish diverse prosodic patterns and text-speech duration alignments".
- **What EXP-031 found:** on quality alone the student is as good as any teacher-based method. The teacher's measurable contribution was pitch variety.
- **So:** the teacher's passes are the price of variety.

**The reference is the paper's own setting.** The requester asked on 2026-11-16 to measure against what the paper evaluated, not the released code's default:
- **The paper's setting:** t_switch = 0.25, 14 teacher steps and 2 student steps.
- **Never measured for variety:** EXP-017 ran this setting but did not measure variety.
- **The thesis's version (W8) differs:** it restarts the student with mostly fresh noise, which may wash out later teacher work.

**What the study answers:**
1. **The paper's 100%:** how much variety does the paper's setting give, and does the thesis's W8 keep it?
2. **The curve:** how much of it does each cheaper amount of teacher work keep (variety and quality per pass)?
3. **Noisy prompts:** does that variety survive a noisy prompt, and does any teacher step keep the student's quality there?
4. **The handover:** does continuing from the teacher's state (the paper's Algorithm 2) differ from re-noising its estimate (the released code)?

The answers set the target for the next handover design: keep the paper's variety for fewer passes than simply stopping the teacher earlier.

## What runs

| Arm | What it is | Passes |
|---|---|---|
| `P25` | **The 100% reference.** The paper's Algorithm 2 at t_switch = 0.25: 14 guided teacher steps on the 32-interval sway grid (t₁₄ = 0.2270), then the student continues from the teacher's state and steps again at 0.6173. The paper does not state the second time; it is assumed, as in EXP-017's `alg2` | 30 |
| `P25R` | The same 14 teacher steps, with the released code's handover: the estimate is re-noised at the student's levels 0.2929 and 0.6173 (EXP-017's `paper`) | 32 |
| `W1`, `W2`, `W3`, `W4`, `W8` | The released sampler on the 16-interval grid: teacher to Step k, then the student's three levels. `W8` is the thesis's version of 0.25 (Chapter 3's reference); `W4` is the released code's default (0.07) | 7, 9, 11, 13, 21 |
| `STU` | The student alone | 4 |
| `TCH` | The teacher alone, 32 guided steps (quality stage only) | 64 |

| Stage | Cases | Arms |
|---|---|---|
| Quality (UTMOS, WER, speaker similarity) | Development speakers 1–30 × clean and noisy15 = 60 | all nine |
| Prosodic variety (CVf0, aligned CVf0, timing spread) | Speakers 1–20 × 12 seeds × clean and noisy15 | all but `TCH` |

- **noisy15:** white noise at 15 dB, the thesis's noisy condition (EXP-002). The noise is the same for every seed of a speaker.
- **The speakers and seeds** are EXP-031's, and the paper's arms match EXP-017's speakers. That allows two reproduction checks:
  - `STU`, `W4` and `W8` must repeat EXP-031's scores;
  - `P25`, `P25R`, `TCH`, `STU`, `W4` and `W8` are compared with EXP-017's scores.
- **Measures:** EXP-031's.
  - **CVf0:** pitch variety across seeds, frame by frame. Timing differences count.
  - **Aligned CVf0:** each seed is first aligned to the first seed in time, so only pitch movement beyond timing counts.
  - **Timing spread:** how far the seeds' timings differ, in frames.
- **Tensor bank** (speakers 1–5, both conditions, all 12 seeds; about 120 MB, kept on Drive and not committed):
  - **contents:** the prompt mel, the student's one-step draft, the teacher's estimates at Steps 1, 2, 3, 4 and 8 and at the paper's Step 14, the outputs of `STU`, `W4`, `W8` and `P25`, and every arm's f0 track;
  - **format:** float16, generated frames only;
  - **hashes:** in `bank_index.json`.
  - **Use:** new handover designs can be screened offline before any GPU run.

## Smoke test

A run stops before synthesis if any of these fails:
- **Exact identities:**
  - the split student loop and a repeat of `P25`'s student (≤ 1e-5);
  - the 32-interval teacher function run on the 16-interval grid, against the trajectory the `W` arms use (≤ 1e-4).
- **Reproductions:**
  - EXP-027's `W4` (≤ 0.01);
  - development speaker 1 under both conditions, `STU`, `W4` and `W8` against EXP-031's stored scores (≤ 0.01).
  - The paper's arms are compared with EXP-017's stored scores and recorded, not gated.
- **The tensor bank:** a two-seed variety run on one speaker writes one tensor-bank file, which is reloaded and checked.

## Estimands and label rules (frozen)

**Variety share:** with V = CVf0 or aligned CVf0, averaged over the 20 speakers:

share_A = (V_A − V_STU) / (V_P25 − V_STU)

- **What it means:** the part of the paper setting's variety gain over the student alone that arm A keeps.
- **Interval:** a 95% speaker-level bootstrap interval (B = 2000, seed 2026).
- **"At least 90%" confirmed:** when L_A = V_A − 0.9·V_P25 − 0.1·V_STU has a bootstrap lower bound above 0. The 90% bar follows the paper's own teacher-guided setting, which kept 89.1% of its teacher's CVf0.

**Quality bars** (EXP-029's clean bars, all three must hold):
- **The bars:** mean ΔUTMOS ≥ −0.08, ΔWER upper bound ≤ +0.01, ΔSIM lower bound ≥ −0.01.
- **Clean prompts:** against `P25`.
- **noisy15 prompts:** against the student alone.

| Label | Rule |
|---|---|
| `cheapest_step_point` | Clean: the smallest k in (1, 2, 3, 4, 8) whose `W_k` keeps ≥ 90% of `P25`'s gain on both measures (point estimates) and passes the quality bars against `P25`; otherwise `none` |
| `cheapest_step_confirmed` | The same, with the 90% confirmed on both measures |
| `w8_matches_paper` | `W8` meets the `cheapest_step_point` conditions |
| `noisy_cheapest_step_point` | noisy15: the smallest k whose `W_k` keeps ≥ 90% of `P25`'s gain on both measures (variety only) |
| `noisy_quality_safe_arms` | noisy15: the teacher arms that pass the quality bars against the student alone |

**Reported descriptively:**
- the full curves with intervals;
- each arm's variety change from clean to noisy15;
- `P25` − `P25R` (the handover mechanism);
- `TCH` − `P25` quality;
- the reproductions.

## What a result would mean

| Result | Meaning |
|---|---|
| `cheapest_step_point` = W2 or W3 | Stopping the teacher that early already keeps the paper's variety. A new handover must beat that curve, not just W4 |
| `w8_matches_paper` false | The thesis's 0.25 (W8) loses variety that the paper's setting has. The re-noising handover or the coarser grid costs variety, which matters for Chapter 3's reference |
| `P25` above `P25R` on variety | Continuing from the teacher's state keeps variety that re-noising loses. *How* the handover crosses matters, not only *when* |
| No teacher arm in `noisy_quality_safe_arms`, but the variety survives noise | On noisy prompts, the variety and the copied noise arrive together. A handover that separates them is the design target |

**Known limits:**
- **Pitch tracking on noisy outputs:** arms that copy the noise may be tracked less reliably, which can inflate or deflate their variety. The voiced share is reported for this reason.
- **No listening test, and no held-out speakers.**

## Results (2026-11-16)

Both stages complete: 60 of 60 quality cases and 40 of 40 variety cases.
- **Checks:** smoke identities exactly 0. `STU`, `W4` and `W8` repeat EXP-031, and every shared arm repeats EXP-017 (largest UTMOS difference 2.6e-3).
- **Recomputation:** `analyze()` rerun locally gives the same results.
- **Full details:** the [results log](../../../docs/progress/log/2026-11-16-exp032-results.md).

**Frozen labels:**

| Label | Value |
|---|---|
| `cheapest_step_point` | `none` |
| `cheapest_step_confirmed` | `none` |
| `w8_matches_paper` | `false` |
| `noisy_cheapest_step_point` | `none`: **not a finding**, because noisy15 pitch variety is undefined (below) |
| `noisy_quality_safe_arms` | `[]` |

**Clean prompts:**

| Arm | Passes | UTMOS | CVf0 | Aligned CVf0 | Share of `P25`'s gain: CVf0 | Aligned |
|---|---:|---:|---:|---:|---|---|
| `STU` | 4 | 3.940 | 0.0521 | 0.0410 | 0% | 0% |
| `W2` | 9 | 3.831 | 0.0862 | 0.0574 | 52% [37%, 67%] | 35% [16%, 52%] |
| `W3` | 11 | 3.822 | 0.0908 | 0.0571 | 59% [42%, 80%] | 34% [13%, 49%] |
| `W4` | 13 | 3.867 | 0.0918 | 0.0584 | 61% [42%, 81%] | 37% [16%, 54%] |
| `W8` | 21 | 3.885 | 0.0912 | 0.0576 | 60% [41%, 80%] | 35% [9%, 53%] |
| `P25` | 30 | 3.709 | 0.1173 | 0.0881 | 100% | 100% |
| `P25R` | 32 | 3.726 | 0.1150 | 0.0889 | 97% [89%, 106%] | 102% [92%, 116%] |

**What it shows:**
- **`W8` does not keep the paper's variety.** The thesis's version of 0.25 keeps about a third of `P25`'s aligned pitch variety gain over the student alone. `P25` is higher for 19 of 20 speakers.
- **The variety plateaus.** On the released grid it stops growing after Step 2–3: `W3`, `W4` and `W8` do not differ (post hoc intervals include zero).
- **The restart level is the likely cause** (exploratory):
  - **The handover mechanism doesn't matter:** `P25` and `P25R` do not differ.
  - **The restart level does:** `P25R`, whose student restarts at 0.2929, is above `W8` (restart at 0.0761) on aligned CVf0 for 20 of 20 speakers. `W8`'s teacher had gone further, so the gap is not missing teacher work.
  - **The confound:** the two arms also differ in grid.
- **The variety has a cost.** It comes with lower predicted naturalness: `P25` is 0.176 UTMOS below `W8`.
- **noisy15:**
  - **Variety:** pYIN voices no frames in the teacher arms' outputs, which copy the white noise. Their variety is undefined, and the noisy label reflects that, not a loss of variety.
  - **Quality:** every teacher arm is about 0.7 UTMOS below the student alone.

## Offline screen on the tensor bank (2026-11-16, post hoc)

The bank was received and checked against `bank_index.json`. The screen vocodes the stored mels on a CPU and tracks their pitch; no TTS network runs.
- **Speakers:** 1–5, clean prompts.
- **Script:** `code/analysis/exp032_bank_pitch_screen.py`.
- **Output:** `offline/bank_pitch_screen.json`.
- **Write-up:** [log](../../../docs/progress/log/2026-11-16-exp032-bank-pitch-screen.md).

**What it found:**
- **The teacher decides the melody between Steps 3 and 8.** Its Step 8 estimate is within a semitone of `P25`'s final melody on 82% of frames. Two different seeds agree on 36%.
- **`W8`'s restart at 0.0761 loses most of it:** the output keeps 53% of the Step 8 melody. The paper's restarts keep 79% (`P25R`, 0.2929) and 86% (`P25`).
- **This is the melody-level view of EXP-026's result:** under j = 1 the restart level, not the teacher's steps, limits the variety.
- **noisy15:** the teacher's estimates are almost unvoiced from Step 2 on, so their melody cannot be measured.

**Earlier restart experiments against `P25` (post hoc):**
- **The setup:** EXP-026–029 already ran the student restarted at 0.2929 (`W4R2`, `W5R2`, `W6R2`, `W8R2`). EXP-026 scored 30 of the same clean cases, with a matching `W8`.
- **The result:** on those cases `W8R2` (20 passes) matches `P25` (30 passes) in UTMOS (+0.003 [−0.126, +0.112]). Its CVf0 on the same speakers is about the same (0.1171 against 0.1173; different seed sets, not paired).
- **Details:** point 8 of the results log, and `diagnostics.json`.

## Data and evidence status

- **Development evidence only.**
- **Audio and the tensor bank** are not committed. `.npz` files are ignored.
- **Results** are stored under `results/`, byte-identical to the run's zip:
  - with a SHA-256 manifest;
  - with `bank_index.json`, the tensor bank's file hashes (120 files, 97,720,820 bytes; the bank itself stays on Drive);
  - and `diagnostics.json`, the post hoc diagnostics from `code/analysis/exp032_diagnostics.py`.
