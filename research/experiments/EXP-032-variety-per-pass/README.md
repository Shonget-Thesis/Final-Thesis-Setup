# EXP-032: how much variety each teacher pass buys, against the paper's t_switch = 0.25

**Status:** prepared and preregistered on 2026-11-16 (Asia/Manila), before any synthesis; freeze `c952fe555d49...`.

Task EVAL-025 (todo). The run needs a Colab Tesla T4, about 3 hours. Files: [notebook](../../../code/colab/DMOSpeech2_EXP032_Variety_Per_Pass.ipynb), [runner](../../../code/colab/run_variety_per_pass.py), [tests](../../../code/tests/test_variety_per_pass.py).

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

## Data and evidence status

- **Development evidence only.**
- **Audio and the tensor bank** are not committed. `.npz` files are ignored.
- **Results** will be stored under `results/`:
  - with a SHA-256 manifest;
  - with `bank_index.json`, the tensor bank's file hashes.
