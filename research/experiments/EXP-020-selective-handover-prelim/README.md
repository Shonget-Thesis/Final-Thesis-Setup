# EXP-020: Selective Handover

**Status:** run 2026-11-08 UTC on a Colab Tesla T4 (thesis `07fe593`, Shonget-DMOSpeech2 `8a7868f`), 360 of 360 cases and 20 of 20 diversity pairs; results in [results/](results/MANIFEST.json). **It does not advance: C1 to C4 pass, C5 and C6 fail.** Preregistration `7afeef06fabefaa8...`. Task EVAL-013 (review). Development evidence only; nothing here is adopted by the thesis, and no adviser has reviewed it.

An earlier version of this experiment, Voice-Swap only (preregistration `8fb1600f2355604e...`, commit `9554165`), was replaced before any synthesis. Voice-Swap is the primary combination here.

- Runner: [run_selective_handover_prelim.py](../../../code/colab/run_selective_handover_prelim.py)
- Notebook: [DMOSpeech2_EXP020_Selective_Handover.ipynb](../../../code/colab/DMOSpeech2_EXP020_Selective_Handover.ipynb)
- Tests: [test_selective_handover.py](../../../code/tests/test_selective_handover.py)
- Preregistration: [preregistration.json](preregistration.json)

## Idea

Handover methods found so far pass the whole state across. Selective Handover asks which parts of the teacher's estimate the student actually needs. No model is trained: the released teacher and student are used unchanged.

1. The teacher runs to step k and gives its estimate T.
2. One student pass from independent noise gives a draft S (1 forward pass).
3. On generated frames, T and S are each split into three parts that add back exactly: **voice** (per-band average over time), **slow** (a moving average over 9 frames, about 96 ms, of the remainder) and **fine** (what is left).
4. Each part is taken from the teacher or the student, and the joined result goes to the student's normal restart.

A combination is named by three letters for (voice, slow, fine). TTT is today's handover, STT is Voice-Swap, and SSS is the student alone (stored student-only stands in for it, with its own noise draw).

## Per-input step choice

Prompt SNR picks a rung of a ladder for each input, from cheap to expensive:

| Rung | Output | Forward passes |
|---|---|---:|
| 1 | Plain handover at Step 4 | 13 |
| 2 to 5 | STT at Steps 4, 5, 6, 7 | 14, 16, 18, 20 |
| Otherwise | Reference, Step 8 | 21 |

## What is run

- **The map:** the six mixed combinations at Step 4 on 360 cases, together with stored TTT and SSS, give the effect of each part on similarity, WER, failures, UTMOS and CVf0.
- **The later steps:** STT at Steps 5, 6 and 7.
- **Diversity:** the first 20 development speakers, clean prompt, 20 seeds each, CVf0 from librosa pYIN. The DMOSpeech 2 paper gives no formula, seed handling or voicing rule, so those choices are this study's and are recorded in the preregistration.

Quality uses the EXP-012 Quick set (60 development speakers x clean, noisy15 and Gaussian 18/20/22/25 dB). These speakers were inspected before this design. Speakers 61-100 are not used, so they stay available to EXP-019 or a later confirmation.

## Preregistered criteria

The primary combination is STT, fixed in advance so that eight combinations on 60 speakers cannot become a search for the best one. The other five mixed combinations form the descriptive map. Selective Handover advances to a frozen test on fresh speakers only if all six criteria pass. Otherwise it is reported as a negative result, with no retuning on these data. The map is reported either way.

| Criterion | Requirement |
|---|---|
| C1 noisy failures | `STT_4` at most 75% of `plain_4` failures on the five noisy conditions, fewer, exact McNemar p < 0.05 |
| C2 clean failures | `STT_4` failures at most `plain_4` + 1 |
| C3 the voice is the cause | Main effect of taking the voice part from the student on noisy similarity: 95% speaker-bootstrap lower bound > 0 |
| C4 diversity | CVf0 of `STT_4` above student only (lower bound > 0) and not below `plain_4` (lower bound > -0.02) |
| C5 beats the calibrated SNR rule | Out-of-fold `ladder` has lower mean passes than `snr_theta` with no more failures |
| C6 beats the ladder without selection | Out-of-fold `ladder` has lower mean passes than `ladder_plain` with no more failures |

Failures use Definition 3.1 against plain Step 8, whose noise floor on noisy prompts is documented in EXP-017.

**Calibration.** The ladder has one fixed-sequence Learn-Then-Test chain over its base cut-off for each rung spacing in {1, 2, 3} dB, with the risk level delta split equally across the chains (Bonferroni). The cheapest certified setting is used.

## Results

The smoke test reproduced the 8 stored Step-4 and Step-8 labels (largest difference about 1e-7). Report: [SELECTIVE_HANDOVER_REPORT.md](results/SELECTIVE_HANDOVER_REPORT.md).

**Verdict: Selective Handover does not advance.** As preregistered, it is reported as a negative result with no retuning on these data. Speakers 61-100 stay unused.

| Criterion | Measured | Result |
|---|---|---|
| C1 noisy failures | `STT_4` 74 of 300 against `plain_4` 126 (59%); only `plain_4` fails on 86, only `STT_4` on 34; p = 2.3e-6 | Pass |
| C2 clean failures | 2 against 2 | Pass |
| C3 the voice is the cause | Noisy similarity +0.0752 [+0.0635, +0.0874] | Pass |
| C4 diversity | CVf0 against student +0.0383 [+0.0240, +0.0555]; against `plain_4` -0.0044 [-0.0144, +0.0018] | Pass |
| C5 beats the calibrated SNR rule | `ladder` 19.68 passes, 9 failures; `snr_theta` 18.71 passes, 7 failures | **Fail** |
| C6 beats the ladder without selection | `ladder` 19.68 passes, 9 failures; `ladder_plain` 19.27 passes, 8 failures | **Fail** |

**The map.** Each part does a different job (main effect of taking it from the student, Step 4, 95% speaker interval).

| Part | Noisy similarity | Noisy failure rate | CVf0 (clean prompts) | Reading |
|---|---|---|---|---|
| voice | +0.0752 [+0.0635, +0.0874] | -0.2267 [-0.2942, -0.1617] | -0.0020 [-0.0086, +0.0019] | Carries the prompt's noise; no detectable diversity cost |
| slow | -0.0028 [-0.0087, +0.0031] | +0.0150 [-0.0325, +0.0667] | -0.0402 [-0.0589, -0.0249] | Carries the pitch diversity; keep the teacher's |
| fine | +0.0064 [+0.0042, +0.0087] | -0.0150 [-0.0333, +0.0042] | -0.0010 [-0.0027, +0.0011] | Little effect |

## Diagnostics (not preregistered)

These were computed after the results were seen, so they explain the verdict and cannot change it. [exp020_diagnostics.py](../../../code/analysis/exp020_diagnostics.py) prints every number here from the committed CSV files without importing the runner.

| Output | Passes | Noisy failures (of 300) | Noisy WER | Noisy similarity | Noisy UTMOS | Clean failures (of 60) |
|---|---:|---:|---:|---:|---:|---:|
| student only | 4 | 51 | 0.049 | 0.862 | 2.69 | 8 |
| `plain_4` | 13 | 126 | 0.109 | 0.777 | 1.48 | 2 |
| `STT_4` | 14 | 74 | 0.075 | 0.847 | 2.54 | 2 |
| `STT_6` | 18 | 61 | 0.069 | 0.850 | 2.53 | 0 |
| `STT_7` | 20 | 61 | 0.079 | 0.849 | 2.48 | 0 |
| `plain_8` (reference) | 21 | 0 by definition | 0.075 | 0.785 | 1.46 | 0 |

Failure rate by measured prompt SNR:

| Prompt SNR (dB) | Cases | `plain_4` | `STT_4` | `plain_6` | `STT_6` | `plain_7` | `STT_7` | Student only |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 or more | 49 | 4.1% | 4.1% | 0% | 0% | 0% | 0% | 14.3% |
| 25 to 30 | 74 | 16.2% | 16.2% | 9.5% | 14.9% | 9.5% | 13.5% | 20.3% |
| 22 to 25 | 107 | 34.6% | 19.6% | 23.4% | 15.9% | 18.7% | 17.8% | 14.0% |
| 20 to 22 | 67 | 52.2% | 29.9% | 32.8% | 25.4% | 23.9% | 20.9% | 7.5% |
| 18 to 20 | 20 | 55.0% | 40.0% | 35.0% | 30.0% | 45.0% | 20.0% | 10.0% |
| Under 18 | 43 | 72.1% | 30.2% | 55.8% | 23.3% | 39.5% | 32.6% | 34.9% |

- **The ladder had no room to win.** At 25 dB or more the swap has no failure advantage over plain handover and costs one more pass; at Steps 6 and 7 between 25 and 30 dB it is worse. Below 25 dB it reduces failures, but the rate stays near 20% or above, far over the 5% limit, so no noisy rung can be certified.
- **The 5% target could not be met on noisy prompts by this design.** Step 8 and the 32-interval s = 0.25 reference, which have equal mean quality, flag each other on 78 and 81 of 300 noisy inputs (EXP-017). That floor was known before this design and the criteria used the rule anyway, which is a weakness of the preregistration. It does not change the verdict. `STT_4`'s 74 is the same size as the floor but not the same kind: 43 of its failures are on the word arm, against 27 for the floor.
- **The swap fixes voice failures, not word failures.** Against `plain_4` on noisy prompts, the voice arm falls from 96 to 34 and the word arm only from 47 to 43, although mean WER improves (0.109 to 0.075).
- **Against the reference, on means.** At 14 against 21 passes, `STT_4` on noisy prompts has higher similarity (+0.062 [+0.048, +0.076]) and UTMOS (+1.08 [+0.93, +1.23]) and equal WER (+0.0002 [-0.018, +0.018]). On clean prompts no difference is detected in similarity, WER or UTMOS, and CVf0 is -0.005 [-0.015, +0.001]. The reference copies the prompt's noise, so doing better than it on noisy prompts is a low bar.
- **Student only is still better on noisy prompts.** It has 51 failures against 74 (p = 0.004), higher similarity, lower WER and higher UTMOS at 4 passes against 14. `STT_4`'s advantages are clean prompts (2 failures against 8) and pitch diversity on clean prompts (0.0898 against 0.0515). CVf0 was not measured on noisy prompts, which is where the swap matters, so a case for `STT_4` over the student alone on noisy prompts is not established.
- **Later teacher steps add little once the voice is swapped.** Against `STT_4`, Steps 5 to 7 show no detectable change in noisy similarity or WER, and UTMOS at Step 7 is lower (-0.053 [-0.108, -0.006]). Failures are 74, 73, 61 and 61 (p = 1.0, 0.07 and 0.07 against Step 4). A per-input choice among Steps 4 to 7 has little to gain here.

## How to run

Open the notebook on a Colab T4 and run all cells. The run took about 4.6 hours of per-case time (3.2 quality, 1.5 diversity), against an estimate of 5 to 6. Step 2b keeps the checkpoints and the English dataset in `MyDrive/dmospeech2_cache/` after the first run, so later runs copy them from Drive instead of downloading them; the checkpoint hashes are still checked every time. The runner refuses to start if the preregistration or the runner has changed since the freeze, if the GPU is not a T4, or if the smoke test did not reproduce the stored labels.

## Prior work and novelty

Not established. The handover and model-switching methods found so far (DMOSpeech 2, Gandikota and Bau, SRDiffusion) pass the whole state across, and no paper measuring or selecting which parts cross a teacher-to-student handover was found in targeted searches, which is not proof of absence. Treating low and high frequencies differently is known in diffusion guidance, and channel statistics are known from voice conversion. Any claim of novelty needs a full literature review.

## Caveats

- Development evidence on inspected speakers, white-noise prompts only.
- The student's draft has its own timing, so mixing its parts with the teacher's may give inconsistent input; the map is meant to show where this matters.
- Noise is not only in the averages, and prosody and timbre are entangled (noted by DiffStyleTTS and Diffsody).
- A similar shift toward the noisy prompt's averages made failures worse in EXP-014 (SEA, 140 against 126).
- The failure rule is relative to a reference that is poor on noisy prompts, so similarity and CVf0 are also reported.
- `student_sample` and the torch path through the operators ran only on the T4 in this notebook, not locally, because torch is not installed here.
- Costs are counted in forward passes. The wall-clock time of each combination was not measured separately.
- CVf0 was measured only on clean prompts. Its scale (`plain_8` 0.0948) differs from the DMOSpeech 2 paper's teacher-guided value (0.5932), and the paper gives no formula, so only the ordering can be compared.
