# EXP-020: Selective Handover

**Status:** preregistered 2026-11-08 (`7afeef06fabefaa8...`), not yet run. Task EVAL-013 (todo). Development evidence only; nothing here is adopted by the thesis, and no adviser has reviewed it.

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

## How to run

Open the notebook on a Colab T4 and run all cells. About 5 to 6 hours by estimate, not measured. The runner refuses to start if the preregistration or the runner has changed since the freeze, if the GPU is not a T4, or if the smoke test did not reproduce the stored labels.

## Prior work and novelty

Not established. The handover and model-switching methods found so far (DMOSpeech 2, Gandikota and Bau, SRDiffusion) pass the whole state across, and no paper measuring or selecting which parts cross a teacher-to-student handover was found in targeted searches, which is not proof of absence. Treating low and high frequencies differently is known in diffusion guidance, and channel statistics are known from voice conversion. Any claim of novelty needs a full literature review.

## Caveats

- Development evidence on inspected speakers, white-noise prompts only.
- The student's draft has its own timing, so mixing its parts with the teacher's may give inconsistent input; the map is meant to show where this matters.
- Noise is not only in the averages, and prosody and timbre are entangled (noted by DiffStyleTTS and Diffsody).
- A similar shift toward the noisy prompt's averages made failures worse in EXP-014 (SEA, 140 against 126).
- The failure rule is relative to a reference that is poor on noisy prompts, so similarity and CVf0 are also reported.
- `student_sample` and the torch path through the operators have not been run locally, because torch is not installed. The notebook's smoke test is the first run of that path.
