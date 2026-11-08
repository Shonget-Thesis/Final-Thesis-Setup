# EXP-020: Voice-Swap Handover

**Status:** preregistered 2026-11-08 (`8fb1600f2355604e...`), not yet run. Task EVAL-013 (todo). Development evidence only; nothing here is adopted by the thesis, and no adviser has reviewed it.

- Runner: [run_voice_swap_prelim.py](../../../code/colab/run_voice_swap_prelim.py)
- Notebook: [DMOSpeech2_EXP020_Voice_Swap_Handover.ipynb](../../../code/colab/DMOSpeech2_EXP020_Voice_Swap_Handover.ipynb)
- Tests: [test_voice_swap.py](../../../code/tests/test_voice_swap.py)
- Preregistration: [preregistration.json](preregistration.json)

## Idea

Teacher-guided sampling passes the teacher's whole endpoint estimate to the student. Voice-Swap Handover keeps the part that changes over time from the teacher and takes the per-band average over time from one quick student draft. No model is trained: the released teacher and student are used unchanged.

1. The teacher runs to Step 4 and gives its estimate T.
2. One student pass from independent noise gives a draft: S = eps + g(eps, 0).
3. On generated frames, x = T - bandavg(T) + bandavg(S). Prompt frames are unchanged.
4. The student's normal restart runs from x.

Prompt SNR chooses the route for each input:

| Prompt | Route | Forward passes |
|---|---|---:|
| SNR at or above theta_hi | Plain handover at Step 4 | 13 |
| theta_lo to theta_hi | Voice-Swap at Step 4 | 14 |
| Below theta_lo | Reference, Step 8 | 21 |

## Why this experiment

- At Step 4, 96 of 126 noisy-prompt failures are speaker-similarity failures (EXP-013).
- DMOSpeech 2 states that early steps set prosody and late steps refine voice quality and speaker identity. It reports no direct measurement of that split, and neither does any source found so far.
- In voice conversion, channel statistics over time carry speaker information (AdaIN-VC, AGAIN-VC).
- Noisy prompts carry their noise into the generated speech (Wang et al., 2024).

## Data

- **Quality:** the EXP-012 Quick set, 60 development speakers x clean, noisy15 and Gaussian 18/20/22/25 dB (360 cases). These speakers were inspected before this design.
- **Diversity:** the first 20 development speakers, clean prompt, 20 seeds each, CVf0 from librosa pYIN. The DMOSpeech 2 paper gives no formula, seed handling or voicing rule, so those choices are this study's and are recorded in the preregistration.
- Speakers 61-100 are not used, so they stay available to EXP-019 or a later confirmation.

Plain Steps 4 and 8 (EXP-018) and student-only (EXP-017) are reused. The smoke test must reproduce the stored labels before the full run can start.

## Preregistered criteria

Voice-Swap advances to a frozen test on fresh speakers only if all five pass. Otherwise it is reported as a negative result, with no retuning on these data.

| Criterion | Requirement |
|---|---|
| C1 noisy failures | `vswap_4` at most 75% of `plain_4` failures on the five noisy conditions, fewer, exact McNemar p < 0.05 |
| C2 clean failures | `vswap_4` failures at most `plain_4` + 1 |
| C3 the voice is the cause | Noisy similarity, `vswap_4` minus the reversed control `rswap_4`: 95% speaker-bootstrap lower bound > 0 |
| C4 diversity | CVf0 above student only (lower bound > 0) and not below `plain_4` (lower bound > -0.02) |
| C5 beats the calibrated SNR rule | Out-of-fold `route3` has lower mean passes than `snr_theta` with no more failures |

Failures use Definition 3.1 against plain Step 8, whose noise floor on noisy prompts is documented in EXP-017.

**Calibration of the two cut-offs.** Each theta_hi in {25, 30, 35, 40} dB has one fixed-sequence Learn-Then-Test chain over theta_lo, from theta_hi down to 12 dB. The risk level delta is split equally across the four chains (Bonferroni), which keeps the procedure valid when two cut-offs are chosen from the same data. The cheapest certified pair is used.

## How to run

Open the notebook on a Colab T4 and run all cells. About 2.5 hours by estimate, not measured. The runner refuses to start if the preregistration or the runner has changed since the freeze, if the GPU is not a T4, or if the smoke test did not reproduce the stored labels.

## Prior work and novelty

Not established. Swapping channel statistics is known from voice conversion and appears in diffusion style transfer. Handover and model-switching methods found so far (DMOSpeech 2, Gandikota and Bau, SRDiffusion) pass the whole state across. No paper applying this swap at a teacher-to-student handover was found in targeted searches, which is not proof of absence. Any claim of novelty needs a full literature review.

## Caveats

- Development evidence on inspected speakers, white-noise prompts only.
- Noise is not only in the averages, and prosody and timbre are entangled (noted by DiffStyleTTS and Diffsody).
- A similar shift toward the noisy prompt's averages made failures worse in EXP-014 (SEA, 140 against 126).
- The failure rule is relative to a reference that is poor on noisy prompts, so similarity and CVf0 are also reported.
- `student_sample` and the torch path through the operators have not been run locally, because torch is not installed. The notebook's smoke test is the first run of that path.
