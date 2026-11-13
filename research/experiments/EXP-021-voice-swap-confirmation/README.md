# EXP-021: Voice-Swap Handover, frozen confirmation

**Status:** preregistered 2026-11-13 and amended the same day before any synthesis (`33f37d54ea30cdca...`), not yet run. Task EVAL-014 (blocked: parked at the requester's request on 2026-11-13). Nothing here is adopted by the thesis, and no adviser has reviewed it.

**Amendment (2026-11-13, before any synthesis on speakers 61-100): P3 is judged on clean prompts only.**
- **The problem:** the first preregistration (`1082c66869fd2f33...`, in git history at `8b8bc54`) also required P3 on noisy15 prompts. In EXP-022, on 20 development speakers, the reference's outputs on those prompts had no frame voiced in at least half of the 20 seeds. Its CVf0 does not exist there, so P3 could not have passed, whatever Voice-Swap does.
- **What changed:** noisy15 pitch variety is still measured and is now reported without a pass or fail, with the number of speakers whose pitch could be tracked.
- **What did not change:** margins, data, outputs and the other criteria.
- **Requested by:** Rhenel Jhon R. Sajol. See the [amendment log](../../../docs/progress/log/2026-11-13-exp021-p3-amendment.md).

- Runner: [run_voice_swap_confirm.py](../../../code/colab/run_voice_swap_confirm.py)
- Notebook: [DMOSpeech2_EXP021_Voice_Swap_Confirmation.ipynb](../../../code/colab/DMOSpeech2_EXP021_Voice_Swap_Confirmation.ipynb)
- Tests: [test_voice_swap_confirm.py](../../../code/tests/test_voice_swap_confirm.py)
- Preregistration: [preregistration.json](preregistration.json)

## Why this test

EXP-020 did not advance Selective Handover: its adaptive SNR ladder failed. After its results were seen, a diagnostic showed that the fixed Voice-Swap combination (STT at Step 4) matched the 0.25 reference on the 60 development speakers at two-thirds of the forward passes. A diagnostic chosen after the results is not evidence, so this experiment tests that one claim with the pass rules frozen first, on speakers that were never used to design Voice-Swap.

## The method

| Step | What happens | Passes |
|---|---|---:|
| 1 | The guided teacher runs to Step 4 and gives its estimate T | 10 |
| 2 | One student pass from independent noise gives a draft S | 1 |
| 3 | On generated frames, T's voice part (each band's average over time) is replaced by S's; T's slow and fine parts are kept | 0 |
| 4 | The released student loop restarts from that input as usual | 3 |

Total 14 passes, against 21 for the reference (Step 8). Every operation is imported unchanged from the EXP-020 runner, and the smoke test checks that it reproduces EXP-020's stored Voice-Swap scores.

## Design

- **Speakers:** development speakers 61-100 by utterance ID (40 speakers). EXP-012 to EXP-020 used speakers 1-60, so these were never used to design Voice-Swap. EXP-001 and EXP-002 synthesized them for the problem grid (clean and noisy15, Steps 4/5/6/8).
- **Shared with EXP-019:** EXP-019's preregistration names the same 40 speakers as its fresh set. EXP-021 tunes nothing on them, but the team should know that both tests would use them.
- **Prompt conditions:** clean, white noise at 15 and 20 dB, recorded cafeteria babble at 15 dB (DEMAND PCAFETER, channel 1, from Zenodo), synthetic reverberation (RT60 0.9 s) and telephone band. 240 cases in total.
- **Outputs per case:** `ref` and `ref2` (Step 8 with two student noise draws), `vs` and `vs2` (Voice-Swap with two draft and student noise draws), `plain4` (today's Step-4 handover) and `student` (student only). Averaging two draws per method halves the sampling noise in the paired comparison, and `ref2` against `ref` measures the failure rule's own noise floor.
- **Timing:** the reference path and the Voice-Swap path are each run and timed separately on the T4, from the start of the teacher to decoded audio, with CUDA synchronized. The order alternates between cases, and one untimed warm-up case runs first.
- **Pitch variety:** CVf0 as defined in EXP-020, on the first 30 fresh speakers, with clean and noisy15 prompts and 20 seeds each, for `ref`, `vs` and `student`. P3 uses the clean prompts; the noisy15 values are reported only, because the reference's pitch cannot be tracked there (EXP-022).

## Preregistered criteria

Voice-Swap is confirmed on these speakers only if all four pass. Otherwise it is a negative result, with no retuning and no change of margins.

| Criterion | Requirement (speaker-bootstrap 95% CI) |
|---|---|
| P1 word errors | WER of Voice-Swap minus the reference (both averaged over their two draws, capped at 1): upper bound at most +0.01 on clean prompts and at most +0.025 on the five degraded conditions pooled |
| P2 voice similarity | the same difference in speaker similarity: lower bound at least -0.01 on clean and on degraded prompts |
| P3 pitch variety | on clean prompts: CVf0 of Voice-Swap minus the reference, lower bound at least -0.02; Voice-Swap minus the student alone, lower bound above 0 |
| P4 time | measured time of Voice-Swap divided by the reference's, per case: upper bound at most 0.80 |

**How the margins were chosen.** They were sized before any fresh data by resampling 40-speaker studies from the EXP-020 data. If Voice-Swap is exactly as good as it looked in EXP-020:
- **Degraded WER:** it passes the +0.025 margin with probability about 0.8 using single draws; the two draws per method should raise that. It would pass only 0.2 to 0.4 of the time at +0.01 or +0.015, because 40 speakers cannot resolve smaller WER differences.
- **Clean margins:** it passes them with probability about 1.
- **Pitch variety:** it passes with probability about 0.76 at 20 pairs, which is why this test uses 30.
- **Time:** EXP-018's T4 timings predict a ratio of about 0.71.

The descriptive results, which carry no pass or fail, are:
- per-condition differences;
- Definition 3.1 failures of every output against `ref`, with `ref2` as the noise floor;
- McNemar of `plain4` against `vs`;
- Voice-Swap against the student alone;
- real-time factors;
- pitch variety on noisy15 prompts, with the number of speakers whose pitch could be tracked;
- a check that both timed paths produce the same Step-4 estimate.

**Check of the amended P3 on development data.** On EXP-020's clean-prompt pitch data (speakers 1-20, the same three outputs), P3 can be computed ([exp022_diagnostics.py](../../../code/analysis/exp022_diagnostics.py), Part 3):

| Comparison | Result | Requirement |
|---|---|---|
| Voice-Swap minus the reference | -0.0050 [-0.0157, +0.0009] | at least -0.02 |
| Voice-Swap minus the student alone | +0.0383 [+0.0240, +0.0555] | above 0 |

These speakers designed Voice-Swap, so this shows only that the rule is computable, not that EXP-021 will pass.

## How to run

Push the branch, open the notebook on a Colab T4 and run all cells. About 4 hours by estimate, not measured.
- **Caching:** Step 2b keeps the checkpoints and dataset in Google Drive. Step 2c does the same for the DEMAND recording.
- **Smoke test:** it uses development speakers only. It must reproduce the stored scores of the reference (EXP-018), Step 4 (EXP-018), Voice-Swap (EXP-020) and the student alone (EXP-017) before the full run can start.
- **The runner refuses to start if** any of its seven code files, the preregistration or the stored input files have changed; if the GPU is not a T4; if Shonget-DMOSpeech2 is not at `8a7868f`; or if the DEMAND file differs from the smoke test's.
- **On the branch:** the EXP-020 results (`results/selective.csv`) must be there, because the smoke test reads them.

## Prior work and novelty

To our knowledge, no published method chooses which parts of the teacher's estimate cross a teacher-to-student handover. This rests on targeted searches from 2026-11-08 to 2026-11-13, not a full literature review. The closest work found:

- **Whole-state handovers:** DMOSpeech 2's teacher-guided sampling, SRDiffusion, and Gandikota and Bau's diversity distillation all pass the teacher's whole state to the other model.
- **Teacher-student collaboration:**
  - Starodubcev et al. (CVPR 2024) let the student go first and use an estimator to decide whether the teacher refines it.
  - Distillation++ (ICCV 2025) lets the teacher nudge the student's estimates.
  - Neither mixes components of the two models' estimates.
- **Frequency splits:**
  - Frequency-guided diffusion for image translation (ICCV 2025) mixes low and high frequencies, but within one model.
  - Coupled diffusion sampling (CVPR 2026) couples two models' whole trajectories.
  - MrFlow and Bottleneck Sampling change resolution within one model.
- **Noisy prompts in zero-shot TTS:** published approaches found so far are trained (noise-mixing fine-tuning, background-aware masked speech prediction). No training-free fix inside sampling was found.
- **The building block is classic.** A band's log-spectrum average over time is what cepstral mean normalization subtracts to remove stationary channel effects in speech recognition. Voice-Swap uses that quantity differently: it takes it from the student's draft at the handover, rather than subtracting it.

## Caveats

- Development speakers of this project, though unused for Voice-Swap. This is not the thesis's frozen held-out evaluation (EVAL-001/EVAL-002).
- Reverberation and telephone band are synthetic. Babble is one recorded track.
- The degraded-prompt WER margin (+0.025) is set by what 40 speakers can resolve. A tighter WER claim needs more speakers.
- CVf0 follows EXP-020's definition. The DMOSpeech 2 paper gives no formula, so only orderings can be compared with it.
- The run loops were exercised locally on numpy with a fake model. The torch path runs first in the notebook's smoke test.
