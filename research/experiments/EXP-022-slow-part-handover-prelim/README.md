# EXP-022: Slow-Part Handover (development test)

**Status:** preregistered 2026-11-13 (`c311aeb52d424344...`), not yet run. Task EVAL-015 (todo). Development evidence only; nothing here is adopted by the thesis, and no adviser has reviewed it.

- Runner: [run_slow_part_handover_prelim.py](../../../code/colab/run_slow_part_handover_prelim.py)
- Notebook: [DMOSpeech2_EXP022_Slow_Part_Handover.ipynb](../../../code/colab/DMOSpeech2_EXP022_Slow_Part_Handover.ipynb)
- Tests: [test_slow_part_handover.py](../../../code/tests/test_slow_part_handover.py)
- Preregistration: [preregistration.json](preregistration.json)

## Idea

EXP-020 measured what each part of the teacher's estimate carries at the handover. The voice part (each band's average over time) carries the prompt's noise, the slow part (a 9-frame moving average of the rest, about 96 ms: timing and pitch movement) carries the pitch variety, and the fine detail barely matters. The student keeps only 7.6% of whatever it is handed. So the student needs only the teacher's slow part, and the teacher can stop as soon as that part has settled, which may happen at a different step for each input.

## Algorithm

| Step | What happens | Passes |
|---|---|---:|
| 1 | The guided teacher runs one step at a time; after step k its estimate T_k is split into voice, slow and fine parts | 2 per step |
| 2 | From Step 2: if d_slow_k = ‖slow(T_k) − slow(T_k−1)‖ / ‖slow(T_k)‖ ≤ λ, stop | 0 |
| 3 | One student pass from independent noise gives a draft S; the handover input is voice(S) + slow(T_k) + fine(S) | 1 |
| 4 | The released student finishes from it | 3 |

Stopping at Steps 2 to 7 costs 10, 12, 14, 16, 18 or 20 passes. If nothing settles by Step 7, the teacher continues to Step 8 and hands over its whole estimate, exactly as the 0.25 reference does (21 passes, no draft). λ is the only calibrated number. Voice-Swap is the special case that always stops at Step 4 and keeps the teacher's fine part. At Step 4 the two gave the same results in EXP-020.

## Design

- **Quality:** the EXP-012 Quick set (60 development speakers × clean, noisy15 and Gaussian 18/20/22/25 dB, 360 cases).
  - **Synthesized:** the handover at every Step 2 to 7, and the change of the voice, slow and fine parts and of the whole estimate at every teacher step, which needs no extra passes.
  - **Reused:** plain Steps 1 to 8 (EXP-018), student only (EXP-017) and Voice-Swap (EXP-020).
- **Pitch variety:** CVf0 of the handover at Steps 2, 3 and 4 and of Step 8, on the first 20 development speakers × clean and noisy15 × 20 seeds.
- **Calibration:** 5-fold speaker cross-validation. On the training folds, λ moves from conservative (λ = 0, everything goes to Step 8) towards aggressive and stops before the first λ at which the mean WER rises more than 0.01 or the mean similarity falls more than 0.01 against Step 8. The held-out folds are then scored.
- **Comparison rules:** the same procedure on the whole estimate's change (the thesis's drift) and on the voice part's change.
- **Candidate steps:** Steps 4 to 7 always. Steps 2 and 3 are used only if their pitch variety holds (see C3), which is decided before the rules are calibrated.

## Preregistered criteria

Slow-Part Handover advances to a frozen test on new speakers only if all four pass. Otherwise it is reported as a negative result with no retuning.

| Criterion | Requirement |
|---|---|
| C1 cheaper than Voice-Swap | out-of-fold mean passes of the slow-part rule below 14 |
| C2 quality | out-of-fold slow-part rule minus Step 8, speaker bootstrap: capped WER upper bound ≤ +0.025 and similarity lower bound ≥ −0.01, all 360 cases pooled |
| C3 pitch variety at Step 3 | CVf0 at Step 3 minus Step 8: lower bound ≥ −0.02 on clean and on noisy15 |
| C4 the slow part beats the whole estimate | out-of-fold mean passes below those of the same rule on the whole estimate's change |

**Read C1 with the data mix in mind.** Five of the six conditions are noisy, so C1 passes only if noisy inputs also settle by about Step 4. Clean prompts settling early cannot pass it alone. The report shows clean and noisy passes separately.

**Calibration controls the average.** It limits the mean change, so a few inputs may be degraded while the average stays within the limit. Definition 3.1 failures are reported descriptively.

## How to run

Push the branch, open the notebook on a Colab T4 and run all cells. About 4 hours by estimate, not measured.
- **Smoke test:** it must reproduce EXP-020's stored Step-4 handover scores and EXP-018's stored teacher drift at every step before the full run can start.
- **The runner refuses to start if** any of its seven code files, the preregistration or the stored inputs have changed, if the GPU is not a T4, or if Shonget-DMOSpeech2 is not at `8a7868f`.
- **On the branch:** the EXP-020 results (`results/selective.csv`) must be there.
- **When it finishes:** send `DMOSpeech2_EXP022_Slow_Part_Handover_Results.zip`.

## Prior work and novelty

To our knowledge, no published handover passes only part of the teacher's estimate, or stops the teacher based on the change of that part. This rests on targeted searches from 2026-11-08 to 2026-11-13, not a full literature review.
- **Whole-state handovers:** DMOSpeech 2, SRDiffusion, Gandikota and Bau, T-Stitch, Hybrid SD and TRACK pass whole states.
- **Closest relative:** HybridStitch chooses a model per token position, not per kind of information.
- **Stop signals:** existing ones use the whole estimate (SRDiffusion, TeaCache, this project's drift and STAC) or the prompt (SNR).
- **Borrowed:** the per-band average is the quantity cepstral mean normalization uses, and stopping each input at a different step is itself established.

## Caveats

- **Inspected speakers:** these are the speakers that designed EXP-020, with white-noise prompts only.
- **"Stalled, not settled":** on noisy prompts the teacher's estimate barely moves, so the slow part may look settled before it is. Word errors are the thing to watch.
- **Steps 2 and 3:** the teacher's timing has not been tested together with the student's detail there.
- **Risk:** the thesis's 5% failure guarantee is not used here, because the failure rule's floor on noisy prompts is about a quarter (EXP-017). EXP-021 measures that floor.
- **Local testing:** the run loops were exercised locally on numpy with a fake model. The torch path runs first in the notebook's smoke test.
