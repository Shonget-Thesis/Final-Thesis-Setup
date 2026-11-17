# EXP-022: Slow-Part Handover (development test)

**Status:** run 2026-11-13 UTC on a Colab Tesla T4 (thesis `a068a0d`, Shonget-DMOSpeech2 `8a7868f`), 360 of 360 cases and 40 of 40 diversity pairs; results in [results/](results/MANIFEST.json). **It does not advance: C2 and C4 pass, C1 and C3 fail.** Preregistration `c311aeb52d424344...`. Task EVAL-015 (review). Development evidence only; nothing here is adopted by the thesis, and no adviser has reviewed it.

- Runner: [run_slow_part_handover_prelim.py](../../../code/colab/run_slow_part_handover_prelim.py)
- Notebook: [DMOSpeech2_EXP022_Slow_Part_Handover.ipynb](../../../code/colab/DMOSpeech2_EXP022_Slow_Part_Handover.ipynb)
- Tests: [test_slow_part_handover.py](../../../code/tests/test_slow_part_handover.py)
- Preregistration: [preregistration.json](preregistration.json)
- Report: [SLOW_PART_REPORT.md](results/SLOW_PART_REPORT.md); diagnostics: [exp022_diagnostics.py](../../../code/analysis/exp022_diagnostics.py)

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

## Results

The smoke test reproduced the 4 stored Step-4 handover scores (largest difference about 6e-8) and EXP-018's teacher drift at every step (largest difference 7e-8). The full run took about 4.3 hours (first and last CSV rows at 00:43 and 04:58 UTC). Report: [SLOW_PART_REPORT.md](results/SLOW_PART_REPORT.md).

**Verdict: Slow-Part Handover does not advance.** As preregistered, it is reported as a negative result with no retuning on these data.

| Criterion | Measured (out of fold, 95% speaker interval) | Result |
|---|---|---|
| C1 cheaper than Voice-Swap | 14.19 mean passes (clean 15.13, noisy 14.00) | **Fail** |
| C2 quality | WER +0.0031 [-0.0129, +0.0177]; similarity +0.0523 [+0.0404, +0.0651] against Step 8 | Pass |
| C3 pitch variety at Step 3 | clean -0.0025 [-0.0152, +0.0072]; noisy15 not computable | **Fail** |
| C4 the slow part beats the whole estimate | 14.189 against 14.194 mean passes | Pass |

**C3 decided the outcome, and it measured the reference, not the method.**
- **What happened:** on noisy15 prompts, the reference (Step 8) had no frame voiced in at least 10 of 20 seeds for any of the 20 speakers, so its CVf0 does not exist. The handovers could be tracked on 14 or 15 of the 20 pairs.
- **How the rule handled it:** a missing lower bound is not a pass, so Steps 2 and 3 were dropped before calibration.
- **Effect on C1:** the cheapest remaining stop, Step 4, costs 14 passes, the same as Voice-Swap, so C1 (below 14) could no longer pass.
- **Why the reference cannot be tracked:** on white-noise prompts its outputs score UTMOS 1.25 to 2.08 (3.92 on clean prompts), against 1.92 to 3.11 for the Step-4 handover. This fits EXP-020's finding that the teacher's voice part carries the prompt's noise.
- **A design weakness:** EXP-020 measured pitch variety on clean prompts only, and noisy15 was added here without checking that the reference's pitch can be tracked there. This is a weakness of the preregistration, not a reason to change its verdict. EXP-021's P3 had the same weakness and was amended before any synthesis to judge pitch variety on clean prompts only (see its README).

**C4 passed by 2 passes across all 360 cases** (5,108 against 5,110). Both rules stopped every noisy input at Step 4 and routed only 9 clean inputs differently. That is not evidence that the slow part is the better signal.

**The stop rule was barely adaptive.**
- **The quality limit never bound:** in every fold λ went to the top of its grid (about 0.67, the 97.5th percentile of the training folds' signal values over Steps 4 to 7).
- **Effect:** an input went past Step 4 only if its Step-4 signal was above that percentile. All 300 noisy inputs stopped at Step 4; the clean inputs stopped at Steps 4 (33), 5 (20) and 6 (7).

**Fixed handovers (descriptive; against Step 8, 95% speaker interval).**

| Steps | WER | Similarity | Definition 3.1 failures (of 360) |
|---|---|---|---|
| 2 | +0.0164 [-0.0006, +0.0329] | +0.0433 [+0.0318, +0.0549] | 94 |
| 3 | +0.0106 [-0.0079, +0.0276] | +0.0497 [+0.0379, +0.0623] | 88 |
| 4 to 7 | +0.0018 to +0.0036 | +0.052 to +0.056 | 77, 70, 61, 71 |

- **UTMOS** at Step 4 is +0.90 [+0.77, +1.03].
- **Failures** on noisy prompts stay near the rule's floor of about a quarter (EXP-017).
- **Clean pitch variety** holds at Step 3 (-0.0025 [-0.0152, +0.0072], within -0.02) but not at Step 2 (-0.0157 [-0.0278, -0.0071]).

## Diagnostics (not preregistered)

[exp022_diagnostics.py](../../../code/analysis/exp022_diagnostics.py) computed these after the results, reusing the runner's calibration and cross-validation unchanged. They explain the verdict and cannot change it, and they cannot support a claim.

**With Step 3 allowed** (it passes the clean pitch check; Step 2 does not):

| | Slow-part rule (out of fold) | Comparison |
|---|---|---|
| Mean passes | 13.08 (clean 14.20, noisy 12.85) | 6.6% below Voice-Swap, 37.7% below the reference; whole-estimate rule 13.81 |
| WER vs Step 8 | +0.0080 [-0.0081, +0.0219] | within the C2 margin |
| Similarity vs Step 8 | +0.0496 [+0.0370, +0.0627] | within the C2 margin |
| WER vs Voice-Swap | +0.0084 [+0.0013, +0.0159] | higher |

**The signal does not pick the right inputs.**
- **Against random routing:** with Steps 3 to 7, routing at random while keeping the rule's stops per condition gives the same WER change (+0.0086 [+0.0051, +0.0121], against the rule's +0.0080) and the same failures (84.3, against 84).
- **Where the saving comes from:** noisy inputs stop at Step 3 (218 of 300). Their slow part changes less between steps than on clean prompts (mean 0.41 to 0.57, against 0.66 at Step 3). This is the "stalled, not settled" caveat below.

**With Steps 2 and 3 allowed:** 12.58 passes, WER vs Voice-Swap +0.0101 [+0.0025, +0.0179], and random routing again matches the rule.

**Fixed Step 3 (12 passes) against Voice-Swap (14):** WER +0.0110 [+0.0020, +0.0199], similarity -0.0020 [-0.0051, +0.0011]. Fixed Step 4 matches Voice-Swap (WER +0.0035 [-0.0032, +0.0114]).

**Reading.** On these data the adaptive stop adds nothing that a fixed choice of step would not. Handing over at Step 3 instead of Step 4 saves 2 passes and costs about one WER point against Voice-Swap.

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
