# EXP-030: Timing-Only Handover (TOH) pilot

**Status:** prepared and preregistered on 2026-11-15, before any synthesis (`575df8686329...`). Task EVAL-023 (todo). The run needs a Colab Tesla T4: [notebook](../../../code/colab/DMOSpeech2_EXP030_Timing_Only_Handover.ipynb), [runner](../../../code/colab/run_timing_only_handover.py), [tests](../../../code/tests/test_timing_only_handover.py).

- **A development pilot:** it tests whether one new part works. It is not evidence for a final method.
- **Not adopted:** nothing here changes the thesis method, and no adviser has reviewed it.

## The method

**Kept from DMOSpeech 2:** the guided teacher runs k sway steps, then the distilled student finishes from its first restart level with fresh noise.

**Changed: what crosses the handover.**
- **The released handover (W_k)** re-noises the teacher's endpoint estimate T_k. Every spectral property of T_k reaches the student, including the prompt's recording condition, which the teacher copies. EXP-017 and EXP-018 show this at every handover step.
- **TOH** hands over only the teacher's timing:

1. **Student draft:** S₀ = the student's one-step draft from pure noise, which is level 0 of the released student loop (1 pass).
2. **Teacher:** T_k = the guided teacher's endpoint estimate after k sway steps (2k + 2 passes).
3. **Time map:** w is a monotone dynamic-time-warping map from T_k's frames to S₀'s frames.
   - **Features:** 10 band-group log energies, smoothed over 3 frames and z-scored over time.
   - **Steps:** (1,1), (1,2) and (2,1), so every local slope lies between 1/2 and 2.
   - **Band:** a Sakoe-Chiba band of 10% of the length (at least 8 frames).
4. **Re-time and finish:** S₀ is resampled along w, and the student's three remaining levels run from it with the same seeds as the student-alone loop (3 passes).

**Cost:** 2k + 6 passes, so 10 at k = 2 and 12 at k = 3. For comparison: W4, DMOSpeech 2's code default, costs 13, and the reference W8 costs 21.

**What follows from the design:**
- **The condition can't cross.** w only moves the student's own frames in time. The time map is unaffected by any per-band offset or scale in T_k (tested).
- **The student alone is an exact control.** With w = identity, TOH equals the student alone exactly (checked in the smoke test).
- **A bad teacher is bounded.** The slope limits mean it can only re-time the student's draft within bounds.

**What is known and what is new:**
- **Known:** DTW alignment and rhythm transfer, re-noising an edited input (SDEdit), DMOSpeech 2's handover, and noise warping in video diffusion.
- **Not found** in a targeted search on 2026-11-15: a teacher-to-student handover whose only payload is a time map of the student's own draft.
- **Status:** partly novel under the strict rubric. See the [design log](../../../docs/progress/log/2026-11-15-timing-only-handover-design.md).

## What runs

One teacher trajectory to Step 8 and one student draft per case. Every arm shares the same restart noises, so all comparisons are paired.

| Arm | What it is | Passes |
|---|---|---:|
| STU | Student alone | 4 |
| W2, W3, W4, W8 | Released handover at Steps 2, 3, 4 and 8; W8 is the reference | 9, 11, 13, 21 |
| TOH2, TOH3 | Timing-Only Handover from T₂, T₃ | 10, 12 |
| MIR3 | Control: TOH3's map mirrored in time. The same amount and kind of re-timing, placed where the teacher did not put it | 12 |

- **Quality:** EXP-029's 30 development speakers (61–90) × clean, gauss25, noisy15 (white noise at 25 and 15 dB) and telephone, so 120 cases.
- **Pitch variety:** EXP-029's 20 speakers (61–80) × 12 seeds, clean prompts.
- **Untouched:** the remaining calibration speakers and all test speakers.
- **Why reuse EXP-029's cases:** W4 and W8 repeat EXP-029's code path on the same cases and seeds. Their scores are compared with EXP-029's stored ones on the 60 shared quality cases and the 20 diversity speakers, as a reproduction check.

**Smoke test:** a run stops before synthesis if any of these fails:
- **Exact identities (≤ 1e-5):**
  - the split student loop matches the full one;
  - the identity map leaves the draft unchanged;
  - TOH with the identity map equals the student alone.
- **Time map on real teacher estimates:** monotone, slope-limited and inside its band.
- **EXP-027's W4:** its scores are reproduced (≤ 0.01).

## Hypotheses and verdict rule (frozen)

| | Prediction | Test |
|---|---|---|
| H1a | The teacher's timing adds pitch variety | CVf0(TOH3) − CVf0(STU): lower bound > 0 |
| H1b | Timing alone keeps W3's variety | CVf0(TOH3) − CVf0(W3): lower bound ≥ −0.010 |
| H2a | The prompt's noise does not cross | Per-speaker mean over gauss25 and noisy15 of UTMOS(TOH3) − UTMOS(W4): lower bound > 0 |
| H2b | As good as the student alone on noisy prompts | Same pooling, UTMOS(TOH3) − UTMOS(STU): lower bound ≥ −0.10 |
| H3 | Clean quality holds | Against W8: mean ΔUTMOS ≥ −0.08, ΔWER upper bound ≤ +0.01, ΔSIM lower bound ≥ −0.01 (EXP-029's bars) |

Bounds are 95% speaker-level bootstrap intervals (B = 2000, seed 2026).

**Verdict:**

| Verdict | When |
|---|---|
| **Supported** | All five hold for TOH3. |
| **Timing does not carry the variety** | H2a, H2b and H3 hold but H1a or H1b fails. The noise is blocked, but the variety lives in pitch movement rather than timing. |
| **Not supported** | H2a, H2b or H3 fails. |

- **TOH2:** judged by the same rule against W2, as a secondary result.
- **Exploratory:** telephone prompts, MIR3, runaway outputs (WER > 1) and the EXP-029 reproduction.
- **No retuning on these data.**

## What a result would mean

- **Supported:** TOH becomes the new part of the thesis method. The adaptive layer (SNR sending very noisy prompts straight to the student) would be added as a known part and tested on new speakers.
- **Timing does not carry the variety:** TOH still blocks the condition but does not replace the teacher's sound. The teacher's variety would then be pitch movement, which the time map cannot carry.
- **Not supported:** TOH is dropped. No novelty claim is forced.

## Data and evidence status

- **Development evidence only:** these speakers informed EXP-029.
- **Audio** is not committed.
- **Results** will be stored under `results/` with a SHA-256 manifest.
