# 2026-11-17: The Commit-Point Quantile Handover and its successors (local screens)

Follow-up to the [literature map](2026-11-17-literature-map.md); tracked as EVAL-028.
- **The request:** Rhenel Jhon R. Sajol asked for a new algorithm ("WE NEED NEW ALGORITHM"), then approved the plan: map the literature, design, screen locally, then test on the T4 ("do it").
- **Status:** a local CPU screen, so screening only. Nothing is adopted, and no adviser has reviewed it.

## The algorithm, as one changed term

**The original:** DMOSpeech 2's teacher-guided sampling, at the commit point that EXP-033 found (`CPH8`):

    x̂ = the teacher's estimate at t = 0.2929       (sway grid of 8: five guided evaluations, 10 passes)
    x = x̂
    for τ in (0.2929, 0.6173):                       (two student passes)
        x = student( (1 − τ)·ε_τ + τ·x ,  τ )

**The problem:**
- **On a noisy prompt, the teacher copies the prompt's spread-out background into x̂,** and the restart carries it into the output ([spread-weighted SNR log](2026-11-17-spread-weighted-snr.md)).
- **The student alone stays cleaner but flat.**
- **A gate can only choose between the two.** That is true of Chapter 3's SNR and of the spread-weighted SNR alike. No published inference-only method keeps the teacher's prosody without its copied noise (gap G1 of the literature map).

**The changed term:** `x = x̂` becomes `x = Q(x̂, s₀)`. Here s₀ is the student's one-pass draft from the same seed.

    Q(x̂, s₀)_b(t) = sort( s₀_b )[ rank_t( x̂_b )(t) ]      for every mel band b, over the generated frames

**In words:** in each band, the student's own values, placed in the teacher's time order.
- **The teacher decides when:** when each band rises and falls, which carries the melody, timing and energy contour.
- **The student decides how loud:** the level of each rank, including its own clean floor.

**Why it drops the copied background** (the [invariance theory note](2026-11-15-handover-invariance-theory.md), Theorem 2):
- **Every value in the payload is the student's.** The teacher contributes only each band's time order.
- **A stationary floor or channel leaves that order unchanged,** because it changes a band by a strictly increasing map. Q is the most informative payload with this property.
- **A fluctuating floor can reorder the quietest frames,** but only among the student's own quiet values.
- **If the teacher differs from the student only by such a map,** Q returns the student's draft.

**Why at the commit point:**
- **The earlier attempt:** EXP-031 tested Q at Step 3, followed by the student's first restart (0.0761). That restart erases the melody (EXP-032 bank screen).
- **The fix:** EXP-033 found that the melody is committed by t ≈ 0.29 and survives a restart there.

**Cost:**

| Arm | Passes |
|---|---:|
| The student alone (`STU`) | 4 |
| `CPH8` | 12 |
| **CP-QH** (the teacher's 10, the student's draft 1, two restarts 2) | **13** |
| The paper's setting (`P25`) | 30 |

The operator itself is one sort and one argsort per band. It needs no noise estimate and no gate: the same procedure runs on every prompt.

## What is new and what is known

| Part | Status |
|---|---|
| Teacher first, then the student | Known ([DMOSpeech 2](https://arxiv.org/abs/2507.14988); [Gandikota and Bau](https://arxiv.org/abs/2503.10637)) |
| Handing over at t = 0.2929 | This thesis's EXP-033 configuration; the principle that early steps decide is known |
| The operator, per-band quantile mapping | Known ([Hilger and Ney, 2001](https://www.isca-archive.org/eurospeech_2001/hilger01_eurospeech.html)) |
| Q as the most informative handover payload invariant to per-band increasing maps | Not found (the theory note's six searches; the literature map) |
| The student's own draft as the reference distribution at a handover | Not found |
| An inference-only sampler that keeps the teacher's prosody on noisy prompts without the copied noise | Not found (gap G1) |

## Screen 1: CP-QH (`code/analysis/cpqh_screen.py`)

**Setup:**
- **Sentences:** the three shortest clean EXP-033 quality cases, six seeds each (input seed + 1000 + s).
- **Conditions:** clean and `noisy15` (the experiments' white noise at 15 dB).
- **Arms:** the student alone (4 passes), `CPH8` (12) and CP-QH (13), with the same seeds and student noises.
- **Output:** `research/experiments/EXP-033-cheaper-teacher-path/offline/cpqh_screen.json` (36 runs, 536 s of synthesis).

| Measure | Condition | Student alone | `CPH8` | CP-QH |
|---|---|---:|---:|---:|
| UTMOS | clean | 4.12 | 3.93 | 4.03 |
| UTMOS | `noisy15` | 1.68 | 1.28 | **2.74** |
| Outputs below UTMOS 2.0 | `noisy15` | 14 of 18 | 18 of 18 | **2 of 18** |
| Voiced share of frames | `noisy15` | 0.31 | 0.00 | **0.71** |
| CVf0 | clean | 0.051 | 0.084 | 0.069 |
| Aligned CVf0 | clean | 0.040 | 0.067 | 0.049 |
| CVf0 | `noisy15` | 0.038 | none (unvoiced) | **0.072** |
| Aligned CVf0 | `noisy15` | 0.039 | none (unvoiced) | **0.059** |
| Frames within a semitone of `CPH8`'s melody | clean | 0.52 | — | 0.82 |

| Pre-set rule | Result |
|---|---|
| 1. `noisy15` UTMOS ≥ the student's − 0.10 | **Pass** (2.74 against 1.68) |
| 2. `noisy15` variety above the student's, on both measures | **Pass** (0.072 against 0.038; 0.059 against 0.039) |
| 3. Clean variety: ≥ 75% of `CPH8`'s gain over the student, on both measures | **Fail** (54% and 34%) |
| 4. Clean UTMOS ≥ `CPH8`'s − 0.10 | **Pass** (4.03 against 3.93) |

**Verdict:** CP-QH fails rule 3, so it is not recommended for a T4 test as one handover for all prompts.

**Where the clean shortfall comes from:**

| Sentence | Student CVf0 | `CPH8` CVf0 | CP-QH CVf0 |
|---|---:|---:|---:|
| First | 0.039 | 0.114 | 0.070 |
| Second | 0.051 | 0.067 | 0.061 |
| Third | 0.063 | 0.071 | 0.076 |

Most of the shortfall is in the first sentence, where `CPH8`'s variety is highest.

**The diagnosis:**
- **QH is invariant to every per-band increasing map,** so only the time order may come from the teacher.
- **Every value is the student's,** and so is each band's share of loud frames. That carries the student's narrower pitch range.
- **So CP-QH follows the teacher's melody, within the student's range:** 82% of frames lie within a semitone of `CPH8` on clean prompts, against 52% for the student alone.

**What it shows on noisy prompts:** this is gap G1, the teacher's prosody without its copied noise.
- **The most voiced output:** CP-QH is the only arm whose `noisy15` output is mostly voiced.
- **Cleaner and more varied than the student alone,** which Chapter 3's gate treats as the safe choice. It has 1.06 more UTMOS and almost twice the pitch variety.
- **The reason first suggested was refuted by the controls below.** It was that the teacher's estimate is smooth and its order smooths away the student's copied texture. In fact, the teacher's estimate is rougher than the student's draft on these prompts, and the cleaning comes from the student's restart levels.

## Screen 2: the Floor-Canonical Handover, and CP-QH on new sentences (`code/analysis/floor_canonical_screen.py`)

**Why a second candidate:**
- **QH's limit:** QH gives up the teacher's values because it is invariant to every increasing map.
- **What the noise actually does is less:** a stationary additive background adds a constant per band, in power.
- **The exact invariant of that smaller group** keeps the teacher's values relative to its own floor. The Floor-Canonical Handover (FCH) is its canonical form, with the floor fixed by the student's draft:

      FCH(x̂, s₀)_b(t) = ½ log( max( P_T(t) − q10(P_T) + q10(P_S),  q01(P_S) ) ),    P = exp(2x)

- **What it does:** it keeps the teacher's values above its floor and replaces only the floor with the student's. When the floors agree, it returns the teacher's estimate.
- **Cost:** 13 passes, as for CP-QH.

**Tests** (`code/tests/test_floor_canonical.py`):
- the payload is unchanged by any added per-band floor;
- its 10th-percentile power equals the student's;
- it returns the teacher's estimate when the floors agree;
- the torch and numpy versions agree.

**Setup:**
- **Sentences:** the 4th to 6th shortest clean EXP-033 quality cases, not used in screen 1.
- **Seeds and conditions:** six seeds; clean and `noisy15`.
- **Arms:** the student alone, `CPH8`, CP-QH and FCH.
- **Size:** 36 runs, 692 s.

| Measure | Condition | Student alone | `CPH8` | CP-QH | FCH |
|---|---|---:|---:|---:|---:|
| UTMOS | clean | 3.66 | 3.58 | 3.57 | 3.57 |
| UTMOS | `noisy15` | 2.14 | 1.25 | **2.84** | 1.25 |
| Outputs below UTMOS 2.0 | `noisy15` | 8 of 18 | 18 of 18 | **0 of 18** | 18 of 18 |
| Voiced share of frames | `noisy15` | 0.25 | 0.00 | 0.55 | 0.00 |
| CVf0 | clean | 0.044 | 0.095 | 0.091 | 0.096 |
| Aligned CVf0 | clean | 0.037 | 0.078 | 0.063 | 0.075 |
| CVf0 | `noisy15` | 0.105 | none | 0.084 | none |
| Aligned CVf0 | `noisy15` | 0.104 | none | 0.070 | none |
| Frames within a semitone of `CPH8`'s melody | clean | 0.45 | — | 0.90 | 0.99 |

| Pre-set rule | FCH | CP-QH (replication) |
|---|---|---|
| 1. `noisy15` UTMOS ≥ the student's − 0.10 | **Fail** (1.25 against 2.14) | Pass (2.84) |
| 2. `noisy15` variety above the student's | **Fail** (unvoiced output) | **Fail** (0.084 against 0.105; 0.070 against 0.104) |
| 3. Clean variety ≥ 75% of `CPH8`'s gain | Pass (102% and 95%) | **Fail** (92% and 63%) |
| 4. Clean UTMOS ≥ `CPH8`'s − 0.10 | Pass | Pass |

**Verdict:** FCH fails. CP-QH misses rule 3 again and, on these sentences, rule 2.

**About CP-QH's rule 2 failure:**
- **Few voiced frames behind the student's number:** its `noisy15` CVf0 rests on 3% and 10% voiced frames in two of the three sentences, so its pitch track there mostly follows noise.
- **One sentence is a fair comparison:** there the student's output is mostly voiced (61%), and its CVf0 is above CP-QH's (0.078 against 0.059).
- **The rule stays failed.** Any later test should require a minimum voiced share before comparing variety.

**What FCH shows:** matching the floor changes nothing on noisy prompts. All 18 outputs are below UTMOS 2.0, as with `CPH8`, although FCH lowered the teacher's floor by a median of 7.9 dB (12.5 dB in the upper bands).
- **Caveat:** FCH matched the 10th percentile. How much of the copied noise lies above that floor is measured in the controls below.

**CP-QH over both screens** (six sentences):

| Measure | Student alone | `CPH8` | CP-QH |
|---|---:|---:|---:|
| `noisy15` UTMOS | 1.91 | 1.27 | **2.79** |
| `noisy15` outputs below 2.0 | 22 of 36 | 36 of 36 | **2 of 36** |
| `noisy15` voiced share | 0.28 | 0.00 | 0.63 |
| Clean UTMOS | 3.89 | 3.75 | 3.80 |
| Clean CVf0 | 0.047 | 0.090 | 0.080 (77% of `CPH8`'s gain) |
| Clean aligned CVf0 | 0.038 | 0.072 | 0.056 (52% of the gain) |

CP-QH's `noisy15` UTMOS is above both the student's and `CPH8`'s in all six sentences.

## Controls: where does CP-QH's cleaning come from? (`code/analysis/cpqh_controls_screen.py`)

CP-QH differs from the student alone in two ways: it skips the student's first restart level (0.0761), and it uses the teacher's time order. Two controls separate them.

**The arms** (the six sentences of screens 1 and 2, six seeds, clean and `noisy15`, 72 runs):

| Arm | What it is | Passes |
|---|---|---:|
| Student alone | Draft, then restarts at 0.0761, 0.2929 and 0.6173 | 4 |
| **Draft restarted at 0.2929** | The draft, then 0.2929 and 0.6173 only: CP-QH with the student's own order in place of the teacher's | 3 |
| Smoothed self-order | The draft's values in the order of the draft smoothed over 9 frames (about 96 ms), then 0.2929 and 0.6173 | 3 |
| CP-QH | As before | 13 |

**Readings fixed before the run** (on `noisy15` UTMOS):
- **Q1, the teacher's order:** it matters if CP-QH beats the draft at 0.2929 by ≥ 0.30. The restart levels explain the cleaning if the difference is below 0.10.
- **Q2, any smooth order:** yes if the smoothed self-order is within 0.10 of CP-QH.

**Reproduction:** the student-alone and CP-QH scores reproduce the earlier screens exactly (largest UTMOS difference 0.0).

| Measure | Condition | Student alone | Draft at 0.2929 | Smoothed self-order | CP-QH |
|---|---|---:|---:|---:|---:|
| UTMOS | clean | 3.89 | 3.89 | 3.85 | 3.80 |
| UTMOS | `noisy15` | 1.91 | **3.23** | 3.08 | 2.79 |
| Outputs below UTMOS 2.0 | `noisy15` | 22 of 36 | **1 of 36** | 2 of 36 | 2 of 36 |
| Voiced share of frames | `noisy15` | 0.28 | 0.54 | 0.56 | 0.63 |
| CVf0 | clean | 0.047 | 0.041 | 0.045 | 0.080 |
| Aligned CVf0 | clean | 0.038 | 0.034 | 0.038 | 0.056 |
| CVf0 | `noisy15` | 0.072 | 0.050 | 0.049 | **0.078** |
| Aligned CVf0 | `noisy15` | 0.072 | 0.046 | 0.048 | **0.065** |

The student alone's `noisy15` CVf0 rests on 28% voiced frames, mostly a pitch track of noise.

**The readings:**
- **Q1: the restart levels.** CP-QH minus the draft at 0.2929 is −0.44 UTMOS. The cleaning on noisy prompts comes from skipping the student's first restart level, not from the teacher's order.
- **Q2: yes.** The smoothed self-order is as clean as CP-QH; it is 0.29 better.

**What this changes:**
- **The student copies a noisy prompt's background at its first restart level (0.0761).** Restarting the draft at 0.2929 instead takes `noisy15` from 1.91 to 3.23 UTMOS, and from 22 to 1 of 36 outputs below 2.0. It costs one pass less, with the same clean quality and slightly less clean variety (CVf0 0.041 against 0.047).
  - **Not run before here:** the result tables of EXP-011 to EXP-033 have no student-only arm without the 0.0761 level.
- **Both of CP-QH's parts are needed on noisy prompts:**
  - **Skipping 0.0761 keeps out the student's own copying.**
  - **Replacing the teacher's values keeps out the teacher's copying.** With the same restart levels, the arms that hand over the teacher's values stay noisy: `CPH8` and FCH had 36 of 36 and 18 of 18 outputs below 2.0.
- **CP-QH's role, compared with the clean student path, is the teacher's prosody.** Among the clean paths, it carries about 1.5 times the pitch variety of the draft at 0.2929 (CVf0 0.078 against 0.050; aligned 0.065 against 0.046). That costs 0.44 UTMOS and 10 more passes.
- **The teacher's estimate carries the copied background as texture.** On `noisy15`, its temporal roughness (mean squared frame-to-frame change over the variance) is 0.14, against 0.07 for the student's draft. On clean prompts it is 0.06, against 0.10. FCH removed the floor but kept this texture, which explains its failure.

## Probe: does roughness tell the conditions apart band by band? (`code/analysis/roughness_band_probe.py`)

This is an exploratory design input on the same six sentences (two seeds each), not a screen.

| Condition | Median ratio of the teacher's band roughness to the draft's | Bands where the teacher is rougher |
|---|---:|---:|
| Clean (12 cases) | 0.58 | 1.8% (no case above 10%) |
| `noisy15` (12 cases) | 1.96 | 85% (91–100% in 10 cases; about 20% in one sentence's two seeds) |

**The idea it supports:** in each band, the teacher's estimate can be checked against the student's draft. Copied background shows up as roughness that the draft does not have. That suggested the next candidate, screened below on new sentences.

## Screen 3: the Roughness-Selected Quantile Handover (`code/analysis/rqh_screen.py`)

**The candidate (RQH):** in each band, use QH where the teacher's estimate is rougher in time than the student's draft, and the teacher's values elsewhere.
- **The reference:** the student's draft is the clean reference. There is no prompt statistic and no fitted threshold.
- **Its limits:** it reduces to `CPH8` when no band is rougher, and to CP-QH when all are.
- **Prompt frames:** the teacher's, as in `CPH8`.
- **Cost:** 13 passes.
- **Tests:** `code/tests/test_rqh_screen.py`.

**Setup:**
- **Sentences:** the 7th to 12th shortest clean EXP-033 quality cases, not used in any earlier screen.
- **Seeds and conditions:** six seeds; clean and `noisy15`; 72 runs.
- **Arms:** the student alone, the draft at 0.2929, `CPH8`, CP-QH and RQH.

| Pre-set rule | Result |
|---|---|
| 1. `noisy15` UTMOS ≥ CP-QH's − 0.10, and ≤ 10% of outputs below 2.0 | **Fail** (1.42 against 2.13; 34 of 36 below 2.0) |
| 2. `noisy15` variety above the draft at 0.2929, where both voice ≥ 30% of frames | Pass (two sentences only) |
| 3. Clean variety ≥ 75% of `CPH8`'s gain | Pass (98% and 104%) |
| 4. Clean UTMOS ≥ `CPH8`'s − 0.10 | Pass (3.74 against 3.73) |

**Verdict:** RQH fails.
- **The detector misses some noisy prompts.** On two sentences the teacher was rougher than the draft in only 2–43% of the bands (12% to 74% on a third), so RQH stayed close to `CPH8`.
- **It also fails where QH covers nearly every band.** It still scored below CP-QH there (for example 1.38 against 2.46 with all bands). The only remaining difference is the prompt frames: the teacher's in RQH, the draft's in CP-QH. That is tested below.
- **A broken teacher on a clean prompt:** on one sentence the teacher fails even on the clean prompt (`CPH8` UTMOS 1.38–2.48). RQH follows it.

## All screened sentences (12 sentences × 6 seeds; CPU screening)

| Measure | Student alone (4 passes) | Draft at 0.2929 (3) | `CPH8` (12) | CP-QH (13) |
|---|---:|---:|---:|---:|
| Clean UTMOS | 4.04 | 4.05 | 3.74 | 3.94 |
| Clean outputs below 2.0 | 0 of 72 | 0 of 72 | 4 of 72 | 0 of 72 |
| `noisy15` UTMOS | 1.74 | **2.81** | 1.27 | 2.46 |
| `noisy15` outputs below 2.0 | 50 of 72 | **15 of 72** | 72 of 72 | 20 of 72 |
| `noisy15` voiced share | 0.22 | 0.48 | 0.00 | 0.56 |
| Clean CVf0 | 0.047 | 0.042 | **0.109** | 0.086 |
| Clean aligned CVf0 | 0.039 | 0.034 | **0.083** | 0.062 |
| `noisy15` CVf0, ten sentences where both voice ≥ 30% | — | 0.051 | — | **0.095** |
| `noisy15` aligned CVf0, same sentences | — | 0.046 | — | **0.086** |

**How much of `CPH8`'s clean variety CP-QH keeps:**
- **All twelve sentences:** 63% of the gain (51% aligned).
- **Without the sentence where the teacher breaks:** 75% (56% aligned).

**The trade-off on noisy prompts:**
- **The draft at 0.2929 is the cleanest and cheapest path.**
- **CP-QH has higher pitch variety than it in all ten comparable sentences,** almost twice as much, for 0.35 lower UTMOS and 10 more passes.
- **The prompts are hard:** on two of the twelve sentences, every arm fails on `noisy15`.

## Control: the payload's prompt frames (`code/analysis/prompt_region_control.py`)

**Why:**
- **Where each arm takes its prompt frames:** a payload covers the prompt frames as well as the generated ones. `CPH8`, FCH and RQH hand over the teacher's estimate there. CP-QH and the student-only arms hand over the draft.
- **The question:** whether that difference matters. The earlier screens did not test it.

**Setup:**
- **Arms:** four swaps, all restarted at 0.2929 and 0.6173, plus the draft at 0.2929.
- **Cases:** the six sentences of screens 1 and 2, six seeds, clean and `noisy15`; 72 runs.

| Arm | Generated frames | Prompt frames | Clean UTMOS | `noisy15` UTMOS | `noisy15` below 2.0 |
|---|---|---|---:|---:|---:|
| `CPH8` | Teacher | Teacher | 3.75 | 1.27 | 36 of 36 |
| `CPH8` with the draft's prompt | Teacher | Draft | 3.83 | 1.27 | 36 of 36 |
| CP-QH | QH | Draft | 3.80 | **2.79** | **2 of 36** |
| CP-QH with the teacher's prompt | QH | Teacher | 3.73 | 1.98 | 20 of 36 |
| Draft at 0.2929 | Draft | Draft | 3.89 | 3.23 | 1 of 36 |

**Readings fixed before the run:**
- **Q3, do the teacher's prompt frames carry its copying?** **No** (+0.002). The copying is in the teacher's generated frames.
- **Q4, are the draft's prompt frames part of CP-QH's cleanliness?** **Yes** (+0.81). With the teacher's prompt frames, 20 of 36 outputs fall below 2.0.

**On clean prompts the swap is harmless:**
- **Quality:** the draft's prompt frames leave `CPH8`'s quality as it was.
- **Variety:** also unchanged (CVf0 0.095 against 0.089).
- **Melody:** the swapped arm follows `CPH8`'s on 98% of frames.

**What it changes in the earlier verdicts:** FCH and RQH used the teacher's prompt frames, so their noisy-prompt failures are partly confounded.
- **FCH:** its generated frames also kept the teacher's texture.
- **RQH:** its detector also missed whole sentences.
- **What was not run:** neither was rerun with the draft's prompt frames.

## Where this leaves the method

**The copied background reaches the output through three routes,** each found by a control:

| Route | Fix that worked |
|---|---|
| The student's own restart at 0.0761 | Restart the student at 0.2929 and 0.6173 only |
| The teacher's generated values (temporal texture, not a floor offset) | Hand over the draft's values in the teacher's order (QH) |
| The teacher's prompt frames, once the generated values are clean | Hand over the draft's prompt frames |

**CP-QH closes all three routes, and it is the only teacher-carrying arm that does:**
- **White-noise prompts:** clean (2.46 UTMOS over twelve sentences), with almost twice the pitch variety of the clean student path.
- **Clean prompts:** it keeps 51–75% of `CPH8`'s variety gain. That fails the 75% bar used throughout.
- **Why it falls short on clean prompts:** the student's values carry the student's narrower pitch range.

**No gate-free candidate passed all its pre-set rules.** The evidence supports two configurations for a T4 test:

| Configuration | Clean prompts | White-noise prompts | Passes |
|---|---|---|---:|
| **CP-QH everywhere** (no gate) | Quality like `CPH8`, 51–75% of its variety gain | Clean, with the teacher's timing | 13 |
| **A noise gate, then `CPH8` or CP-QH** | `CPH8`: the paper's variety at 12 passes (T4, EXP-033) | CP-QH, or the draft at 0.2929 for 3 passes and flat prosody | 12–13, or 3 |

**What is new, so far as the searches went:**
- **Not found in the literature:**
  - QH as the condition-safe payload at a handover, with the student's draft as the reference;
  - the three copying routes;
  - the student copying the background at its first restart level.
- **Known:** the operators (quantile mapping, restart schedules). The literature check for the restart-level finding has not been done.

## Caveats

- **CPU screening only:** twelve development sentences (the shortest), six seeds, UTMOS and pYIN pitch only. No WER or speaker similarity yet.
- **One noise type:** white noise at 15 dB only. Other noise types and SNRs are untested for these arms.
- **Designed after looking:** RQH, the controls and the probe were designed after seeing earlier screens. Their rules were fixed before each run; RQH ran on sentences not used to design it.
- **Hard sentences:** on two of the twelve, every arm fails on `noisy15`.
- **A broken teacher:** on one clean prompt the teacher fails (`CPH8` 1.38–2.48), which inflates `CPH8`'s clean CVf0.

## Next

A preregistered T4 run (EXP-034) on development speakers not used here would decide between the two configurations:
- **Arms:** the student alone, the draft at 0.2929, `CPH8`, CP-QH and `P25`.
- **Conditions:** clean, white noise at 15 and 25 dB, pink, DEMAND babble, reverberation and telephone.
- **Measures:** WER, speaker similarity, UTMOS, voiced share, CVf0 and aligned CVf0, with a minimum voiced share before comparing variety.
- **The gate:** the spread-weighted SNR (EVAL-027) computed on each prompt, so that both configurations are evaluated from one run.

## Checks run

- **The screens:** screens 1–3, the two controls and the probe all exited 0. Their JSON is in `research/experiments/EXP-033-cheaper-teacher-path/offline/` (`cpqh_screen`, `floor_canonical_screen`, `cpqh_controls_screen`, `roughness_band_probe`, `rqh_screen`, `prompt_region_control`).
- **Reproduction:** the control screen reproduced the earlier screens' student-alone and CP-QH scores exactly.
- **Tests:**
  - `test_cpqh_screen.py` (4);
  - `test_floor_canonical.py` (4);
  - `test_cpqh_controls_screen.py` (3);
  - `test_rqh_screen.py` (4);
  - `test_prompt_region_control.py` (3).
- **Workspace check:** `python scripts/check_workspace.py` PASSED.
- **Full suite:** `python -m unittest discover -s code/tests` ran 413 tests, OK.
- **Manuscript:** not changed, so not rebuilt.
