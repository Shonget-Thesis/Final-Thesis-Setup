# 2026-11-17: The Commit-Point Quantile Handover (local screen)

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
- **A possible reason (untested):** the teacher's estimate at t = 0.29 is smooth. Placing the student's values in its time order may also smooth away the student's own copied texture.
