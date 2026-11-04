# The Problem, Its Validation, and Our Candidate Solutions

**Date:** 2026-10-27, updated 2026-11-02 · **Branch:** `rhenelAudit` · **Status:** explanation of existing results. All numbers are development evidence from the team's experiments; nothing here is adviser-approved. Sources are listed at the end. On 2026-10-27 the team removed Appendix E (TAS and SWC) from the manuscript and the presentation; it is kept as `paper/chapters/appendix-e-hidden.tex`, and the research record is unchanged.

**Purpose:** explain in plain language what problem the thesis solves, how we showed the problem is real, what the quality measures mean, and how the solution ideas (SNR-only, SNR + drift, SNR + TAS, SNR + SWC) work and compare.

---

## 1. The system in one picture

DMOSpeech 2 is a zero-shot text-to-speech system: give it a few seconds of someone's voice (the **prompt**) and a text, and it speaks the text in that voice. It generates speech in two stages:

```
voice prompt + text
        │
        ▼
Teacher (large model), steps 0, 1, 2, …, k   ← 2 forward passes per step
        │   hands over at step k  (the "switching time")
        ▼
Student (small distilled model)               ← 3 forward passes
        │
        ▼
Vocoder → speech in the prompt's voice
```

A **forward pass** is one run of a neural network. Each teacher step costs 2 passes: the model runs once with the prompt and text and once without, and the two results are combined (classifier-free guidance). The student always costs 3. A handover at step k therefore costs 2(k + 1) + 3 passes:

| Handover at | Step 4 | Step 5 | Step 6 | Step 8 |
|---|---:|---:|---:|---:|
| Forward passes | 13 | 15 | 17 | **21** |

The DMOSpeech 2 paper evaluates one teacher-guided configuration, switching time s = 0.25, which in the released sampler hands over at **Step 8 for every input**. That is the **reference**: the quality every cheaper option is compared against. The released code's own default, s = 0.07, is the opposite fixed choice: **Step 4 for every input**. Neither value was shown to be the best (see the R03 note in Section 2.1).

## 2. The problem

### 2.1 One switching time for everyone

The paper's setting pays 21 passes for every input, whether the prompt is a clean studio recording or a noisy clip; the code's default pays 13 for every input. The thesis asks: can we choose the switching time **per input**, spending fewer passes on easy inputs, without letting quality drop?

**Was 0.25 shown to be the best value? (task R03)** No. In arXiv v1 (20 Jul 2025), 0.25 appears once, in the list of compared configurations in Section 4.1 (p. 9): "(tswitch = 0.25, with teacher handling 14 steps and student handling 2 steps, for a total of 16 steps)". The paper evaluates no other switching time and gives no reason for 0.25. Section 3.3.2 (p. 8) only suggests "around 0.4-0.5" when diversity is critical and "around 0.1-0.2" when efficiency is paramount, without results for those values; 0.25 lies in neither range. The released code's `generate()` defaults to 0.07. The thesis cites the AAAI 2026 version, whose wording and pages have not yet been checked against this.

### 2.2 What "wasted computation" means

Take one input and generate it twice with the same random draws: once handing over at Step 4 (13 passes) and once at Step 8 (21 passes). Score both (Section 4). If the Step-4 output is just as intelligible and sounds just as much like the speaker, within small margins, then the extra 8 passes (38% of the cost) bought **no measurable improvement**. Those 8 passes are wasted computation.

It is like sending every parcel by the most expensive express courier, including the ones that would have arrived on time by regular mail.

How much is wasted? In the validation test below (Section 3):

- **488 of 500 clean prompts** were already fine at Step 4. Each still paid for Step 8: 8 extra passes × 488 = **3,904 passes** with no measurable gain.
- If we had known in advance the cheapest safe step for each of the 1,000 inputs, the average cost would have been **14.85 passes instead of 21 (−29%)**. That 29% is an upper bound: the most a perfect per-input decision could save on that data.

### 2.3 Why not hand over early for everyone?

Because noisy prompts break. At Step 4, **322 of 500 noisy prompts (64%)** failed against the reference: the teacher needs its later steps to pull the voice out of the noise. A cheaper fixed setting saves compute on clean prompts but damages noisy ones.

### 2.4 The problem in one sentence

A fixed switching time either wastes compute on easy inputs (Step 8 for everyone) or damages hard inputs (an early step for everyone), so we need a **per-input decision** that is cheap to make and keeps the failure rate under a stated limit.

## 3. How we validated the problem

### 3.1 The controlled test (Appendix D, "Fixed Switching Times")

- **Data:** all 500 speakers of the study's three splits (Seed-TTS English test set), each under two prompt conditions: **clean** (the original recording) and **noisy** (white Gaussian noise added at 15 dB). That makes 1,000 instances.
- **What was run:** every instance was generated at Steps 4, 5, 6 and 8 with the **same random draws**, on the Tesla T4 GPU.
- **How it was scored:** each early output was compared with the Step-8 output of the same instance using the added-failure rule (Section 4).

### 3.2 Results

| Handover | Passes | Clean failures (of 500) | Noisy failures (of 500) | Overall failure rate |
|---|---:|---:|---:|---:|
| Step 4 | 13 | 12 | 322 | 33.4% |
| Step 5 | 15 | 9 | 269 | 27.8% |
| Step 6 | 17 | 5 | 247 | 25.2% |
| Step 8 (reference) | 21 | 0 | 0 | 0% (by definition) |

The cheapest safe step, found in hindsight, was Step 4 for 666 instances, Step 5 for 100, Step 6 for 55 and Step 8 for 179 (all of them noisy).

### 3.3 What this shows

1. **The waste is real:** clean prompts almost never need Step 8 (12 failures of 500 at Step 4).
2. **The risk is real:** noisy prompts often do (322 of 500 fail at Step 4).
3. **The right step differs per input** (666 / 100 / 55 / 179 across Steps 4 / 5 / 6 / 8). No single fixed step is both cheap and safe, so a per-input decision is needed, and it could save up to 29%.

### 3.4 Why adding noise on purpose is a valid test

It is a **controlled experiment**. Speaker, text and random draws stay the same; only the prompt's noise changes, so any difference in failures is caused by the noise. It also tells us in advance which inputs are hard, so we can check whether a decision rule recognizes them.

### 3.5 What the later tests added: the blind spot of SNR-only

Later tests (EXP-011, EXP-012, EXP-015) added more noise levels. The SNR estimate reads a little above the true added-noise level, so prompts with moderate noise slip through a 20 dB gate:

| Prompt condition | Estimated SNR (EXP-011 pilot, 20 speakers) | SNR-only failures (EXP-015, of 60) |
|---|---|---:|
| Clean | 26.5–50.0 dB | 2 |
| Noise added at 25 dB | 24.2–28.1 dB | 8 |
| Noise added at 22 dB | 22.7–25.1 dB | 19 |
| Noise added at 20 dB | 21.4–23.2 dB | 26 |
| Noise added at 18 dB | 19.9–21.2 dB | 31 |
| Noise added at 15 dB | 17.5–18.2 dB | 0 (all sent to the reference) |

Prompts with 18–22 dB of added noise read above 20 dB, pass the gate, get the cheap Step-4 handover, and fail. This is the gap the candidate solutions close.

### 3.6 Caveat

These are **development** results. The splits had been examined in earlier analyses, and the extra noise levels are white noise only. The independent test on held-out speakers comes after the proposal.

## 4. The measures in plain words

| Measure | What it means | Better is |
|---|---|---|
| **WER** (word error rate) | A speech recognizer (Whisper-large-v3) transcribes the generated speech. WER = (wrong + missing + extra words) ÷ number of words in the text. It measures **intelligibility**. | Lower |
| **SIM** (speaker similarity) | A speaker-verification model (WavLM-Base+) turns speech into a "voiceprint" vector. SIM is the cosine similarity between the voiceprints of the generated speech and of the speaker's real recording (at most 1). It measures **how much the output sounds like the person**. | Higher |
| **UTMOS** | A model that predicts how natural listeners would rate the speech (scale 1–5). Reported only; it is not used in the failure rule because it is unreliable on single utterances. | Higher |
| **Forward passes (NFE)** | Number of network runs (Section 1). The main cost measure: exact and independent of hardware. Wall-clock time on the T4 will be measured in the full study. | Lower |

WER example: text "the cat sat on the mat" (6 words), transcript "the cat sat in the mat". One word is wrong, so WER = 1/6 = 16.7%.

**Added failure** (Definition 3.1). An early output fails if it is clearly worse than the Step-8 output of the same input:

- WER > max(12.5%, reference WER + 1 point), **or**
- SIM < reference SIM − 0.02.

Example: the reference has WER 3% and SIM 0.70. An early output with WER 4.5% passes the WER test, because the limit is max(12.5%, 4%) = 12.5%. With SIM 0.67 it fails the SIM test, because the limit is 0.68. Since both outputs share the same random draws, the rule counts only the damage done by stopping the teacher early.

**Failure rate (risk):** the share of inputs with an added failure. Our limit is **5%** (α = 0.05).

**Learn-Then-Test (α = 0.05, δ = 0.10):** the calibration method. It picks the most cost-saving threshold whose failure rate can be certified at or below 5% with 90% confidence on calibration data.

**SNR (Ŝ) and decibels.** The prompt's signal-to-noise ratio says how much stronger the voice is than the background. Our estimate compares the energy of the loudest half of the prompt's 25 ms frames with that of the quietest 10% (Definition 3.3). On the decibel scale, every +10 dB means 10× more power:

| SNR | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---:|---:|---:|---:|
| Voice power ÷ noise power | ≈ 32× | 100× | ≈ 316× | 1,000× |

"Clean" means the original recording with no noise added (estimated at about 23–50 dB on the development speakers).

**Drift (dᵢ)** (Definition 3.2). At every teacher step, the model predicts what the final output will be. Drift is how much that prediction changed since the previous step, relative to its size. Small drift means the prediction has settled.

**Fixed / new and the McNemar test.** To compare a policy with SNR-only on the same inputs, count the SNR-only failures it avoids (**fixed**) and the failures it adds (**new**). The exact McNemar test asks whether that difference could be chance.

**Out of fold.** Thresholds are chosen on four-fifths of the speakers and scored on the remaining fifth, rotating, so no speaker is judged with a threshold tuned on that speaker.

## 5. The solutions

All results in this section come from the EXP-015 set: 60 development speakers × 6 prompt conditions (clean, and white noise added at 15, 18, 20, 22 and 25 dB) = 360 instances, out of fold.

| Policy | Rule in one line | Failures (of 360) | Rate | Passes |
|---|---|---:|---:|---:|
| Reference (released) | Step 8 for everyone | 0 | 0% | 21.00 |
| SNR-only | Step 4 if Ŝ ≥ 20 dB, else Step 8 | 86 | 23.9% | 14.40 |
| SNR + drift (Chapter 3) | First of Steps 4–6 where Ŝ ≥ θ and drift ≤ λ | 0 | 0% | 21.00 (no threshold certified) |
| Calibrated SNR gate | Step 4 if Ŝ ≥ 26 dB, else Step 8 | 7 | 1.9% | 18.71 |
| **SNR + TAS** | Step 4 if Ŝ ≥ 20 dB and the trajectory looks normal | **12** | 3.3% | **18.07\*** |
| **SNR + SWC**, clean every input below the gate | Step 4 for everyone; noisy prompts cleaned first | **20** | 5.6% | **13.00†** |
| **SNR + SWC**, clean heavy noise only | As above, but moderately noisy prompts go to Step 8 | **11** | 3.1% | **16.98†** |

\* Corrected TAS using d₁…d₄, from an offline re-analysis of the committed data; the correction is not committed yet. EXP-015, the candidates report and the hidden Appendix E file currently show 18.20 passes from the version that also read d₅ and d₆, values that only exist after Step 4.
† SWC outputs were generated on an A100 GPU and judged against references from the T4 (Section 7).

### 5.1 SNR-only (the baseline)

- **Idea:** noisy prompts are risky, so measure the prompt's SNR and let only clean-looking prompts hand over early.
- **Rule:** Step 4 if Ŝ ≥ 20 dB; otherwise Step 8.
- **Result:** the cheapest rule (14.40 passes), but **86 failures (23.9%)**, almost five times the 5% limit, because prompts with 18–22 dB of noise read above 20 dB (Section 3.5).
- **Verdict:** the right idea, but too blunt. It is the baseline every candidate must beat.

### 5.2 SNR + drift (the Chapter 3 method, presented at the concept defense)

- **Idea:** stop the teacher once its prediction has settled, as long as the prompt is clean enough.
- **Rule (Definition 3.4):** stop at the first candidate step (4, 5 or 6) where Ŝ ≥ θ and dᵢ ≤ λ; otherwise Step 8.
- **Result:**
  - On clean and 15 dB prompts, drift added nothing beyond SNR. With proper calibration the rule reduced to SNR-only on the test split: 4 of 200 failures at 17.04 passes for both (October 25 audit).
  - With moderate noise it reads the signal backwards. Noise makes the prediction settle **too early**: median d₄ falls from 0.61 on clean prompts to 0.17 at 15 dB, because the model has less to follow, not because the output is ready. Noisy prompts look "ready" and would fail. No threshold could be certified, so the rule fell back to Step 8 for everyone: safe, but with no saving.
- **Verdict:** this finding, which we call **premature stabilization**, is what led to TAS.

### 5.3 SNR + TAS: Trajectory Anomaly Score (candidate 1)

- **Idea:** flip the drift logic. Under a noisy prompt the trajectory goes still too early, so **unusual stillness is a warning, not a go signal**.
- **How it works:**
  1. Collect the drift profile z = (log d₁, …, log d₄), which the teacher has already computed by Step 4.
  2. Learn what normal looks like from **clean prompts only**: the mean μ and covariance Σ of z. No failure labels are needed for this step.
  3. Score how unusual an input is with a Mahalanobis distance, A(x) = (z − μ)ᵀ Σ⁻¹ (z − μ). It counts only when the trajectory is stiller than normal; otherwise A(x) = 0.
  4. Hand over at Step 4 if Ŝ ≥ 20 dB **and** A(x) ≤ κ̂; otherwise run to Step 8. The threshold κ̂ is certified by Learn-Then-Test.
- **Result:** 12 failures (3.3%, within the limit) at 18.07 passes. It fixes 74 SNR-only failures and adds **0 new** ones. The 0 holds by design: TAS only vetoes handovers, and a vetoed input runs the full teacher, which cannot fail.
- **Trade-off:** it buys safety with compute. It uses about 25% more passes than SNR-only but saves about 14% against the reference. It adds no network passes, only arithmetic on four numbers.
- **Keeps the title:** its inputs are still SNR and drift.

### 5.4 SNR + SWC: SNR-Wiener Conditioning (candidate 2)

- **Idea:** instead of avoiding noisy prompts, **clean them** so that the cheap early handover becomes safe. Most noisy-prompt failures were speaker-similarity failures: the output inherits the prompt's noise.
- **How it works:**
  1. In the model's own mel spectrogram, estimate the noise level N_b of each frequency band from the quietest 20% of the prompt frames.
  2. Multiply every prompt frame by the gain G = min{1, max{0.1, 1 − β(Ŝ)·N_b / m}}, where m is the frame's magnitude in that band. This subtracts the noise and never cuts a frame below 10% of its value.
  3. Set the strength from the SNR: β(Ŝ) = min{1, max{0, (30 − Ŝ)/10}}. Prompts at 30 dB or above are left untouched; prompts at 20 dB or below are cleaned at full strength.
  4. Give the cleaned prompt to both the teacher and the student, then hand over at Step 4 (13 passes).
  5. Route by SNR with two certified thresholds θ_e ≤ θ_hi: Step 4 with the original prompt if Ŝ ≥ θ_hi; Step 4 with the cleaned prompt if Ŝ < θ_e; Step 8 in between.
- **Result:**
  - On the 300 noisy instances, SWC's Step-4 output beat the full teacher's output on the uncleaned prompt: speaker similarity **+0.126** (higher in 90% of instances) and WER 4.8 points lower.
  - Cleaning every input below the gate gave 20 failures at **13.00 passes**. It is the only policy with fewer failures **and** less compute than SNR-only, although 5.6% is just over the limit.
  - Cleaning heavy noise only gave 11 failures (3.1%) at 16.98 passes.
- **Trade-off:** it changes the input instead of only deciding, and it assumes steady background noise (white noise here). On its own it uses SNR but not drift; the drift side of the title comes from TAS.

### 5.5 How the two candidates fit together

TAS **decides** which inputs may go early; SWC **repairs** the input so that more can go early. One combination was tested: the TAS gate with SWC at the handover gave 12 failures at 18.13 passes, no better than either alone. The other order, cleaning the prompt first and then letting TAS check the cleaned trajectory, is untested and would need a new GPU run.

## 6. Why TAS and SWC are our candidate methods

### 6.1 How they were chosen

Across EXP-012 to EXP-015 the team tested more than ten ideas. In the policy tables many variants look acceptable, but mostly because the calibrated SNR gate does the work: the methods themselves made the early output worse or no better. Two ideas did something on their own:

- **SWC** made the early output itself better: noisy Step-4 failures fell from 126 to 21 of 300, and it is the only policy with fewer failures and less compute than SNR-only.
- **TAS**, like the simpler drift veto, turned the team's drift finding into a working safety check that keeps the SNR + drift design: 86 failures down to 12, with 0 new.

The ideas that did not work:

| Idea | Result |
|---|---|
| Drift-Extrapolated Handover (EXP-013) | Noisy failures 126 → 90, but clean failures 2 → 25 |
| Three handover operators: cDEH, TCR, SEA (EXP-014) | No compute saving; TCR clearly harmful |
| Self-Noise Negative Guidance and SNR-Compensated Guidance (EXP-015) | Worse than the plain Step-4 handover: 160–164 and 142–153 noisy failures vs 126 |
| Output-side noise cancellation and an antithetic student pair (EXP-015) | Worse (156 noisy failures) or no effect (124) |

### 6.2 What we designed, and why it counts as novel

Both candidates are **built from established components, combined in a new way**. That is a legitimate kind of novelty, and most methods are presented this way, as long as three things hold:

1. **It has not been done before.** To our knowledge, nobody has combined these parts this way; a proper literature search is still needed before the defense.
2. **It is not obvious.** The design follows from an insight.
3. **It works.** It beats the baseline, here on development data.

| | SNR + TAS | SNR + SWC |
|---|---|---|
| Established components | Drift score (Chapter 3), Mahalanobis anomaly score, a model of normal learned from clean prompts, Learn-Then-Test | Spectral subtraction (Boll, 1979); SNR-dependent strength and a spectral floor (Berouti, Schwartz & Makhoul, 1979)‡; noise estimated from the quietest frames |
| What we designed | Drift read in reverse: early stillness is a reason **not** to stop, used as a certified veto on the teacher-to-student handover | Cleaning of the sampler's own conditioning (teacher and student), with strength from the scheduler's SNR, so an early handover is safe without extra teacher steps |
| Insight behind it | Premature stabilization, the team's finding (Section 5.2) | Noisy-prompt failures are mostly the voice inheriting the noise |
| Evidence it works | 86 → 12 failures against SNR-only | 86 → 20 failures with fewer passes than SNR-only |
| Strength of the novelty claim | Stronger: new decision logic built on a new finding | Weaker: the formula is unchanged, and a pretrained denoiser did about as well (similarity +0.117) |

‡ Berouti et al. is not yet in the thesis bibliography; check the reference before citing it.

**How to say it:**

> "We propose two methods built from established components. TAS combines the teacher's drift profile, a one-sided anomaly score learned from clean prompts, and Learn-Then-Test calibration; to our knowledge, it is the first to treat premature stabilization as a reason not to hand over early. SWC applies SNR-scaled spectral subtraction to the sampler's conditioning so that an early handover becomes safe."

**Do not claim:**

- that the formulas or components themselves are new;
- that the candidates beat the calibrated SNR gate (they trade a few more failures for less compute);
- that they hold on new speakers or real-world noise (not tested yet).

**Be ready for:**

- *"Why not the simple veto (d₄ too small, so run to Step 8)? It got 13 failures against TAS's 12."* TAS is the general form of that rule. The new part, reading drift in reverse, is shared by both, and on these data the anomaly score adds no measurable gain yet; the confirmation run will test whether it does.
- *"Why is it called Wiener?"* The gain is magnitude spectral subtraction, not a Wiener filter. Either rename the method or explain the name.

## 7. Limits and next steps

**Limits**

- **Development evidence only:** 60 development speakers and white noise only. The methods were designed after looking at these data; TAS was even first scored on the same data before EXP-015's preregistration was written.
- **Replay, not a live run:** each input was generated at Step 4 and Step 8, and each policy's choice was looked up afterwards. Passes are counted, not timed.
- **TAS correction pending:** the as-run TAS also read d₅ and d₆. The corrected version (d₁…d₄) gives 12 failures at 18.07 passes, but it is not committed yet.
- **SWC hardware caveat:** SWC outputs came from an A100 and the references from the T4. The student's random draws differ between those GPUs, which most likely adds failures to SWC.
- **The 5% limit holds only on average across noise levels.** At the worst level, TAS fails on 10% (6 of 60 at 22 dB) and SWC on 8.3%.
- **The calibrated SNR gate (26 dB) is not beaten on both counts.** Its threshold was tuned on these same noise levels.

**Next steps**

1. Commit the TAS correction (d₁…d₄) to the runner, the hidden Appendix E file, the TAS figure and the reports.
2. **EVAL-008 confirmation:** generate SWC and its references on the same T4, with an SWC-at-Step-8 control; test both candidates, with settings fixed in advance, on speakers not used here; add babble, reverberation and telephone-band noise.
3. **Full study after the proposal:** run the frozen protocol on the held-out speakers, live on the T4, with real timing.

## Sources

- Chapter 3, measures and Definitions 3.1–3.4: [03-methodology.tex](../../../paper/chapters/03-methodology.tex)
- Appendix D (problem validation): [appendices.tex](../../../paper/chapters/appendices.tex)
- Appendix E (candidates), removed from the manuscript on 2026-10-27: [appendix-e-hidden.tex](../../../paper/chapters/appendix-e-hidden.tex)
- SNR readings per noise level: [EXP-011 pilot report](../../experiments/EXP-011-followup-t4/results/pilot/PILOT_REPORT.md)
- SNR + drift reduces to SNR-only; d₄ shows no discrimination on clean prompts: [October 25 audit](../../audits/2026-10-25/FINDINGS.md)
- Ideas that did not work: [EXP-013](../../experiments/EXP-013-endpoint-extrapolation-prelim/README.md), [EXP-014](../../experiments/EXP-014-handover-operators-prelim/README.md)
- Candidate results and failures per noise level: [EXP-015 report](../../experiments/EXP-015-noise-guidance-prelim/results/NOISE_GUIDANCE_REPORT.md), [candidates report](PROPOSED_CANDIDATES_TAS_SWC_REPORT.md)
