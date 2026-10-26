# Proposed Candidates: Trajectory Anomaly Score (TAS) and SNR-Wiener Conditioning (SWC)

**Preliminary development evidence from EXP-012 to EXP-015 (Google Colab, Tesla T4 and A100), 26 October 2026.**
**Thesis:** A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching Text-to-Speech Inference

> **Status.** Development evidence on 60 development speakers with synthetic Gaussian noise. These are candidates for the proposal, not adviser-approved results. Independent confirmation on unseen speakers is pending, and the SWC numbers carry a hardware caveat (Section 8). Every number below is computed from committed CSV files; nothing is typed in.

---

## 1. Summary

The study's problem is that one fixed teacher-to-student handover is **unsafe for noisy voice prompts** (early handover damages the speech) and **wasteful for clean prompts** (late handover spends compute for no measurable gain). The baseline is **SNR-only**: hand over at Step 4 (13 forward passes, NFE) if the prompt SNR is at least 20 dB, otherwise run the full teacher (Step 8, 21 NFE). On the 360-case preliminary set, SNR-only fails **86 of 360** cases (23.9%), almost five times the study's 5% risk limit, because prompts with 18–22 dB of noise measure above 20 dB and pass the gate.

After four rounds of preliminary experiments, two candidates remain. They do different jobs:

| Candidate | Role | What it is | Novelty |
|---|---|---|---|
| **TAS: Trajectory Anomaly Score** | **Detects** inputs that are unsafe to hand over early | A one-sided anomaly score on the teacher's own drift profile, learned from clean prompts, used with the SNR gate and certified by Learn-Then-Test | A new detection algorithm (Section 4.4) |
| **SWC: SNR-Wiener Conditioning** | **Repairs** noisy prompts so early handover becomes safe | An SNR-scheduled spectral gain that removes the prompt's noise from the conditioning inside the sampler, before the teacher and student read it | A new use of a classic operation (Section 5.4) |

**Key results against SNR-only** (out of fold, speakers held out, pooled Learn-Then-Test calibration, exact McNemar on the same 360 cases):

| Policy | Failures | Risk | Within 5% limit | Mean NFE | SNR-only failures fixed / new failures |
|---|---:|---:|---|---:|---|
| SNR-only (20 dB) | 86 | 23.9% | no | 14.40 | — |
| **SNR + TAS** | **12** | **3.3%** | **yes** | 18.20 | 74 / 0 (p ≈ 1e-22) |
| **SNR + SWC on every input below the calibrated gate** | **20** | 5.6% | no (just over) | **13.00** | 75 / 9 (p ≈ 4e-14) |
| **SNR + SWC, heavy-noise routing** | **11** | **3.1%** | **yes** | 16.98 | 79 / 4 (p ≈ 4e-19) |
| Full teacher always (reference) | 0 | 0% | yes | 21.00 | — |

- **TAS** meets the risk limit that SNR-only breaks, at 26% more compute than SNR-only and 13% less than the full teacher.
- **SWC** produces fewer failures than SNR-only **and** uses less compute. On noisy prompts its Step-4 speech is better than the full teacher's (speaker similarity +0.126, better in 90% of 300 cases, p ≈ 4e-44; word error rate 0.048 lower).

---

## 2. Data, measures and procedure

- **Cases:** the EXP-012 Quick set, the first 60 development speakers (by utterance ID) of Seed-TTS test-en × 6 prompt conditions (clean as recorded; white Gaussian noise at 15, 18, 20, 22 and 25 dB) = 360 cases. Development speakers only; no held-out data is used.
- **Reference and failure rule:** each case's plain Step-8 output (21 NFE) is the reference. A candidate output fails (Definition 3.1) if its WER exceeds max(0.125, WER_ref + 0.01) or its speaker similarity is more than 0.02 below the reference's. WER uses Whisper-large-v3, similarity uses WavLM-Base+ SV against the clean ground-truth recording.
- **Stored references:** the plain Step-4 and Step-8 results were synthesized on the Tesla T4 (EXP-002 for clean and noisy15, EXP-011 pilot and EXP-012 for the Gaussian levels).
- **Calibration and evaluation:** every policy threshold is certified by fixed-sequence Learn-Then-Test (α = 0.05, δ = 0.10) on training speakers only, in 5-fold speaker-grouped cross-validation (seed 2026), and evaluated on the held-out fold. "Pooled" certifies the total risk; "stratified" requires every condition to pass.
- **EXP-015 run:** 26 October 2026 on a Colab NVIDIA A100-SXM4-40GB; thesis commit `bdee11e`, Shonget-DMOSpeech2 `8a7868f`; `preregistration.json` written before synthesis. Results with SHA-256 hashes: [`research/experiments/EXP-015-noise-guidance-prelim/results/`](../../experiments/EXP-015-noise-guidance-prelim/results/MANIFEST.json); full generated report: [`NOISE_GUIDANCE_REPORT.md`](../../experiments/EXP-015-noise-guidance-prelim/results/NOISE_GUIDANCE_REPORT.md).

---

## 3. How the candidates were found

| Experiment | Tested | Finding |
|---|---|---|
| EXP-012 | Gates on free signals (drift, guidance gap, SNR–drift consistency) | Noise makes the teacher's endpoint **stop moving early**: median drift d₄ falls from 0.61 (clean) to 0.17 (15 dB). Definition 3.4 ("stop when drift is low") is therefore backwards under noise and certifies no threshold. Low drift works as a **veto**. |
| EXP-013 | Drift-extrapolated handover (DEH) | The Step-8 endpoint cannot be predicted from Step 4: the endpoint changes direction between steps. |
| EXP-014 | Zero-cost handover operators (coupled restart, coherence-gated DEH, envelope anchoring) | No operator saves compute; 96 of 126 noisy Step-4 failures are **speaker-similarity** failures, so the prompt's noise is leaking into the voice. |
| Offline analysis | Paired clean/noisy outputs of the same speakers and seeds | A Step-4 method must raise noisy-prompt similarity by about **+0.08** to win; methods inside the sampler reached at most +0.024; the noise itself costs **0.06–0.24** similarity. At equal compute, no re-ranking of inputs (SNR, drift, guidance gap, learned combinations) beats SNR-only (85–89 failures). |

Two conclusions followed: a **detector** can make early handover safe by sending the right inputs to the teacher (TAS), and only **removing the prompt's noise** can make early handover both safe and cheap (SWC).

---

## 4. Candidate 1: Trajectory Anomaly Score (TAS)

### 4.1 Definition

For each input, the teacher's own drift profile (computed anyway; no extra forward pass) is

  **z = (log d₁, …, log d₆)**, where dᵢ = ‖x̂₁(tᵢ) − x̂₁(tᵢ₋₁)‖_M / ‖x̂₁(tᵢ)‖_M.

From the **clean** prompts of the calibration speakers (no failure labels), estimate the mean μ and covariance Σ of z. The score is

  **A(z) = (z − μ)ᵀ Σ⁻¹ (z − μ)** if the mean of (z − μ) is negative (the trajectory is stiller than normal), and **A(z) = 0** otherwise.

**Handover rule:** hand over at Step 4 if **Ŝ ≥ 20 dB and A(z) ≤ τ̂**; otherwise run the full teacher. τ̂ is certified by Learn-Then-Test (≤ 5% added failures at 90% confidence).

### 4.2 Why it works

From the endpoint estimate x̂₁(tᵢ) = xᵢ + (1 − tᵢ)vᵢ and the Euler step xᵢ = xᵢ₋₁ + (tᵢ − tᵢ₋₁)vᵢ₋₁, the change of the endpoint estimate is exactly

  **Δᵢ = x̂₁(tᵢ) − x̂₁(tᵢ₋₁) = (1 − tᵢ)(vᵢ − vᵢ₋₁)**,

so drift measures how much the generation direction is still turning. A noisy prompt weakens the conditioning: in EXP-012 the guidance gap ‖v_c − v_u‖/‖v_c‖ falls from 0.236 (clean) to 0.116 (15 dB) and moves with drift (Spearman 0.89). The trajectory then straightens early, which ordinary early-exit logic reads as "ready". TAS reads it as **premature stabilization**.

### 4.3 Results (out of fold, 360 cases)

| Policy | Pooled: failures | Pooled: NFE | Stratified: failures | Stratified: NFE |
|---|---:|---:|---:|---:|
| SNR-only (20 dB) | 86 | 14.40 | 86 | 14.40 |
| SNR + drift as written (Definition 3.4) | 0 | 21.00 (no threshold certified) | 0 | 21.00 |
| Calibrated SNR gate (θ ≈ 26 dB) | 7 | 18.71 | 1 | 20.84 |
| SNR + simple drift veto (−d₄) | 13 | 18.20 | 1 | 20.29 |
| **SNR + TAS** | **12** | **18.20** | **1** | **20.82** |

- **Against SNR-only:** 74 failures fixed, 0 new (p ≈ 1e-22); within the 5% limit that SNR-only breaks; 26% more compute.
- **Against Definition 3.4:** TAS saves 13% compute where the original rule saves nothing.
- **Against the simple veto:** a tie (12 vs 13 failures at the same NFE).
- **Against the calibrated SNR gate:** a trade-off, not a win. TAS hands over 126 inputs early instead of 103 (18.20 vs 18.71 NFE) but has 12 instead of 7 failures; the difference is not significant (paired p = 0.23). The gate's θ ≈ 26 dB was tuned on these noise levels; TAS's trajectory model needs only clean prompts (its threshold, like the gate's, is certified with failure labels).

### 4.4 Novelty

Mahalanobis distance and Learn-Then-Test are existing tools. New, to our knowledge: (1) the phenomenon TAS detects, premature stabilization of the teacher's trajectory caused by corrupted conditioning; (2) the one-sided design, which treats "too still" as danger, the opposite of ordinary early-exit logic; (3) the trajectory model learned from clean conditions only; (4) its use to gate a certified teacher-to-student handover. The closest work, DiffPath (Heng, Thiery and Soh, NeurIPS 2024; arXiv 2405.11881), uses the rate of change and curvature of diffusion paths to detect out-of-distribution images in an unconditional model; it does not detect corrupted conditioning or drive a handover. TAS needs only the sampler's endpoint estimates, so it applies to any guided flow-matching or diffusion sampler.

### 4.5 Evidence status

TAS chooses between plain Step 4 and Step 8, whose outcomes were measured on the T4, so its results are hardware-consistent and unaffected by Section 8. They come from the same development cases on which premature stabilization was discovered, so they are development evidence, not independent confirmation.

---

## 5. Candidate 2: SNR-Wiener Conditioning (SWC)

### 5.1 Definition

Before the teacher and the student read the prompt, its conditioning features (the log-mel frames of the prompt region, for both the teacher's and the student's conditioning) are cleaned:

1. Per mel band b, the **noise floor** N_b is the mean linear magnitude of the prompt's quietest 20% of frames (pauses that contain only background).
2. Each prompt frame with magnitude M gets the gain **G = clip(1 − β(Ŝ)·N_b / M, 0.1, 1)**.
3. The **strength is set by the prompt SNR**: **β(Ŝ) = β_max · clip((30 − Ŝ) / 10, 0, 1)**, so prompts at 30 dB or above are untouched and prompts at 20 dB or below get full strength (β_max = 1.0; a half-strength variant uses 0.5).

The teacher then runs to Step 4 with the cleaned conditioning, and the student completes as usual. **Cost: 13 NFE; no extra network.**

### 5.2 Results

**Quality of SWC at Step 4 (13 NFE) against the full-teacher reference (21 NFE)**, paired, mean differences:

| Prompt | Speaker similarity | Word error rate |
|---|---:|---:|
| Clean | −0.001 | −0.002 |
| Noise 15 dB | **+0.205** | **−0.142** |
| Noise 18 dB | **+0.170** | −0.047 |
| Noise 20 dB | **+0.129** | −0.034 |
| Noise 22 dB | **+0.089** | −0.010 |
| Noise 25 dB | +0.037 | −0.007 |
| **All noisy (300)** | **+0.126** (better in 90%, p ≈ 4e-44) | **−0.048** (p ≈ 9e-12) |

**Failures with every input at Step 4:** plain Step 4 fails 126 of 300 noisy cases; **SWC fails 21** (15 dB: 4, 18 dB: 2, 20 dB: 2, 22 dB: 5, 25 dB: 8). Half strength (β_max 0.5) fails 33.

**Policies against SNR-only** (pooled, out of fold): SWC for every input below the calibrated gate (θ_hi = 26 dB in every fold) gives **20/360 at 13.00 NFE**, fewer failures and less compute than SNR-only (75 fixed, 9 new, p ≈ 4e-14), but its 5.6% risk is just above the 5% limit. Routing only heavy noise to SWC (θ_e = 18–21 dB by fold) gives **11/360 (3.1%) at 16.98 NFE**, within the limit. Under stratified calibration the routed policy falls back to 1/360 at 20.80 NFE.

### 5.3 Comparison with alternatives that clean the noise

| Method | Noisy similarity vs reference | Noisy failures at Step 4 | Extra network |
|---|---:|---:|---|
| **SWC** | **+0.126** | **21** | none |
| Pretrained enhancer on the prompt (DNS64 Demucs), full | +0.117 | 31 | yes (median 15.6 ms per prompt on the A100) |
| Pretrained enhancer, SNR-scheduled | +0.119 | 27 | yes |
| Prompt-Informed Environment Cancellation (cleaning the **output**) | −0.018 (WER +0.32) | 156 | none |

SWC matches or exceeds a pretrained neural enhancer without any extra network, and cleaning the **input** works where cleaning the **output** fails. Applied to every prompt, the full-strength enhancer also harmed clean prompts (15 clean failures); SWC's SNR schedule leaves prompts at 30 dB or above untouched and cleans clean prompts between 23 and 30 dB only lightly (mean clean similarity change −0.001).

### 5.4 Novelty

The gain itself is classic spectral subtraction (Boll, IEEE Trans. ASSP, 1979) with a minimum-energy noise estimate, so SWC is **not** a new formula. New, to our knowledge: (1) applying it to the **conditioning features inside the sampler**, for both teacher and student, so the model never imitates the noise; (2) the **SNR-scheduled strength**; (3) using conditioning repair as a **substitute for teacher computation**, which is what makes early handover safe; (4) the finding that input-side repair beats output-side cancellation and matches a neural enhancer. Related work on noisy prompts uses trained prompt denoisers (Lu et al., Interspeech 2025; arXiv 2505.13830), enhancement front-ends, or separation models that keep the background on purpose (arXiv 2608.03011; arXiv 2502.07345). SWC suits stationary background noise; non-stationary noise is untested.

### 5.5 Evidence status

SWC was preregistered in EXP-015 with its parameters fixed in advance (not tuned on outcomes), so its effect size is close to a confirmatory result on these speakers. Its outputs were synthesized on the A100 and judged against T4 references (Section 8).

---

## 6. Methods that did not win (EXP-015)

| Method | Noisy failures at Step 4 (plain: 126) | Noisy similarity vs reference | Verdict |
|---|---:|---:|---|
| Self-Noise Negative Guidance, η 0.5 / 1.0 (18 NFE) | 160 / 164 | −0.021 / −0.024 | worse |
| SNR-Compensated Guidance, full / capped (13 NFE) | 142 / 153 | −0.016 / −0.020 | worse |
| Antithetic student pair (16 NFE) | 124 | −0.007 | no effect |
| Prompt-Informed Environment Cancellation (13 NFE) | 156 | −0.018 | worse |

Methods that steer or amplify guidance act on a condition that still contains the noise, and output-side cancellation cannot remove noise the model has already generated.

---

## 7. Preregistered criteria

C1: noisy Step-4 failures drop by at least 25% with p < 0.05. C2: clean Step-4 failures rise by at most 1. C3: a calibrated policy (route or heavy) has lower mean NFE than the **calibrated** SNR gate with no more failures.

| Method | C1 | C2 | C3 |
|---|---|---|---|
| SWC (β_max 1.0 / 0.5) | PASS / PASS | FAIL (5 / 6 vs 2) | FAIL |
| Enhancer at Step 4 / Step 6 / SNR-scheduled | PASS / PASS / PASS | FAIL (15 / 13 / 4) | FAIL |
| SWC + PIEC | PASS | FAIL (5) | FAIL |
| PIEC, SNNG, SCG, antithetic | FAIL | FAIL except PIEC | FAIL |

C3 compares with the calibrated SNR gate (7/360 at 18.71 NFE), not with SNR-only: SWC's routed policy uses less compute (16.98) but has more failures (11), so it fails C3. C2 failures are inflated by the hardware difference in Section 8: methods that leave most clean prompts unchanged also show clean failures, for example SNR-Compensated Guidance (scale exactly 2, the plain setting, for prompts above about 33 dB) with 7 and the antithetic pair with 7, against 2 for the stored plain output.

---

## 8. Caveat: A100 versus T4

The smoke test re-scored stored T4 results on the A100. The teacher reproduced exactly (endpoint difference 0, drift difference 0.000), but the student outputs did not:

| Case | Step | WER (A100 / stored T4) | Similarity (A100 / stored T4) |
|---|---|---|---|
| Speaker 18770703, clean | 4 / 8 | 0.000 / 0.000; 0.000 / 0.000 | 0.955 / 0.945; 0.958 / 0.940 |
| Speaker 18770703, 20 dB | 4 / 8 | 0.143 / 0.071; 0.143 / 0.071 | 0.826 / 0.903; 0.796 / 0.855 |
| Speaker 18829843, clean | 4 / 8 | 0.000 / 0.000; 0.000 / 0.000 | 0.952 / 0.967; 0.956 / 0.962 |
| Speaker 18829843, 20 dB | 4 / 8 | 0.000 / 0.154; 0.077 / 0.077 | 0.637 / 0.720; 0.683 / 0.648 |

The teacher's initial noise is drawn on the CPU and matches. The student's noise is drawn on the GPU with a seeded generator, and the drawn values evidently differ between GPU models, so each new A100 output is judged against a T4 reference made with a different student sample.

- **Unaffected:** SNR-only, the calibrated SNR gate, the drift veto and TAS, which use only T4 outcomes.
- **Affected:** every new-method output (SWC, enhancer, guidance variants). The mismatch adds random disagreements, which inflate failures; for a method that improves quality as much as SWC, same-hardware failures are expected to be no higher.
- The full-FP32 (no TF32) setting added before the run does not prevent this, because the difference comes from the random draws, not from arithmetic precision.

---

## 9. Recommendations

1. **Same-hardware confirmation of SWC.** Run SWC on the T4 (about 1–1.5 hours for the 300 noisy cases), or recompute the plain Step-4 and Step-8 references on the A100, and add an SWC-at-Step-8 control so the gain from cleaning can be separated from the safety of the early handover (as done for the enhancer: enhanced Step 4 judged against enhanced Step 8 fails 18 of 360).
2. **Independent confirmation of both candidates** on speakers not used here (for example, calibration-split speakers with new noise draws), with SWC's parameters and TAS's procedure fixed in advance.
3. **Other degradations:** babble, reverberation and telephone-band prompts, where SNR is unreliable and SWC's stationary-noise assumption is weakest.
4. **Proposal text:** whether and how to present TAS and SWC in Chapter 3 is a team and adviser decision. Suggested claims: TAS is a new detection algorithm; SWC is a new use of spectral subtraction for conditioning repair, not a new formula; both are development evidence.

---

## 10. Files

| What | Where |
|---|---|
| EXP-015 results with hashes | `research/experiments/EXP-015-noise-guidance-prelim/results/` (`MANIFEST.json`) |
| Generated EXP-015 report | `research/experiments/EXP-015-noise-guidance-prelim/results/NOISE_GUIDANCE_REPORT.md` |
| Runner (TAS, SWC and all methods) | `code/colab/run_noise_guidance_prelim.py` (commit `bdee11e` for the run) |
| Notebook | `code/colab/DMOSpeech2_Noise_Guidance_Prelim.ipynb` |
| Earlier experiments | `research/experiments/EXP-012…014` (READMEs and results) |
| Re-run the analysis without a GPU | copy the results folder elsewhere (the committed files are hashed evidence), then `python code/colab/run_noise_guidance_prelim.py analyze --out <copy>` |
