# EXP-024: Analytic Unconditional Guidance (development pilot)

**Status:** prepared, not yet run. Preregistration frozen 2026-11-14 before any synthesis (`993c306d13cfb32d...`). Task EVAL-017 (todo). The run needs a Colab Tesla T4: [notebook](../../../code/colab/DMOSpeech2_EXP024_Analytic_Guidance_Pilot.ipynb), [runner](../../../code/colab/run_analytic_guidance_pilot.py), [tests](../../../code/tests/test_analytic_guidance.py).

- **A pilot:** it decides whether a frozen test on new speakers is worth running, and cannot support a claim.
- **Not adopted:** nothing here changes the thesis method, and no adviser has reviewed it.

## Idea

Classifier-free guidance runs the teacher twice per step: the conditional pass sees the text and the voice prompt, and the unconditional pass sees neither.
- **The theory:** in flow matching, x_t = (1 − t)·ε + t·x₁. The best unconditional velocity is (E[x₁ | x_t] − x_t) / (1 − t).
  - **At t = 0:** the state is pure noise and tells nothing about x₁, so E[x₁ | x₀] is the average mel μ. The unconditional velocity is then exactly μ − x.
  - **For small t:** E[x₁ | x_t] = μ + O(t).
- **Why it fits the handover:** teacher-guided sampling hands over at Step 4 (t = 0.0761, the student's restart level) or Step 8 (t = 0.2929, the 0.25 reference). So the teacher only runs while the signal is weak.
- **The method:** at Steps 0..K, Analytic Unconditional Guidance (AUG) replaces the network's unconditional evaluation by v_u = (μ − x_i) / (1 − t_i). The conditional evaluation and the guidance strength are unchanged. Each of those steps costs one pass instead of two.
- **μ:** one value per mel band. It is the network's own unconditional estimate at t = 0, pooled over every frame of 20 calibration cases from speakers the pilot does not use (Quick-set positions 41–60). Nothing is trained.

| Output | What it is | Passes |
|---|---|---:|
| W4 / W8 | released teacher, handover at Step 4 / Step 8 (the 0.25 reference) | 13 / 21 |
| W4_K0 / W8_K0 | closed form at Step 0 only (diagnostic) | 12 / 20 |
| W4_K2 / W8_K2 | closed form at Steps 0–2 | 10 / 18 |
| W4_K4 / W8_K4 | closed form at Steps 0–4 | 8 / 16 |

AUG changes no handover decision, so it would make any policy cheaper, including SNR + drift.

## Novelty check (family-level searches, 2026-11-14)

No method was found that keeps the conditional pass and computes the unconditional term of classifier-free guidance in closed form. The closest methods all differ:
- **Lite-CFG** ([DC-DiT](https://arxiv.org/html/2603.06351)): gives the unconditional pass fewer tokens, and needs a trained multi-budget model.
- **Path-Drop Guidance** ([SPRINT](https://arxiv.org/html/2510.21986v2)): the unconditional pass skips the middle blocks, which needs a trained architecture.
- **The guidance interval** ([Kynkäänniemi et al.](https://arxiv.org/pdf/2404.07724)): switches guidance off at high noise, which changes the output.
- **[CFG-Zero⋆](https://arxiv.org/pdf/2503.18886v2.pdf):** zeroes the first steps.
- **Independent-condition and time-step guidance** ([arXiv 2407.02687](https://arxiv.org/pdf/2407.02687)): still run a network pass.
- **Gaussian analytical teleportation** ([Wang and Vastola](https://arxiv.org/pdf/2412.09726)): replaces the whole model to skip early steps.
- **Analyses of CFG in Gaussian or linear settings** ([2505.19210](https://www.alphaxiv.org/abs/2505.19210), [2509.22007](https://arxiv.org/html/2509.22007v1), [2610.13369](https://arxiv.org/html/2610.13369)): no sampling shortcut. The 2026 paper's sampler still runs the unconditional pass.

The theory (Gaussian structure at high noise) is known. This is not a full literature review.

**Correction after the freeze (2026-11-14, before any synthesis).** A search of practitioner tools found community ComfyUI nodes named "Uncond Zero" ([1](https://www.runcomfy.com/comfyui-nodes/Uncond-Zero-for-ComfyUI), [2](https://comfy.icu/node/Uncond-Zero)). They skip the unconditional pass by treating its prediction as zero. So "replace the unconditional pass with a cheap fixed substitute to save compute" is already practised, though not published as a method. AUG's substitute is the derived closed form, restricted to the window where it is accurate. That makes it a principled variant of a known idea, not a new kind of mechanism. By the requester's standard it is partly novel. The frozen preregistration's sentence ("no method that computes the unconditional term in closed form") is still literally true, but it understated the neighbourhood.

## What runs

| Block | Data | What it measures |
|---|---|---|
| Calibration | 20 cases, Quick-set positions 41–60, clean | μ, and how much each case's band means differ from it |
| Diagnostic and quality | 40 speakers (positions 1–40) × clean and noisy15 = 80 cases, synthesis seed = input seed | (1) the gap between the closed form and the network's unconditional estimate at Steps 0–8, in units of the guidance signal; (2) at Step 0, how much of that gap is seed-specific (3 further noise draws per case); (3) how far AUG with K = 0, 2, 4 moves the Step-4 and Step-8 estimates (whole and 9-frame slow part), next to refining the grid from 16 to 32 intervals; (4) W4_K and W8_K scored on clean prompts, W8_K on noisy15 |
| Pitch variety | 10 speakers × clean × 20 seeds (EXP-020's seeds) | CVf0 of the four AUG outputs; the released W4 is re-synthesized for the first speaker to check EXP-020's stored CVf0 |
| Timing | 10 clean cases, 3 repeats, alternating order | T4 time of the teacher to Step 4 and Step 8, released against K = 2 and K = 4 |

The released teacher's outputs for the same cases and seeds are reused from EXP-018 (W4, W8), and their CVf0 from EXP-020 (plain_4, plain_8).

The smoke test must pass three checks before the full run is allowed:
1. it reproduces the stored W4 and W8 scores;
2. this runner's teacher loop matches the released teacher;
3. with AUG switched off, the loop is the released teacher (≤ 1e-4 for both loop checks).

## Criteria (frozen)

| Criterion | Rule |
|---|---|
| quality_clean, per K = 2, 4 | W4_K and W8_K against the released W4 and W8 of the same case and seed, clean prompts (speaker bootstrap, 95%). W4: capped-WER upper bound ≤ +0.01. W8: capped-WER upper bound ≤ +0.015. Both: similarity lower bound ≥ −0.01 and UTMOS lower bound ≥ −0.10. |
| pitch_variety, per K | mean CVf0 of W4_K and of W8_K minus EXP-020's stored CVf0 of the released handover at the same step, both ≥ −0.01 |
| quality_noisy_reference, per K (secondary) | W8_K against the released W8 on noisy15: WER ≤ +0.025, similarity ≥ −0.01, UTMOS ≥ −0.10 |

**Decision rule:**
- **Advance:** AUG goes to a frozen test on new speakers only if some K meets quality_clean and pitch_variety. The K carried forward is the largest that does, which is the cheapest.
- **Noisy prompts:** quality_noisy_reference at that K decides whether the frozen test also covers noisy-prompt teacher runs.
- **Otherwise:** the pilot is reported as negative, with no retuning on these data.

## Pre-run power check

[`code/analysis/exp024_power_check.py`](../../../code/analysis/exp024_power_check.py) produced [power_check.json](power_check.json) before the freeze. It uses stored outputs only.
- **The proxy:** a change known to be small, namely the 16- against the 32-interval grid at the same sway time (EXP-018).
- **Clean Step 4:** with 40 speakers the proxy passes the bounds in 100% of 300 random subsets. With 20 speakers, UTMOS passes only 69%, hence 40 speakers.
- **Clean Step 8:** Step 7 stands in, because its 32-interval twin was stored and Step 8's was not. The WER bound +0.01 passes the proxy in 33% of subsets and +0.015 in 100%, hence +0.015.
- **Noisy15:**
  - **The proxy fails here:** it passes all noisy bounds in only 0% (Step 4) and 2% (Step 7) of subsets, because the grid change itself moves similarity (−0.010 [−0.020, −0.002] at Step 4).
  - **So noisy-prompt Step-4 handovers are not judged.**
  - **So the noisy-prompt reference criterion is secondary.** A failure there does not show that AUG is worse than a grid change.

## Risks known before the run

- **Shared errors:** the theorem assumes an ideal network. If the two branches share an error that CFG cancels, the closed form will not cancel it. The Step-0 split measures this directly.
- **Stationary μ:** μ is the same for every frame position and canvas length; the position part of the Step-0 split measures what that misses.
- **Accuracy at later steps:** the first-order term the closed form leaves out grows with t, so K = 4 (t up to 0.076) is the riskiest arm.

## Data and evidence status

Development evidence only. The 60 Quick-set speakers designed earlier experiments, and the prompts are white noise only. Results will be stored under `results/` with a SHA-256 manifest. Audio is not committed.
