# 2026-11-17: Literature map for a new handover algorithm (teacher–student sampling and noisy-prompt TTS, 2025–2026)

Follow-up to EVAL-026 and EVAL-027; no new task ID.
- **The request:** after the spread-weighted SNR turned out to be a known tool in a new place, Rhenel Jhon R. Sajol asked for a new algorithm ("WE NEED NEW ALGORITHM") and approved a literature map before any design ("do it").
- **The aim:** find where an inference-only handover algorithm would collide with published work, and where it would not.
- **Status:** targeted web searches on 2026-11-16 and 2026-11-17. They returned abstracts and snippets, not full texts, so the numbers quoted in the sources are unverified here. The map is not exhaustive.

## A. Teacher first, distilled model after (inference-only)

| Work | What it does | Overlap with this thesis |
|---|---|---|
| [DMOSpeech 2](https://arxiv.org/abs/2507.14988) | Teacher-guided sampling: the teacher runs the first steps, the student finishes | The method this thesis modifies |
| [Gandikota and Bau](https://openaccess.thecvf.com/content/WACV2026/html/Gandikota_Distilling_Diversity_and_Control_in_Diffusion_Models_WACV_2026_paper.html) (WACV 2026; [arXiv 2503.10637](https://arxiv.org/abs/2503.10637)) | The base model for the first step only, then the distilled model; diversity is set by the first steps | "The teacher decides early" is published for images |
| [A First-Step Bias Restores Lost Diversity](https://arxiv.org/html/2610.37110) (ReSeed, 2026) | Corrects only the student's first step | Same principle |
| [Restoring Diversity by Probing Activations](https://arxiv.org/html/2611.18550v1) (2026) | Perturbs activations along the teacher's principal directions at the first step, training-free | Same principle, inside the network |
| [T-Stitch](https://arxiv.org/abs/2402.14167) (ICLR 2025) | A small model first, a large one later, on a fixed schedule | Model switching on a fixed schedule |

## B. Training the early segment for diversity

| Work | What it does |
|---|---|
| [DP-DMD](https://arxiv.org/pdf/2602.03139) (2026) | Trains the first step to follow the teacher's early path and uses the mode-seeking DMD loss for the rest |
| [CrossDistill](https://arxiv.org/pdf/2609.14725) (2026) | Trajectory distillation at high noise and distribution matching at low noise; the reverse order loses both diversity and quality |
| [DistillAlign](https://arxiv.org/pdf/2607.26811v1) (2026) | Pairs DMD with a mode-covering consistency constraint (video) |

**What this means:** the trained version of "the teacher decides early, the student finishes" (the commit map raised on 2026-11-16) is published.

## C. Fewer teacher steps and adaptive solvers

Covered in the [adaptive teacher log](2026-11-16-adaptive-teacher-screen.md):
- [Fast F5-TTS / EPSS](https://arxiv.org/abs/2505.19931) prunes F5-TTS's sway schedule to 7 evaluations.
- [Gotta Go Fast](https://arxiv.org/abs/2105.14080) adapts the step size to an error tolerance.
- ["Error as Signal"](https://en.papernotes.org/ICLR2026/image_generation/error_as_signal_stiffness-aware_diffusion_sampling_via_embedded_runge-kutta_guid/) (ICLR 2026) gates a correction on each step's built-in error estimate.

## D. Noisy voice prompts in zero-shot TTS

**The problem is known:**
- **[Noise robustness of flow-matching zero-shot TTS](https://arxiv.org/abs/2406.05699)** (Interspeech 2024): quality drops sharply with noisy prompts.
- **[LauraTTS noise robustness](https://arxiv.org/pdf/2505.13830)** (2025): the original model's metrics collapse with noisy prompts.

**Every published remedy changes the model or the prompt:**

| Remedy | Works |
|---|---|
| Retraining with noise or filtered data | [arXiv 2406.05699](https://arxiv.org/abs/2406.05699) (DNSMOS filtering and noise mixing) |
| A model that removes the noise itself | [OZSpeech](https://aclanthology.org/2025.acl-long.1043.pdf) (ACL 2025) |
| Cleaning the prompt or its tokens first | [LauraTTS](https://arxiv.org/pdf/2505.13830) (2025) |
| Noise-robust speaker or style features | [SSL adapters](https://arxiv.org/pdf/2401.05111) (2024); NoreSpeech (Interspeech 2023, found in snippets only) |
| Controllable removal or preservation of the background | [arXiv 2502.07345](https://arxiv.org/pdf/2502.07345) (2025) |
| Separating speech and background for editing | [SeamlessEdit](https://arxiv.org/html/2505.14066v2) (2025) |
| Copying the scene on purpose | [AST-LDM](https://arxiv.org/pdf/2406.12688) |

**The enhancement trade-off:**
- **What happens:** enhancing the prompt raises perceived quality but lowers speaker similarity ([WildSpoof 2026 BSC submission](https://arxiv.org/abs/2602.05770); also arXiv 2406.05699).
- **The open call:** the WildSpoof paper names adaptive strategies as future work.
- **Already tested here:** this thesis tested prompt enhancement, prompt Wiener gains and noise-floor cancellation in EXP-015 and EXP-016.

## E. Per-band quantile mapping

- **Quantile histogram equalization** of mel bands for noise-robust ASR is classic ([Hilger and Ney, Eurospeech 2001](https://www.isca-archive.org/eurospeech_2001/hilger01_eurospeech.html)). See the [invariance theory note](2026-11-15-handover-invariance-theory.md).
- **Per-band quantile mapping** is used for climate-model bias correction (for example [arXiv 2301.01253](https://ar5iv.labs.arxiv.org/html/2301.01253)).
- **Not found:**
  - rank-based transfer of temporal structure between two generative models;
  - quantile mapping at a teacher-to-student handover.

## Related: why the student drops texture

- **Mode seeking:** distribution-matching students are mode-seeking (reverse KL) and lose subtle textures ([Adaptive Video Distillation](https://arxiv.org/html/2603.21864), 2026). Low-noise refinement losses address it ([Flash-DMD](https://arxiv.org/html/2511.20549v1)).
- **Fit with our principle:** this agrees with the spread principle, in which the teacher copies spread-out background texture and the student drops it ([spread-weighted SNR log](2026-11-17-spread-weighted-snr.md)).
- **Not found:** a measurement of this for a TTS prompt's background.

## The gaps

| Gap | What is missing | Evidence of the gap |
|---|---|---|
| **G1** | An inference-only teacher–student sampler for TTS that keeps the teacher's prosody on a noisy prompt without the copied noise | Section D: every remedy retrains or enhances; WildSpoof lists adaptive strategies as future work |
| **G2** | A measurement of how much of a prompt's background a teacher and its distilled student each reproduce | Not found; the searches surfaced no study measuring background reproduction |
| **G3** | A characterization of which handover payloads can carry a recording condition, and quantile matching at a handover with the student's own draft as the reference | Not found here or in the 2026-11-15 audit |

**Where new work would collide:**
- "Teacher early, student late" in any general form (sections A and B).
- Adaptive teacher steps (section C).
- Routing by noise type through spectral flatness (the [spread-weighted SNR log](2026-11-17-spread-weighted-snr.md)).

**What follows:**
- **The candidate:** an algorithm aimed at G1 that uses G3's operator, the Commit-Point Quantile Handover (`code/analysis/cpqh_screen.py`). Its screen is logged separately.
- **G2:** a measurement result for Chapter 4, not an algorithm.

## Queries (2026-11-16 and 2026-11-17)

1. Spectral flatness of background noise to classify noise type and choose a processing path.
2. Zero-shot TTS prompt quality filtering with DNSMOS; selecting a model or enhancing the prompt by noise type.
3. Distilled few-step diffusion reproducing less noise texture than the teacher; noisy conditioning prompts.
4. Distilling only the early trajectory segment to preserve diversity, with a mode-seeking model for later steps.
5. Gandikota and Bau's base-model-first-step hybrid.
6. Noisy reference prompts in 2025–2026 zero-shot TTS, keeping prosody without enhancement artifacts.
7. Few-step distilled TTS prosody diversity and DMOSpeech 2 follow-ups.
8. Per-band quantile mapping or histogram equalization between two generative models; invariance to stationary additive noise.
9. Diffusion models copying the background of reference audio; acoustic environment leakage.

The adaptive-solver queries are listed in the [adaptive teacher log](2026-11-16-adaptive-teacher-screen.md).

## Checks run

- **Searches:** the nine above, run on 2026-11-16 and 2026-11-17. Each source here appeared in their results.
- **Code:** none in this log.
