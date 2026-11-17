# Adaptive per-input computation (steps, early exit, switch point, model choice, guidance) and risk/quality guarantees: speech/audio and diffusion/flow generation

Scope note for the report writer. These notes look for work that could threaten or support the novelty of a thesis contribution. That contribution is a **training-free, per-input decision of when a guided teacher (F5-TTS) hands over to a distilled student (DMOSpeech 2)**. It uses signals the sampler already computes (prompt SNR, endpoint-stabilization drift) and a threshold calibrated with **Learn-Then-Test (LTT)** to bound the rate of added quality failures. Each work entry gives these fields: **Adapted** (what changes per input), **Signal**, **Training** (training-free or trained), **Guarantee**, **Domain**, **Configs** (models and datasets evaluated), **Results**, and **Verification** (how much of the primary source I actually read). Searched 2 Nov 2026. "Abstract-level" means only the arXiv abstract page or a search-result quotation of it was read, not the full paper.

---

## Speech/audio: adaptive or early-exit diffusion/flow TTS, per-utterance step counts, speculative decoding/early exit in codec-LM TTS, adaptive-NFE vocoders, Shallow Flow Matching, per-input teacher/student switching

### Takeaway
I found **no published TTS work that switches from a teacher to a student per utterance**, and no flow/diffusion TTS work that picks the number of steps or the stopping point per utterance with a statistical guarantee. The nearest speech work is Shallow Flow Matching (a trained head predicts a per-utterance start time, and a global α chosen on validation PMOS scales it, with no guarantee). Chatterbox-Flash is also close: training-free, confidence-based early termination per block in a discrete block-diffusion TTS, again with no guarantee. Speech speculative decoding adapts compute per input in autoregressive codec LMs, but it uses trained drafts and gives no risk bound.

### Cited Findings

**Shallow Flow Matching (SFM): verified. The coordinator's "trained head, global α, no guarantee" is confirmed.**
- Yang, Cai, Saito, Wang, Saruwatari, "Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis", arXiv 2505.12226 (May 2025; v2 Oct 2025). The arXiv comments say it was accepted at NeurIPS 2025. — [arXiv abs](https://arxiv.org/abs/2505.12226)
- **Adapted:** the start time t_h of the flow ODE, **per utterance**. The SFM head outputs X̂_h, t̂_h and log σ̂²_h from the coarse hidden states. The time output goes through a sigmoid and is **averaged over frames, giving one value per utterance**. Inference (Algorithm 2) builds the starting state from this t_h. — [arXiv HTML v2](https://arxiv.org/html/2505.12226v2)
- **Global α:** an "SFM strength" α ≥ 1 scales t̂_h and σ̂_h at inference (Eq. 22). It is chosen **on each model's validation set** by a coarse search that starts at α = 1.0, steps by 1.0 while tracking mean pseudo-MOS (PMOS), then checks α ± 0.5. The reported values are Matcha-TTS 2.5 (LJSpeech) and 3.5 (VCTK), StableTTS 3.0, CosyVoice 2.0, and CosyVoice-DiT 2.5. The authors avoid finer grids to limit overfitting. — [arXiv HTML v2](https://arxiv.org/html/2505.12226v2)
- **Training:** the head is trained jointly with the backbone. The total loss is L_coarse + L_t + L_σ + L_μ + L_CFM, and the head has two Conv1d layers. It is **not training-free**. — [arXiv HTML v2](https://arxiv.org/html/2505.12226v2)
- **Configs:** Matcha-TTS (LJSpeech, VCTK), StableTTS (VCTK), CosyVoice (LibriTTS, flow module only), and the authors' CosyVoice-DiT (LibriTTS). **F5-TTS is not a backbone.** — [arXiv HTML v2](https://arxiv.org/html/2505.12226v2)
- **Results (test sets):**
  - Matcha-TTS (LJSpeech): UTMOS 4.186 → 4.257, WER 3.308 → 3.413 (slightly worse).
  - Matcha-TTS (VCTK): UTMOS 4.008 → 4.106, WER 1.534 → 0.952.
  - CosyVoice: UTMOS 4.191 → 4.194, UTMOSv2 3.303 → 3.480, WER 3.513 → 3.810 (worse).
  - CosyVoice-DiT: UTMOS 3.499 → 3.751.
  - The authors themselves call the WER/SIM gains inconsistent.
  - — [arXiv HTML v2](https://arxiv.org/html/2505.12226v2)
- **Speed:** on Matcha-TTS (LJSpeech) at α = 5.0, the RTF reductions with adaptive-step solvers are:
  - Heun(2): 60.8%, NFE 306.72 → 120.07
  - Fehlberg(2): 51.0%, NFE 45.28 → 22.14
  - Bosh(3): 59.3%
  - Dopri(5): 47.6%, NFE 121.46 → 63.74

  The paper reports that fixed-step solvers do not get this acceleration. — [arXiv HTML v2](https://arxiv.org/html/2505.12226v2)
- **Guarantee:** none. Theorems 1–2 are properties of the flow construction (Wasserstein-2 continuity, equivalence of the piecewise flow). They say nothing about prediction quality. α is chosen empirically. The authors observe that the learned t_h tends to be small. — [arXiv HTML v2](https://arxiv.org/html/2505.12226v2)

**Chatterbox-Flash (block-diffusion TTS with confidence-based early decoding): verified from HTML**
- Seo, Park, Nam (Resemble AI), "Chatterbox-Flash: Prior-Calibrated Block Diffusion for Streaming Zero-Shot TTS", arXiv 2605.30748 (May 2026, preprint). — [arXiv HTML v1](https://arxiv.org/html/2605.30748v1)
- **Adapted:** the number of denoising iterations **per block**, in a discrete masked block-diffusion decoder fine-tuned from an AR TTS decoder (T3). A block stops when all positions are unmasked or K steps are reached. — [arXiv HTML v1](https://arxiv.org/html/2605.30748v1)
- **Signal:** a prior-calibrated PMI score, s_i = log p_i(x̂_i) − log p̄(x̂_i). At step k, positions above the q_k-quantile of the block's masked-position scores are unmasked, with q_k = max(0, 1 − α(k+1)/K). — [arXiv HTML v1](https://arxiv.org/html/2605.30748v1)
- **Training:** the decoding rule works at inference time only, on top of a fine-tuned block-diffusion model. The block prior p̄ is computed once and cached. **Guarantee:** none, since α is a hyperparameter and the threshold is an empirical quantile. — [arXiv HTML v1](https://arxiv.org/html/2605.30748v1)
- **Results:**
  - Main table (K = 8, α = 0.5): about 20% fewer steps per block.
  - LibriSpeech-PC test-clean: SIM-o 0.717 → 0.713, WER 1.67 → 1.67, UTMOS 4.29 → 4.28.
  - Seed-TTS test-en: SIM-o 0.704 → 0.704, WER 1.96 → 2.04, UTMOS 4.09 → 4.08.
  - At K = 10, savings are about 20%, 32% and 41% for α = 0.5, 0.75 and 1.0.
  - A figure caption labels about 4.6 average steps as "~41%", which is internally inconsistent: from K = 10 that would be a 54% reduction.
  - No noisy-prompt evaluation.
  - — [arXiv HTML v1](https://arxiv.org/html/2605.30748v1)

**SPAR-K (early exit in spoken language models): abstract-level**
- Huang, Chiang, Lee, arXiv 2603.09215 (Mar 2026, v2 Aug 2026, preprint). The exit layer is **scheduled, not chosen per input**: most speech positions exit at a fixed intermediate layer, with periodic full-depth refresh steps. Configs are Step-Audio-2-mini and GLM-4-Voice. Results: QA accuracy drops by at most 0.82%, and speech decoding depth falls by up to 11% (Step-Audio-2-mini) and 5% (GLM-4-Voice). The authors report that **confidence-based early exit, common in text LLMs, performs worse on speech tokens**. No guarantee. — [arXiv abs](https://arxiv.org/abs/2603.09215)

**Speculative decoding in codec-LM TTS (per-input compute varies with draft acceptance)**
- VADUSA: Li, Wang, Zhang, Guo, Yu, "Fast and High-Quality Auto-Regressive Speech Synthesis via Speculative Decoding", ICASSP 2025, arXiv 2410.21951. It uses draft heads (MEDUSA-style, applied to VALL-E according to the search summary) and a tolerance mechanism during sampling. The abstract gives no numbers, and it does not say explicitly whether the draft heads are trained. No guarantee. — [arXiv abs](https://arxiv.org/abs/2410.21951)
- SSD: Lin et al., "Accelerating Autoregressive Speech Synthesis Inference With Speech Speculative Decoding", Interspeech 2025, arXiv 2505.15380. A lightweight separate draft model is verified in parallel by the target, giving a **1.4× speedup** over AR decoding. Subjective evaluations say target quality is preserved. No guarantee is stated. — [arXiv abs](https://arxiv.org/abs/2505.15380)
- Nguyen et al., "Accelerating Codec-based Speech Synthesis with Multi-Token Prediction and Speculative Decoding", arXiv 2410.13839. It uses multi-token prediction with Viterbi-based speculative decoding. *Search-result level only; not opened.* — [arXiv HTML](https://arxiv.org/html/2410.13839v1)

**Fixed (non-adaptive) step reductions in flow TTS and vocoders: for contrast only**
- Fast F5-TTS / EPSS (Zheng et al., Interspeech 2025, arXiv 2505.19931) is a training-free, **fixed** non-uniform 7-step schedule for F5-TTS. It reports RTF 0.030, about 4× faster, and also works on E2 TTS. The schedule is global, not per utterance. *Abstract quoted in search result; full paper not opened.* — [arXiv abs](https://arxiv.org/abs/2505.19931v2); [ISCA archive](https://www.isca-archive.org/interspeech_2025/zheng25d_interspeech.html)
- ZipVoice (arXiv 2506.13053) uses flow distillation with a fixed NFE K as the speed/quality knob. *Search-result level.* — [arXiv HTML](https://arxiv.org/html/2506.13053v2)
- InferGrad (Chen et al., ICASSP 2022, arXiv 2202.03751) is a diffusion vocoder trained with the few-step inference schedule in the loop. It matches WaveGrad quality with 2 instead of 6 iterations on LJSpeech, using a fixed step count. *Abstract via search result.* — [arXiv](https://arxiv.org/abs/2202.03751v1)

**Classifier-free guidance scheduling in zero-shot TTS (timestep-based, not per input)**
- Zheng & Maleki, "Selective Classifier-free Guidance for Zero-shot Text-to-speech", arXiv 2509.19668 (Sep 2025; v2 Mar 2026). They report that image-domain CFG variants generally fail to help TTS. Speaker similarity improves when standard CFG is used early and selective CFG only in later timesteps. The abstract does not say whether the switch point is global or per utterance, and does not name the base models. — [arXiv abs](https://arxiv.org/abs/2509.19668)

### Inferences
- **The coordinator's SFM description is accurate**, with two refinements. First, the per-utterance start time comes from a **trained** head and is **scaled by a global α tuned on validation PMOS**. Second, SFM works within **one model**, starting the ODE from a coarse estimate; there is no teacher→student handover. SFM's adaptivity is per utterance but learned. Its α tuning is a validation search with no risk statement. A thesis that offers a training-free, LTT-calibrated, per-utterance decision is therefore distinguishable on all three axes (training, decision variable, guarantee).
- Chatterbox-Flash is the closest **training-free, confidence-driven, per-input stopping rule in TTS**. It works on discrete masked tokens, per block, and gives no guarantee. It does not threaten a teacher→student handover claim, but it is relevant prior art for "TTS samplers can stop adaptively using signals they already compute" and should be cited.
- SPAR-K's finding that confidence-based early exit underperforms on speech tokens can be cited to motivate why speech-specific signals (prompt SNR, endpoint drift) are worth studying instead of transferring LLM confidence scores directly. It is evidence from autoregressive spoken LMs, not flow TTS.
- Adaptive-step ODE solvers (Dopri5, Bosh3, as used in SFM's speed evaluation) already make NFE vary per input through local error control. That is a per-input adaptive-compute baseline with **numerical** tolerances, not quality-risk guarantees. A careful reviewer could raise it.

### Gaps
- **No per-input teacher→student (guided teacher → distilled student) switching in TTS was found.** Searches covered TTS and diffusion together with phrases such as early exit, adaptive steps, model switching and hybrid; none returned it. Absence of evidence from a limited search (about 15 queries), not proof of absence.
- I found **no adaptive-NFE (per-input) diffusion or flow vocoder**. Every vocoder hit used fixed few-step schedules.
- VADUSA's quantitative speedups and whether its drafts are trained, and Nguyen et al.'s numbers, were not verified because I read only abstracts.
- The base models for Selective CFG were not confirmed.
- A 2026 "joint residual reweighting" CFG paper on F5-TTS/CosyVoice2 appeared only in an aggregator snippet (emergentmind 2606.25672) and is unverified. It is not listed above.

---

## General diffusion/flow: early-exit diffusion, per-sample adaptive step schedules, convergence/stability-based stopping, adaptive guidance, per-sample model switching (which are training-free?)

### Takeaway
Per-input adaptivity in image and video diffusion is well established. **Learned** policies include AdaDiff (Zhang et al.), cost-aware routing and RouteT2I. **Training-free** stability rules include Adaptive Guidance (cosine-similarity convergence → drop CFG), AdaptiveDiffusion (third-order latent difference → skip steps) and Jot (per-token stability). Model switching along the trajectory exists (T-Stitch: small→large; Hybrid SD: large→small) but uses a **fixed** switch point. The closest structural analog to the thesis is Castillo et al.'s Adaptive Guidance. It makes a training-free, **per-sample** switch from expensive guided steps to cheaper unguided steps, driven by a convergence signal, with a hand-set global threshold and **no guarantee**, in the image domain.

### Cited Findings

**Adaptive Guidance (AG): the closest structural analog (training-free, per-sample, convergence-triggered switch from guided to cheaper steps)**
- Castillo, Kohler, Pérez, Pérez, Pumarola, Ghanem, Arbeláez, Thabet, "Adaptive Guidance: Training-free Acceleration of Conditional Diffusion Models", arXiv 2312.12487 (Dec 2023). The arXiv page lists no peer-reviewed venue, so the venue is **unverified**. — [arXiv abs](https://arxiv.org/abs/2312.12487)
- **Signal:** the cosine similarity γ_t between conditional and unconditional noise predictions. Once γ_t > γ̄, AG switches from CFG steps (2 NFE) to conditional-only steps (1 NFE). γ̄ is its only hyperparameter. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12487)
- **Per sample:** yes. The truncation point depends on γ̄, the starting noise and the prompt, so each sample switches at a different step. The rule was motivated by a differentiable-NAS search over per-step choices, then simplified by hand. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12487)
- **Training:** training-free and plug-and-play. **Guarantee:** none. γ̄ is set by hand. — [arXiv abs](https://arxiv.org/abs/2312.12487)
- **Configs:** LDM-512 (about 900M parameters, used for the search) and EMU-768 (2.7B). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12487)
- **Results:**
  - About 25% less compute overall.
  - At γ̄ = 0.991, 29.6 average NFE versus 40 for CFG.
  - Human study on 1,000 prompts: CFG 502 vs AG 498, Wilcoxon p = 0.603, so no significant difference.
  - About 50% of the speed-up of Guidance Distillation, without training.
  - LinearAG saves more but deviates from the baseline.
  - — [ar5iv](https://ar5iv.labs.arxiv.org/html/2312.12487); [arXiv abs](https://arxiv.org/abs/2312.12487)

**AdaptiveDiffusion: training-free, stability-based step skipping**
- Ye, Yuan, Xia, Yan, Chen, Yan, Shi, Zhang, "Training-Free Adaptive Diffusion with Bounded Difference Approximation Strategy", NeurIPS 2024, arXiv 2410.09873. — [arXiv abs](https://arxiv.org/abs/2410.09873)
- **Adapted:** which noise-prediction steps are skipped, reusing earlier predictions. **Signal:** stability between timesteps, measured by a **third-order latent difference**. **Training:** none. **Guarantee:** none. **Domain/configs:** image and video diffusion models (abstract-level; specific models not confirmed). **Results:** about 2–5× average speedup with outputs reported as matching the full-step result. — [arXiv abs](https://arxiv.org/abs/2410.09873)

**Truncated Jump Sampling (TJS) / "endpoint decodability": training-free early stop that returns the decoded x0**
- Peng & Gao, "x-Prediction Is All You Need: Training-Free Accelerated Generation via Endpoint Decodability", arXiv 2607.06114 (Jul 2026, v4 Aug 2026, preprint). The method stops the diffusion/flow ODE at t* and returns E[x0 | x_t]. It needs no retraining, distillation or architecture change. On SDXL, SD3.5M, Z-Image-Turbo and three class-conditional benchmarks it reports 20–70% fewer NFEs with near-matched quality. — [arXiv abs](https://arxiv.org/abs/2607.06114)
- **Adaptivity:** the abstract describes a single exit time. A search-result excerpt of the PDF defines "TJS-γ" as early exit at a **fixed fraction** γ (NFE = ⌈γK⌉+1). It also describes structure-aware or adaptive early-exit policies as future work. *The fixed-γ and future-work points are excerpt-level.* — [arXiv PDF](https://arxiv.org/pdf/2607.06114)

**AdaDiff (Tang et al.; DeeDiff): trained, uncertainty-driven layer-wise early exit inside each step**
- Tang, Wang, Ding, Liang, Li, Xu, "AdaDiff: Accelerating Diffusion Models through Step-Wise Adaptive Computation", arXiv 2309.17074 (v1 Sep 2023, v3 Aug 2024). The Fast-yet-Safe paper (below) refers to this model as **DeeDiff**. An ECCV 2024 venue is **unverified**. — [arXiv abs](https://arxiv.org/abs/2309.17074); [arXiv HTML v3](https://arxiv.org/html/2309.17074v3)
- **Adapted:** network depth at each denoising step, exiting when uncertainty is low. **Signal:** a timestep-aware uncertainty estimation module (UEM) at intermediate layers. **Training:** yes. It starts from pretrained U-ViT and trains with L_simple + λL_u + βL_UAL. **Threshold:** set by hand in the range 0.2–0.01. **Guarantee:** none. — [arXiv HTML v3](https://arxiv.org/html/2309.17074v3)
- **Results (FID, layer reduction):**

  | Dataset | FID | Layer reduction |
  |---|---|---|
  | CIFAR-10 | 3.7 | 47.7% |
  | CelebA | 3.9 | 46.2% |
  | ImageNet-256 (U-ViT-L) | 4.5 | 45.2% |
  | MS-COCO | 7.40 | 43.6% |

  BERxiT and CALM-style exits reach about FID 20 at about 20% layer reduction on CIFAR-10. — [arXiv HTML v3](https://arxiv.org/html/2309.17074v3)

**A Simple Early Exiting Framework (ASE): time-dependent, not per-sample**
- Moon et al., ICML 2024, arXiv 2408.05927. The method skips part of the score network following a **time-dependent exit schedule**, so the adaptivity is across timesteps, not across samples. It is compatible with fast solvers. Whether it needs fine-tuning was **not confirmed** from the abstract. — [arXiv abs](https://arxiv.org/abs/2408.05927)

**AdaDiff (Zhang et al.): learned per-prompt step count**
- Zhang, Wu, Xing, Shao, Jiang, "AdaDiff: Adaptive Step Selection for Fast Diffusion Models", AAAI 2025, arXiv 2311.14768. — [arXiv abs](https://arxiv.org/abs/2311.14768)
- **Adapted:** the number of denoising steps **per prompt**, conditioned on the richness of the prompt text. **Training:** a lightweight policy trained by policy gradient on a reward that balances time and quality. **Guarantee:** none. **Configs:** three image and two video generation benchmarks (not named in the abstract). **Results:** quality similar to a fixed 50-step baseline with 33–40% less inference time. — [arXiv abs](https://arxiv.org/abs/2311.14768)

**Model switching along the trajectory (fixed switch point)**
- **T-Stitch:** Pan et al., ICLR 2025, arXiv 2402.14167. It uses a **small** model for early steps, then a **large** one. It is training-free. About 40% of early steps of DiT-XL can go to a 10× faster DiT-S with no measured drop on ImageNet, and it also applies to Stable Diffusion. The switch fraction is a global setting. *Abstract-level via search.* — [arXiv abs](https://arxiv.org/abs/2402.14167); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/120339238f293d4ae53a7167403abc4b-Paper-Conference.pdf)
- **Hybrid SD:** Yan et al., arXiv 2408.06646 (Aug 2024, preprint). A **large** cloud model handles early steps (semantic planning) and a **small** edge model handles later steps (detail), which is the same direction as teacher→student. The framework is described as a "training-free inference framework", but the edge U-Net is pruned and a lightweight VAE is trained. It reports a 66% cloud-cost reduction and a 225.8M-parameter edge model. The abstract does not say whether the switch is per prompt. — [arXiv abs](https://arxiv.org/abs/2408.06646)

**Per-sample draft/verify (speculative sampling for diffusion)**
- De Bortoli, Galashov, Gretton, Doucet, "Accelerated Diffusion Models via Speculative Sampling", arXiv 2501.05370 (Jan 2025; v2 Jul 2025). A draft model proposes and the target diffusion model accepts or rejects, so target NFE vary per sample. One drafting strategy needs **no draft-model training**. It roughly **halves NFE while producing exact samples from the target**. The guarantee is distributional exactness, not risk control. — [arXiv abs](https://arxiv.org/abs/2501.05370)

**Learned per-prompt routing across step counts or models (trained, no guarantee)**
- Cost-Aware Routing (CATImage): Li, Chen, Su, Jitkrittum, Sun, Sangkloy, TMLR, arXiv 2506.14753. A learned router sends each prompt to a number of denoising steps or to one of nine pretrained T2I models, including a small distilled model for simple prompts. Evaluated on COCO and DiffusionDB, its average quality is higher than any single model's. No guarantee. — [arXiv abs](https://arxiv.org/abs/2506.14753)
- RouteT2I (ICCV 2025, arXiv 2411.13787) routes each prompt between a large cloud model and a light edge model using a learned multi-metric quality predictor. *Search-result level.* — [arXiv abs](https://arxiv.org/abs/2411.13787); [mlanthology ICCV 2025](https://mlanthology.org/iccv/2025/xin2025iccv-adaptive)

**Per-position convergence stopping in diffusion language models (training-free)**
- Jot: "Just on Time: Token-Level Early Stopping for Diffusion Language Models", arXiv 2602.11133 (Feb 2026), listed at NeurIPS 2026. It freezes each token once its top-2 probability-ratio confidence passes a threshold that is adjusted by proximity to already-resolved tokens. It is training-free. *The speedups (up to 5.54× on GSM8K and 19.60× on HumanEval for Dream-7B, within about 3% accuracy) come from a secondary review and are unverified.* — [arXiv HTML](https://arxiv.org/html/2602.11133); [NeurIPS 2026 listing](https://neurips.cc/virtual/2026/173817)

### Inferences
- **The novelty of "training-free per-input switch driven by a convergence signal" is partly threatened** by Adaptive Guidance. AG already shows that a per-sample, training-free, threshold-on-convergence rule can drop expensive guided evaluations in favour of cheaper ones. The thesis differs in four ways:
  - (a) the cheaper branch is a **different, distilled model**, so this is a teacher→student handover, not one model losing guidance;
  - (b) the domain is **zero-shot speech** with a **prompt-quality (SNR) signal**;
  - (c) the threshold is **calibrated with LTT** rather than set by hand;
  - (d) AG's evidence is image SSIM plus one human-preference test.

  The thesis should cite AG prominently and frame its signal (endpoint-stabilization drift) as belonging to the same family as AG's cosine-similarity convergence, AdaptiveDiffusion's third-order latent difference, TJS's endpoint decodability, Jot's per-token stability and CALM's hidden-state saturation. **"Using a signal the sampler already computes" is not new by itself.**
- **Switching between models along the trajectory is established** (T-Stitch, Hybrid SD), but every instance I verified uses a **fixed, global switch point**. Hybrid SD's large→small direction matches teacher→student. A **per-input** switch point between two different pretrained models, especially with a calibrated guarantee, was not found in diffusion literature either.
- Per-prompt adaptivity in image diffusion is mostly **learned** (Zhang et al.'s AdaDiff, CATImage, RouteT2I). The training-free methods (AG, AdaptiveDiffusion, TJS, Jot) use **hand-set** thresholds. None of these gives a statistical guarantee.
- Speculative sampling (De Bortoli et al.) is an alternative to threshold-based handover with a **different guarantee type**: exact target sampling, at the cost of target verification calls. A reviewer could ask why the thesis uses an LTT bound rather than an exactness-preserving draft/verify scheme.

### Gaps
- **No primary source found for a training-free, per-sample rule that stops when the predicted x0 stabilizes in image/flow diffusion.** TJS uses x0 decoding but a fixed exit fraction. AG and AdaptiveDiffusion use related stability signals for a different decision (dropping guidance, skipping steps).
- Venues not verified: AG (arXiv only), Tang et al. AdaDiff (ECCV 2024 suspected but not confirmed), Hybrid SD (arXiv only).
- AdaptiveDiffusion's exact models and per-model speedups, ASE's training requirement, and T-Stitch's per-sample behaviour come from abstracts only.
- Jot's numbers are secondary-source only.
- Prompt-aware serving (arXiv 2502.06798) and HADIS (arXiv 2509.00642), which use learned or rule-based routers with output re-checking, appeared in search results but were not opened.

---

## Guarantees: conformal prediction, Learn-Then-Test, or risk control applied to generative models, early exit or adaptive computation (CALM, conformal early exit, risk-controlled generation), and whether any is applied to diffusion, flow matching or speech

### Takeaway
**Calibrating an adaptive-computation threshold with LTT to bound degradation relative to the full model is established methodology.** CALM (NeurIPS 2022) did it for early-exit LMs with LTT and fixed-sequence testing. **Fast yet Safe (NeurIPS 2024) did it for early-exit diffusion (DeeDiff on CelebA, LPIPS consistency risk)** and for a big/little speculative-decoding model pair (BiLD), comparing CRC, UCB and LTT. I found **no** application of LTT, conformal prediction or risk control to **flow-matching or speech generation compute decisions**. The thesis's statistical method is therefore a transfer of a known recipe to a new decision variable (handover time) and a new domain (zero-shot TTS). It is not a new guarantee technique, and CALM and Fast yet Safe must be cited as direct precedents.

### Cited Findings

**Learn-Then-Test (the method itself)**
- Angelopoulos, Bates, Candès, Jordan, Lei, "Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control", arXiv 2110.01052 (Oct 2021; v5 Sep 2022). It reframes risk control as **multiple hypothesis testing** and gives explicit finite-sample guarantees for any model, without refitting it. Its examples are in computer vision and tabular medical data, not generation. The journal venue was not verified on the page read. — [arXiv abs](https://arxiv.org/abs/2110.01052)

**CALM: LTT-calibrated early exit for generation (the direct methodological precedent)**
- Schuster, Fisch, Gupta, Dehghani, Bahri, Tran, Tay, Metzler, "Confident Adaptive Language Modeling", **NeurIPS 2022 (oral)**, arXiv 2207.07061. — [arXiv abs](https://arxiv.org/abs/2207.07061)
- **Adapted:** decoder depth **per input and per generated token**.
- **Signals:** three confidence measures:
  - softmax response (top-2 gap; no training);
  - **hidden-state saturation** (cosine similarity of consecutive layers' hidden states; parameter-free, no training);
  - an early-exit classifier (trained, with the rest of the model frozen).
  - — [ar5iv](https://ar5iv.labs.arxiv.org/html/2207.07061)
- **Calibration:** **LTT with fixed-sequence testing** over a descending grid of thresholds in steps of about 0.05. It picks the lowest threshold that passes and falls back to the full model (λ = 1). p-values come from a Hoeffding-type bound; Hoeffding–Bentkus is used in practice. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2207.07061)
- **Guarantee:** distribution-free and finite-sample, given i.i.d. prompts and losses in [0,1]. It holds with probability ≥ 1−ε over the calibration draw, with ε = 0.05 in all experiments (CALM's notation: ε is the failure probability, δ the tolerance). The two notions are:
  - **Textual consistency:** expected distance to the full-model output ≤ δ, with δ = 0.1 and 0.25.
  - **Risk consistency:** expected increase in risk versus the full model, measured against references, ≤ δ, with δ = 0.02 and 0.05.

  The guarantee is on **expected sequence-level** consistency, not per token. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2207.07061)
- **Configs:** 8-layer T5 1.1 on CNN/DM (ROUGE-L), WMT15 EN-FR (BLEU), and SQuAD 1.1 (Token-F1). **Results (softmax measure, test set):**

  | Dataset | Consistency notion | δ | Avg. layers | Speedup |
  |---|---|---|---|---|
  | CNN/DM | Textual | 0.1 | 5.73 | 1.41× |
  | CNN/DM | Textual | 0.25 | 2.62 | 2.57× |
  | CNN/DM | Risk | 0.05 | 1.73 | 3.53× |
  | WMT | Textual | 0.25 | — | 2.83× |
  | WMT | Risk | 0.05 | — | 2.73× |
  | SQuAD | Textual | 0.25 | — | 1.88× |
  | SQuAD | Risk | 0.05 | — | 1.63× |

  Gains are largest on long outputs. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2207.07061)

**Fast yet Safe: risk-controlled early exit, including diffusion and a big/little model pair**
- Jazbec, Timans, Hadži Veljković, Sakmann, Zhang, Naesseth, Nalisnick, "Fast yet Safe: Early-Exiting with Risk Control", **NeurIPS 2024**, arXiv 2405.20915. It is a "distribution-free, post-hoc" tuning of the exit threshold. — [arXiv abs](https://arxiv.org/abs/2405.20915)
- **Frameworks:**
  - **CRC** (expected risk), extended to early exit and relaxed from monotone losses to monotone risks;
  - **UCB** (high-probability control, Waudby-Smith–Ramdas bound, δ = 0.1), recommended over LTT;
  - **LTT** as the baseline, via CALM.
  - — [arXiv HTML v2](https://arxiv.org/html/2405.20915v2)
- **Risks:** a "performance gap" risk (against labels) and a **"consistency" risk that uses the full model's outputs as pseudo-labels, so no labels are needed**. Losses are 0-1, mIoU, ROUGE-L/Token-F1, BLEU and **LPIPS (diffusion)**, plus Brier-score risks. — [arXiv HTML v2](https://arxiv.org/html/2405.20915v2)
- **Configs:**
  - ImageNet classification (MSDNet, DViT, L2W-DEN, Dyn-Perc);
  - Cityscapes segmentation (ADP-C);
  - T5 early-exit LMs (CNN/DM, SQuAD; replicating CALM);
  - **early-exit diffusion with DeeDiff on CelebA** (CIFAR in the appendix);
  - **speculative decoding with BiLD** (a big/little decoder pair) on IWSLT De–En.
  - — [arXiv HTML v2](https://arxiv.org/html/2405.20915v2)
- **Results:**
  - Classification (n = 100, ε = 0.05): about 61% fewer layers with CRC and about 46% with UCB.
  - Segmentation (mIoU gap risk): about 35.7% (ε = 0.05) and about 53.3% (ε = 0.1) reduction in mean exit layer.
  - LM: UCB saves more than LTT at the same guarantee.
  - **Diffusion: risk controlled at all tested levels, with modest per-step savings (about 15% at ε = 0.05) that accumulate over about 500 sampling steps. Looser ε gives visibly worse images, still within the stated control.**
  - BiLD: CRC/UCB speed up more than LTT while keeping control.
  - — [arXiv HTML v2](https://arxiv.org/html/2405.20915v2)

**Conformal sampling and stopping for generative LMs**
- Quach, Fisch, Schuster, Yala, Sohn, Jaakkola, Barzilay, "Conformal Language Modeling", **ICLR 2024**, arXiv 2306.10193. It calibrates a **stopping rule for sampling** candidates and a **rejection rule**, so that the output set contains at least one acceptable answer with high probability. Tasks are open-domain QA, summarization and radiology report generation. The abstract does not name the calibration framework. — [arXiv abs](https://arxiv.org/abs/2306.10193)

**Risk control for diffusion *outputs* (not compute)**
- Teneggi, Tivnan, Stayman, Sulam, "How to Trust Your Diffusion Model: A Convex Optimization Approach to Conformal Risk Control", ICML 2023, arXiv 2302.03791. K-RCPS gives entrywise calibrated intervals for diffusion samples in image denoising (faces, abdominal CT). It does not control sampling steps or compute. *Abstract via search result.* — [arXiv abs](https://web3.arxiv.org/abs/2302.03791v3)

**Calibrated deferral or delegation between cheap and expensive models (LLM domain)**
- Calibrate-Then-Delegate: Pona et al., arXiv 2604.14251 (Apr 2026, preprint). A delegation-value probe decides when to escalate from a cheap safety probe to an expensive expert. The threshold is calibrated on held-out data with **multiple hypothesis testing**, giving finite-sample guarantees on **either delegation budget or risk**, in LLM safety monitoring. The abstract does not name LTT; a search snippet says LTT, which is **unconfirmed**. — [arXiv abs](https://arxiv.org/abs/2604.14251)
- Conformal Cascade: Dou, Lian, Li, arXiv 2607.25018 (Jul 2026, preprint). A small model defers to a larger one unless its conformal prediction set is a singleton. Coverage is ≥ 1−Kα, tightening to 1−α under a condition. It is training-free and black-box. Evaluated on 18 multiple-choice QA benchmarks; open-ended generation is left to future work. — [arXiv abs](https://arxiv.org/abs/2607.25018)

**Speech: conformal methods appear only for recognition, not synthesis**
- Ernez et al. (2023, PMLR vol. 204) apply conformal risk control to ASR candidate-transcription sets, reportedly guaranteeing WER < 2% with 80% confidence on wav2vec 2.0. *Search-result level; adjacent task (ASR), not synthesis.* — [PMLR](https://proceedings.mlr.press/v204/ernez23a.html)

### Inferences
- **This is the most important threat or support finding.** The thesis's calibration logic (pick a threshold on a sampler signal, test a descending grid with LTT and fixed-sequence testing, bound the added-failure rate relative to the full or teacher-only output, fall back to the full model if no threshold passes) is **essentially CALM's recipe**. Fast yet Safe has already **applied the same family of methods to diffusion sampling** (LPIPS consistency to the full model) and to a **big/little model handoff** (BiLD). The thesis should therefore claim novelty in:
  - (i) the **decision variable**: the per-utterance teacher→student handover time in a guided-teacher/distilled-student flow TTS pipeline;
  - (ii) the **domain**: zero-shot TTS with noisy prompts;
  - (iii) the **signals**: prompt SNR plus endpoint-stabilization drift.

  It should **not** claim a new risk-control method or the first use of LTT for adaptive generative computation.
- **This is supportive evidence for the thesis:** it can borrow Fast yet Safe's "consistency risk" definition (added failure relative to the teacher-only output, so no human labels are needed). It can also borrow Fast yet Safe's evidence that LTT tends to be **more conservative** than UCB or CRC at the same guarantee. That helps anticipate a weaker-than-hoped compute saving and justify the choice (LTT handles non-monotone risk, which may matter if failure rate is not monotone in the handover threshold).
- Every guarantee here (CALM, Fast yet Safe, LTT) assumes **exchangeable (i.i.d.) calibration and test inputs**. This matches the repository's audit distinction: a bound computed on a development replay set does not transfer to an independent held-out set unless that set comes from the same distribution. Noisy-prompt distribution shift would break the guarantee unless calibration covers it.
- Calibrate-Then-Delegate and Conformal Cascade show the 2026 trend of calibrated cheap→expensive delegation with finite-sample guarantees. Both are in LLM classification or monitoring, not generation, so they support the framing without overlapping the domain.

### Cross-cutting matrix (all verified works; "—" means none or not applicable)

| Work (venue, year) | Adapted per input | Signal | Training-free? | Guarantee | Domain |
|---|---|---|---|---|---|
| SFM (NeurIPS 2025) | ODE start time per utterance (+ global α) | coarse hidden states → head | No (trained head) | — | TTS (Matcha, StableTTS, CosyVoice) |
| Chatterbox-Flash (arXiv 2026) | denoising steps per block | prior-calibrated PMI confidence | Rule yes (model fine-tuned) | — | TTS (discrete block diffusion) |
| SPAR-K (arXiv 2026) | none (scheduled layer exit) | fixed schedule | n/c | — | spoken LMs |
| VADUSA (ICASSP 2025) / SSD (Interspeech 2025) | accepted draft tokens | target verification | No (draft heads/model) | — | codec-LM TTS |
| Adaptive Guidance (arXiv 2023) | CFG→cond switch step per sample | cos-sim cond vs uncond | Yes | — | T2I |
| AdaptiveDiffusion (NeurIPS 2024) | skipped steps per trajectory | 3rd-order latent difference | Yes | — | image/video |
| TJS (arXiv 2026) | exit time (fixed γ) | — (x0 decoding) | Yes | — | T2I |
| AdaDiff/DeeDiff (Tang; arXiv) | layers per step | trained uncertainty module | No | — | images |
| AdaDiff (Zhang; AAAI 2025) | steps per prompt | policy net on text | No (RL) | — | image/video |
| T-Stitch (ICLR 2025) | — (fixed small→large switch) | — | Yes | — | images |
| Hybrid SD (arXiv 2024) | — (large→small switch, per-prompt not stated) | — | Inference yes | — | T2I |
| Spec. sampling diffusion (arXiv 2025) | target NFE per sample | accept/reject | One variant yes | exact sampling | diffusion |
| CATImage (TMLR) / RouteT2I (ICCV 2025) | model/steps per prompt | learned quality predictor | No | — | T2I |
| CALM (NeurIPS 2022) | layers per token | softmax / saturation / classifier | 2 of 3 measures yes | **LTT + FST** | text generation |
| Fast yet Safe (NeurIPS 2024) | exit layer (incl. diffusion, BiLD) | exit confidence | Yes (post-hoc) | **CRC / UCB / LTT** | vision, LM, **diffusion** |
| Conformal LM (ICLR 2024) | # samples drawn | calibrated stop/reject | Yes (post-hoc) | conformal set coverage | text generation |

### Gaps
- **No LTT, CRC or conformal method applied to flow-matching TTS, speech generation, or a teacher→student handover was found.** This is the gap the thesis would fill. Absence is based on targeted searches, not an exhaustive survey.
- Whether Conformal Language Modeling uses LTT internally was not confirmed (abstract only).
- Fast yet Safe's appendix (CIFAR diffusion, full per-ε diffusion table) and the remaining 55k characters of its HTML were not read. The exact diffusion compute saving in total NFE-equivalent terms is not given in the main text.
- A "risk-control routing" paper (arXiv 2603.14623, high-probability deviation guarantee) appeared only in a search snippet and was not verified.

---

## Robustness to noisy prompts in zero-shot TTS that adapts computation per input

### Takeaway
Noisy-prompt robustness in zero-shot TTS is handled with **training-time** methods: noise-mixing fine-tuning, data filtering, masked-denoising pre-training, and codec-token denoisers. I found **no work that changes the amount or type of computation per input according to prompt noise or SNR**. Using prompt SNR as a per-utterance compute or handover signal appears unaddressed in the literature I searched.

### Cited Findings
- Wang, Eskimez, Thakker, Yang, Zhu, Tang, Xia, Li, Zhao, Li, Kanda, "An Investigation of Noise Robustness for Flow-Matching-Based Zero-Shot TTS", **Interspeech 2024**, arXiv 2406.05699. It uses training strategies (masked speech-denoising pre-training, multi-speaker detection plus DNSMOS filtering of pre-training data, fine-tuning with random noise mixing). The authors report significant intelligibility, speaker-similarity and quality gains over enhancing the prompt with speech enhancement. **No per-input adaptive computation.** The abstract gives no numbers. — [arXiv abs](https://arxiv.org/abs/2406.05699)
- Lu et al. (Interspeech 2025, arXiv 2505.13830) propose a codec denoiser for LLM-based zero-shot TTS that predicts clean first-group acoustic tokens from noisy-prompt tokens. They report it beats state-of-the-art speech enhancement front-ends and note that LLM TTS tends to copy the prompt's acoustic environment. *Search-result level; not opened.* — [arXiv HTML](https://arxiv.org/html/2505.13830v2)
- Chatterbox-Flash reports **no noisy-prompt evaluation**. Its only harder-input test is EmergentTTS-Eval, which compares schedules with early decoding disabled. — [arXiv HTML v1](https://arxiv.org/html/2605.30748v1)
- Selective CFG for zero-shot TTS schedules guidance by timestep. Its abstract does not tie guidance to prompt quality. — [arXiv abs](https://arxiv.org/abs/2509.19668)

### Inferences
- Using **prompt SNR as an input-level signal to choose a teacher→student handover point** has no direct precedent in the sources found. This supports the novelty of the signal choice. The thesis should still cite the training-based robustness work (Wang et al. 2024) and speech-enhancement-front-end baselines as the conventional alternative.
- Because the LTT guarantee assumes exchangeability, a calibration set with a different SNR mix than deployment would invalidate the bound. Stratifying by SNR, or calibrating on the target noise distribution, is a methodological need the literature does not address for TTS.

### Gaps
- Quantitative noisy-prompt results for Wang et al. (2024) and Lu et al. (2025) were not extracted (abstract or search level only).
- No source found on per-input guidance strength or step budget conditioned on prompt noise. A 2026 aggregator item on speaker/joint-residual CFG reweighting for F5-TTS/CosyVoice2 (emergentmind 2606.25672) was not verified and may or may not include noise experiments.
