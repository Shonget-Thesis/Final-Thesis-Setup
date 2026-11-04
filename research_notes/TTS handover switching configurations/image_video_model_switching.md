# Multi-model / teacher-student switching along one diffusion or flow sampling trajectory (image and video): how the switching point is chosen and evaluated

Scope: images and video only (speech is covered elsewhere). Compiled 2026-11-02 from arXiv HTML/abs pages, ar5iv, CVF open access, NVIDIA Research and GitHub. Every number below was read from the linked primary source unless marked UNVERIFIED. Where the fetch tool returned a paraphrase of the paper, a quote is given only when the source text was returned verbatim. Where a paper has several arXiv versions, numbers come from the version served at the time of fetching (noted where known).

---

## Q1. T-Stitch (Pan et al., arXiv 2402.14167): what is the switching parameter, which fractions were evaluated, and does the best fraction differ by model pair? Is the venue ICLR 2025?

### Takeaway
Verified. T-Stitch runs a small denoiser for the first fraction r1 of the T steps and the large denoiser for the rest. It sweeps r1 from 0% to 100% in 10% steps for every model pair. The fraction that keeps quality differs by pair: about 40% for DiT-S to DiT-XL, best FID at 40% (still better than all-large at 50%) for LDM-S to LDM, and 20–30% for BK-SDM Tiny to SD v1.4. In few-step LCM settings, any small-model share hurts FID. The fraction is fixed per configuration, not chosen per input. The venue is ICLR 2025.

### Cited Findings
- **Venue.** NVIDIA Research lists T-Stitch as a conference paper at ICLR, April 2025. Authors are Zizheng Pan, Bohan Zhuang, De-An Huang, Weili Nie, Zhiding Yu, Chaowei Xiao, Jianfei Cai and Anima Anandkumar. The ICLR 2025 proceedings PDF is headed "Published as a conference paper at ICLR 2025" — [NVIDIA Research](https://research.nvidia.com/labs/lpr/publication/t-stitch2024); [ICLR 2025 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/120339238f293d4ae53a7167403abc4b-Paper-Conference.pdf); [arXiv 2402.14167](https://arxiv.org/abs/2402.14167)
- **Switching parameter.** Quoted: "let D1 take the first ⌊r1T⌉ steps and D2 takes the last ⌊r2T⌉ steps", with r2 = 1 − r1. D1 is the small model, so the small model goes first (Sec. 3) — [arXiv HTML](https://arxiv.org/html/2402.14167)
- **Configurations evaluated** — [arXiv HTML](https://arxiv.org/html/2402.14167):
  - Fractions 0% to 100% in 10% steps (Tables 1, 2, 5, 9).
  - A three-model DiT-S/B/XL grid in 0.1 increments, giving 66 configurations (Figs. 5–6).
  - For LCM: 0/50/100% at 2 steps (Table 10) and 0–100% in 25% steps at 4 steps (Table 11).
  - Samplers: DDIM 100 steps (main DiT setting), DDPM 250, DPM-Solver++ 50, PNDM 50 (SD).
  - Classifier-free guidance (CFG): DiT 1.5, LDM 3.0, SD 7.5.
- **DiT-S to DiT-XL** (DDIM-100; Table 9, FID on 5K images) — [arXiv HTML](https://arxiv.org/html/2402.14167):
  - FID by small-model fraction: 0% 9.20; 10% 9.17; 20% 8.99; 30% 9.03; 40% 8.95; 50% 10.06; 60% 12.46; 100% 33.46.
  - Recall stays 0.74–0.75 up to 60%. Precision falls from 0.81 to 0.76 at 50%.
  - Abstract, quoted: "40% of the early timesteps can be safely replaced with a 10x faster DiT-S without performance drop".
  - Fig. 1 / Table 8: the 50% setting gives FID 10.06 at 1.76× speedup.
- **LDM-S to LDM** (guidance 3.0; Table 1) — [arXiv HTML](https://arxiv.org/html/2402.14167):
  - FID: 20.11 at 0%, 18.60 at 40% (lowest), 19.33 at 50%, 40.92 at 100%.
  - Time falls from 7.1 s to 2.9 s.
  - Inception Score: 199.24 at 0%, 204.01 at 30%, 193.62 at 40%.
- **BK-SDM Tiny to SD v1.4** (Table 2) — [arXiv HTML](https://arxiv.org/html/2402.14167):
  - FID: 13.07 at 0%, 12.29 at 20%, 12.54 at 30%, 13.65 at 40%.
  - CLIP score: 0.2957 at 0%, 0.2910 at 30%, 0.2860 at 40%.
  - Time falls from 3.1 s to 1.9 s.
  - The paper says the early 30% of steps can be taken by BK-SDM Tiny "without a significant performance drop in Inception Score and CLIP Scores while achieving even better FID."
- **DiT-S to U-ViT-H** (DPM-Solver++ 50; Table 5): FID 15.04 at 0%, 12.44 at 20%, 14.19 at 40%, 17.6 at 50% — [arXiv HTML](https://arxiv.org/html/2402.14167)
- **Few-step distilled (LCM) regime.** Adding the small model only hurts here — [arXiv HTML](https://arxiv.org/html/2402.14167):
  - LCM-SSD-1B to LCM-SDXL at 2 steps (Table 10): FID 21.98 at 0%, 24.36 at 100%; time 768 ms to 634 ms.
  - At 4 steps (Table 11): FID 17.05 at 0%, 21.28 at 50%, 25.50 at 100%; time 1,029 ms to 823 ms.
- **Order matters** (Table 7, DiT-S/XL at 50%) — [arXiv HTML](https://arxiv.org/html/2402.14167):
  - FID: small-to-large 10.06 (default), large-to-small 27.61, interleave 19.02, "decreasing probability" 12.94.
- **How the fraction is selected.** It is chosen offline against a compute budget, not per input — [arXiv HTML](https://arxiv.org/html/2402.14167):
  - Quoted: "given a compute budget such as time cost, we can efficiently find a few configurations that satisfy this constraint via a pre-computed lookup table" (Sec. 3.2).
  - Multi-model combinations "effectively obtain a smooth Pareto Frontier for both FID and Inception Score" (Sec. 4.1).
  - The optimal trade-off guidelines are left "for future work". There is no per-sample switching.
- **Stated limitations.** "T-Stitch requires a smaller model that has been trained on the same data distribution as the large model". It also has extra memory cost, and "the speedup gain is bounded by the efficiency of the small model" (Sec. 5) — [arXiv HTML](https://arxiv.org/html/2402.14167)

### Inferences
- The prior belief ("about 40% DiT, 50% LDM, 20–30% SD") is mostly right. The LDM figure is better stated as "lowest FID at 40%, still better than all-large at 50%". The SD figure comes from BK-SDM Tiny with SD v1.4: lowest FID at 20%, and "safe" up to 30% by the authors' wording.
- The best fraction depends on the model pair, the sampler and the step count. When the large model is already a few-step distilled model (LCM), the speed-for-quality trade disappears: every small-model share raises FID. This is direct cross-domain evidence that the switch point depends on the setting.
- The 27.61 vs 10.06 FID gap between large-to-small and small-to-large ordering shows that the direction of the handoff matters as much as where it happens.

### Gaps
- The numbers above come from the arXiv HTML. The served page identified as 2402.14167v1, so this should be confirmed. I did not cross-check them line by line against the ICLR 2025 camera-ready PDF, which may differ. UNVERIFIED whether the camera-ready tables match.
- The intro reports DiT-S at 1.7 s, while Table 3 gives 1.6 s per 8 images. This is a minor internal inconsistency in the paper.

---

## Q2. Distilling Diversity and Control in Diffusion Models (Gandikota & Bau, arXiv 2503.10637): base model for the first k steps, then the distilled model. Is k varied (Fig. 7), and is k=1 the pick? Is the venue WACV 2026?

### Takeaway
Verified. The paper runs the base model for the first k timestep(s) and the distilled model for the rest. Figure 7(b) varies k, and the authors pick k=1. With k=1, SDXL-Base followed by SDXL-DMD2 at 4 steps restores diversity (DreamSim 0.350 vs base 0.337 vs DMD2 0.264) and improves COCO-30k FID (10.79 vs base 12.74 vs DMD2 15.52) at the distilled model's latency (0.64 s vs 9.22 s). k is fixed for all prompts, and the authors list per-prompt adaptive transition points as future work. The venue is WACV 2026.

### Cited Findings
- **Venue.** The paper appears in the WACV 2026 proceedings on CVF Open Access (Gandikota and Bau, Northeastern). Search-result metadata gives pages 1304–1313 — [CVF WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Gandikota_Distilling_Diversity_and_Control_in_Diffusion_Models_WACV_2026_paper.html); [WACV 2026 poster page](https://wacv.thecvf.com/virtual/2026/poster/883); [project page](https://distillation.baulab.info/)
- **Method (Algorithm 1).** For timesteps t > T−k the base model denoises, and the distilled model does the rest. The paper describes using the base model "for the critical first timestep(s)" — [arXiv HTML v4](https://arxiv.org/html/2503.10637)
- **Models.** The main experiments use SDXL-Base with SDXL-DMD2. Appendix D also tests PixArt-Alpha with PixArt-Delta, and SD 2.1 with SD-Turbo. SDXL-Turbo, SDXL-Lightning and LCM appear only in the control-transfer part (Sec. 3) — [arXiv HTML](https://arxiv.org/html/2503.10637)
- **Step counts.** Base uses 50 steps; distilled and hybrid use 4 steps; the skip-first-step ablation uses 3 steps (Table 3) — [arXiv HTML](https://arxiv.org/html/2503.10637)
- **Figure 7 panels:** (a) varies the base model's guidance scale, (b) varies k, and (c) varies the base model's total timesteps. The text says "using the base model for just the first timestep (k=1) provides substantial diversity gains with minimal computational overhead". The Fig. 7(c) caption says "replacing 1-1 timesteps of distilled with base is most ideal" — [arXiv HTML](https://arxiv.org/html/2503.10637)
- **Diversity** (Table 2, DreamSim mean pairwise distance, 100 seeds per prompt; base / distilled / hybrid) — [arXiv HTML](https://arxiv.org/html/2503.10637):

  | Prompt | Base | Distilled | Hybrid |
  |---|---|---|---|
  | Sunset beach | 0.396 | 0.271 | 0.373 |
  | Cute puppy | 0.233 | 0.199 | 0.265 |
  | Futuristic city | 0.237 | 0.198 | 0.283 |
  | Person | 0.484 | 0.347 | 0.461 |
  | Van Gogh art | 0.337 | 0.305 | 0.366 |
  | **Average** | **0.337** | **0.264** | **0.350** |

- **Quality and speed** (Table 3, COCO-30k) — [arXiv HTML](https://arxiv.org/html/2503.10637):

  | Method | Steps | FID | IS | CLIP | Time |
  |---|---|---|---|---|---|
  | Base | 50 | 12.74 | 24.74 | 31.83 | 9.22 s |
  | DMD2 | 4 | 15.52 | 27.20 | 31.69 | 0.64 s |
  | Hybrid (k=1) | 4 | 10.79 | 26.13 | 32.12 | 0.64 s |
  | Skip-first-timestep | 3 | 10.12 | 24.69 | 31.71 | 0.53 s |

- **Where the base step goes** (Table E.1, "image of a car", 100 samples; DreamSim diversity) — [arXiv HTML](https://arxiv.org/html/2503.10637):
  - Replacing the first timestep with the base model gives 0.350 at 0.64 s.
  - Replacing the final timesteps gives 0.251 at 6.61 s.
  - SDXL-Base alone gives 0.357; DMD2 alone gives 0.264.
- **Fixed for all prompts.** Limitations, quoted: "our approach treats all prompts uniformly". Adaptive transition points are proposed as future work. Other stated limitations: both models must be loaded in memory; diversity is visual (DreamSim) only; the FID gain over base is a hypothesis about guidance in the final steps — [arXiv HTML](https://arxiv.org/html/2503.10637)

### Inferences
- This is the clearest image-domain example of teacher-first-then-student handoff. A single teacher step decides the diversity and composition of the sample. Placing the teacher at the end instead (Table E.1) gives no diversity benefit.
- The switch point was picked once for the model pair, from Fig. 7(b), and applied to every prompt. The authors themselves name per-prompt adaptivity as an open problem. This supports the thesis claim that switch points are usually fixed rather than chosen per input.

### Gaps
- The text available to me gives no numeric list of the k values plotted in Fig. 7(b), and no per-k metric values. The exact sweep range is UNVERIFIED and would need the figure itself.
- Inconsistency in the paper: the Fig. 7(b) caption says "Running distilled model from first timestep (k=1)", while the body text says the base model takes the first timestep at k=1.
- I did not check whether the WACV camera-ready changes any numbers relative to arXiv v4.

---

## Q3. OMS-DPM (Liu et al., ICML 2023): model-schedule search under a time budget

### Takeaway
Verified as ICML 2023. OMS-DPM treats the "model schedule" (which pretrained model from a zoo runs at each step, plus the step count) as a search space. A predictor-guided search finds the schedule once per dataset and time budget. The best placement of small and large models differs by dataset: large models near the data end help on CIFAR-10 and ImageNet-64 under DDIM, and the opposite holds on LSUN-Church. It also differs by budget. There is no per-input schedule.

### Cited Findings
- **Venue and authors.** "Accepted by ICML2023". Authors are Enshu Liu, Xuefei Ning, Zinan Lin, Huazhong Yang and Yu Wang — [arXiv 2306.08860](https://arxiv.org/abs/2306.08860)
- **Abstract claims.** The paper calls the "model schedule" an overlooked dimension. It reports that "small models, though having worse generation quality when used alone, could outperform large models" at certain steps, and that the method speeds up Stable Diffusion sampling 2× while keeping quality — [arXiv abs](https://arxiv.org/abs/2306.08860)
- **Definition.** The schedule is "the model assignments to each of the denoising steps". The number of sampling steps is also searched, and a "null model" can skip a timestep (Sec. 1, 4.1) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08860)
- **Model zoos.** Sizes are 6, 6, 7 and 6 models for CIFAR-10, CelebA, ImageNet-64 and LSUN-Church (App. B.1). Stable Diffusion uses the four official checkpoints with DPM-Solver (Sec. 5.2). Both DPM-Solver and DDIM are evaluated — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08860)
- **Results** — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08860):
  - CIFAR-10 with DPM-Solver (Table 1a): at a 7.0×10³ ms budget, single-model baseline 3.56 vs OMS-DPM 3.25±0.01; at 2.5×10³ ms, 3.93 vs 3.19±0.05.
  - The paper summarises this as "better FID (3.19 v.s. 3.56)" at 2.8× speedup.
  - At a low budget: "6.08 v.s. 8.73 on CIFAR-10".
  - Stable Diffusion on MS-COCO 256 (Table 3): OMS-DPM at 12 NFE gets FID 11.34. The best single model gets 12.11 at 12 NFE and 11.81 at 24 NFE.
- **Placement depends on the dataset.** "using large models at steps near the generated data is more likely to achieve better performances" holds on ImageNet-64 and CIFAR-10 (DDIM). On LSUN-Church "the case is just the opposite". Optimal hyperparameters "are different for different datasets" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08860)
- **Placement depends on the budget.** Under low budgets "using smaller models with more steps is more likely to gain a better generation quality", and the third-order solver "is only used under an adequate budget" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08860)
- **Motivating experiment** (Sec. 3). The authors "randomly pick models in each of the denoising steps". One random mix reaches FID about 5.4 and beats the best single model on both quality and speed. They conclude "smaller models can actually outperform larger models at a wide range of denoising steps" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08860)
- **Search scope.** The search runs once per dataset and budget ("for any given platform or budget"). No per-input adaptation is described — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08860)
- **Limitations.** A new dataset or task needs new predictor training data, which "incurs a substantial overhead". Also, "the quality and size of the model zoos matter for the performances" — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08860)

### Inferences
- OMS-DPM is the strongest evidence that the right handoff depends on the setting. The same search over the same kind of zoo puts large models late on CIFAR-10 and ImageNet-64 but early on LSUN-Church, and the answer shifts with the time budget.
- Because the schedule is fixed per dataset and budget, OMS-DPM supports the claim that switch points are chosen "once per model/dataset" rather than per input.

### Gaps
- No CLIP scores were reported for the Stable Diffusion schedule in the text I retrieved.
- I did not extract the exact searched schedules (which model at which step) for each dataset. Only the qualitative trends above are verified.

---

## Q4. Other works: expert denoisers by noise level, step-aware models, big-little cascades, distillation hybrids, SDEdit, Restart

### Takeaway
Two families exist across image and video:
- **Fixed noise-level partitions trained into the model:** eDiff-I, ERNIE-ViLG 2.0 and Wan2.2's high-/low-noise experts.
- **Inference-time handoffs between independently trained models:** Hybrid SD, T-Stitch, OMS-DPM, DDSM and TRACK, plus SRDiffusion, the only inference-time handoff here that is adaptive per input.

The 2026 "diversity distillation" papers (CrossDistill, DP-DMD, 1.x-Distill) move the early/late split into training as a fixed boundary rather than an inference-time teacher handoff. In all of them the boundary is a single fixed hyperparameter, sometimes swept over a small grid.

### Cited Findings

**A. Expert denoisers by noise level (fixed partition, trained in)**
- **eDiff-I** (Balaji et al., arXiv 2211.01324, 2022; no peer-reviewed venue checked) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2211.01324):
  - Experts are created by splitting the training noise distribution, "splitting p(σ) equally by area". The final model has a high-noise expert, a low-noise expert and one expert "for learning all intermediate noise intervals".
  - Table 1 (zero-shot COCO FID-30K at 256²): Config-A 7.35, Config-B 7.26, Config-C 7.11, Config-D 6.95. The text says Config-C uses two base experts and Config-D three.
  - Figure 7: the expert ensemble "consistently outperforms the baseline model on the entire FID-CLIP score trade-off curve". No numbers are given in the text.
  - The splits are fixed at training time.
- **ERNIE-ViLG 2.0** (Feng et al., CVPR 2023) — [CVF CVPR 2023 PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Feng_ERNIE-ViLG_2.0_Improving_Text-to-Image_Diffusion_Model_With_Knowledge-Enhanced_Mixture-of-Denoising-Experts_CVPR_2023_paper.pdf); [arXiv 2210.15257](https://arxiv.org/abs/2210.15257):
  - Denoising steps are split into stages, each with its own expert. The final model has 10 experts (24B parameters), with zero-shot COCO FID-30k 7.23, or 6.75 with reranking.
  - An appendix ablation (Fig. 11) compares 1, 2, 5 and 10 experts at equal training samples.
  - Verified only from search-result excerpts of the primary PDF. Per-expert-count FIDs were not extracted.
- **Wan2.2 A14B (video)** — [Wan2.2 README](https://github.com/Wan-Video/Wan2.2); [wan_t2v_A14B.py config](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_t2v_A14B.py):
  - Two 14B experts (27B total, 14B active per step): "a high-noise expert for the early stages, focusing on overall layout" and a low-noise expert for refinement.
  - The README defines the switch step t_moe as the step whose SNR is half of SNR_min. The model switches to the low-noise expert when t < t_moe.
  - The T2V-A14B config sets `boundary = 0.875`, 40 sample steps, shift 12.0, and different guidance per expert: (3.0, 4.0) for (low-noise, high-noise).
  - The README compares validation loss for Wan2.1 alone, each mixed variant, and the full MoE (the MoE "achieves the lowest validation loss"), but only as images.
  - The boundary is a fixed constant per model config.

**B. Step-aware model sizes (searched once per dataset)**
- **DDSM** (Yang et al., ICLR 2024 per arXiv comments, 2310.03337) — [arXiv abs](https://arxiv.org/abs/2310.03337):
  - Uses "a spectrum of neural networks whose sizes are adapted according to the importance of each generative step", with the per-step sizes found by evolutionary search.
  - Reported compute savings: CIFAR-10 49%, CelebA-HQ 61%, LSUN-bedroom 59%, AFHQ 71%, ImageNet 76%, "without compromising the generation quality".

**C. Big–little cascades and switching along the trajectory**
- **Hybrid SD** (Yan et al., arXiv 2408.06646; no peer-reviewed venue shown) — [arXiv HTML](https://arxiv.org/html/2408.06646):
  - Large model first, then small, as an edge-cloud split. Quoted: "For steps where t>k, θ_large is used for denoising, otherwise, θ_small is employed".
  - Only k=10 and k=5 of 25 DPM-Solver steps were evaluated for SD-v1.4, and k=4 of 8 for LCM (Table 1, COCO-30K).

  | Configuration | FID | IS | CLIP |
  |---|---|---|---|
  | SD-v1.4 only | 12.22 | 37.63 | 0.2993 |
  | + OursTiny, k=10 | 13.75 | 34.49 | 0.2887 |
  | + OursTiny, k=5 | 15.39 | 31.50 | 0.2734 |
  | + Small, k=10 | 14.29 | 36.67 | 0.2921 |
  | + Small, k=5 | 15.48 | 34.02 | 0.2805 |
  | + Tiny, k=10 | 14.59 | 35.70 | 0.2909 |
  | + Tiny, k=5 | 15.92 | 32.66 | 0.2780 |
  | SD-v1.4 LCM only | 16.31 | 37.24 | 0.2825 |
  | OursTiny LCM only | 23.42 | 25.56 | 0.2309 |
  | Hybrid LCM, k=4 | 16.19 | 31.76 | 0.2698 |

  - Cost: 66% lower cloud cost (5.4 T vs 15.8 T FLOPs, Table 3).
  - k is a preset, tunable constant; "reducing k can be a trade-off strategy". There is no rule for choosing it and no per-prompt choice.
- **TRACK** (Munir et al., NVIDIA, arXiv 2609.30096, Sep 2026; video) — [arXiv HTML](https://arxiv.org/html/2609.30096):
  - An offline, per-step routing policy. It scores large–small disagreement on calibration prompts and routes the K lowest-disagreement steps to the small model. Schedules can be non-contiguous. The policy is "prompt-independent".
  - K is set as the largest number of small-model steps that keeps quality comparable to all-large. K=28 on Cosmos 3 and K=40 on Wan 2.1 "degrade quality severely".
  - Table 1 results:
    - Wan 2.1 14B/1.3B: 30/50 steps switched, 1.95×.
    - Cosmos 3 Super/Nano: 24/35, 2.04×. Cosmos 3 Super/Edge: 24/35, 2.73×.
    - TurboDiffusion: 3/4, 2.69×. FastVideo: 2/3, 2.17×.
    - VBench sub-scores stay within about ±0.6 of all-large. Example: Wan subject consistency 94.62 to 94.81.
  - Equal-budget comparison on Wan 2.1, K=30/50 (Table 2, DreamSim retention): TRACK Top-K 98.3%, Last-K 97.8%, First-K 84.6%, Random-K 85.7%, All-Small 79.4%.
  - Energy savings 45.6–59.7% (App. C).
  - Inconsistency in the paper: TurboDiffusion is listed as 3/4 switched in Table 1 but 2/4 in the App. C table.
- **SRDiffusion** (Cheng et al., arXiv 2505.19151, May 2025; video; no peer-reviewed venue listed) — [arXiv HTML](https://arxiv.org/html/2505.19151):
  - Large model "sketches" first, then the small model "renders". The switch is per sample at runtime: it fires when the second-order change signal D_t^deriv falls below a threshold δ (and is above 0), after a minimum FIX_STEP = 5.
  - Thresholds evaluated:
    - Wan 14B to 1.3B: δ ∈ {0.002, 0.01, 0.03} gives VBench 83.87 / 84.06 / 84.12 at 2.23× / 2.84× / 3.21×, vs 84.05 for 14B alone (Tables 2, 6).
    - CogVideoX 5B to 2B: δ ∈ {0.008, 0.01, 0.015, 0.03} gives 80.83 / 80.85 / 80.51 / 80.12 at 1.65× / 1.82× / 2.05× / 2.32×, vs 80.89 (Tables 2, 7).
  - At δ=0.01 the average switch step is about 10 for Wan and about 15 for CogVideoX.
  - Adaptive vs a fixed switch step (Table 4):

    | Model | Setting | VBench | LPIPS | PSNR | SSIM |
    |---|---|---|---|---|---|
    | Wan | fixed T=10 | 84.05 | 0.196 | 20.53 | 0.732 |
    | Wan | adaptive δ=0.01 | 84.06 | 0.197 | 20.53 | 0.734 |
    | CogVideoX | fixed T=15 | 80.85 | 0.182 | 22.75 | 0.789 |
    | CogVideoX | adaptive δ=0.01 | 80.85 | 0.177 | 22.93 | 0.795 |

    So per-sample adaptivity gives only marginal average gains. The paper reports lower PSNR variance (Fig. 6) and later switching on hard prompts.
  - Limitation: it works only within one model family sharing a VAE.

**D. 2026 "first steps for diversity" distillation papers (training-time boundary, not inference handoff)**
- **CrossDistill** (Liu, Li et al., arXiv 2609.14725, Sep 2026; video) — [arXiv HTML](https://arxiv.org/html/2609.14725):
  - Splits the flow-matching time axis at a crossover τ⋆. The high-noise interval [τ⋆, 1] gets a trajectory-preserving objective (PCM-style); the low-noise interval [0, τ⋆] gets distribution matching (DMD).
  - Grid evaluated: τ⋆ ∈ {0.90, 0.92, 0.94, 0.96} on Wan2.1-T2V-1.3B (Fig. 4; no per-τ numbers in the text). Default τ⋆ = 0.94 for both 1.3B and 14B (Table 4), described as giving "the best quality–diversity balance".
  - τ⋆ is fixed, described as "a stable scheduling prior". No inference-time teacher-then-student baseline is reported.
- **DP-DMD** (Wu, Li, Zhang, Ma, arXiv 2602.03139, Feb 2026; images) — [arXiv HTML](https://arxiv.org/html/2602.03139):
  - Only the first student step gets a teacher-derived target-prediction (diversity) loss; the remaining steps use standard DMD.
  - Ablated: the teacher anchor step K (Table 3, SD3.5-M, Pick-a-Pic, DINO/CLIP diversity):

    | Setting | DINO | CLIP |
    |---|---|---|
    | DMD | 0.137 | 0.133 |
    | K=1 | 0.157 | 0.155 |
    | K=3 | 0.175 | 0.178 |
    | K=5 (default) | 0.179 | 0.182 |
    | K=10 | 0.187 | 0.185 |
    | K=30 | 0.187 | 0.181 |

  - Teacher at 60 NFE (Table 1): 0.240 / 0.221.
  - Fixed: the method "uses a fixed anchor step and loss weight". Future work: choose them "according to timestep, prompt complexity, teacher uncertainty, or student capacity".
  - It does not compare against the Gandikota & Bau inference hybrid.
- **1.x-Distill** (Li et al., arXiv 2604.04018, Apr 2026; SD3-Medium, SD3.5-Large) — [arXiv HTML](https://arxiv.org/html/2604.04018):
  - "1.x NFE" comes from a 2-step student whose second step reuses cached DiT block contributions: 1.67 / 1.75 effective NFE on SD3-M, 1.74 on SD3.5-L, and up to 33× speedup vs 28×2 NFE.
  - No teacher is used at inference.
  - A training-time boundary α (teacher CFG disabled on one side) is swept in Fig. 9. "When α<0.92, the distilled model exhibits clear degradation"; α = 0.94 is chosen.
  - Inconsistency in the paper: the caption and Eq. 4 disagree on which side of α has CFG disabled.

**E. Noise-level handoff analogues (not model switching)**
- **SDEdit** (Meng et al., arXiv 2108.01073) — [ar5iv](https://ar5iv.labs.arxiv.org/html/2108.01073):
  - t0 is the noise level at which the reverse SDE starts from a noised guide. Larger t0 gives more realism and less faithfulness.
  - Sweep: t0 = 0.01, then 0.1 to 1.0 in steps of 0.1 (LSUN church; Fig. 3, App. D.2), plotted as KID and L2. No numeric table.
  - Recommendation: "For reasonable guides, we find that t0∈[0.3,0.6] works well."
  - Selection: binary-search t0 on one randomly chosen image, then fix it for the whole task ("the shared t0 works well for all reasonable guides in the same task").
  - Settings used: 0.5 for stroke-based synthesis; 0.5 (VP) and 0.45 (VE) for stroke-based editing; 0.35 (VE) for compositing.
- **Restart sampling** (Xu et al., arXiv 2306.14878) — [arXiv abs](https://arxiv.org/abs/2306.14878):
  - Alternates forward noise injection with backward ODE steps. Reports 10× speedup on CIFAR-10 and 2× on ImageNet-64 vs prior best SDE results, and a better text–image alignment/quality vs diversity balance on Stable Diffusion.
  - The restart interval [t_min, t_max] and how it is chosen were not on the page I checked.

### Inferences
- Handoff direction depends on the goal and the setting:
  - Small-first is right for T-Stitch's FID and speed trade on DiT/LDM/SD.
  - Large- or teacher-first is right for diversity and layout: Gandikota & Bau, Hybrid SD, SRDiffusion, Wan2.2's high-noise expert, and CrossDistill/DP-DMD's early-step diversity objectives.
  - In TRACK's video experiment, putting the small model on the first K steps keeps far less diversity (84.6%) than putting it on the last K (97.8%).
  - OMS-DPM finds the preferred placement flips between datasets.
- Boundaries cluster near the high-noise end in recent flow models: Wan2.2 T2V boundary 0.875, CrossDistill τ⋆ = 0.94, 1.x-Distill α = 0.94. Each is a single constant per model, chosen from a small sweep or defined analytically (Wan's SNR rule). Note that CrossDistill and 1.x-Distill share authors (Haoyu Li, Kai Zhang), so their identical 0.94 may not be independent evidence.
- Many switch values that are actually deployed were tested on only a few settings: Hybrid SD k ∈ {5, 10}; CrossDistill four τ⋆ values; SRDiffusion three or four δ values; Gandikota & Bau one Fig. 7(b) sweep. This fits the thesis claim that switch points are often lightly tested.

### Gaps
- The following were not checked and are UNVERIFIED (no claims made): MEME (multi-architecture multi-expert diffusion), Phased DMD, MoDM / DiffServe (per-query model routing at serving time), DMD/DMD2 follow-ups beyond the three listed, and the Wan2.2 I2V boundary value (only the T2V config was read).
- eDiff-I: I could not confirm from the retrieved text what Config-A and Config-B consist of. The paper also has an FID inconsistency (7.04 in the text vs 6.95 in Table 1 for Config-D).
- ERNIE-ViLG 2.0: FID per number of experts (Fig. 11) was not extracted.
- DDSM: not verified whether the searched step-wise schedules differ by dataset.
- Venue for SDEdit (commonly cited as ICLR 2022) and Restart (commonly cited as NeurIPS 2023) is not confirmed from the pages fetched; both are UNVERIFIED.
- "Denoising, Fast and Slow" (arXiv 2604.19141; listed on CVF as CVPR 2026) adapts per sample via patch-level difficulty, not model switching. Details are UNVERIFIED and seen only in search results — [arXiv HTML](https://arxiv.org/html/2604.19141).

---

## Q5. Which works choose the switch per input, and do any give guarantees?

### Takeaway
Of the inference-time model handoffs reviewed, only SRDiffusion chooses the switch per sample, using a runtime latent-change threshold. Its own ablation shows almost no average gain over a well-chosen fixed step. Speculative sampling for diffusion (De Bortoli et al.) decides draft acceptance per sample and per step, and is the only work found with a formal guarantee: exact samples from the target model (Prop. 3.1). Every other work fixes the switch per model, dataset or budget. Gandikota & Bau and DP-DMD explicitly leave per-prompt adaptivity to future work.

### Cited Findings
- **SRDiffusion** is per-sample and online. "During runtime, we record the second-order derivative of predicted noise", and the switch fires on a threshold δ. The introduction claims a "guaranteed speed-quality trade-off", but no formal guarantee or proof is given; evidence is empirical (VBench, LPIPS, PSNR, SSIM) — [arXiv HTML](https://arxiv.org/html/2505.19151)
- **Speculative sampling for diffusion** (De Bortoli, Galashov, Gretton, Doucet, arXiv 2501.05370) — [arXiv HTML](https://arxiv.org/html/2501.05370):
  - A cheap drafter proposes L steps, and the target model checks them in parallel. Each draft is accepted with probability min(1, q/p); on rejection, reflection maximal coupling is used.
  - Proposition 3.1: the algorithm "outputs Y∼q" (exact target distribution).
  - Image results on CIFAR-10 and LSUN 64×64 (pixel U-Nets). CIFAR-10: "FID score of 2.34 with only 35 calls to the target model", vs 2.45 with 100 calls for classical sampling. The abstract claims "halving the number of function evaluations".
  - The window L is fixed; acceptance varies per sample.
- **TRACK** is offline and "prompt-independent". Its "discrete optimal solution" result holds only for a surrogate divergence objective, not as a quality guarantee — [arXiv HTML](https://arxiv.org/html/2609.30096)
- **Fixed, not per-input, by the authors' own description:**
  - T-Stitch: lookup table per compute budget — [arXiv HTML](https://arxiv.org/html/2402.14167)
  - Gandikota & Bau: "treats all prompts uniformly" — [arXiv HTML](https://arxiv.org/html/2503.10637)
  - OMS-DPM: per dataset and budget — [ar5iv](https://ar5iv.labs.arxiv.org/html/2306.08860)
  - Hybrid SD: preset k — [arXiv HTML](https://arxiv.org/html/2408.06646)
  - CrossDistill: "stable scheduling prior" — [arXiv HTML](https://arxiv.org/html/2609.14725)
  - DP-DMD: "fixed anchor step and loss weight" — [arXiv HTML](https://arxiv.org/html/2602.03139)
  - Wan2.2: config constant `boundary = 0.875` — [config](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_t2v_A14B.py)
  - SDEdit: t0 binary-searched on one image, then shared across the task — [ar5iv](https://ar5iv.labs.arxiv.org/html/2108.01073)

**Summary table** (all entries from the sources linked above):

| Work (venue/year) | Domain | Switch parameter | Values evaluated | Fixed or per input | Guarantee |
|---|---|---|---|---|---|
| [T-Stitch](https://arxiv.org/html/2402.14167) (ICLR 2025) | Image | small-model fraction r1 (small first) | 0–100% in 10% steps; 66-config 3-model grid; LCM 0/25/50/75/100% | Fixed per model pair/budget (lookup table) | None |
| [Gandikota & Bau](https://arxiv.org/html/2503.10637) (WACV 2026) | Image | k base-model steps first, then distilled (4 total) | Fig. 7(b) sweep, exact values not given; k=1 chosen | Fixed for all prompts | None |
| [OMS-DPM](https://ar5iv.labs.arxiv.org/html/2306.08860) (ICML 2023) | Image | model per step + step count | Predictor-based search per dataset × budget | Fixed per dataset/budget | None |
| [DDSM](https://arxiv.org/abs/2310.03337) (ICLR 2024) | Image | network size per step | Evolutionary search | Fixed per dataset (search) | None |
| [eDiff-I](https://ar5iv.labs.arxiv.org/html/2211.01324) (arXiv 2022) | Image | noise intervals per expert (split p(σ) equally by area) | Configs A–D (2 or 3 experts) | Fixed at training | None |
| [ERNIE-ViLG 2.0](https://arxiv.org/abs/2210.15257) (CVPR 2023) | Image | stage per expert | 1/2/5/10 experts | Fixed at training | None |
| [Wan2.2](https://github.com/Wan-Video/Wan2.2) (2025 release) | Video | high-/low-noise expert boundary (SNR rule; T2V boundary 0.875) | Validation-loss comparison only | Fixed per model config | None |
| [Hybrid SD](https://arxiv.org/html/2408.06646) (arXiv 2024) | Image | k large-model steps first | k ∈ {5, 10} of 25; k=4 of 8 (LCM) | Fixed preset | None |
| [TRACK](https://arxiv.org/html/2609.30096) (arXiv 2026) | Video | K steps routed to small model (non-contiguous) | Per pipeline, e.g. 30/50, 24/35, 3/4, 2/3 | Fixed offline, prompt-independent | Optimality for surrogate only |
| [SRDiffusion](https://arxiv.org/html/2505.19151) (arXiv 2025) | Video | latent-change threshold δ (large first) | δ ∈ {0.002, 0.01, 0.03} Wan; {0.008, 0.01, 0.015, 0.03} CogVideoX; fixed T=10/15 baselines | Per sample (runtime) | Claimed "guaranteed" trade-off; no proof |
| [Speculative sampling](https://arxiv.org/html/2501.05370) (arXiv 2025) | Image (+ robotics) | draft window L; per-step accept/reject | Several drafters and L values | Per sample (acceptance) | Exact target samples (Prop. 3.1) |
| [CrossDistill](https://arxiv.org/html/2609.14725) (arXiv 2026) | Video | crossover τ⋆ between training objectives | {0.90, 0.92, 0.94, 0.96} | Fixed | None (appendix "not a formal theory") |
| [DP-DMD](https://arxiv.org/html/2602.03139) (arXiv 2026) | Image | first student step gets teacher target; anchor K | K ∈ {1, 3, 5, 10, 30} | Fixed | None |
| [1.x-Distill](https://arxiv.org/html/2604.04018) (arXiv 2026) | Image | training CFG boundary α | Fig. 9 sweep; α=0.94 | Fixed | None |
| [SDEdit](https://ar5iv.labs.arxiv.org/html/2108.01073) (arXiv 2021) | Image | start noise level t0 | 0.01, 0.1…1.0 | Fixed per task (binary search on one image) | Bound on deviation from the guide only (Prop. 1) |

### Inferences
- Across image and video, the handoff point is almost always a single constant, chosen once per model pair, dataset, budget or task. This supports the thesis claim that per-input switching is rare.
- The one adaptive model-handoff work (SRDiffusion) found per-sample switching roughly equal to a tuned fixed step on average. Its claimed benefits are lower variance and later switching on hard prompts. So per-input switching is not automatically better, and any thesis claim of advantage would need direct evidence.
- Formal guarantees exist only where the method verifies against the target model at each step (speculative sampling). That gives exactness but requires target-model evaluations. None of the cheap single-handoff schemes (T-Stitch, Gandikota & Bau, Hybrid SD, TRACK, SRDiffusion) give a quality guarantee.

### Gaps
- I did not verify whether the speculative-sampling paper was published at ICML 2025. The arXiv page only lists "Machine Learning, ICML" as keywords, so the venue is UNVERIFIED. The LSUN numbers are in App. J, which I did not read.
- Apart from SRDiffusion and speculative sampling, I found no peer-reviewed or preprint image/video work that picks a teacher-to-student switch point per input. This absence comes from targeted searches, not an exhaustive survey.
