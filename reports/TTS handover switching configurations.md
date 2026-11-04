# TTS teacher handovers remain fixed and untested

DMOSpeech 2 (AAAI 2026) is the only speech-generation work found that runs a guided teacher (F5-TTS with classifier-free guidance) for the early part of one sampling trajectory and hands the rest to a distilled student. It evaluates exactly one switching time: **t_switch = 0.25, with 14 teacher steps and 2 student steps**. That one configuration matches the 4-step student's intelligibility (Seed-TTS English WER **1.738 vs 1.752**) and recovers **89.1% of the teacher's pitch diversity**, at about **three times the student's real-time factor**. Its suggested ranges (0.1–0.2 for efficiency, 0.4–0.5 for diversity) are untested. The released code defaults to a different stopping time (0.07) and uses a different handover mechanism. No citing paper, GitHub issue or fork evaluates any other value. Nothing in the published record justifies 0.25 beyond the authors' choice.

Across speech, image and video, the handover point is almost always **one constant per model, dataset or compute budget**. Four works adapt per input, and none of them bounds quality loss:

- **Shallow Flow Matching** uses a trained head to set each utterance's start time.
- **SRDiffusion** uses a runtime threshold to switch video models.
- **Adaptive Guidance** makes a training-free switch from guided to cheaper steps.
- **Chatterbox-Flash** decides per block when to stop.

Finite-sample guarantees on adaptive computation do exist. CALM calibrates early exit with Learn-Then-Test, and Fast yet Safe extends risk control to early-exit diffusion and to a big/little model pair. Neither has been applied to flow-matching or speech generation.

The thesis's gap is real but narrow. It lies in the combination of three things: a training-free, per-utterance teacher-to-student handover time; speech-specific signals; and an LTT bound. It does not lie in any one of them alone. One result should shape how the benefit is framed. SRDiffusion's per-sample switch only matched a tuned fixed switch on average quality. The defensible claim is therefore calibrated compute savings with controlled risk, not higher average quality.

## Summary table: one TTS teacher-to-student switch among many fixed handovers

The three panels below cover speech and audio, image and video, and adaptive computation with or without guarantees. Markers:

- **†**: checked at abstract or search-result level only.
- **‡**: venue not confirmed from a primary source.
- **§**: a number derived from code or arithmetic rather than stated by the authors.

Some numbers in panel A were extracted from arXiv/ar5iv HTML by a fetch-and-summarise tool and need re-checking against the PDF before they enter the manuscript: DiffSinger, DiffGAN-TTS, SFM, BBED, CRP and CTFSE. The DOSE and Fast Grad-TTS PDFs were read directly. Panel B numbers come from arXiv HTML versions (T-Stitch v1, Gandikota & Bau v4) and have not been cross-checked against camera-ready PDFs.

### A. Speech and audio

| Paper | Venue, year | Domain | Handover parameter | Configurations evaluated | Key findings | Per-input or fixed | Guarantee |
|---|---|---|---|---|---|---|---|
| [DMOSpeech 2](https://arxiv.org/html/2507.14988v1) | AAAI 2026 | Zero-shot TTS | t_switch: F5-TTS teacher (CFG) for K steps, DMD student for M steps | One: t_switch = 0.25, 14 + 2 = 16 steps | Seed-TTS en WER 1.738 (student 1.752, teacher 1.947). CV_f0 0.5932, which is 89.1% of the teacher's 0.6659 (student 0.4640). RTF 0.0941 (student 0.0316, teacher 0.1671). zh naturalness CMOS +0.45 over the student, marked significant | Fixed, global | None |
| [DiffSinger / DiffSpeech](https://ar5iv.labs.arxiv.org/html/2105.02446) | AAAI 2022 (arXiv comment) | Singing synthesis, TTS | Shallow step k: the auxiliary decoder's mel is diffused to step k, then k reverse steps run | k = 54, 25, none (PopCS, T = 100); k = 70 (LJSpeech) | No shallow diffusion: −0.500 CMOS. k = 25: −0.053 CMOS vs k = 54. RTF −45.1% (SVS), −29.2% (TTS) | Fixed per dataset: a boundary predictor averages per-mel k′ | None |
| [DiffGAN-TTS](https://ar5iv.labs.arxiv.org/html/2201.11972) | Preprint; ICML 2022 workshop talk | TTS (internal Mandarin) | Two-stage: coarse mel diffused to t = 1, then one denoising step | T = 1, 2, 4; two-stage | Two-stage MOS 4.17 vs 3.97 at T = 1, RTF 0.0097. Cosine similarity 0.783 vs 0.819 | Fixed (t = 1, trained for it) | None |
| [Shallow Flow Matching (SFM)](https://arxiv.org/html/2505.12226) | NeurIPS 2025 | TTS (Matcha, StableTTS, CosyVoice) | ODE start time t̂_h from a trained head, scaled by a global strength α | α = 1.0–10.0 | Matcha LJ: α = 2.5 gives mean start 0.248; UTMOS 4.186 → 4.257 but WER 3.308 → 3.413. CosyVoice SFM-t: α = 1.0 gives WER 14.932. Heun-solver RTF −60.8% at α = 5 | **Per utterance** (trained head); α global | Continuity and path-equivalence theorems only |
| [DOSE](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e966a12c2d6307adb8809aaa9acf057-Paper-Conference.pdf) | NeurIPS 2023 | Speech enhancement | Two start points τ1 > τ2 | Grid {1, 5, …, 50}²; fixed vs hand-picked | Hand-picked 2-step PESQ 2.56 (VB). Mismatched 1-step PESQ 1.51 hand-picked vs 1.38 fixed | Fixed (ERM on validation); per-sample choice named as future work | Existence propositions only |
| [BBED](https://ar5iv.labs.arxiv.org/html/2302.14748) | arXiv ‡ | Speech enhancement | Reverse start t_rs | 0.5, 0.999 | PESQ 3.05 vs 3.08 with half the iterations. The OUVE baseline loses 0.31 PESQ at 0.5 | Fixed | None |
| [CRP](https://ar5iv.labs.arxiv.org/html/2309.09677) | arXiv, IEEE notice ‡ | Speech enhancement | Start t_rsp; number of steps | t_rsp = 0.5 fixed; 1–5 steps | PESQ 2.29 (1 step) and 2.55 (5 steps), vs 2.36 for a 60-NFE generative baseline | Fixed | None |
| [CTFSE](https://arxiv.org/html/2508.06842) | Interspeech 2025 (arXiv comment) | Speech enhancement | Second flow starts at t = 1 around the first flow's estimate (σ = 0.5); NFE | NFE per setting | PESQ 3.05 at 5 NFE vs StoRM 2.90 at 101 | Fixed (NFE chosen to maximise mean WB-PESQ) | None |
| [AD-FlowTSE](https://arxiv.org/abs/2510.16995) † | Preprint | Target speaker extraction | Start point set from the mixing ratio | Not read | Not read | Per input (claimed in the abstract) | Not read |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc), [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC), [openvpi DiffSinger](https://github.com/openvpi/DiffSinger/blob/main/docs/BestPractices.md) | Code | Singing voice conversion and synthesis | k_step (default 100 of 1000), t_start (default 0.0), T_start (example 0.4) | User flags | No numbers. openvpi warns T_start "should not be set too large" | Fixed per config | None |
| [Chatterbox-Flash](https://arxiv.org/html/2605.30748v1) | Preprint 2026 | TTS (discrete block diffusion) | Denoising steps per block, from a prior-calibrated confidence quantile with rate α | K = 8, α = 0.5; K = 10, α ∈ {0.5, 0.75, 1.0} | About 20% fewer steps. Seed-en WER 1.96 → 2.04, SIM-o unchanged at 0.704 | Per block (training-free rule) | None |
| [SSD](https://arxiv.org/abs/2505.15380) †, [VADUSA](https://arxiv.org/abs/2410.21951) † | Interspeech 2025; ICASSP 2025 | Codec-LM TTS | Speculative draft acceptance | Not extracted | SSD: 1.4× speedup, quality preserved in subjective tests | Per input (acceptance) | None |

Several speech methods that look like handovers start the *full* reverse process from a data-dependent prior, so they have no intermediate switching step:

- **Grad-TTS** starts at T from N(μ, τ⁻¹I) with τ = 1.5 ([Grad-TTS](https://ar5iv.labs.arxiv.org/html/2105.06337)).
- **PriorGrad** does the same for both its acoustic model and its vocoder ([PriorGrad](https://ar5iv.labs.arxiv.org/html/2106.06406)).
- **Bridge-TTS** starts from the deterministic text latent ([Bridge-TTS](https://ar5iv.labs.arxiv.org/html/2312.03491)).
- **StoRM** centres its start on a predictor's output at T ([StoRM](https://ar5iv.labs.arxiv.org/html/2212.11851)).
- **ResGrad** generates a residual from pure noise ([ResGrad](https://ar5iv.labs.arxiv.org/html/2212.14518)).
- **Voice-conversion models** such as Diff-HierVC use a data-driven prior, though this was checked at abstract level only ([Diff-HierVC](https://arxiv.org/abs/2311.04693) †).

No per-input adaptive-NFE vocoder was found. InferGrad, for example, trains a vocoder for a fixed 2-step schedule ([InferGrad](https://arxiv.org/abs/2202.03751) †).

One TTS paper switches guidance rather than models over time. Selective CFG reports that standard CFG early and selective CFG later improves speaker similarity, but its abstract does not say whether the switch is per utterance ([Selective CFG](https://arxiv.org/abs/2509.19668) †).

### B. Image and video

| Paper | Venue, year | Domain | Handover parameter | Configurations evaluated | Key findings | Per-input or fixed | Guarantee |
|---|---|---|---|---|---|---|---|
| [T-Stitch](https://arxiv.org/html/2402.14167) | ICLR 2025 | Image | Fraction r1 of steps given to the small model, which runs first | 0–100% in 10% steps; a 66-configuration three-model grid; LCM at 2 and 4 steps | DiT-S → DiT-XL FID: 9.20 at 0%, 8.95 at 40%, 10.06 at 50%. BK-SDM Tiny → SD: lowest FID at 20%. LCM: every small-model share hurts (FID 17.05 → 21.28 at 50%). Large-to-small order: FID 27.61 vs 10.06 | Fixed per model pair and budget (lookup table) | None |
| [Gandikota & Bau](https://arxiv.org/html/2503.10637) | WACV 2026 | Image | k base-model (teacher) steps first, then DMD2 | Fig. 7(b) sweep, values not extractable; k = 1 chosen | DreamSim diversity 0.350 (hybrid), 0.337 (base), 0.264 (DMD2). FID 10.79 / 12.74 / 15.52. 0.64 s vs 9.22 s. Base model on the final steps instead: 0.251 | Fixed for all prompts; adaptive switching named as future work | None |
| [OMS-DPM](https://ar5iv.labs.arxiv.org/html/2306.08860) | ICML 2023 (arXiv comment) | Image | Model assigned to each step, plus step count | Predictor-guided search per dataset and budget | CIFAR-10 FID 3.19 vs 3.56 at 2.8× speedup. Large models work best late on CIFAR-10 and ImageNet-64, but early on LSUN-Church | Fixed per dataset and budget | None |
| [Hybrid SD](https://arxiv.org/html/2408.06646) | Preprint 2024 | Image (edge–cloud) | k large-model steps first, then the small model | k ∈ {10, 5} of 25; k = 4 of 8 (LCM) | FID 12.22 (SD alone), 13.75 (k = 10), 15.39 (k = 5), OursTiny. 66% lower cloud cost | Fixed preset | None |
| [TRACK](https://arxiv.org/html/2609.30096) | Preprint 2026 | Video | Set of K steps routed to the small model (can be non-contiguous) | Per pipeline, e.g. 30 of 50 steps on Wan 2.1 | 1.95–2.73× speedup with VBench within about ±0.6. DreamSim retention: Top-K 98.3%, Last-K 97.8%, First-K 84.6% | Fixed offline, prompt-independent | Optimality for a surrogate objective only |
| [SRDiffusion](https://arxiv.org/html/2505.19151) | Preprint 2025 | Video | Threshold δ on a second-order latent-change signal; large model runs first | Wan: δ ∈ {0.002, 0.01, 0.03}; CogVideoX: four δ values; fixed-step baselines | Wan VBench 84.06 at 2.84×, vs 84.05 for the large model alone. Adaptive δ = 0.01 vs fixed step 10: 84.06 vs 84.05 | **Per sample, at runtime** | "Guaranteed" trade-off claimed, no proof |
| [Wan2.2 A14B](https://github.com/Wan-Video/Wan2.2) | 2025 release | Video | Boundary between high-noise and low-noise experts (SNR rule) | `boundary = 0.875` (T2V config) | Validation-loss plots only | Fixed per config | None |
| [CrossDistill](https://arxiv.org/html/2609.14725), [DP-DMD](https://arxiv.org/html/2602.03139), [1.x-Distill](https://arxiv.org/html/2604.04018) | Preprints 2026 | Video, image | Training-time boundary: τ⋆, anchor step K, or α | τ⋆ ∈ {0.90, 0.92, 0.94, 0.96}; K ∈ {1, 3, 5, 10, 30}; α sweep | Default τ⋆ = 0.94. DP-DMD DINO diversity 0.137 → 0.179 at K = 5, vs 0.240 for the teacher. 1.x-Distill degrades for α < 0.92 | Fixed | None |
| [Speculative sampling for diffusion](https://arxiv.org/html/2501.05370) | Preprint 2025 ‡ | Image | Draft window L; per-step accept/reject | Several drafters and L values | CIFAR-10 FID 2.34 with 35 target calls, vs 2.45 with 100 | Per sample (acceptance) | **Exact target samples** (Prop. 3.1) |
| [SDEdit](https://ar5iv.labs.arxiv.org/html/2108.01073) | arXiv 2021 ‡ | Image editing | Start noise level t0 | 0.01, then 0.1–1.0 | "t0∈[0.3,0.6] works well" | Fixed per task (binary search on one image) | Bound on deviation from the guide only |

### C. Adaptive computation with and without guarantees

| Paper | Venue, year | Domain | Handover / adaptive parameter | Configurations evaluated | Key findings | Per-input or fixed | Guarantee |
|---|---|---|---|---|---|---|---|
| [Adaptive Guidance](https://ar5iv.labs.arxiv.org/html/2312.12487) | Preprint 2023 ‡ | Text-to-image | Step at which CFG (2 NFE) gives way to conditional-only steps (1 NFE), once the cosine similarity γ_t exceeds γ̄ | LDM-512, EMU-768; γ̄ = 0.991 | 29.6 vs 40 NFE. Human study: 498 vs 502 preferences, p = 0.603 | **Per sample, training-free** | None (γ̄ set by hand) |
| [CALM](https://ar5iv.labs.arxiv.org/html/2207.07061) | NeurIPS 2022 | Text generation (T5) | Exit layer per token, chosen by a confidence threshold | Softmax, hidden-state saturation and trained-classifier measures; textual δ = 0.1/0.25, risk δ = 0.02/0.05 | CNN/DM 3.53× at risk δ = 0.05; WMT 2.73×; SQuAD 1.63× | Per input and per token | **LTT with fixed-sequence testing**, ε = 0.05 |
| [Fast yet Safe](https://arxiv.org/html/2405.20915v2) | NeurIPS 2024 | Vision, language models, **diffusion** (DeeDiff), BiLD big/little decoder | Exit threshold | CRC, UCB and LTT at several ε | Diffusion: risk controlled at every tested level, about 15% per-step savings at ε = 0.05. UCB and CRC save more than LTT | Per input | **CRC / UCB / LTT** |
| [AdaptiveDiffusion](https://arxiv.org/abs/2410.09873) † | NeurIPS 2024 | Image, video | Skipped steps, chosen from a third-order latent difference | Not extracted | About 2–5× speedup | Per trajectory, training-free | None |
| [TJS](https://arxiv.org/abs/2607.06114) † | Preprint 2026 | Text-to-image | Exit time t*, returning E[x0 \| x_t] | SDXL, SD3.5M, Z-Image-Turbo | 20–70% fewer NFE | Fixed fraction γ (from a PDF excerpt) | None |
| [AdaDiff (Zhang et al.)](https://arxiv.org/abs/2311.14768) †, [CATImage](https://arxiv.org/abs/2506.14753) †, [RouteT2I](https://arxiv.org/abs/2411.13787) † | AAAI 2025; TMLR; ICCV 2025 | Image, video | Step count or model per prompt | Learned policy or router | AdaDiff: 33–40% less time at similar quality | Per prompt, **trained** | None |
| [Calibrate-Then-Delegate](https://arxiv.org/abs/2604.14251) †, [Conformal Cascade](https://arxiv.org/abs/2607.25018) † | Preprints 2026 | LLM monitoring, QA | Threshold for deferring from a cheap model to an expensive one | — | Cascade coverage ≥ 1 − Kα | Per input | Finite-sample (multiple testing; conformal) |

## Handover position trades diversity, intelligibility and speed differently in every setting

The direction of the handover matters as much as its position.

**Teacher first preserves diversity.** In images, one base-model step at the *start* of a 4-step DMD2 run lifts DreamSim diversity from **0.264 to 0.350**, above the 50-step base model's 0.337. Putting base-model steps at the *end* instead gives **0.251** ([Gandikota & Bau](https://arxiv.org/html/2503.10637)). In video, routing a small model's steps to the beginning retains only **84.6%** of DreamSim similarity, against **97.8%** when they go at the end ([TRACK](https://arxiv.org/html/2609.30096)). DMOSpeech 2 follows the same logic in speech: 14 teacher steps restore pitch variation (CV_f0) from the student's **0.4640 to 0.5932**. The authors read the remaining gap as mostly prosodic, because mean-centred UTMOS distributions "remain consistent across all models" ([DMOSpeech 2](https://arxiv.org/html/2507.14988v1)).

**T-Stitch shows the opposite ordering when the goal is efficiency.** Its small, non-distilled model goes first. Reversing the order raises FID from **10.06 to 27.61** ([T-Stitch](https://arxiv.org/html/2402.14167)). My reading of the contrast: a *distilled* student is strong near the data end, where DMOSpeech 2's student beats its own teacher on WER, so it belongs last. A merely *smaller* model is weakest at fine detail, so it belongs first.

**Speech-specific evidence says early, noisy steps carry most of the quality.**

- F5-TTS loses only **0.11 WER points** going from 32 to 16 NFE (2.42 → 2.53) at half the RTF ([F5-TTS](https://arxiv.org/html/2410.06885)).
- A uniform 7-step schedule collapses to WER **4.16** (Chinese WER 7.96). The EPSS schedule, which keeps early steps dense, holds **2.45** at the same cost ([EPSS](https://arxiv.org/html/2505.19931)).
- In EPSS's ablation, dropping one *early* index costs more (WER **2.71**) than dropping a *late* one (**2.41**) ([EPSS](https://arxiv.org/html/2505.19931)).
- Distilled students are flat or non-monotonic in step count. CoMoSpeech's FD is best at 10 steps and worse at 50 (0.725 vs 0.850) ([CoMoSpeech](https://arxiv.org/abs/2305.06908)). FlashSpeech peaks at 2 NFE ([FlashSpeech](https://arxiv.org/abs/2404.14700)). CM-TTS's MOS dips at T = 2 ([CM-TTS](https://arxiv.org/abs/2404.00569)).
- Grad-TTS shows that a better-informed start state matters most when few steps follow: listeners preferred the data-dependent prior **93.8%** of the time at 10 steps, but only 60.3% at 50 ([Grad-TTS](https://ar5iv.labs.arxiv.org/html/2105.06337)).

Together these results support DMOSpeech 2's design of a teacher for the noisy segment and only two student steps at the end. They do not locate the best boundary.

**Metrics disagree about where the handover should sit, which makes it a risk decision rather than a single optimum.**

- **SFM:** in CosyVoice's SFM-t variant, an early start (α = 1.0, mean start 0.088) gives WER **14.932**, against 4.882 at α = 2.0. A late start (α = 4.0, mean 0.349) lowers pseudo-MOS to 3.395 and similarity to 0.868 ([SFM](https://arxiv.org/html/2505.12226)). At the selected α on test sets, SFM raises UTMOS but *worsens* WER on Matcha-TTS LJ (3.308 → 3.413) and CosyVoice (3.513 → 3.810).
- **DiffGAN-TTS:** the two-stage model gains MOS (4.17 vs 3.97) but loses cosine similarity (0.783 vs 0.819) ([DiffGAN-TTS](https://ar5iv.labs.arxiv.org/html/2201.11972)).
- **DMOSpeech 2:** teacher-guided sampling wins a significant Chinese naturalness CMOS (+0.45) over the student, with no significant English difference (+0.01) ([DMOSpeech 2](https://arxiv.org/html/2507.14988v1)).
- **Speech enhancement:** the cost of a later start depends on how well the start state matches the target. BBED loses about 0.03 PESQ at t_rs = 0.5 while OUVE loses 0.31 ([BBED](https://ar5iv.labs.arxiv.org/html/2302.14748)). DOSE's hand-picked start points beat equal intervals by up to **0.13 PESQ** on mismatched data ([DOSE](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e966a12c2d6307adb8809aaa9acf057-Paper-Conference.pdf)).

That mismatch-dependence is exactly the per-input variation a per-utterance rule would exploit.

**The best handover shifts with model pair, sampler, dataset and budget.**

- T-Stitch's safe small-model fraction is about **40%** for DiT and **20–30%** for Stable Diffusion. It turns harmful at every fraction when the large model is a few-step LCM ([T-Stitch](https://arxiv.org/html/2402.14167)).
- OMS-DPM finds that large models belong late on CIFAR-10 and ImageNet-64 but early on LSUN-Church. Small models with more steps win under tight budgets ([OMS-DPM](https://ar5iv.labs.arxiv.org/html/2306.08860)).
- The LCM result matters here because DMOSpeech 2's student is also a few-step distilled model. Mixing trades that pay off between many-step models can vanish once a few-step distilled model is involved.

**Cost savings are bounded by the hybrid's footprint.**

- DMOSpeech 2's teacher-guided mode loads **0.6B parameters** against 0.3B for the student alone, and runs at RTF 0.094 against 0.032 ([DMOSpeech 2](https://arxiv.org/html/2507.14988v1)).
- Every teacher call costs two forward passes because of CFG ([DMOSpeech2 code](https://github.com/yl4579/DMOSpeech2/blob/main/src/infer.py)).
- Gandikota & Bau and T-Stitch both list holding two models in memory as a limitation.

**Per-input adaptivity has so far bought compute, not average quality.**

- SRDiffusion's adaptive switch matched a fixed switch step almost exactly: VBench **84.06 vs 84.05** on Wan and **80.85 vs 80.85** on CogVideoX. Its claimed benefits are lower PSNR variance and later switching on hard prompts ([SRDiffusion](https://arxiv.org/html/2505.19151)).
- Adaptive Guidance cut NFE by about a quarter at human-rated parity ([Adaptive Guidance](https://ar5iv.labs.arxiv.org/html/2312.12487)).
- Chatterbox-Flash saved about 20% of steps with essentially unchanged metrics ([Chatterbox-Flash](https://arxiv.org/html/2605.30748v1)).

## Nothing published justifies DMOSpeech 2's t_switch = 0.25

**The paper evaluates 0.25 and nothing else.** Section 4.1 lists teacher-guided sampling as a baseline, "t_switch=0.25, with teacher handling 14 steps and student handling 2 steps, for a total of 16 steps". Section 3.3.2 says the teacher is used for "approximately 6-14 steps instead of 32" and the student for "2-3 instead of 4". It recommends "a higher t_switch value (around 0.4-0.5)" for diversity and "a lower t_switch value (around 0.1-0.2)" for efficiency "with minimal degradation in perceptual quality". **No table or figure reports any other t_switch, K or M** ([DMOSpeech 2](https://arxiv.org/html/2507.14988v1)). The recommendations are therefore untested claims. The results are reported only as test-set averages, with no per-utterance analysis of which inputs benefit from more teacher steps.

The predecessor, DMOSpeech (ICML 2025), contains no hybrid at all. It reports only a 128-step teacher and 1- and 4-step students. Its 4-step student cut LibriSpeech WER from **9.51 to 1.94** but lowered CV_f0 from **0.70 to 0.58** ([DMOSpeech](https://arxiv.org/html/2410.11097v2)). That loss of pitch variation is the mode shrinkage DMOSpeech 2's teacher guidance was introduced to repair.

**The released code and demo do not reproduce a clean "0.25".**

- **Code defaults.** `generate()` defaults to `teacher_steps=16, teacher_stopping_time=0.07, student_start_step=1` ([infer.py](https://github.com/yl4579/DMOSpeech2/blob/main/src/infer.py)). §This yields five teacher calls ending at t ≈ 0.076, followed by three student steps at t ≈ 0.076, 0.293 and 0.617, for eight network calls in total. That is a low-switch operating point the paper never evaluates.
- **The demo's "16 steps" cell.** It describes "14 steps from the teacher and 2 steps from the student" but sets `teacher_steps=24, teacher_stopping_time=0.3`, with the comment "0.25 means students go for the last two steps (0.26ish, 0.6ish)" ([demo.ipynb](https://github.com/yl4579/DMOSpeech2/blob/main/src/demo.ipynb)). §The teacher's grid ends at t ≈ 0.341, and the student re-enters at a fixed t ≈ 0.293.
- **The "stochastic duration" cell.** It uses 0.25 and §yields 13 + 2 calls.
- **A different mechanism.** Algorithm 2 continues the student from the teacher's state at the switch time. The code instead takes the teacher's clean estimate x̂₁ and re-noises it with fresh Gaussian noise at a student time fixed by `student_start_step`, independently of `teacher_stopping_time` ([infer.py](https://github.com/yl4579/DMOSpeech2/blob/main/src/infer.py)). The extra noise at the handover may contribute to the recovered diversity.

The paper does not say which setting or mechanism produced the Table 1 row. Its diversity metric was computed "with the DDPM sampler", and the code's teacher defaults to stochastic sampling (`eta=1.0`). CV_f0 values from deterministic ODE sampling may not be comparable ([DMOSpeech 2](https://arxiv.org/html/2507.14988v1)).

**One reconciliation is my own arithmetic, not the authors' statement.** Apply Algorithm 2's rule (the largest k with t_k ≤ t_switch) to a 32-interval sway-sampled grid with s = −1, so that t_k = 1 − cos(πk/64). Then t₁₄ ≈ 0.227 ≤ 0.25 < t₁₅ ≈ 0.259, which gives **exactly 14 teacher steps**. The paper's "14 + 2" is therefore consistent with Algorithm 2 on F5-TTS's default 32-step grid. The demo reaches the same count by a different route.

This also shows what 0.25 means in practice. The teacher covers only the noisiest quarter of the time axis (t = 0 is noise in this convention), but because of sway sampling that quarter holds **14 of the 32 grid steps (44%)**.

**No one else has evaluated or analysed other switching times.**

- **GitHub.** As of 2 November 2026 the repository had 305 stars, 39 forks and 18 issues and PRs. None concerns switching time, teacher steps or diversity. Every fork with a readable `infer.py` keeps the defaults ([GitHub API](https://api.github.com/repos/yl4579/DMOSpeech2); [issues](https://github.com/yl4579/DMOSpeech2/issues?q=is%3Aissue)).
- **Semantic Scholar** lists 10 citing papers. Their citation contexts concern the GRPO duration predictor or metric optimisation, never teacher-guided sampling ([Semantic Scholar citations](https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.14988/citations?fields=title,year,venue,externalIds,contexts&limit=100)). The duration predictor lies outside this thesis's scope.
- **OpenAlex** reports zero citations ([OpenAlex](https://api.openalex.org/works?search=DMOSpeech%202%20reinforcement%20learning%20duration)).
- **arXiv metadata search** for "teacher-guided sampling" returned no records ([arXiv API](http://export.arxiv.org/api/query?search_query=all:%22teacher-guided%20sampling%22)).

**Four checks remain open, so "none found" is not proof of absence.**

- The AAAI camera-ready PDF was blocked by bot protection, so whether it adds a t_switch ablation is **unverified**.
- Google Scholar was not queried.
- GitHub code search for other `teacher_stopping_time` values was not run.
- The Hugging Face model card and the project page dmospeech2.github.io were not checked.

**Indirect evidence makes a teacher-early, student-late split sensible but says nothing about 0.25 specifically.** Early-step importance (EPSS, sway sampling), flat student step curves, and teacher-first diversity recovery in images all support the shape of the design. Comparable boundaries elsewhere cover a wide range:

- **SFM** selected α = 2.5 for Matcha-TTS on LJSpeech, which gives a mean start of **0.248** on the same 0-is-noise axis ([SFM](https://arxiv.org/html/2505.12226)). That is a coincidence, not a justification: SFM hands over in the opposite direction (a cheap coarse model first), uses a trained head, and works with different models.
- **Gandikota & Bau** use one teacher step out of four.
- **Hybrid SD** tested 5 or 10 large-model steps out of 25.
- **T-Stitch** puts its safe fraction at 20–40%, with the small model first.
- **Wan2.2** sets its high-noise expert boundary at 0.875 ([Wan2.2 config](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_t2v_A14B.py)).
- **CrossDistill**'s training-time crossover τ⋆ = 0.94 applies on axes where 1 is noise ([CrossDistill](https://arxiv.org/html/2609.14725)).

On F5-TTS's axis, the Wan2.2 and CrossDistill boundaries correspond to high-noise segments covering the noisiest 12.5% and 6% of time. That is closer to the code's 0.07 than to 0.25. This conversion is mine, and schedule shifts (Wan's shift of 12.0, F5's sway) change how many *steps* fall in each segment, so the values do not transfer directly. CrossDistill and 1.x-Distill also share authors, so their matching 0.94 values are not independent evidence.

**The defensible position: 0.25 is the published operating point, not a demonstrated optimum.** A fixed-switch sweep that names its mechanism would be the first published evidence on DMOSpeech 2 at other switching times. It should cover at least the code default 0.07, the paper's 0.25, and the recommended 0.1–0.2 and 0.4–0.5 ranges.

## Per-input and risk-calibrated neighbours narrow the novelty claim

No single work combines a training-free, per-utterance, teacher-to-student handover time with a calibrated guarantee. Several works each take one or two of those ingredients, and four of them are serious threats to an over-broad claim.

| Work | What it already does | How the thesis differs | Threat to which claim |
|---|---|---|---|
| [SFM](https://arxiv.org/html/2505.12226) (NeurIPS 2025) | Picks a per-utterance ODE start time in speech flow matching | Thesis is training-free (SFM trains a head and retrains the flow module). It hands over between two pretrained models, expensive guided teacher first; SFM moves from a coarse predictor into the same model. It adds an LTT bound (SFM's α is tuned on mean pseudo-MOS with no risk statement). F5-TTS is not an SFM backbone | **High** for "first per-input handover in TTS"; low for the full combination |
| [Adaptive Guidance](https://ar5iv.labs.arxiv.org/html/2312.12487) (preprint ‡) | Training-free, per-sample switch from expensive guided steps to cheaper ones, triggered by a convergence signal | Thesis's cheaper branch is a *different, distilled model*; AG only drops guidance within one model. Thesis is in speech, adds a prompt-SNR signal, and calibrates the threshold with LTT; AG's γ̄ is set by hand | **High** for "training-free convergence-triggered switch" |
| [SRDiffusion](https://arxiv.org/html/2505.19151) (preprint) | Per-sample runtime switch from a large to a small model, triggered by a latent-change threshold | Thesis uses a teacher and its distilled student, works in speech, and gives a formal bound. SRDiffusion claims a "guaranteed" trade-off without proof | **High** for "first per-input model switch in generation" |
| [CALM](https://ar5iv.labs.arxiv.org/html/2207.07061) (NeurIPS 2022) | LTT with fixed-sequence testing over a descending threshold grid, falling back to the full model. Uses a parameter-free hidden-state-saturation signal and bounds degradation relative to the full model | Thesis changes the decision variable (handover time in a flow sampler, not decoder depth), the domain (TTS) and the signals (prompt SNR, endpoint drift) | **High** for "risk-calibrated adaptive generation" |
| [Fast yet Safe](https://arxiv.org/html/2405.20915v2) (NeurIPS 2024) | Risk control (CRC/UCB/LTT) of early exit in **diffusion** with a label-free consistency risk; also a big/little decoder pair (BiLD) | Thesis applies this to flow-matching TTS and a teacher-to-student handover time | **High** for "first risk-controlled diffusion compute" |
| [Chatterbox-Flash](https://arxiv.org/html/2605.30748v1) (preprint) | Training-free per-block confidence stopping in TTS | Chatterbox-Flash works on discrete masked tokens with one model; thesis uses continuous flow with two models, adds a guarantee, and tests noisy prompts (Chatterbox-Flash has none) | Medium |
| [Speculative sampling](https://arxiv.org/html/2501.05370) (preprint ‡) | Per-sample draft/verify with exact target-distribution samples | Thesis avoids target verification calls and bounds *added failures* rather than guaranteeing exactness | Medium (a reviewer alternative) |
| [AdaptiveDiffusion](https://arxiv.org/abs/2410.09873) †, [TJS](https://arxiv.org/abs/2607.06114) † | Training-free stability or endpoint-decodability signals for skipping or stopping | They skip or stop steps; thesis switches models and calibrates the threshold | Medium for "sampler-internal signals" |
| [AD-FlowTSE](https://arxiv.org/abs/2510.16995) † | Per-input start point for flow-based target speaker extraction | Different task, one model, no guarantee (abstract only) | Low to medium |
| [AdaDiff (Zhang et al.)](https://arxiv.org/abs/2311.14768) †, [CATImage](https://arxiv.org/abs/2506.14753) †, [RouteT2I](https://arxiv.org/abs/2411.13787) † | Per-prompt choice of step count or model | Trained routers with no guarantee | Low |
| [Calibrate-Then-Delegate](https://arxiv.org/abs/2604.14251) †, [Conformal Cascade](https://arxiv.org/abs/2607.25018) † | Finite-sample-calibrated deferral between cheap and expensive models | LLM classification and monitoring, not a generative trajectory | Low to medium |
| [SSD](https://arxiv.org/abs/2505.15380) †, [VADUSA](https://arxiv.org/abs/2410.21951) † | Per-input compute through speculative acceptance in codec-LM TTS | Autoregressive models, draft models or heads, no bound | Low |

**SFM is the speech-domain threat.** Its head predicts one start time per utterance (frame values are averaged through a sigmoid), and inference starts the flow ODE from a state built at that time. A thesis therefore cannot say "first per-input handover in speech flow matching". SFM differs on every other axis:

- It requires training: the head plus joint losses, with CosyVoice's flow module retrained.
- Its handover runs cheap-first within one model.
- Its global α is chosen by a coarse validation search on mean pseudo-MOS.
- Its theorems cover Wasserstein continuity and path equivalence, not quality.
- The authors themselves call its WER and similarity gains "more variable" ([SFM](https://arxiv.org/html/2505.12226)).

SFM's adaptive-step solvers (Heun, Dopri5) also vary NFE per input through local error control. A reviewer could raise that as a per-input baseline, although its tolerances are numerical, not quality-risk bounds.

**CALM and Fast yet Safe are the guarantee-side threat.** The thesis's calibration logic follows CALM's recipe:

- threshold a sampler signal;
- test a descending grid with Learn-Then-Test and fixed-sequence testing;
- bound expected degradation relative to the full model;
- fall back to the full model if no threshold passes.

CALM's guarantee holds with probability at least 1 − ε over the calibration draw (ε = 0.05), for i.i.d. inputs and losses in [0, 1]. It applies to *expected* sequence-level consistency, not per-token behaviour ([CALM](https://ar5iv.labs.arxiv.org/html/2207.07061)). Learn-Then-Test itself frames risk control as multiple hypothesis testing with finite-sample guarantees for any model without refitting ([LTT](https://arxiv.org/abs/2110.01052)).

Fast yet Safe has already moved this family to diffusion sampling, using LPIPS consistency to the full model as a label-free risk. It reports that risk stayed controlled at every tested level, with about 15% per-step savings at ε = 0.05 accumulating over roughly 500 steps. It also found that UCB and CRC save more compute than LTT at the same guarantee ([Fast yet Safe](https://arxiv.org/html/2405.20915v2)).

The thesis should borrow two things from Fast yet Safe:

- **The consistency-risk definition.** Added failures are measured relative to the teacher-only or fixed-switch output, so no human labels are needed.
- **The finding that LTT is conservative.** This is a reason to expect modest savings. LTT is still justified if the failure rate is not monotone in the threshold.

**Adaptive Guidance is the closest structural analogue, and SRDiffusion is the closest model-switching one.** AG's rule fires per sample when the conditional and unconditional noise predictions converge (cosine similarity above γ̄). It saves about 25% of compute with no significant human-preference difference ([Adaptive Guidance](https://ar5iv.labs.arxiv.org/html/2312.12487)).

This places endpoint-stabilisation drift in an established family of training-free convergence signals: AG's cosine similarity, AdaptiveDiffusion's third-order latent difference, TJS's endpoint decodability, and CALM's hidden-state saturation. "Using a signal the sampler already computes" is not new on its own. SRDiffusion shows a per-sample model switch works but gains little on average over a tuned fixed step ([SRDiffusion](https://arxiv.org/html/2505.19151)).

**Two findings support the thesis.**

- **Three works name per-sample adaptivity as open.** DOSE writes that its hyperparameters, chosen under empirical risk minimisation, "may not be optimal for every individual sample" and calls per-sample selection "an important direction for future research" ([DOSE](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e966a12c2d6307adb8809aaa9acf057-Paper-Conference.pdf)). Gandikota & Bau state that "our approach treats all prompts uniformly" ([Gandikota & Bau](https://arxiv.org/html/2503.10637)). DP-DMD lists choosing its anchor step "according to timestep, prompt complexity, teacher uncertainty, or student capacity" as future work ([DP-DMD](https://arxiv.org/html/2602.03139)).
- **Speech-specific signals have a motivation.** SPAR-K reports that confidence-based early exit, common in text LLMs, performs worse on speech tokens ([SPAR-K](https://arxiv.org/abs/2603.09215) †). That supports using speech-specific signals such as prompt SNR, although it comes from autoregressive spoken LMs, not flow TTS.

## The remaining gap is a specific combination, and the claim should say so

**The gap that survives the search** has four parts:

- a **training-free, per-utterance** choice of
- the **time at which a classifier-free-guided flow-matching teacher hands its trajectory to a distilled student** (F5-TTS → DMOSpeech 2),
- driven by **prompt SNR and endpoint-stabilisation drift**,
- with a **Learn-Then-Test-calibrated finite-sample bound on added quality failures**.

The evidence for each part:

- **Speech:** no work was found that chooses a teacher-to-student switch per input, and no LTT, CRC or conformal method has been applied to a compute decision in flow-matching or speech generation. Conformal methods appear in speech only for recognition, e.g. risk-controlled ASR candidate sets ([Ernez et al.](https://proceedings.mlr.press/v204/ernez23a.html) †).
- **Noisy prompts:** noisy-prompt robustness in zero-shot TTS is handled at training time, through noise-mixing fine-tuning, data filtering and masked-denoising pre-training ([Wang et al.](https://arxiv.org/abs/2406.05699) †). No work was found that adapts computation to prompt SNR, so the signal choice is unprecedented in the sources searched.
- **Image and video:** the only per-input model switch is SRDiffusion, which has no guarantee. The only guaranteed per-sample scheme is speculative sampling, which buys exactness with target verification calls.

These absences come from targeted searches rather than exhaustive surveys. They include arXiv metadata queries that do not search full text, and about fifteen queries for per-input TTS switching. The claim needs a "to our knowledge" qualifier and a search date.

**A wording that stays inside the evidence:**

> To our knowledge, as of November 2026 no published text-to-speech method chooses, per utterance, the time at which a classifier-free-guided teacher hands a flow-matching sampling trajectory to a distilled student, and no published speech-generation method attaches a finite-sample risk guarantee to such a sampling-compute decision. We propose a training-free handover rule for F5-TTS → DMOSpeech 2 sampling that uses the reference prompt's SNR and the stabilisation drift of the sampler's endpoint estimate, and we calibrate its threshold with Learn-Then-Test so that, for test utterances exchangeable with the calibration set, the rate of quality failures added relative to [the stated reference sampler] stays below δ with probability at least 1 − ε. The calibration procedure follows CALM and Fast yet Safe; the contribution is its application to a new decision variable and domain, not a new risk-control method.

**What the thesis must not claim:**

- "first per-input handover in TTS" (SFM);
- "first to use signals the sampler already computes" (Adaptive Guidance, CALM, AdaptiveDiffusion);
- "first LTT-calibrated adaptive generation" (CALM);
- "first risk-controlled diffusion compute" (Fast yet Safe);
- "per-input switching improves quality over a fixed switch". SRDiffusion found parity on average, and the workspace's own audit treats a hybrid advantage as unestablished.

**Five obligations make the claim defensible.**

1. **Name the handover mechanism.** State whether it follows Algorithm 2 (continue from the teacher's state) or the released code (re-noise the teacher's clean estimate). These are different samplers with different stochasticity.
2. **Report fixed-switch baselines across a sweep.** Include 0.07, 0.25 and the 0.1–0.2 and 0.4–0.5 ranges, since none has been published. The adaptive rule's value is measured against the *best* fixed switch, not only against 0.25.
3. **Define "added failure" against an explicit reference.** Use teacher-only output or the fixed 0.25 switch, following Fast yet Safe's label-free consistency risk. Report WER and speaker similarity separately from MOS proxies, because SFM and DiffGAN-TTS show these move in opposite directions.
4. **State that the bound assumes exchangeability.** It holds only for test utterances drawn like the calibration set. A different prompt-SNR mix at deployment breaks it unless calibration is stratified by SNR or drawn from the target noise distribution. For the same reason, a bound computed on a development replay set does not transfer to an independent held-out set from a different distribution.
5. **Anticipate conservative savings.** LTT saved less than UCB or CRC in Fast yet Safe's comparisons. The thesis should say why it prefers LTT, for example that the failure rate may not be monotone in the threshold.

## Conclusion

The literature has treated "where to hand over" as a per-setting tuning question, answered with a lookup table, a validation grid or one fixed constant. The few adaptive studies show the payoff of adaptivity appears in variance and hard cases rather than in mean quality. That favours the thesis's framing: a bound on the *rate of added failures* is a tail statement, and tails are where per-input switching has shown value. A thesis that promises a better average would be arguing against the closest evidence. A thesis that promises fewer teacher steps at a certified failure rate is arguing with it.

The DMOSpeech 2 operating point is also under-specified. The paper's t_switch, the code's stopping time and the demo's settings describe different samplers, and only one of them was ever measured. The thesis's first empirical contribution can therefore be a modest but citable one: the first published switching-time sweep for a guided-teacher/distilled-student TTS pipeline, with its mechanism stated. The adaptive, LTT-calibrated rule would then be evaluated against the best of those fixed points rather than against an unexamined default.

## Reference list with verification status

Verification levels:

- **Full text**: the paper, tables or code were read.
- **Extraction**: full text was read through an HTML-summarising tool; re-check the numbers against the PDF.
- **Abstract** (†): abstract or search-result level only.
- **‡** in the venue column: the venue was not confirmed from a primary source.

| Work | Link | Venue as verified | Verification level |
|---|---|---|---|
| Li et al., DMOSpeech 2 | [arXiv 2507.14988v1](https://arxiv.org/html/2507.14988v1); [AAAI OJS](https://ojs.aaai.org/index.php/AAAI/article/view/40450) | AAAI 2026, vol. 40(38), pp. 31814–31822 (OpenAlex/OJS); camera-ready PDF unread | Full text (arXiv v1) |
| DMOSpeech 2 code and demo | [infer.py](https://github.com/yl4579/DMOSpeech2/blob/main/src/infer.py); [demo.ipynb](https://github.com/yl4579/DMOSpeech2/blob/main/src/demo.ipynb) | Official repository | Full text; schedules derived without running the model (§) |
| Li, Kumar, Jin, DMOSpeech | [PMLR](https://proceedings.mlr.press/v267/li25ay.html); [arXiv 2410.11097](https://arxiv.org/html/2410.11097v2) | ICML 2025 | Full text |
| Chen et al., F5-TTS | [arXiv 2410.06885](https://arxiv.org/html/2410.06885) | ACL 2025 (OpenAlex) | Full text (tables) |
| Zheng et al., Fast F5-TTS / EPSS | [arXiv 2505.19931](https://arxiv.org/html/2505.19931) | Interspeech 2025 | Full text (tables) |
| Liu et al., DiffSinger | [ar5iv 2105.02446](https://ar5iv.labs.arxiv.org/html/2105.02446) | AAAI 2022 (arXiv comment) | Extraction |
| Liu, Su, Yu, DiffGAN-TTS | [ar5iv 2201.11972](https://ar5iv.labs.arxiv.org/html/2201.11972); [ICML 2022 listing](https://icml.cc/virtual/2022/21294) | Preprint; ICML 2022 workshop talk | Extraction |
| Yang et al., Shallow Flow Matching | [arXiv 2505.12226](https://arxiv.org/html/2505.12226); [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/6ee0ebd0983f0926496b7d4e0d48b8e3-Abstract-Conference.html) | NeurIPS 2025 | Full text / extraction |
| Popov et al., Grad-TTS | [ar5iv 2105.06337](https://ar5iv.labs.arxiv.org/html/2105.06337) | ICML 2021 (per a citing paper's references) | Extraction |
| Vovk et al., Fast Grad-TTS | [PDF](https://www.hse.ru/data/2025/11/09/135179301/Fast%20Grad-TTS%20Towards%20Efficient%20Diffusion-Based%20Speech%20Generation%20on%20CPU.pdf); [ISCA](https://www.isca-archive.org/interspeech_2022/vovk22_interspeech.html) | Interspeech 2022 | Full text |
| Lee et al., PriorGrad | [ar5iv 2106.06406](https://ar5iv.labs.arxiv.org/html/2106.06406) | ICLR 2022 | Extraction |
| Chen et al., ResGrad | [ar5iv 2212.14518](https://ar5iv.labs.arxiv.org/html/2212.14518) | Not stated ‡ | Extraction |
| Chen et al., Bridge-TTS | [ar5iv 2312.03491](https://ar5iv.labs.arxiv.org/html/2312.03491) | ‡ | Extraction |
| Tai et al., DOSE | [NeurIPS PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e966a12c2d6307adb8809aaa9acf057-Paper-Conference.pdf) | NeurIPS 2023 | Full text |
| Lay et al., BBED | [ar5iv 2302.14748](https://ar5iv.labs.arxiv.org/html/2302.14748) | ‡ | Extraction |
| Lay et al., CRP | [ar5iv 2309.09677](https://ar5iv.labs.arxiv.org/html/2309.09677) | ‡ (IEEE notice) | Extraction |
| Lemercier et al., StoRM | [ar5iv 2212.11851](https://ar5iv.labs.arxiv.org/html/2212.11851) | IEEE/ACM TASLP 2023 | Extraction |
| Trachu et al., Thunder | [ar5iv 2406.06139](https://ar5iv.labs.arxiv.org/html/2406.06139) | Preprint | Extraction |
| CTFSE (cascaded two flows) | [arXiv 2508.06842](https://arxiv.org/html/2508.06842) | Interspeech 2025 (arXiv comment) | Extraction |
| Hsieh & Kim, AD-FlowTSE | [arXiv 2510.16995](https://arxiv.org/abs/2510.16995) | Preprint | Abstract † |
| so-vits-svc; DDSP-SVC; openvpi DiffSinger | [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc); [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC); [openvpi](https://github.com/openvpi/DiffSinger/blob/main/docs/BestPractices.md) | Code documentation | Full text (README/docs) |
| Seo, Park, Nam, Chatterbox-Flash | [arXiv 2605.30748v1](https://arxiv.org/html/2605.30748v1) | Preprint | Full text |
| Lin et al., SSD | [arXiv 2505.15380](https://arxiv.org/abs/2505.15380) | Interspeech 2025 | Abstract † |
| Li et al., VADUSA | [arXiv 2410.21951](https://arxiv.org/abs/2410.21951) | ICASSP 2025 | Abstract † |
| Huang, Chiang, Lee, SPAR-K | [arXiv 2603.09215](https://arxiv.org/abs/2603.09215) | Preprint | Abstract † |
| Wang et al., noise robustness for flow-matching zero-shot TTS | [arXiv 2406.05699](https://arxiv.org/abs/2406.05699) | Interspeech 2024 | Abstract † |
| Zheng & Maleki, Selective CFG for zero-shot TTS | [arXiv 2509.19668](https://arxiv.org/abs/2509.19668) | Preprint | Abstract † |
| Diff-HierVC | [arXiv 2311.04693](https://arxiv.org/abs/2311.04693) | Interspeech 2023 (arXiv comment) | Abstract † |
| Chen et al., InferGrad | [arXiv 2202.03751](https://arxiv.org/abs/2202.03751) | ICASSP 2022 | Abstract † |
| CoMoSpeech; FlashSpeech; CM-TTS | [2305.06908](https://arxiv.org/abs/2305.06908); [2404.14700](https://arxiv.org/abs/2404.14700); [2404.00569](https://arxiv.org/abs/2404.00569) | ‡ (arXiv only) | Full text (tables) |
| Pan et al., T-Stitch | [arXiv 2402.14167](https://arxiv.org/html/2402.14167); [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/120339238f293d4ae53a7167403abc4b-Paper-Conference.pdf) | ICLR 2025 | Full text (arXiv v1; camera-ready tables not cross-checked) |
| Gandikota & Bau, Distilling Diversity and Control | [arXiv 2503.10637](https://arxiv.org/html/2503.10637); [CVF WACV 2026](https://openaccess.thecvf.com/content/WACV2026/html/Gandikota_Distilling_Diversity_and_Control_in_Diffusion_Models_WACV_2026_paper.html) | WACV 2026 | Full text (arXiv v4); Fig. 7(b) values unverified |
| Liu et al., OMS-DPM | [ar5iv 2306.08860](https://ar5iv.labs.arxiv.org/html/2306.08860) | ICML 2023 (arXiv comment) | Full text |
| Yan et al., Hybrid SD | [arXiv 2408.06646](https://arxiv.org/html/2408.06646) | Preprint | Full text |
| Munir et al., TRACK | [arXiv 2609.30096](https://arxiv.org/html/2609.30096) | Preprint | Full text |
| Cheng et al., SRDiffusion | [arXiv 2505.19151](https://arxiv.org/html/2505.19151) | Preprint | Full text |
| Wan2.2 | [README](https://github.com/Wan-Video/Wan2.2); [T2V config](https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_t2v_A14B.py) | Model release | Code/config |
| CrossDistill; DP-DMD; 1.x-Distill | [2609.14725](https://arxiv.org/html/2609.14725); [2602.03139](https://arxiv.org/html/2602.03139); [2604.04018](https://arxiv.org/html/2604.04018) | Preprints | Full text |
| De Bortoli et al., speculative sampling for diffusion | [arXiv 2501.05370](https://arxiv.org/html/2501.05370) | Preprint; ICML 2025 ‡ | Full text (App. J unread) |
| Meng et al., SDEdit | [ar5iv 2108.01073](https://ar5iv.labs.arxiv.org/html/2108.01073) | ‡ | Full text |
| Angelopoulos et al., Learn then Test | [arXiv 2110.01052](https://arxiv.org/abs/2110.01052) | Journal venue ‡ | Abstract † |
| Schuster et al., CALM | [ar5iv 2207.07061](https://ar5iv.labs.arxiv.org/html/2207.07061) | NeurIPS 2022 (oral) | Full text |
| Jazbec et al., Fast yet Safe | [arXiv 2405.20915v2](https://arxiv.org/html/2405.20915v2) | NeurIPS 2024 | Full text (main body; appendix unread) |
| Castillo et al., Adaptive Guidance | [ar5iv 2312.12487](https://ar5iv.labs.arxiv.org/html/2312.12487) | ‡ (arXiv only) | Full text |
| Ye et al., AdaptiveDiffusion | [arXiv 2410.09873](https://arxiv.org/abs/2410.09873) | NeurIPS 2024 | Abstract † |
| Peng & Gao, TJS | [arXiv 2607.06114](https://arxiv.org/abs/2607.06114) | Preprint | Abstract + PDF excerpt † |
| Zhang et al., AdaDiff | [arXiv 2311.14768](https://arxiv.org/abs/2311.14768) | AAAI 2025 | Abstract † |
| Li et al., CATImage | [arXiv 2506.14753](https://arxiv.org/abs/2506.14753) | TMLR | Abstract † |
| RouteT2I | [arXiv 2411.13787](https://arxiv.org/abs/2411.13787) | ICCV 2025 | Search result † |
| Pona et al., Calibrate-Then-Delegate | [arXiv 2604.14251](https://arxiv.org/abs/2604.14251) | Preprint | Abstract † (use of LTT unconfirmed) |
| Dou, Lian, Li, Conformal Cascade | [arXiv 2607.25018](https://arxiv.org/abs/2607.25018) | Preprint | Abstract † |
| Ernez et al., conformal risk control for ASR | [PMLR v204](https://proceedings.mlr.press/v204/ernez23a.html) | PMLR vol. 204, 2023 | Search result † |
| Citation and repository searches for DMOSpeech 2 | [Semantic Scholar API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.14988/citations?fields=title,year,venue,externalIds,contexts&limit=100); [OpenAlex](https://api.openalex.org/works?search=DMOSpeech%202%20reinforcement%20learning%20duration); [GitHub API](https://api.github.com/repos/yl4579/DMOSpeech2); [arXiv API](http://export.arxiv.org/api/query?search_query=all:%22teacher-guided%20sampling%22) | Queried 2 Nov 2026 | Google Scholar, GitHub code search, HF model card and project page not checked |
