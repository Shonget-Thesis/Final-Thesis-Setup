# Speech generation methods that hand over between models/stages at a chosen step (shallow diffusion, two-stage, residual refinement, coarse-to-fine flow matching, prior-start)

Scope note: notes compiled 2026-11-02 from primary sources: the paper itself (arXiv/ar5iv/proceedings PDF) or official repositories. Venue lines come from the arXiv "comments"/journal-ref fields (retrieved through the arXiv API), the proceedings site, or the conference site. "UNVERIFIED" marks anything not confirmed from a primary source. Distillation and adaptive step-size methods are mentioned only where they also involve a handover. Many SE works use time conventions where t=T (or t=1) is the noisy end. Flow-matching works such as SFM use t=0 for noise and t=1 for data. Each finding states its convention.

Fetch provenance: Most numbers below came from full-text extraction of ar5iv/arxiv-HTML pages through a fetch-and-summarise tool. The DOSE and Fast Grad-TTS PDFs were read directly, page by page. Table and row numbers were quoted by the extraction step. Re-check them against the PDF before using them in the manuscript.

---

## Q1. DiffSinger / DiffSpeech (Liu et al., AAAI 2022, arXiv 2105.02446): shallow step k, the boundary predictor, and whether k is per dataset

### Takeaway
DiffSinger's handover parameter is the shallow step k (out of T = 100 for SVS). A trained boundary predictor (BP) picks k once per dataset, and the denoiser is trained only on t ∈ {1..k}. Reported values are k = 54 (PopCS, SVS) and k = 70 (LJSpeech, DiffSpeech). The single ablation shows that no shallow diffusion costs −0.500 CMOS and k = 25 costs −0.053 CMOS relative to k = 54. Shallow diffusion cut RTF by 45.1% (SVS) and 29.2% (TTS). k is never chosen per input, and the paper gives no guarantee.

### Cited Findings
- **Venue**: the arXiv comment says "Accepted by AAAI 2022". The paper covers SVS (DiffSinger), TTS (DiffSpeech) and the shallow diffusion mechanism — [arXiv 2105.02446](https://arxiv.org/abs/2105.02446)
- **Handover mechanism**: an auxiliary decoder produces a coarse mel M̃. It is diffused to step k as M̃_k = √ᾱ_k·M̃ + √(1−ᾱ_k)·ε, and k reverse steps run from there instead of from pure noise at T. Algorithm 1 trains the denoiser with t ∼ Uniform({1,…,k}), so the chosen k is also a training-time hyperparameter. The denoiser is never trained above k — [DiffSinger, ar5iv](https://ar5iv.labs.arxiv.org/html/2105.02446)
- **Boundary predictor (how k is chosen)**:
  - BP is a classifier with a noising module. Diffused ground-truth mels M_t are labelled 1 and diffused auxiliary-decoder mels M̃_t are labelled 0, trained with cross-entropy L_BP = −E[log BP(M_t,t) + log(1−BP(M̃_t,t))].
  - For each training mel, the method finds the earliest k′ such that the margin between BP(M_t,t) and BP(M̃_t,t) is below a threshold for 95% of steps t ∈ [k′, T]. k is the average of k′ over the training set.
  - The threshold is 0.4, "set empirically". The auxiliary decoder is trained first, then BP is trained for 30k steps to obtain k.
  — [DiffSinger, ar5iv](https://ar5iv.labs.arxiv.org/html/2105.02446)
- **Alternatives in Appendix B**: choose the smallest t at which an expected KL-divergence condition holds, or set k manually by brute-force search on the validation set. The paper says k "actually can be chosen manually" — [DiffSinger, ar5iv](https://ar5iv.labs.arxiv.org/html/2105.02446)
- **k is per dataset, not per input**: BP selects "the hyperparameter k for the whole dataset". k = 54 with T = 100 on PopCS (SVS) and k = 70 on LJSpeech (DiffSpeech) — [DiffSinger, ar5iv](https://ar5iv.labs.arxiv.org/html/2105.02446)
- **Ablation (Table 2, CMOS vs row 1; columns C = channels, L = layers, w/ shallow, k)**: these are the only k values evaluated —
  - 256/20/✓/k=54: 0.000
  - 256/20/×/–: −0.500
  - 256/20/✓/k=25: −0.053
  - 128/20/✓/k=54: −0.071
  - 512/20/✓/k=54: −0.044
  - 256/10/✓/k=54: −0.293
  - 256/30/✓/k=54: −0.445
  — [DiffSinger, ar5iv](https://ar5iv.labs.arxiv.org/html/2105.02446)
- **MOS**:
  - Table 1 (SVS, Mel + PWG): "DiffSinger Naive" 3.71 ± 0.10 vs DiffSinger 3.85 ± 0.11.
  - Table 3 (TTS, Mel + HiFi-GAN): "DiffSpeech Naive" 3.69 ± 0.05 vs DiffSpeech 3.92 ± 0.06.
  — [DiffSinger, ar5iv](https://ar5iv.labs.arxiv.org/html/2105.02446)
- **Speed**: shallow diffusion gives a 45.1% speedup for SVS (RTF 0.191 vs 0.348) and 29.2% for TTS (RTF 0.121 vs 0.171). No RTF is reported per k value (e.g. k = 25) — [DiffSinger, ar5iv](https://ar5iv.labs.arxiv.org/html/2105.02446)
- **Independent re-runs**:
  - DiffGAN-TTS ran DiffSpeech with "up to 60 diffusion steps" (60 used) on its internal Mandarin corpus: MOS 4.28 ± 0.08, RTF 0.2224 on a T4 — [DiffGAN-TTS, ar5iv](https://ar5iv.labs.arxiv.org/html/2201.11972)
  - ResGrad reports DiffSpeech MOS/RTF of 3.78/0.182 (LJSpeech), 2.50/0.263 (LibriTTS) and 3.44/0.857 (VCTK) on a V100 — [ResGrad, ar5iv](https://ar5iv.labs.arxiv.org/html/2212.14518)
- **Community follow-up code (openvpi DiffSinger)**: this is not the original authors' code. It exposes `use_shallow_diffusion`, `T_start`/`T_start_infer` (reflow; the example sets both to 0.4, with `T_start_infer` ≥ `T_start`) and `K_step`/`K_step_infer` (DDPM, "currently not recommended", with T_start = 1 − K_step/timesteps). It warns that "`T_start` should not be set too large" because the auxiliary decoder's poor outputs then cannot be refined. It claims shallow diffusion "can improve quality and save inference time" but gives no numbers. Values sit in the config file, so they are fixed per configuration — [openvpi/DiffSinger BestPractices.md](https://github.com/openvpi/DiffSinger/blob/main/docs/BestPractices.md)

### Inferences
- In practice k is a dataset-level hyperparameter selected by a heuristic statistic (BP margin under 0.4 for 95% of steps, averaged over the training set). It is not a per-utterance decision and carries no error control. The averaging step throws away per-utterance variation in k′ that the method computes internally. A per-input variant would be a direct extension.
- The k ablation is very coarse (54 vs 25 vs none, one CMOS test). The −0.053 CMOS gap between k = 25 and k = 54 is small. The thesis cannot cite DiffSinger as evidence of a sharp quality optimum in k.
- The "Naive" rows appear to be the variants without shallow diffusion: the speedup figures compare the same pair. The fetched text did not quote the exact definition, so check the PDF.

### Gaps
- The paper reports no RTF or MOS per k other than the Table 2 CMOS rows. No WER or speaker-similarity numbers are reported for any k.
- T for DiffSpeech (LJSpeech) was not explicitly confirmed in the extracted text. Only T = 100 for SVS was confirmed.
- The exact form of the Appendix B KL condition was not quoted.
- The reason DiffGAN-TTS used 60 steps for DiffSpeech (vs k = 70 in the original) is not explained. It is likely their own configuration on a different corpus (UNVERIFIED).

---

## Q2. DiffGAN-TTS (Liu, Su, Yu, arXiv 2201.11972): venue, T = 1/2/4, and the two-stage "active shallow diffusion"

### Takeaway
DiffGAN-TTS is an arXiv preprint (comment "Preprint. 16 pages") that was presented as a contributed talk at the ICML 2022 Workshop on Machine Learning for Audio Synthesis. It is not a main-track paper. It evaluates denoising-diffusion-GAN models with T = 1, 2 and 4, plus a two-stage model. In the two-stage model, a pretrained basic acoustic model's coarse mel is diffused to step t = 1 and denoised in a single step. The handover step is fixed, not per input, and the paper gives no guarantees.

### Cited Findings
- **Venue**: arXiv comment "Preprint. 16 pages" — [arXiv API record for 2201.11972](https://arxiv.org/abs/2201.11972). ICML's virtual site lists it as a contributed talk in the "Workshop on Machine Learning for Audio Synthesis" at ICML 2022 — [ICML 2022 virtual listing](https://icml.cc/virtual/2022/21294)
- **Evaluated configurations and results (Table 1; RTF on an NVIDIA T4)**. Columns are SSIM, MCD24, F0 RMSE, cosine similarity, #params, RTF, MOS:
  - T=1: 0.529, 5.08, 48.45, 0.819, 32.81M, 0.0069, 3.97 ± 0.10
  - T=2: 0.524, 5.06, 47.24, 0.823, 32.81M, 0.0105, 4.01 ± 0.09
  - T=4: 0.532, 4.93, 45.68, 0.806, 32.81M, 0.0176, 4.22 ± 0.08
  - two-stage: 0.531, 5.09, 46.26, 0.783, 42.64M, 0.0097, 4.17 ± 0.08
  - Baselines: FastSpeech 2 MOS 4.03 ± 0.08 (RTF 0.0058); GANSpeech 4.12 ± 0.11; DiffSpeech (60 steps) 4.28 ± 0.08 (RTF 0.2224); GT mel + HiFi-GAN 4.36 ± 0.07; GT 4.73 ± 0.07.
  — [DiffGAN-TTS, ar5iv](https://ar5iv.labs.arxiv.org/html/2201.11972)
- **Data**: a single internal corpus of 200 hours from 228 Mandarin speakers, with 1,024 validation and 1,024 test utterances — [DiffGAN-TTS, ar5iv](https://ar5iv.labs.arxiv.org/html/2201.11972)
- **Active shallow diffusion (Sec. 3.3, Eq. 10)**:
  - Stage 1: a basic acoustic model (transformer encoder, variance adaptor, 4-block FFT mel decoder) is trained for 200k steps. Its objective sums, over diffusion steps t, the expected distance (MAE) between the diffused ground-truth mel and the diffused coarse prediction. This "forces the base acoustic model to actively learn" to make its diffused samples match those of the ground truth.
  - Stage 2: the pretrained weights initialise the generator and are frozen. The diffusion decoder is conditioned on the coarse mel x̂0.
  - Inference: x̂0 is diffused to t = 1, then one denoising step runs (Algorithm 4). The authors write: "We focus on reducing the number of denoising steps to one."
  — [DiffGAN-TTS, ar5iv](https://ar5iv.labs.arxiv.org/html/2201.11972)
- **Contrast with DiffSinger**: DiffGAN-TTS describes DiffSpeech's scheme as "training a boundary prediction to find the starting diffusion step at inference". Its own model instead learns to "actively fuse the diffusion processes starting from the ground-truth mel spectrogram and the coarse mel spectrogram". The paper does not use the word "passive" — [DiffGAN-TTS, ar5iv](https://ar5iv.labs.arxiv.org/html/2201.11972)
- **Noise schedule**: VP-SDE discretisation with β range 0.1–40. The paper prints "βmin = 0.1 and βmin = 40", an apparent typo for βmax — [DiffGAN-TTS, ar5iv](https://ar5iv.labs.arxiv.org/html/2201.11972)
- **Guarantees**: none. All claims are empirical, and "significantly better" (two-stage vs T = 1) is stated without a significance test — [DiffGAN-TTS, ar5iv](https://ar5iv.labs.arxiv.org/html/2201.11972)
- **External reproduction**: ResGrad's Table 1 reports DiffGAN-TTS on LJSpeech at MOS 2.14 ± 0.09 and RTF 0.009. The reproduction settings were not detailed in the extracted text — [ResGrad, ar5iv](https://ar5iv.labs.arxiv.org/html/2212.14518)

### Inferences
- The two-stage handover gains MOS over one-stage T = 1/2 (4.17 vs 3.97/4.01) at an RTF between them (0.0097). It loses cosine similarity (0.783 vs 0.819/0.823). Even in this early work, the handover trades naturalness against a fidelity-type metric.
- The handover step is fixed by design (t = 1). The model is trained for that handover, so it is not a training-free switch.

### Gaps
- The training T for the two-stage model is not stated.
- No ablation over the handover step for the two-stage model exists, so there is no T > 1 two-stage variant to compare.
- Figure 5 (inference time vs text length) could not be read numerically.

---

## Q3. Shallow Flow Matching (SFM; Yang, Cai, Saito, Wang, Saruwatari; NeurIPS 2025; arXiv 2505.12226)

### Takeaway
SFM is the closest prior work to a per-input handover. A trained, lightweight SFM head predicts a per-utterance start time t̂_h (and σ̂_h) from the weak (coarse) generator's hidden states. Inference starts the flow-matching ODE at that intermediate point. A single global strength α ≥ 1, tuned on validation data, scales t̂_h.

SFM requires training (head plus joint losses), so it is not training-free. Its theorems cover continuity and path equivalence, not quality or risk. Naturalness proxies improve, but WER/SIM changes are mixed, and WER sometimes gets worse. Too small an α can badly hurt WER, and too large an α lowers PMOS and SIM.

### Cited Findings
- **Venue**: NeurIPS 2025 main track. The arXiv comment says "Accepted by NeurIPS 2025" — [NeurIPS 2025 proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/6ee0ebd0983f0926496b7d4e0d48b8e3-Abstract-Conference.html); [arXiv 2505.12226](https://arxiv.org/abs/2505.12226)
- **Core idea (abstract)**: "Unlike conventional FM modules, which use the coarse representations from the weak generator as conditions, SFM constructs intermediate states along the FM paths from these representations". An orthogonal projection determines the temporal position during training — [arXiv 2505.12226](https://arxiv.org/abs/2505.12226)
- **Handover parameter**:
  - Training: t_h = max(0, E[sg[X̂_h]·X₁ / X₁·X₁]) (Eq. 13), the projection coefficient of the head output onto the target. Appendix C later removes the t_h ≥ 0 constraint because L_coarse makes t_h "generally larger than 0 during training".
  - Intermediate state: X_{t̃_h} is built on the CondOT path (Eqs. 14–15). The ODE is integrated from t̃_h to 1, where t = 1 is data.
  — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)
- **Per-utterance prediction (training required)**:
  - The SFM head h_ψ maps the weak generator's final hidden states to X̂_h, t̂_h and σ̂²_h (Eq. 12). t̂_h passes through a sigmoid and is averaged over frames, giving one value per utterance.
  - Losses: L_t = (t̂_h − t̃_h)², L_σ, L_μ, and in total L_SFM = L_coarse + L_t + L_σ + L_μ + L_CFM (Eq. 21).
  - The authors "adopt and train the original models as baseline models". For CosyVoice, only the flow module is trained.
  — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)
- **Inference strength α (Eq. 22)**: Δ = max(α((1−σ_min)t̂_h + σ̂_h), 1) and t̃_h = (α/Δ)·t̂_h. α is global, chosen on validation sets by mean pseudo-MOS (PMOS). The search starts at 1.0, steps by 1.0, then checks ±0.5 around the best value (Sec. 4.5) — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)
- **Mean start times (validation)**: for Matcha-TTS (LJ Speech), the mean of t̃_g over validation utterances is
  - α 1.0 → 0.099
  - 2.0 → 0.198
  - 2.5 → 0.248
  - 3.0 → 0.297
  - 3.5 → 0.347
  - 4.0 → 0.397
  - 5.0 → 0.496
  - 6.0–10.0 → 0.520 (saturated by the Δ clamp)

  Only means are reported, with no per-utterance distribution or variance; the paper says adaptively determined t_h "tends to be small" — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)
- **α sweep, Matcha-TTS LJ Speech validation (Table 3)**. Values are PMOS / UTMOS / UTMOSv2 / Distill-MOS / WER % / SIM:
  - α 1.0: 4.036 / 4.194 / 3.721 / 4.192 / 4.641 / 0.972
  - α 2.0: 4.158 / 4.305 / 3.834 / 4.337 / 3.496 / 0.973
  - α 2.5: 4.176 / 4.276 / 3.872 / 4.381 / 3.556 / 0.972
  - α 10.0: 3.987 / 3.955 / 3.620 / 4.386 / 3.315 / 0.958

  Tables 3 and 6–8 cover α = 1.0–10.0 for Matcha-TTS (LJ and VCTK), StableTTS and CosyVoice — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)
- **α sweep, CosyVoice "SFM-t" variant (Table 9; encoder takes only speech tokens)**. Values are mean t̃_g / σ̃_g / PMOS / UTMOS / UTMOSv2 / Distill-MOS / WER % / SIM:
  - α 1.0: 0.088 / 0.152 / 3.567 / 3.501 / 2.957 / 4.242 / **14.932** / 0.917
  - α 2.0: 0.176 / 0.304 / 4.000 / 4.089 / 3.392 / 4.520 / 4.882 / 0.924
  - α 2.5: 0.219 / 0.380 / 4.005 / 4.078 / 3.395 / 4.543 / 4.834 / 0.911
  - α 3.0: 0.263 / 0.456 / 3.877 / 3.940 / 3.213 / 4.477 / 4.355 / 0.896
  - α 4.0: 0.349 / 0.605 / 3.395 / 3.498 / 2.599 / 4.087 / 4.355 / 0.868
  - α 5–10: about 0.364 / 0.636, PMOS ≈ 3.25–3.26, WER 4.33–4.43, SIM 0.865
  — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)
- **Test-set results at the selected α (Table 4)**. Values are UTMOS / UTMOSv2 / Distill-MOS / WER / SIM / CMOS (SMOS where given); * = significant vs SFM at p < 0.05:
  - Matcha-TTS LJ: baseline 4.186 / 3.692 / 4.282 / **3.308** / 0.971 / −0.48; SFM α=2.5: 4.257 / 3.848 / 4.386 / **3.413** / 0.972 / 0.00
  - Matcha-TTS VCTK: baseline 4.008 / 2.978 / 3.870 / 1.534 / 0.939 / −0.31*; SFM α=3.5: 4.106 / 3.105 / 3.898 / 0.952 / 0.937 / 0.00
  - StableTTS VCTK: ablated 3.328 / 2.958 / 3.929 / 1.798 / 0.932 / −0.34*; SFM α=3.0: 3.516 / 3.020 / 3.953 / 1.745 / 0.933 / 0.00
  - CosyVoice LibriTTS: baseline 4.191 / 3.303 / 4.481 / **3.513** / 0.932 / −0.21*, SMOS 3.47; SFM α=2.0: 4.194 / 3.480 / 4.541 / **3.810** / 0.931 / 0.00, SMOS 3.67
  - CosyVoice-DiT: ablated 3.499 / 3.086 / 4.316 / 3.614 / 0.936 / −0.31*, SMOS 3.15; SFM α=2.5: 3.751 / 3.171 / 4.502 / 3.598 / 0.932 / 0.00, SMOS 3.21
  — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)
- **Speed (Table 5, Matcha-TTS LJ, adaptive solvers)**:
  - Heun(2): ablated RTF 0.401 (NFE 306.72); SFM α=2.0 RTF 0.299 (−25.541%, NFE 229.43); α=5.0 RTF 0.157 (−60.825%, NFE 120.07)
  - Dopri(5): ablated RTF 0.145; α=2.0 RTF 0.115 (−20.484%); α=5.0 RTF 0.076 (−47.605%)
  - Tables 10–11: StableTTS VCTK Heun RTF 2.153 → 0.586 at α=5.0; CosyVoice LibriTTS Heun RTF 2.532 → 0.960 at α=5.0
  — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)
- **Guarantees**: Theorem 1 shows the transformed output distribution varies continuously in t_m and σ_m under the Wasserstein-2 metric. Theorem 2 shows the two-segment piecewise flow matches the CondOT flow. Neither gives a quality or risk guarantee, and the paper says WER/SIM improvements are "more variable" — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)
- **Stated limitations (Sec. 7)**: the SFM head is simple, and a "more powerful SFM head could be used when the weak generator is unstable". Flow conditions could be enriched. For CosyVoice, the "naive fusion strategy may negatively affect cross-modal alignment". F5-TTS is not a backbone and is cited only for its evaluation protocol — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)

### Inferences
- SFM already does per-utterance start-time selection. A thesis cannot claim to be the first per-input handover in speech flow matching. It can claim differences that SFM does not have:
  - training-free selection (SFM trains a head and retrains the flow module);
  - switching between two pretrained models (teacher → student) rather than from a coarse predictor into the same FM model;
  - a calibrated risk/quality guarantee (SFM has none);
  - selection driven by a quality/risk target rather than a regression to a projection coefficient.
- The handover direction is reversed relative to the thesis. In SFM (and DiffSinger/DiffGAN-TTS), a cheap coarse model goes first and the expensive iterative model finishes. In the thesis, the expensive guided teacher goes first and the cheap student finishes. The "where to hand over" analogy holds, but quality evidence does not transfer directly.
- The CosyVoice SFM-t sweep shows non-monotone, metric-dependent behaviour. Too-early handover (α = 1) gives WER 14.932, and too-late handover lowers PMOS and SIM. This supports treating the handover point as a risk-sensitive choice and reporting WER separately from MOS proxies.

### Gaps
- No per-utterance distribution of t̂_h is reported, so the actual spread of per-input start times is unknown.
- Whether SFM models were trained from scratch or fine-tuned from existing checkpoints is not stated.
- The full α sweep tables for StableTTS and the original CosyVoice (Tables 6–8) were only partly extracted.

---

## Q4. Other speech works with a configurable handover / start step: ResGrad, PriorGrad, Grad-TTS, ProDiff, Fast Grad-TTS, Grad-StyleSpeech, DiffProsody, SpecDiff-GAN, DOSE and other SE "start-from-intermediate" works, coarse-to-fine TTS (CosyVoice, Seed-TTS), and others

### Takeaway
Most of these works do not hand over at an intermediate step. They start the full reverse process from a data-dependent prior (Grad-TTS, Grad-StyleSpeech, PriorGrad, Bridge-TTS, EdiTTS, StoRM), or they learn a residual from noise (ResGrad).

Explicit intermediate start parameters, with sweeps, appear mainly in speech enhancement:
- DOSE: τ1, τ2, grid-searched on {1, 5, …, 50}, with fixed vs hand-picked compared.
- BBED: t_rs = 0.5 vs 0.999.
- CRP: t_rsp = 0.5, fixed.
- Thunder and CTFSE: start points set by fixed hyperparameters.

Community singing voice conversion (SVC) code (so-vits-svc, DDSP-SVC, openvpi DiffSinger) exposes k_step / t_start as user flags. Only DOSE's paper explicitly names per-sample selection as future work. AD-FlowTSE (target speaker extraction) claims per-input, mixing-ratio-aware initialisation, verified at abstract level only.

### Cited Findings

**Residual / refinement diffusion on top of an acoustic model**
- **ResGrad (Chen et al., arXiv 2212.14518; arXiv comment "13 pages, 5 figures", so no venue stated)**:
  - Mechanism: the diffusion model learns the residual (ground-truth mel − FastSpeech 2 output). Inference starts "from standard Gaussian noise", not from an intermediate step, and the generated residual is added to the FastSpeech 2 mel. It uses a small Grad-TTS-style U-Net (2.03M params).
  - Steps evaluated: only 4 (ResGrad-4) and 50 (ResGrad-50), with no per-input selection.
  - Table 1 (V100 RTF), MOS / RTF on LJSpeech, LibriTTS, VCTK:
    - FastSpeech 2: 3.29/0.003, 2.58/0.003, 3.14/0.004
    - ResGrad-4: 4.07/0.018, 3.00/0.020, 3.78/0.022
    - ResGrad-50: 4.14/0.169, 3.41/0.223, 3.93/0.244
    - GradTTS-50: 4.14/0.193, 2.38/0.259, 3.85/0.292
  - Ablations (CMOS): no GT pitch in the residual −0.17 (Table 2); ResUNet −0.18 (Table 3); ResGrad-L +0.09 on LibriTTS (Table 4) and +0.02 on VCTK (Table 5).
  — [ResGrad, ar5iv](https://ar5iv.labs.arxiv.org/html/2212.14518)

**Prior-start diffusion (full reverse process from a data-dependent prior, with no intermediate start step)**
- **Grad-TTS (Popov et al., arXiv 2105.06337; ICML 2021, PMLR vol. 139, pp. 8599–8608 per the Fast Grad-TTS reference list)**:
  - Prior: the terminal condition is X_T ∼ N(μ, I) with μ from the encoder. Inference uses N(μ, τ⁻¹I) with τ = 1.5 for all models. No τ ablation and no start-time/intermediate-start ablation are reported.
  - Table 2, steps N with RTF and MOS: N=1000 RTF 3.663, MOS 4.44 ± 0.05; N=100 RTF 0.363, MOS 4.38 ± 0.06; N=10 RTF 0.033, MOS 4.38 ± 0.06; N=4 RTF 0.012, MOS 3.96 ± 0.07. Glow-TTS MOS 4.11, GT 4.53.
  - Table 1 (preference test of N(μ, I) start vs N(0, I) start): at N=10, N(0, I) was worse 93.8%, identical 0.5%, better 5.7%; at N=20, 82.3/2.9/14.8; at N=50, 60.3/5.7/34.0. Reported as p < 0.005 (sign test) for N=10.
  — [Grad-TTS, ar5iv](https://ar5iv.labs.arxiv.org/html/2105.06337); venue per [Fast Grad-TTS paper, ref. 6](https://www.hse.ru/data/2025/11/09/135179301/Fast%20Grad-TTS%20Towards%20Efficient%20Diffusion-Based%20Speech%20Generation%20on%20CPU.pdf)
- **Fast Grad-TTS (Vovk et al., Interspeech 2022, pp. 838–842)**:
  - Methods: an ML-SDE solver, DDIM progressive distillation, a denoising-diffusion GAN (GAN-DPM), and an LSGM with an NVAE latent DPM whose prior is N(z̄¹, I) with a trainable global geometric-mixture weight α ∈ [0,1]. No method starts at an intermediate time.
  - Original Grad-TTS reference setting: ODE, first-order Euler, prior N(X̄, τ⁻¹I) with τ = 1.5, step 0.1, so 10 steps.
  - Table 1 (Intel 2.60 GHz CPU RTF), MOS / RTF / size:
    - Grad-TTS-Vanilla-10: 4.00 ± 0.09 / 0.68 / 14.8m
    - Grad-TTS-ML-4: 3.93 ± 0.09 / 0.28 / 14.8m
    - Grad-TTS-Distilled-2: 3.82 ± 0.09 / 0.15 / 14.8m
    - GAN-DPM-4: 3.71 ± 0.10 / 0.65 / 64.1m
    - LSGM-NVAE-ML-4: 3.36 ± 0.11 / 0.18 / 24.1m
    - FastSpeech2: 3.58 ± 0.11 / 0.02
    - Glow-TTS: 3.35 ± 0.10 / 0.03
    - GT: 4.38 ± 0.07
  — [Fast Grad-TTS PDF](https://www.hse.ru/data/2025/11/09/135179301/Fast%20Grad-TTS%20Towards%20Efficient%20Diffusion-Based%20Speech%20Generation%20on%20CPU.pdf); [ISCA archive](https://www.isca-archive.org/interspeech_2022/vovk22_interspeech.html)
- **Grad-StyleSpeech (Kang et al., arXiv 2211.09383; ICASSP 2023 per the arXiv comment)**: reverse diffusion starts from N(μ, I) with μ from a style-adaptive hierarchical transformer encoder (a "data-driven prior noise distribution"). It uses 100 steps with the ML-SDE solver. No ablation over start time, prior, temperature or steps is reported — [Grad-StyleSpeech, ar5iv](https://ar5iv.labs.arxiv.org/html/2211.09383)
- **PriorGrad (Lee et al., ICLR 2022, arXiv 2106.06406)**:
  - Prior: data-dependent N(μ, Σ). For the vocoder, μ = 0 and Σ comes from frame energy, clipped at std 0.1. For the acoustic model, the per-phoneme mean and variance come from MFA-aligned training frames. Reverse diffusion starts at the full T (Algorithm 2), not an intermediate step.
  - Vocoder (Table 7, 1M steps), DiffWave/PriorGrad MOS: 4.10/4.20 at T_infer 6; 4.15/4.29 at 12; 4.19/4.33 at 50.
  - Acoustic model (Table 8, HiFi-GAN), Baseline/PriorGrad MOS: T_infer 2: 2.80/4.25 (RTF 0.0069); 6: 3.67/4.29 (0.0113); 12: 4.14/4.39 (0.0176).
  - Theory: Proposition 2 shows a lower or equal minimum loss under the data-dependent prior, assuming a linear ε_θ and det Σ = det I. This is not a quality guarantee.
  — [PriorGrad, ar5iv](https://ar5iv.labs.arxiv.org/html/2106.06406)
- **Bridge-TTS (Chen et al., arXiv 2312.03491; venue UNVERIFIED)**:
  - Prior: a Schrödinger bridge from a clean, deterministic prior (the text-encoder latent z = E(y)), whereas Grad-TTS uses N(z, τ_d⁻¹I) with τ_d = 1.5. Sampling always starts at t = 1 from the latent, with no handover parameter.
  - Table 2, MOS / RTF: 50 steps Bridge-VP 4.09 ± 0.07 / 0.117 vs Grad-TTS 3.99 ± 0.07 / 0.116; 1000 steps Bridge-gmax 4.07 vs Grad-TTS 3.98.
  - Table 3: 4 steps Bridge-gmax 4.10 ± 0.06 (RTF 0.013) vs Grad-TTS 3.88, FastGrad-TTS 3.87, ResGrad 4.02; 2 steps Bridge 4.04 vs CoMoSpeech 3.87.
  - Table 4 (CMOS at NFE 4 / 1000): mutable prior −0.13/−0.17; constant g(t) −0.12/−0.14; VP −0.03/−0.08; vanilla SDE −0.07/−0.19; ODE −0.10/+0.00.
  — [Bridge-TTS, ar5iv](https://ar5iv.labs.arxiv.org/html/2312.03491)
- **EdiTTS (Tae, Kim, Kim, arXiv 2110.02584; venue UNVERIFIED)**: editing with coarse perturbations of the Grad-TTS Gaussian prior (pitch-shift kernel, soft-concatenated content chunks) plus masks. Reverse steps are T = 1000. No start-time/noise-level parameter is named or swept, and the algorithms loop from t = T. Results: pitch-shift MOS 3.95 ± 0.07 (Table 1); content replacement WER 9.1129%, CER 3.2187% (Table 2) — [EdiTTS, ar5iv](https://ar5iv.labs.arxiv.org/html/2110.02584)
- **Voice conversion with data-driven priors (abstract-level only)**:
  - DiffVC (Popov et al., arXiv 2109.13821; ICLR 2022 per the Fast Grad-TTS reference list) introduces an ML-SDE solver for fast sampling — [arXiv 2109.13821](https://arxiv.org/abs/2109.13821)
  - Diff-HierVC (Interspeech 2023, oral) uses the converted mel "as a data-driven prior" plus a "masked prior" — [arXiv 2311.04693](https://arxiv.org/abs/2311.04693)
  - DiffSVC (ASRU 2021 per the Fast Grad-TTS reference list) denoises from the forward-process noise, conditioned on PPGs, F0 and loudness — [arXiv 2105.13871](https://arxiv.org/abs/2105.13871)
  - Handover/start-time details were not checked in full text.

**Not handover methods (classified from abstracts only; full text not read)**
- ProDiff (ACM Multimedia 2022): a progressive fast diffusion model that predicts clean data and reduces data variance through knowledge distillation. This is distillation, outside scope — [arXiv 2207.06389](https://arxiv.org/abs/2207.06389)
- DiffProsody (IEEE/ACM TASLP 2024, vol. 32, pp. 2654–2666): a diffusion-based latent prosody generator using denoising diffusion GANs, plus prosody-conditional adversarial training. No handover described in the abstract — [arXiv 2307.16549](https://arxiv.org/abs/2307.16549)
- SpecDiff-GAN (ICASSP 2024): a HiFi-GAN-based vocoder that uses a forward diffusion process with spectrally shaped noise to stabilise GAN training. It is not a stage handover — [arXiv 2402.01753](https://arxiv.org/abs/2402.01753)

**Speech enhancement / extraction: explicit intermediate start points**
- **DOSE (Tai et al., NeurIPS 2023)**:
  - Adaptive prior (Eq. 8): sampling starts at τ < T from p(x_τ|y) = N(√ᾱ_τ·y, (1−ᾱ_τ)I).
  - Two-step sampler (Algorithm 2): choose τ1 and τ2 with τ2 < τ1 ≤ T. Step 1 draws y_τ1 ∼ N(√ᾱ_τ1·y, (1−ᾱ_τ1)I) and computes x̂0 = f_θ(y_τ1, y, τ1). Step 2 draws x̂_τ2 ∼ N(0.5√ᾱ_τ2(x̂0 + y), (1−ᾱ_τ2)I) and computes x̂0 = f_θ(x̂_τ2, y, τ2).
  - Setup: DiffWave architecture, T = 50, linear β ∈ [1e-4, 0.035]. τ1 and τ2 are chosen on a validation split (10% of training data).
  — [DOSE, NeurIPS 2023 PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e966a12c2d6307adb8809aaa9acf057-Paper-Conference.pdf)
- **DOSE sweep and results**:
  - τ1 and τ2 were grid-searched over {1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}. Figure 14 shows PESQ/STOI heatmaps, and τ1 > τ2 performs better, except STOI on matched VB, where the spread is negligible (0.930–0.934).
  - Table 3 (fixed = equal-interval vs hand-picked steps), PESQ / STOI% on matched VB and mismatched CHiME-4:
    - fixed 1 step: 2.47 / 93.0; 1.38 / 82.8
    - hand-picked 1 step: 2.50 / 93.4; 1.51 / 86.4
    - fixed 2 steps: 2.48 / 93.1; 1.44 / 83.6
    - hand-picked 2 steps: 2.56 / 93.6; 1.52 / 86.6
  - Table 1 (2 steps): VB STOI 93.6, PESQ 2.56, CSIG 3.83, CBAK 3.27, COVL 3.19. For comparison, SGMSE (50 steps) has PESQ 2.34 and DR-DiffuSE (6 steps) 2.50.
  — [DOSE PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e966a12c2d6307adb8809aaa9acf057-Paper-Conference.pdf)
- **DOSE theory and stated limitation**:
  - Proposition 1 is an existence result: for any tolerance ξ there is a τ such that D_KL(p(x_t|x) ‖ p(x_t|y)) ≤ ξ for all τ ≤ t ≤ T.
  - Proposition 2 is a recovery condition favouring small σ_t² = (1−ᾱ_t)/ᾱ_t. The paper calls this a trade-off: τ must be small for recovery but not too small.
  - Grid-search cost is O(N·T!/(T−K)!).
  - Conclusion (quoted): "These manually defined hyper-parameters are selected based on the Empirical Risk Minimization (ERM) principle and may not be optimal for every individual sample. Thus, an important direction for future research is to develop methods that can adaptively choose hyper-parameters for different samples."
  — [DOSE PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e966a12c2d6307adb8809aaa9acf057-Paper-Conference.pdf)
- **BBED, "Reducing the Prior Mismatch of SDEs for Diffusion-based SE" (Lay, Welker, Richter, Gerkmann; arXiv 2302.14748; venue UNVERIFIED)**:
  - Reverse starting point t_rs: 0.5 vs 0.999 for BBED (T = 0.999), with step size h = T/30 fixed. Starting at 0.5 halves the number of iterations.
  - Table 1 (WSJ0-CHiME3), POLQA / PESQ / ESTOI / SI-SDR: OUVE (t_rs = 1) 3.71 / 2.92 / 0.92 / 17.78; BBED t_rs = 0.5: 3.97 / 3.05 / 0.93 / 18.96; BBED t_rs = 0.999: 4.01 / 3.08 / 0.94 / 19.26.
  - The text says OUVE loses 0.31 PESQ at t_rs = 0.5. The start point is fixed, and there is no guarantee.
  — [BBED, ar5iv](https://ar5iv.labs.arxiv.org/html/2302.14748)
- **CRP, "Single and Few-step Diffusion for Generative SE" (Lay, Lemercier, Richter, Gerkmann; arXiv 2309.09677; IEEE copyright notice; exact venue UNVERIFIED)**:
  - The second-stage fine-tuning starts the reverse process at a fixed t_rsp = 0.5 (vs T = 0.999), choosing it from prior work rather than sweeping it. n_steps ∈ {1, …, 5}, with t_ε = 0.03.
  - Table 1 (VBD → WSJ0-C3, mismatched), POLQA / PESQ: generative baseline (60 NFE) 3.61 / 2.36; predictive (1) 3.39 / 2.14; CRP (1) 3.47 / 2.29; CRP (5) 3.65 / 2.55.
  — [CRP, ar5iv](https://ar5iv.labs.arxiv.org/html/2309.09677)
- **StoRM (Lemercier et al., IEEE/ACM TASLP 2023)**: the reverse process starts at T from x_T = D_θ(y) + σ(T)z, centred on the predictive estimate rather than at an intermediate time. The default is N = 50 steps. Fig. 6 sweeps N from 10 to 50, and the text says StoRM stays near its optimum at 10 steps. Compute is 4.5·10¹¹ MAC/s at 20 steps vs 2.1·10¹² for SGMSE+M at 50 steps with a corrector. There is no start-time ablation and no guarantee — [StoRM, ar5iv](https://ar5iv.labs.arxiv.org/html/2212.11851)
- **Thunder (Trachu et al., arXiv 2406.06139; preprint, "will be submitted in the interspeech conference")**:
  - Mechanism: a Brownian-bridge model. Regression mode sets t ≈ 1 and predicts clean speech directly from x₁ = y. The output is a mixture of regression and diffusion outputs with a fixed α = 0.8, "chosen via grid search on a validation set".
  - Table 3 (Thunder L), PESQ / SI-SDR / RTF: N=30 3.02 / 19.4 / 0.538; N=15 3.02 / 19.4 / 0.284; N=1 2.99 / 19.6 / 0.038.
  — [Thunder, ar5iv](https://ar5iv.labs.arxiv.org/html/2406.06139)
- **Joint generative and predictive decoders (Shi et al., arXiv 2305.10734; venue not stated)**: an "initial fusion" starts diffusion from the predictive decoder's output, and a final fusion combines both decoders. The abstract claims faster decoding and higher PESQ than StoRM and SGMSE+ on VoiceBank. The start step and numbers were not available on the abstract page — [arXiv 2305.10734](https://arxiv.org/abs/2305.10734)
- **CTFSE, "Speech Enhancement based on cascaded two flows" (arXiv 2508.06842; arXiv comment "Accepted at Interspeech 2025")**:
  - Mechanism: a one-step first flow gives D_θ(x₁, y). The second flow starts at t = 1 from N(D_θ, σ²I) with σ = 0.5, not at an intermediate time, and t_δ = 0.03.
  - NFE was fixed per setting, "selected to maximize the average WB-PESQ scores".
  - Example (VB-DMD, Table 1), PESQ / ESTOI / SI-SDR: CTFSE (5 NFE) 3.05 / 0.88 / 19.13; FlowSE (5) 2.98 / 0.87 / 18.97; StoRM (101) 2.90 / 0.87 / 18.48.
  — [CTFSE, arXiv HTML](https://arxiv.org/html/2508.06842)
- **AD-FlowTSE (Hsieh and Kim, arXiv 2510.16995; preprint, venue not stated; target speaker extraction)**: the flow runs between background and source, governed by the mixing ratio (MR). "MR-aware initialization" starts each input at an adaptive point on the trajectory instead of a fixed reverse schedule, and auxiliary MR estimation is added. Abstract-level only; the mechanism and numbers were not read — [arXiv 2510.16995](https://arxiv.org/abs/2510.16995)
- **Diffiner (Sawata et al., Interspeech 2023 per the arXiv comment)**: a diffusion refiner trained on clean speech only. It uses denoising diffusion restoration to mix newly generated clean parts into degraded parts left by a preceding SE method. Start-step details were not checked — [arXiv 2210.17287](https://arxiv.org/abs/2210.17287)

**Coarse-to-fine LLM + flow TTS**
- **CosyVoice (arXiv 2407.05407; comment "work in progress")**: an LLM generates supervised semantic tokens, and a conditional flow-matching model reconstructs the speech (abstract) — [arXiv 2407.05407](https://arxiv.org/abs/2407.05407). The SFM paper treats CosyVoice's flow module as a coarse-to-fine FM module that conventionally uses coarse representations "as conditions" rather than as start states. SFM retrains only the flow module (Table 4: WER 3.513 → 3.810, SMOS 3.47 → 3.67) — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)
- **Seed-TTS (arXiv 2406.02430)**: the abstract describes autoregressive TTS models, a self-distillation method for speech factorisation, an RL method, and a further (non-autoregressive) variant. The abstract shows no handover or start-step mechanism; the full text was not read — [arXiv 2406.02430](https://arxiv.org/abs/2406.02430)

**Official / community code with a user-set handover step (singing voice conversion)**
- **so-vits-svc (svc-develop-team)**: shallow diffusion on top of the so-vits output is enabled with `-shd`. `-ks/--k_step` defaults to 100, and "the larger the number of k_steps, the closer it is to the result of the diffusion model". `timesteps` defaults to 1000. Training can be limited to `k_step_max` steps, but then `only_diffusion` cannot be used. Shallow diffusion "can solve some electrical sound problems". `--second_encoding` "can yield varying results - sometimes positive and sometimes negative". There are no numbers and no per-input selection — [so-vits-svc README](https://github.com/svc-develop-team/so-vits-svc)
- **DDSP-SVC (yxlllc)**: the DDSP output is enhanced by "a shallow diffusion model (later versions) or a rectified-flow model (current version)". `t_start` is "the start time point of ODE … recommended to keep it equal [to the config value] (the default is 0.0)". `infer_step` and `method` (euler/rk4) are run-time options. There are no quality or speed numbers — [DDSP-SVC README](https://github.com/yxlllc/DDSP-SVC)
- **openvpi DiffSinger**: see Q1 (`T_start` example 0.4, `K_step`; config-fixed) — [BestPractices.md](https://github.com/openvpi/DiffSinger/blob/main/docs/BestPractices.md)

### Inferences
- **Two kinds of "prior-start"**. Grad-TTS-family priors make the start state data-dependent but keep the start time at T. DiffSinger-family, DOSE, BBED and CRP make the start time a tunable hyperparameter. The thesis's switching parameter (the step at which the student takes over the teacher's trajectory) is the second kind. Grad-TTS's Table 1 is evidence that a better-informed start state matters most when the step budget is small (93.8% preference at N = 10, falling to 60.3% at N = 50).
- **Results are mixed across tasks**. Quality as a function of the handover point is not universally monotone:
  - DOSE finds τ1 > τ2 best, and hand-picked beats equal-interval by up to 0.13 PESQ (mismatched 1-step).
  - BBED finds t_rs = 0.5 costs about 0.03 PESQ vs 0.999, while OUVE loses 0.31.
  - The cost of a later or earlier handover depends on how well the prior/start state matches the target. This is the per-input mismatch that a per-input switch would exploit.
- **The SE literature is the most explicit** about the handover point being selected on a validation set (ERM). DOSE's conclusion is the clearest published statement that a per-sample choice is an open problem.

### Gaps
- DOSE's actual selected τ1/τ2 values per dataset (Appendix A.10) were not read.
- StoRM's Fig. 6 values and the CRP/Thunder figure values were not available as numbers.
- No per-step numbers were found for Shi et al. 2305.10734.
- The composite predictive-generative USE paper ([arXiv 2505.24576](https://arxiv.org/abs/2505.24576), TASLP per the arXiv comment) was found only at abstract level; its initialisation/start-step details are unverified.
- Full text was not read for CosyVoice, Seed-TTS, ProDiff, DiffProsody, SpecDiff-GAN, DiffVC, Diff-HierVC, DiffSVC or Diffiner. Their classification as "not a step handover" rests on abstracts.
- Venues are UNVERIFIED for Bridge-TTS, EdiTTS, BBED (2302.14748), CRP (2309.09677), ResGrad and Shi et al.
- A web search found no primary source showing a Seed-TTS or CosyVoice refinement stage that starts from an intermediate time.

---

## Q5. Which works choose the switch per input, which fix one value, and does any give a quality/risk guarantee?

### Takeaway
Among the works verified here, only SFM (TTS, trained head) chooses the handover point per input. AD-FlowTSE (target speaker extraction) also claims to, verified at abstract level only.

Every other work fixes the handover step or start time per model and dataset, usually by validation search or a heuristic: DiffSinger k, DiffGAN-TTS t = 1, DOSE τ1/τ2, BBED/CRP t_rs = 0.5, Thunder α, CTFSE NFE, and CLI k_step/t_start in community SVC code.

No work gives a calibrated quality or risk guarantee for the handover. The theory that exists is an existence or continuity result (DOSE Prop. 1/2, SFM Thm. 1/2) or a loss-bound under linear assumptions (PriorGrad Prop. 2).

### Cited Findings
Summary table (H = handover parameter):

| Work (venue) | H | Values evaluated | Per-input? | Guarantee? |
|---|---|---|---|---|
| DiffSinger/DiffSpeech (AAAI 2022) | shallow step k (T=100 SVS) | k=54, 25, none (Table 2); k=70 LJSpeech | No: BP picks one k per dataset (average of per-mel k′) | None ([ar5iv](https://ar5iv.labs.arxiv.org/html/2105.02446)) |
| DiffGAN-TTS (arXiv preprint; ICML'22 workshop) | T denoising steps; two-stage handover at t=1 | T=1,2,4; two-stage | No: fixed t=1 | None ([ar5iv](https://ar5iv.labs.arxiv.org/html/2201.11972)) |
| SFM (NeurIPS 2025) | start time t̃_h on the CondOT path; global α | α 1.0–10.0 (step 1.0, ±0.5 refinement) | **Yes**: per-utterance t̂_h from a trained head; α global | Continuity/equivalence theorems only ([arXiv HTML](https://arxiv.org/html/2505.12226)) |
| ResGrad (arXiv) | steps on residual from noise | 4, 50 | No | None ([ar5iv](https://ar5iv.labs.arxiv.org/html/2212.14518)) |
| Grad-TTS (ICML 2021) | prior N(μ, τ⁻¹I) at T; τ=1.5; N steps | N=4,10,100,1000; N(0,I) vs N(μ,I) | Prior mean per input; start time fixed | None ([ar5iv](https://ar5iv.labs.arxiv.org/html/2105.06337)) |
| PriorGrad (ICLR 2022) | data-dependent N(μ,Σ) prior at T | T_infer 2/6/12 (acoustic); 6/12/50 (vocoder) | Prior per input; start time fixed | Loss bound under linear ε_θ (Prop. 2) ([ar5iv](https://ar5iv.labs.arxiv.org/html/2106.06406)) |
| Bridge-TTS (arXiv) | deterministic prior at t=1; schedule/sampler | 1000/50/4/2 steps; gmax/VP/constant; SDE/ODE | Prior per input; start fixed | None stated in extraction ([ar5iv](https://ar5iv.labs.arxiv.org/html/2312.03491)) |
| DOSE (NeurIPS 2023) | τ1, τ2 (two-step start points) | grid {1,5,…,50}²; fixed vs hand-picked | No: ERM on a validation split; per-sample choice named as future work | Prop. 1 (existence of τ with KL ≤ ξ), Prop. 2 (recovery condition); no calibration ([PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e966a12c2d6307adb8809aaa9acf057-Paper-Conference.pdf)) |
| BBED (arXiv 2302.14748) | reverse start t_rs | 0.5, 0.999 | No | None ([ar5iv](https://ar5iv.labs.arxiv.org/html/2302.14748)) |
| CRP (arXiv 2309.09677) | t_rsp; n_steps | t_rsp=0.5 fixed; n_steps 1–5 | No | None ([ar5iv](https://ar5iv.labs.arxiv.org/html/2309.09677)) |
| StoRM (TASLP 2023) | start mean = predictor output at T; N | N 10–50 | Start mean per input; time fixed | None ([ar5iv](https://ar5iv.labs.arxiv.org/html/2212.11851)) |
| Thunder (preprint) | regression/diffusion mix α; N | α=0.8; N=30,15,1 | No: grid search on validation | None ([ar5iv](https://ar5iv.labs.arxiv.org/html/2406.06139)) |
| CTFSE (Interspeech 2025) | second flow start from first-flow estimate; NFE | NFE per setting | No: NFE chosen to maximise average WB-PESQ | None ([arXiv HTML](https://arxiv.org/html/2508.06842)) |
| AD-FlowTSE (preprint) | MR-aware initialisation point | not read | **Yes (abstract claim)** | Not read ([arXiv](https://arxiv.org/abs/2510.16995)) |
| so-vits-svc / DDSP-SVC / openvpi DiffSinger (code) | k_step (default 100 of 1000) / t_start (default 0.0) / T_start (example 0.4) | user flags | No: CLI/config | None ([so-vits-svc](https://github.com/svc-develop-team/so-vits-svc), [DDSP-SVC](https://github.com/yxlllc/DDSP-SVC), [openvpi](https://github.com/openvpi/DiffSinger/blob/main/docs/BestPractices.md)) |

- **DiffSinger** computes a per-mel k′ but averages it to one dataset-level k. Appendix B allows manual brute-force validation search instead — [DiffSinger, ar5iv](https://ar5iv.labs.arxiv.org/html/2105.02446)
- **DOSE** states that its hyperparameters "are selected based on the Empirical Risk Minimization (ERM) principle and may not be optimal for every individual sample" and calls adaptive per-sample choice "an important direction for future research" — [DOSE PDF](https://proceedings.neurips.cc/paper_files/paper/2023/file/7e966a12c2d6307adb8809aaa9acf057-Paper-Conference.pdf)
- **SFM** gives a per-utterance start time, but its α is global and tuned by mean validation PMOS. Its theory covers continuity and equivalence, not quality or risk. The paper says WER/SIM gains are "more variable" — [SFM, arXiv HTML](https://arxiv.org/html/2505.12226)

### Inferences
- The verified literature supports the thesis's positioning, narrowed as follows. A training-free, per-input, risk-calibrated handover between two pretrained generators appears to be unaddressed. Per-input start-time selection alone is not new: SFM (trained, NeurIPS 2025) and the AD-FlowTSE abstract already do it.
- No source here offers a distribution-free or calibrated guarantee (conformal, risk-control style) on WER/SIM/MOS at the handover. The only "guarantee-like" statements are DOSE's existence propositions and SFM's continuity theorems.
- Across DiffSinger (CMOS), SFM (WER/PMOS/SIM) and DOSE (PESQ/STOI), quality and speed trade off against the handover point, and the trade-off differs by metric. A risk-calibrated switch should therefore control a specific per-utterance risk such as WER, rather than a single naturalness proxy.

### Gaps
- None of the verified sources reports a per-utterance distribution of the optimal handover point, which would show how much a per-input policy could gain. SFM reports only means, DiffSinger only the dataset average, and DOSE only aggregate grids.
- AD-FlowTSE's per-input mechanism and results were not verified beyond the abstract.
- No speech work was found that switches from a guided teacher to a distilled student at a chosen step. By assignment, other researchers cover that topic. This search did not encounter one among shallow/two-stage works.
