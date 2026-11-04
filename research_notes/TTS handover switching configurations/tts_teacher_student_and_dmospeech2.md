# TTS teacher-student trajectory mixing, few-step TTS step/schedule trade-offs, and DMOSpeech 2's switching time

Research date: 2026-11-02. Sources were read in full text (arXiv HTML tables parsed locally, official GitHub source code downloaded and read) unless marked otherwise. "Unverified" means it was not confirmed from a primary source in this session.

## Q1. Which TTS/speech works run a teacher (or full multi-step model) for some steps and a distilled/few-step student for the rest, or otherwise mix models along one sampling trajectory?

### Takeaway
Within the searched scope, DMOSpeech 2's "teacher-guided sampling" is the only TTS/speech work found that runs a full teacher (F5-TTS with CFG) for the early part of one trajectory and a distilled student (DMD-trained DMOSpeech 2) for the rest. It uses one fixed, global switch and evaluates only one configuration. The closest analogues are image-domain: T-Stitch (small model first, then large) and Adaptive Teacher-Student Collaboration (student first, then a per-input oracle decides whether to refine with the teacher). No speech paper was found that chooses the switch per input.

### Cited Findings
- DMOSpeech 2 introduces "teacher-guided sampling, a hybrid approach leveraging a teacher model for initial denoising steps before transitioning to the student model". It is motivated by "mode shrinkage" in DMD-distilled students. — [DMOSpeech 2, arXiv 2507.14988v1, Abstract and §3.3](https://arxiv.org/html/2507.14988v1)
- The switch parameter is t_switch, with K teacher steps and M student steps. Algorithm 2 finds k* such that t_{k*} ≤ t_switch < t_{k*+1} on a sway-sampled teacher grid, runs the teacher (Euler, with CFG strength λ) for k = 1..k*, then sets the student's first time s_1 = t_{k*} with x_{s_1} ← y_{k*} and s_M = 1. Later student steps re-noise the student's x̂_1 with fresh ε. The switch is fixed and global, not chosen per input. — [DMOSpeech 2, Algorithm 2](https://arxiv.org/html/2507.14988v1)
- Searching the arXiv API for `all:"teacher-guided sampling"` returned 0 records. That search covers title, abstract and comments, not full text. A query combining teacher, student, speech and switch returned only ASR knowledge-distillation papers (2309.08144, 2609.03502). A query for "hybrid sampling" plus speech returned 0. — [arXiv API query](http://export.arxiv.org/api/query?search_query=all:%22teacher-guided%20sampling%22)
- A web search for speech work that runs a large model for early steps and a distilled model for later steps found no speech-specific stitching paper. It surfaced T-Stitch (image) as the closest technique. — [T-Stitch, arXiv 2402.14167](https://arxiv.org/abs/2402.14167)
- T-Stitch (ICLR 2025, image domain) "first leverages a smaller DPM in the initial steps as a cheap drop-in replacement of the larger DPM and switches to the larger DPM at a later stage". It is training-free, and on DiT-XL "40% of the early timesteps can be safely replaced with a 10x faster DiT-S without performance drop". The handover direction is the reverse of DMOSpeech 2 (cheap model first). — [T-Stitch abstract](https://arxiv.org/abs/2402.14167); [ICLR 2025 proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/120339238f293d4ae53a7167403abc4b-Paper-Conference.pdf)
- "Your Student is Better Than Expected" (CVPR 2024, text-to-image) proposes "adaptive collaboration between student and teacher". The distilled model "produces the initial sample, and then an oracle decides whether it needs further improvements with a slow teacher model". This is a per-input decision, but it decides whether to call the teacher after the student, not a teacher-to-student switch time. — [arXiv 2312.10835 (CVPR 2024 camera ready)](https://arxiv.org/abs/2312.10835)
- BELLE ("Bayesian Speech Synthesizers Can Learn from Multiple Teachers", arXiv 2510.24372) learns from multiple TTS "teachers" through synthetic parallel data. From its table of contents, it does not mix models along a sampling trajectory, so it is excluded here. — [arXiv 2510.24372](https://arxiv.org/abs/2510.24372)
- Other distilled TTS systems run the student alone at inference, with no teacher segment: CoMoSpeech (1-step consistency distillation), E1 TTS (1-step DMD), ProDiff (2-step progressive distillation), IntMeanFlow and LFMD (1–4-NFE flow-map students), and DMOSpeech v1 (4-step DMD student). FlashSpeech trains a latent consistency model "without the need for a pre-trained diffusion model as the teacher". — [CoMoSpeech](https://arxiv.org/abs/2305.06908); [E1 TTS](https://arxiv.org/abs/2409.09351); [ProDiff](https://arxiv.org/abs/2207.06389); [IntMeanFlow](https://arxiv.org/abs/2510.07979); [LFMD](https://arxiv.org/abs/2609.36324); [DMOSpeech](https://arxiv.org/abs/2410.11097); [FlashSpeech](https://arxiv.org/abs/2404.14700)

### Inferences
- A thesis on per-input, risk-calibrated teacher-to-student switching in TTS has no direct speech prior art in the searched literature. The nearest per-input precedent is image-domain and post-hoc (student, then oracle, then teacher), not a switching-time selector.
- Every TTS trajectory-mixing configuration found is global and fixed, and only one configuration (DMOSpeech 2's 16-step setting) has published metrics.

### Gaps
- Full-text search across all speech venues (Interspeech, ICASSP, SLT and ASRU proceedings) was not possible. The arXiv API only searches metadata. A speech paper that mixes models along a trajectory without saying so in its abstract could have been missed.
- Image and video model-switching work is out of scope here. The adjacent note `image_video_model_switching.md` in this folder covers it.

## Q2. DMOSpeech 2: what is evaluated about t_switch, what the official code and demo actually do, and who has reproduced, analyzed or cited the teacher-guided sampling

### Takeaway
Confirmed. The paper evaluates exactly one teacher-guided configuration ("t_switch=0.25, with teacher handling 14 steps and student handling 2 steps, for a total of 16 steps"). It recommends 0.4–0.5 for diversity and 0.1–0.2 for efficiency without reporting any data for those ranges. The code defaults are teacher_steps=16, teacher_stopping_time=0.07 and student_start_step=1. The demo's "16 steps" cell uses teacher_steps=24, teacher_stopping_time=0.3 and student_start_step=2. The released code also differs from Algorithm 2: the student re-noises the teacher's x̂₁ estimate to a fixed student time instead of continuing from the teacher state. No paper, issue, fork or blog post was found that evaluates or analyzes any other switching time. None of the 10 citing papers indexed by Semantic Scholar discusses teacher-guided sampling.

### Cited Findings
**Paper, bibliographic**
- arXiv 2507.14988 has a single version (v1, submitted 20 Jul 2025). Authors: Yinghao Aaron Li, Xilin Jiang, Fei Tao, Cheng Niu, Kaifeng Xu, Juntong Song and Nima Mesgarani (Columbia University and NewsBreak). — [arXiv abs](https://arxiv.org/abs/2507.14988); [arXiv HTML v1](https://arxiv.org/html/2507.14988v1)
- Published in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40 (issue 38), pp. 31814–31822, 2026, DOI 10.1609/aaai.v40i38.40450. — [OpenAlex W7138217505](https://api.openalex.org/works?search=DMOSpeech%202%20Reinforcement%20Learning%20for%20Duration%20Prediction); [AAAI OJS page](https://ojs.aaai.org/index.php/AAAI/article/view/40450)

**Paper, what is and is not evaluated (arXiv v1)**
- §4.1, baseline (5): "a teacher-guided sampling approach where the teacher model handles initial denoising steps before transitioning to the student model (t_switch=0.25, with teacher handling 14 steps and student handling 2 steps, for a total of 16 steps)". The teacher is F5-TTS (~300M params, 2M steps on Emilia, ~95k h EN+ZH). The evaluation uses Seed-TTS test-en (1088 samples) and test-zh (2020 samples). — [DMOSpeech 2 §4.1](https://arxiv.org/html/2507.14988v1)
- §3.3.2: the teacher is used for "approximately 6-14 steps instead of 32", and the student for "2-3 instead of 4". "For applications where diversity is critical … a higher t_switch value (around 0.4-0.5) can be used… for applications where efficiency is paramount… a lower t_switch value (around 0.1-0.2) can be employed with minimal degradation in perceptual quality." "The parameters K, t_switch, and M offer flexible control over the trade-off." No table or figure reports any other t_switch, K or M. — [DMOSpeech 2 §3.3.2](https://arxiv.org/html/2507.14988v1)
- Table 1 (Seed-TTS; CMOS anchor = DMOSpeech 2 4-step, where a negative value means DMOSpeech 2 is better; "CV_f0 is computed with the DDPM sampler for fairness"). Columns are WER-en / SIM-en / CER-zh / SIM-zh / CMOS-N-en / CMOS-S-en / CMOS-N-zh / CMOS-S-zh / CV_f0 / RTF:
  - F5-TTS Teacher (32 steps): 1.947 / 0.662 / 1.695 / 0.750 / −0.12* / −0.04 / −0.09 / −0.11* / 0.6659 / 0.1671
  - DMOSpeech 2 (4 steps): 1.752 / 0.698 / 1.527 / 0.760 / 0.0 / 0.0 / 0.0 / 0.0 / 0.4640 / 0.0316
  - w/o duration predictor RL: 3.750 / 0.672 / 2.000 / 0.750 / −0.43** / −0.48** / −0.26* / −0.31* / S/A / S/A
  - Teacher-Guided (16 steps): 1.738 / 0.699 / 1.468 / 0.760 / 0.01 / −0.03 / 0.45** / 0.3* / 0.5932 / 0.0941
  
  — [DMOSpeech 2 Table 1](https://arxiv.org/html/2507.14988v1)
- §4.3.3: the student's CV_f0 is "a 30.3% decrease" from the teacher's. Teacher-guided sampling "recovers much of this diversity (0.5932, 89.1% of teacher's diversity)". Figure 3a shows F0 distributions. Figure 3b shows mean-centered UTMOS distributions that "remain consistent across all models", which the authors read as diversity loss being "primarily in prosodic aspects". The text says teacher-guided sampling is "1.8× faster than the full teacher model" (0.1671/0.0941 ≈ 1.78). — [DMOSpeech 2 §3.3.2, §4.3.3](https://arxiv.org/html/2507.14988v1)
- Table 2 lists "DMOSpeech 2 (Teacher-Guided, 16 steps)" at 0.6B params (teacher plus student) with RTF 0.094, against 0.3B params and RTF 0.032 for student-only. — [DMOSpeech 2 Table 2](https://arxiv.org/html/2507.14988v1)
- Future work: "applying our targeted RL approach to … the teacher model in the hybrid sampling approach". — [DMOSpeech 2 §5](https://arxiv.org/html/2507.14988v1)

**Official code (github.com/yl4579/DMOSpeech2, `src/infer.py`; all 3 commits to this file are dated 2025-07-21)**
- `generate(...)` defaults: `teacher_steps=16, teacher_stopping_time=0.07, student_start_step=1, duration=None, dp_softmax_range=0.7, temperature=0, eta=1.0, cfg_strength=2.0, sway_coefficient=-1.0`. — [infer.py](https://github.com/yl4579/DMOSpeech2/blob/main/src/infer.py)
- The teacher grid is `t = linspace(0,1,teacher_steps+1)`, mapped with sway `t + s*(cos(pi/2*t) - 1 + t)`, then truncated with `t = t[:(t > teacher_stopping_time).float().argmax() + 2]; t = t[:-1]`. It is integrated by Euler `odeint`, and then one extra teacher call forms the clean estimate `test_out = trajectory[-1] + (1 - t[-1]) * pred`. Each teacher call uses CFG (`pred + (pred - null_pred) * cfg_strength`), so it costs two forward passes. With `eta>0` (default 1.0) noise is added to the velocity ("hybrid DDIM/DDPM"). — [infer.py `_teacher_sampling`](https://github.com/yl4579/DMOSpeech2/blob/main/src/infer.py)
- The student schedule is fixed: `steps = [0, 0.25, 0.5, 0.75]`, sway-mapped, then `steps[student_start_step:]`. At every student step (including the first) the input is re-noised with fresh noise, `phi = (1 - t) * x0 + t * x1` where `x0 = randn_like(x1)`, and the output is `phi + (1 - t) * pred`. The student's entry time is therefore set by `student_start_step`, independently of `teacher_stopping_time`. — [infer.py `_student_sampling`](https://github.com/yl4579/DMOSpeech2/blob/main/src/infer.py)
- `generate_teacher_only` defaults to `teacher_steps=32` with stopping time 1.0. — [infer.py](https://github.com/yl4579/DMOSpeech2/blob/main/src/infer.py)
- `grpo_duration_trainer.py` calls generation with `teacher_steps=0`, so GRPO rollouts use the student alone. — [DMOSpeech2 repo src/](https://github.com/yl4579/DMOSpeech2/tree/main/src)

**Official demo notebook (`src/demo.ipynb`), "Comparision between different sampling configurations"**
- "Student only (4 steps)": `teacher_steps=0, student_start_step=0`. The notebook says: "Need to set `teacher_steps` and `student_start_step` to 0 to enable full student sampling." — [demo.ipynb](https://github.com/yl4579/DMOSpeech2/blob/main/src/demo.ipynb)
- "More teacher steps (16 steps)": "Now we use 14 steps from the teacher and 2 steps from the student to have higher diversity (16 steps total)", with `teacher_steps=24, teacher_stopping_time=0.3, # 0.25 means students go for the last two steps (0.26ish, 0.6ish)` and `student_start_step=2`. — [demo.ipynb](https://github.com/yl4579/DMOSpeech2/blob/main/src/demo.ipynb)
- "Stochastic duration": `teacher_steps=24, teacher_stopping_time=0.25, student_start_step=2, temperature=0.8`. — [demo.ipynb](https://github.com/yl4579/DMOSpeech2/blob/main/src/demo.ipynb)

**Effective schedules derived from the official code.** I re-implemented the truncation and sway logic in Python and did not run the model. Teacher calls below include the final x̂₁ call, and each is ×2 forward passes with CFG.
- Code default (16, 0.07, 1): 4 Euler steps, so 5 teacher calls ending at t = 0.0761. The student then runs 3 steps at t = 0.0761, 0.2929, 0.6173, for 8 network calls in total.
- Demo (24, 0.3, 2): 13 Euler steps, so 14 teacher calls ending at t = 0.3407. The student runs 2 steps at t = 0.2929 and 0.6173, for 16 calls in total. This is the configuration that matches the paper's "14 teacher + 2 student".
- Demo (24, 0.25, 2): 12 Euler steps, so 13 teacher calls ending at t = 0.2929, plus 2 student steps, for 15 calls in total.
- Other settings: (16, 0.25, 2) gives 9 + 2 calls; (32, 0.25, 2) gives 16 + 2 calls, with the teacher ending at t = 0.2590.

  — derived from [infer.py](https://github.com/yl4579/DMOSpeech2/blob/main/src/infer.py)

**Repository activity, issues and forks (queried via GitHub REST API on 2026-11-02)**
- 305 stars and 39 forks. The last push to the upstream repository was 2025-07-22. — [GitHub API repo](https://api.github.com/repos/yl4579/DMOSpeech2)
- 18 issues and PRs (#1–#18). Titles cover Colab, gradio UI, training code, long text, a 12 s limit, an Italian model, CTC/SV weights, vocoder path, and automated "[LIBX] Security dependency upgrades". None concerns switching time, teacher steps or diversity. — [Issues](https://github.com/yl4579/DMOSpeech2/issues?q=is%3Aissue)
- Of the 35 forks returned by the API, every fork with a readable `infer.py` keeps the defaults (16, 0.07, 1). The restructured forks JarodMica, eschmidbauer and v3ucn keep the same `infer.py` defaults and the same demo values (24/0.3/2 and 24/0.25/2). wrightmikea adds gradio/FastAPI scripts and keeps the `infer.py` defaults. — [Forks](https://github.com/yl4579/DMOSpeech2/forks); [JarodMica fork](https://github.com/JarodMica/DMOSpeech2)

**Citations (searched 2026-11-02)**
- Semantic Scholar lists 10 citing papers. Their citation contexts concern GRPO duration RL or DMOSpeech-family metric optimization, never teacher-guided sampling or t_switch:
  - Distill Locally, Schedule Globally (arXiv 2609.36324): "DMOSpeech [17] and DMOSpeech 2 [18] optimize perceptual metrics using auxiliary speech models".
  - EmphTTS (2609.27599): "Inspired by DMOSpeech2 … probabilistic duration predictor … as the policy".
  - When Does Predictor-Based RL Align with Human Perception? (2608.31035).
  - VoiceDesigner (2608.13613), which uses the duration predictor.
  - TokAN (2607.03928).
  - GLASS (2606.05889).
  - Int-MeanFlow (2510.07979, ICASSP 2026): "DMOSpeech [22] and DMO-Speech2 [23], reduce NFE in TTS through distillation and reinforcement learning".
  - No Verifiable Reward for Prosody (2509.18531, ICASSP).
  - Synthetic Singers review (2601.13910).

  — [Semantic Scholar citations API](https://api.semanticscholar.org/graph/v1/paper/arXiv:2507.14988/citations?fields=title,year,venue,externalIds,contexts&limit=100)
- OpenAlex reports cited_by_count = 0 for both the arXiv and AAAI records. — [OpenAlex search](https://api.openalex.org/works?search=DMOSpeech%202%20reinforcement%20learning%20duration)
- Web searches found only the paper, its AAAI page and aggregator summaries (papers.cool, emergentmind, liner, takara, sourcepulse). None of them evaluates other switching times. — [papers.cool](https://papers.cool/arxiv/2507.14988); [sourcepulse](https://www.sourcepulse.org/projects/11903397)

### Inferences
- The paper's "t_switch = 0.25" label does not map one-to-one onto the code's `teacher_stopping_time`. The demo setting labelled as the 14+2 = 16-step configuration uses `teacher_stopping_time=0.3`. That teacher ends its grid at sway-time 0.3407, and the student re-enters at fixed sway-time 0.2929, which is slightly noisier than where the teacher stopped. With 0.25 the count is 13+2 = 15. Which exact setting produced the Table 1 row is not stated. This is my derivation from the code; the authors do not say it.
- Algorithm 2 (student continues from y_{k*} at s_1 = t_{k*}) and the released code (teacher x̂₁ prediction, then re-noising with fresh Gaussian noise at a fixed student time) are different handover mechanisms. A thesis reproducing "t_switch" should state which one it implements. The code version adds extra stochasticity at the handover, which may contribute to the recovered diversity.
- The code default of 0.07 (5 teacher calls plus 3 student steps) is a "low-t_switch / efficiency" operating point that the paper never evaluates. Its 0.1–0.2 and 0.4–0.5 recommendations are untested claims in the published record.
- The diversity metric (CV_f0) was computed "with the DDPM sampler", and the code's teacher uses eta = 1.0 (stochastic), so diversity comparisons that use deterministic ODE teacher sampling may not be comparable to Table 1.
- Teacher-guided sampling is reported only as an average over the test sets. No per-utterance analysis exists (for example, which inputs benefit from more teacher steps), which leaves room for per-input switching work.

### Gaps
- The AAAI 2026 camera-ready PDF could not be read: the OJS download returned an Anubis bot-protection page. Whether the camera-ready adds a t_switch, K or M ablation is unverified. The arXiv version has only v1.
- Google Scholar "cited by" was not queried, because it is not accessible to automated fetching in this session. Semantic Scholar (10 citers) and OpenAlex (0) may be incomplete for papers from mid and late 2026.
- GitHub code search (for non-fork repositories using `teacher_stopping_time` with other values) requires authentication and was not run. The Hugging Face model card (yl4579/DMOSpeech2) was not checked.
- The project page dmospeech2.github.io was not fetched in this session.

## Q3. DMOSpeech (v1): student-only setting and any teacher-guided ablations

### Takeaway
DMOSpeech v1 (ICML 2025) reports only a teacher-only setting (N=128) and student-only settings (N=1 and N=4). It contains no teacher-guided or hybrid sampling. Its ablation shows the 4-step DMD student beating the 128-step teacher on WER and SIM but with lower pitch-variation CV_f0, which is the mode shrinkage that DMOSpeech 2 later addresses.

### Cited Findings
- Venue: Proceedings of ICML 2025, PMLR 267:35186–35208. Authors: Yinghao Aaron Li, Rithesh Kumar and Zeyu Jin (Columbia; Adobe Research). arXiv v1 (Oct 2024) was titled "DMDSpeech". — [PMLR](https://proceedings.mlr.press/v267/li25ay.html); [ICML poster](https://icml.cc/virtual/2025/poster/44048); [arXiv 2410.11097](https://arxiv.org/abs/2410.11097)
- The method reduces "sampling steps from 128 to 4 via distribution matching distillation". The teacher samples with DDPM over N steps with CFG ω = 2 ("We set ω=2 both for inference of the teacher model and DMD training"). Student multi-step sampling (Algorithm 1) "is similar to that of the consistency model", re-applying the prompt at each step. — [DMOSpeech arXiv v2 HTML, §1, App. C.2.3–C.2.4](https://arxiv.org/html/2410.11097v2)
- Table 3 (LibriSpeech test-clean; RTF on a V100). Columns are WER / SIM / RTF:
  - DMOSpeech (N=4): 1.94 / 0.69 / 0.07
  - Teacher (N=128): 9.51 / 0.55 / 0.96
  - Ground truth: 2.19 / 0.67

  — [DMOSpeech Table 3](https://arxiv.org/html/2410.11097v2)
- Table 4 ablation. Columns are MOS-N / WER / SIM / CV_f0:
  - Teacher (N=128): 4.22 / 9.51 / 0.55 / 0.70
  - DMD 2 only (N=1): 3.11 / 5.93 / 0.42 / 0.68
  - DMD 2 only (N=4): 4.19 / 5.67 / 0.53 / 0.61
  - +L_CTC only: 4.25 / 1.79 / 0.55 / 0.57
  - +L_SV only: 4.07 / 6.62 / 0.70 / 0.61
  - DMOSpeech (N=4): 4.27 / 1.94 / 0.69 / 0.58
  - Batch size 96→16: slight drop across most metrics.

  — [DMOSpeech Table 4](https://arxiv.org/html/2410.11097v2)
- The authors frame mode shrinkage as "beneficial … focusing on high-probability regions without compromising output diversity across different prompts and text inputs". In Appendix A (continuation task), "the student model closely matches the teacher's distribution in both F0 and energy, demonstrating minimal mode shrinkage in contrast to the results shown in Figure 2, where mode shrinkage was evident." — [DMOSpeech §1, App. A](https://arxiv.org/html/2410.11097v2)
- A full-text keyword search of the v2 HTML for "teacher-guided" and "hybrid" found no hybrid or teacher-then-student configuration. — [DMOSpeech arXiv v2 HTML](https://arxiv.org/html/2410.11097v2)

### Inferences
- The only step-count trade-off DMOSpeech v1 reports for the student is N = 1 versus N = 4 under the DMD2-only ablation (MOS-N 3.11 → 4.19, SIM 0.42 → 0.53). The final DMOSpeech is reported only at N = 4.
- In v1 the CV_f0 drop (0.70 → 0.58) coexists with the claim of "minimal mode shrinkage" in continuation tasks. DMOSpeech 2 later treats the shrinkage as a problem to fix (teacher 0.6659 vs student 0.4640).

### Gaps
- The DMOSpeech v1 demo page's statements about teacher intelligibility at different noise levels appeared only in a search summary and were not verified.

## Q4. F5-TTS NFE 16 vs 32 and sway-sampling ablations (Tables 1, 2, 5, 6), and Fast F5-TTS / EPSS step pruning

### Takeaway
F5-TTS (ACL 2025) shows that 16 versus 32 NFE changes WER by only about 0.1–0.2 points at half the RTF, and that sway sampling (s = −1) is what makes low NFE work. EPSS / Fast F5-TTS (Interspeech 2025) shows that a non-uniform, training-free 7-NFE schedule nearly matches 32 NFE (WER 2.45 vs 2.37 on LibriSpeech-PC) at about 4× speed. Uniform 7 NFE collapses (WER 4.16, zh WER 7.96). Both papers show that steps concentrated early (noisy end) matter most. Neither reports diversity.

### Cited Findings
- F5-TTS was published at ACL 2025 (long), pp. 6255–6271, DOI 10.18653/v1/2025.acl-long.313. — [OpenAlex record](https://api.openalex.org/works?search=F5-TTS%20fairytaler%20fakes%20fluent%20faithful%20speech%20flow%20matching); [arXiv 2410.06885](https://arxiv.org/abs/2410.06885)
- F5-TTS Table 1 (LibriSpeech-PC test-clean; RTF for 10 s of speech on an RTX 3090). Columns are WER / SIM-o / RTF:
  - 16 NFE: 2.53 / 0.66 / 0.15
  - 32 NFE: 2.42 / 0.66 / 0.31

  — [F5-TTS Table 1](https://arxiv.org/html/2410.06885)
- F5-TTS Table 2 (Seed-TTS). Columns are WER / SIM-o / CMOS / SMOS:
  - test-en, 16 NFE: 1.89 / 0.67 / 0.16 / 3.79
  - test-en, 32 NFE: 1.83 / 0.67 / 0.31 / 3.89
  - test-zh, 16 NFE: 1.74 / 0.75 / 0.02 / 3.72
  - test-zh, 32 NFE: 1.56 / 0.76 / 0.21 / 3.83

  — [F5-TTS Table 2](https://arxiv.org/html/2410.06885)
- F5-TTS Table 5 (base model with midpoint solver, sway on or off, s = −1). Columns are WER / SIM-o / UTMOS / RTF:
  - LS-PC: 16 NFE w/ SS 2.43 / 0.66 / 3.87 / 0.26; 32 NFE w/ SS 2.41 / 0.66 / 3.89 / 0.53; 32 NFE w/o SS 2.84 / 0.62 / 3.70 / 0.53
  - Seed-en: 16 w/ SS 1.88 / 0.66 / 3.70; 32 w/ SS 1.87 / 0.66 / 3.72; 32 w/o SS 1.93 / 0.63 / 3.51
  - Seed-zh: 16 w/ SS 1.61 / 0.75 / 2.87; 32 w/ SS 1.58 / 0.75 / 2.91; 32 w/o SS 1.93 / 0.69 / 2.58

  — [F5-TTS Table 5](https://arxiv.org/html/2410.06885)
- F5-TTS Table 6 (solver × sway coefficient). Each cell is WER / SIM-o / UTMOS for LS-PC, Seed-en and Seed-zh, followed by RTF:
  - s = −1, Euler 16: 2.53/0.66/3.88; 1.89/0.67/3.76; 1.74/0.75/2.96; RTF 0.15
  - s = −1, Midpoint 16: 2.43/0.66/3.87; 1.88/0.66/3.70; 1.61/0.75/2.87; RTF 0.26
  - s = −1, Euler 32: 2.42/0.66/3.90; 1.83/0.67/3.76; 1.56/0.76/2.95; RTF 0.31
  - s = −1, Midpoint 32: 2.41/0.66/3.89; 1.87/0.66/3.72; 1.58/0.75/2.91; RTF 0.53
  - s = −1, Heun-3 16: 2.39/0.65/3.87; 1.80/0.66/3.70; 1.55/0.75/2.88; RTF 0.44
  - s = −0.8, Euler 16: 2.82/0.65/3.73; 2.14/0.65/3.70; 2.28/0.72/2.74
  - s = −0.8, Midpoint 16: 2.58/0.65/3.86; 1.86/0.65/3.68; 1.70/0.73/2.83
  - s = −0.8, Euler 32: 2.50/0.66/3.89; 1.81/0.67/3.74; 1.62/0.75/2.94
  - s = −0.8, Midpoint 32: 2.42/0.66/3.89; 1.84/0.66/3.70; 1.62/0.75/2.91
  - s = −0.8, Heun-3 16: 2.40/0.65/3.85; 1.78/0.66/3.68; 1.56/0.74/2.84

  Other s values appear only in a figure (small models), not in tables. — [F5-TTS Table 6](https://arxiv.org/html/2410.06885)
- F5-TTS text: a "more negative s value further improves performance". s = 0 is uniform, the default is s = −1, and the default CFG strength is 2. Inference time doubles with CFG because the model runs once with the condition and once without. — [F5-TTS](https://arxiv.org/html/2410.06885)
- Fast F5-TTS / EPSS (Zheng, Chen, Niu, Ma, Wang, Yu, Chen; SJTU, Shanghai Innovation Institute, Microsoft) appears at Interspeech 2025, pp. 2445–2449, DOI 10.21437/interspeech.2025-2449. EPSS "allocates denser steps with smaller intervals in the initial phase … while pruning steps in the later phase" and is training-free. — [arXiv 2505.19931](https://arxiv.org/abs/2505.19931); [OpenAlex](https://api.openalex.org/works?search=Accelerating%20Flow-Matching-Based%20Text-to-Speech%20via%20Empirically%20Pruned%20Step%20Sampling)
- EPSS Table 2 schedules (sway with s = −1 applied to the listed indices out of 32):
  - 16 NFE: {0–8, 10, 12, 14, 16, 20, 24, 28, 32}
  - 12 NFE: {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32}
  - 10 NFE: {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32}
  - 7 NFE: {0, 2, 4, 6, 8, 16, 24, 32}
  - 6a NFE: {0, 2, 4, 6, 8, 16, 32}
  - 6b NFE: {0, 2, 4, 8, 16, 24, 32}
  - 5 NFE: {0, 2, 4, 6, 8, 32}

  — [EPSS Table 2](https://arxiv.org/html/2505.19931)
- EPSS Table 1. Each cell is WER / SIM-o / UTMOS for LS-PC, Seed-en and Seed-zh, followed by RTF:
  - F5-TTS 32: 2.37/0.66/3.93; 1.70/0.67/3.76; 1.58/0.75/2.96; RTF 0.123
  - F5-TTS 16: 2.44/0.66/3.91; 1.82/0.67/3.78; 1.75/0.75/2.97; RTF 0.065
  - F5-TTS 7 (uniform sway): 4.16/0.60/3.36; 3.01/0.58/3.45; 7.96/0.62/2.40; RTF 0.030
  - Fast F5-TTS (EPSS) 7: 2.45/0.66/3.84; 1.74/0.68/3.78; 1.75/0.76/3.00; RTF 0.030
  - E2 TTS 32: 2.63/0.72/3.72; 1.79/0.73/3.60; 1.82/0.78/2.91; RTF 0.150
  - E2 TTS 16: 2.78/0.71/3.71; 1.88/0.73/3.61; 1.93/0.78/2.93; RTF 0.078
  - E2 TTS 7: 4.04/0.65/3.17; 2.69/0.69/3.39; 3.52/0.72/2.58; RTF 0.036
  - E2 TTS + EPSS 7: 2.73/0.71/3.71; 1.84/0.73/3.69; 1.93/0.78/2.99; RTF 0.036

  — [EPSS Table 1](https://arxiv.org/html/2505.19931)
- EPSS Table 3 ablation (LS-PC). Columns are NFE: RTF / WER / SIM-o / UTMOS:
  - 32: 0.123 / 2.37 / 0.66 / 3.93
  - 16 (EPSS): 0.065 / 2.29 / 0.67 / 3.97
  - 12: 0.050 / 2.44 / 0.66 / 3.93
  - 10: 0.042 / 2.44 / 0.66 / 3.92
  - 7: 0.030 / 2.45 / 0.66 / 3.84
  - 6a: 0.027 / 2.41 / 0.66 / 3.82
  - 6b: 0.027 / 2.71 / 0.64 / 3.79
  - 5: 0.023 / 2.55 / 0.59 / 2.91

  The paper claims a 4× speed-up at 7 NFE (78% less computation than 32 NFE). — [EPSS Table 3](https://arxiv.org/html/2505.19931)

### Inferences
- Dropping 6b's early index 6 hurts more (WER 2.71) than dropping 6a's late index 24 (WER 2.41). Together with sway's benefit, this is direct TTS evidence that early (noise-side) steps carry the most quality-relevant information. DMOSpeech 2 uses the same reasoning to justify giving the early segment to the teacher.
- All of these schedules are global (one per NFE budget) and none is chosen per input. None of these papers reports a diversity metric, so their step reductions say nothing about mode shrinkage.

### Gaps
- F5-TTS Figure 3 (sway s sweep on small models) values are not tabulated and were not extracted.
- F5-TTS v1 (the later checkpoint) step-count results were not checked.

## Q5. Consistency, distilled and few-step TTS models: configurable step counts or schedules, and reported trade-offs (quality, diversity, speed)

### Takeaway
Most distilled TTS models report 1–4-step operating points, and several show non-monotonic quality with more student steps: CoMoSpeech FD is best at 10 steps and worse at 50, FlashSpeech is best at 2, and CM-TTS MOS goes T=1 > T=2 < T=4. Teachers degrade sharply at low NFE. Newer flow-map students (IntMeanFlow, LFMD) explicitly support multiple NFE budgets and search the schedule globally (O3S, TD-DP). Diversity is rarely measured. Exceptions are ProDiff (NDB/JS), E1 TTS (pairwise MFCC/pitch/duration distance), Voicebox (FSD) and DMOSpeech/DMOSpeech 2 (CV_f0). None of these models mixes a teacher into the trajectory.

### Cited Findings
- **CoMoSpeech** (consistency distillation, LJSpeech/Opencpop). Table 1 (TTS), columns NFE / RTF / FD / MOS:
  - Teacher: 50 / 0.1824 / 0.748 / 4.538
  - CoMoSpeech: 1 / 0.0058 / 0.774 / 4.239

  Table 3 gives FD at NFE 1 / 2 / 4 / 10 / 50:
  - Teacher: 7.526 / 4.558 / 2.477 / 1.197 / 0.748
  - CoMoSpeech: 0.774 / 0.762 / 0.784 / 0.725 / 0.850

  Table 4 (SVS) gives the same NFE sweep:
  - Teacher: 7.786 / 7.219 / 4.932 / 3.937 / 3.162
  - CoMoSVS: 3.571 / 3.520 / 3.433 / 3.658 / 3.732

  — [CoMoSpeech arXiv 2305.06908](https://arxiv.org/abs/2305.06908)
- **FlashSpeech** (latent consistency model with adversarial consistency training, from scratch and without a teacher). Table 3 sampling-step ablation, columns UTMOS / Sim-O:
  - NFE 1: 3.99 / 0.51
  - NFE 2: 4.00 / 0.52
  - NFE 4: 3.91 / 0.51

  A separate "prosody generator module enhances the diversity of prosody". — [FlashSpeech arXiv 2404.14700](https://arxiv.org/abs/2404.14700)
- **CM-TTS** (consistency models with weighted samplers). Table 5 (VCTK zero-shot), columns WER / S.Cos / MOS:
  - T=1: 0.0536 / 0.7108 / 3.8715
  - T=2: 0.0536 / 0.7107 / 3.8387
  - T=4: 0.0530 / 0.7110 / 3.9221

  The DiffGAN-TTS comparison rows have MOS 3.4607 / 3.5067 / 3.5893 at T = 1/2/4. Table 10 (LJSpeech) gives CM-TTS MOS 3.7205 / 3.6817 / 3.7113 at T = 1/2/4. — [CM-TTS arXiv 2404.00569](https://arxiv.org/abs/2404.00569)
- **ReFlow-TTS**. Table 1 (LJSpeech), columns NFE / RTF / FD / MOS:
  - 1 step: 1 / 0.00577 / 1.9872 / 4.16
  - 50 steps: 50 / 0.0964 / 0.2759 / 4.43
  - RK45: 152 / 0.3703 / 0.1393 / 4.52

  Table 3 gives 2-ReFlow-TTS MOS of 4.17 (1 step), 4.41 (50) and 4.50 (RK45, 141 NFE). — [ReFlow-TTS arXiv 2309.17056](https://arxiv.org/abs/2309.17056)
- **SlimSpeech** (slim rectified flow, 1 step). Table I baselines (MOS):
  - Matcha-TTS: 3.89 at 4 steps, 3.53 at 1 step
  - Grad-TTS: 3.64 at 4 steps, 3.42 at 1 step
  - 2-ReFlow-TTS (Euler): 4.04 at 4 steps

  Table II gives FAD / FD / CMOS for a 2-rectified-flow TTS at 96 steps (0.607 / 0.810 / 0) and SlimSpeech at 1 step (0.693 / 0.806 / 0). — [SlimSpeech arXiv 2504.07776](https://arxiv.org/abs/2504.07776)
- **E1 TTS** (one-step DMD; "requiring only one neural network evaluation for each utterance"). §IV-G notes that DMD's reverse KL "is also a mode-seeking loss". Table IV gives expected pairwise distances over 100 samples of one text and prompt, as MFCC ×10³ / pitch ×10³ / duration ×10⁻¹ (higher means more diverse):
  - E1 TTS: 10.6 / 8.9 / 4.7
  - StyleTTS 2: 6.2 / 9.2 / 2.8
  - CosyVoice: 7.3 / 7.9 / 4.2
  - ARDiT (B=1, DMD): 11.7 / 10.6 / 6.0

  — [E1 TTS arXiv 2409.09351](https://arxiv.org/abs/2409.09351)
- **ProDiff** (progressive distillation). Table 1 compares MOS for gradient-based against generator-based parameterization, both 4.09 at 128 iterations; at 2 iterations gradient-based falls to 2.96 while generator-based holds 3.98. Table 2 gives ProDiff (2 steps) MOS 4.08, NDB 0.69, JS 0.012 and RTF 0.04, against Grad-TTS (64 steps) at 4.05 / 0.57 / 0.023 / 0.19. Table 3 gives Teacher (T=16) MOS 4.05 and Teacher (T=8) MOS 4.06. — [ProDiff arXiv 2207.06389](https://arxiv.org/abs/2207.06389)
- **DiTTo-TTS**: the main results use 25 diffusion steps (NFE). Ablations use 250 steps for WER and SIM and 25 for inference time. No step sweep was found in its tables. — [DiTTo-TTS arXiv 2406.11427](https://arxiv.org/abs/2406.11427)
- **IntMeanFlow** (integral-velocity MeanFlow distillation, with O3S step search "using a customizable quality metric and a ternary search"). Table 1 (Seed-TTS test-en; F5-TTS Base teacher), columns WER / SIM-o / UTMOS / RTF:
  - Teacher, 32 NFE: 1.87 / 0.67 / 3.70 / 0.243
  - IMF, 1 NFE: 7.27 / 0.48 / 1.84 / 0.009
  - IMF, 2 NFE: 4.48 / 0.59 / 3.35 / 0.013
  - IMF + O3S, 2 NFE: 2.04 / 0.63 / 3.24 / 0.013
  - IMF + O3S, 3 NFE: 1.60 / 0.65 / 3.79 / 0.021

  The 3-NFE student's WER depends on how many teacher NFE were used to build training targets: 1.60 (16), 1.71 (4), 1.83 (2). In token2mel (CosyVoice2 flow, Table 2), CosyVoice2 at 32 NFE gives WER 2.17 / SIM 0.66 / UTMOS 4.36 / RTF 0.510, and IMF at 1 NFE gives 2.18 / 0.63 / 4.28 / 0.026. — [IntMeanFlow arXiv 2510.07979](https://arxiv.org/abs/2510.07979) (listed by Semantic Scholar as ICASSP 2026)
- **LFMD, "Distill Locally, Schedule Globally"** (arXiv 2609.36324v1, 28 Sep 2026, MBZUAI). Students run at 1–4 NFE with EPSS as the baseline schedule ("consistently outperforms uniform sampling"). Teachers use 32 sway-sampled steps with CFG. TD-DP builds "a single interval-cost graph reusable across inference budgets without an external audio metric".
  - After soft-DTW self-distillation, 1 NFE reaches 1.80% WER on Seed-TTS against the 32-NFE teacher's 1.76%, with SIM-o .607 vs .667 and UTMOS 3.41 vs 3.76.
  - Self-distillation lowers 2-NFE WER from 1.76 to 1.68 and 3-NFE WER from 1.72 to 1.61.
  - One-step inference is "approximately 10× faster" than the 32-step teacher.
  - IMF students trained on uniformly sampled teacher trajectories reach about 35% WER at 1 NFE, against about 8% with sway-sampled trajectories.
  - TD-DP and O3S give "broadly similar synthesis quality".

  — [LFMD arXiv 2609.36324](https://arxiv.org/abs/2609.36324)
- **Matcha-TTS** (OT-CFM). Table 1, columns RTF / WER / MOS:
  - MAT-10: 0.038 / 2.09 / 3.84
  - MAT-4: 0.019 / 2.15 / 3.77
  - MAT-2: 0.015 / 2.34 / 3.65
  - GRAD-10: 0.049 / 3.44 / 3.49
  - GRAD-4: 0.019 / 3.69 / 3.20
  - GCFM-4: 0.019 / 2.70 / 3.57

  — [Matcha-TTS arXiv 2309.03199](https://arxiv.org/abs/2309.03199)
- **VoiceFlow** (rectified flow). Table 1 gives MOS at 2 / 10 / 100 steps:
  - VoiceFlow on LJSpeech: 3.92 / 4.10 / 4.17
  - VoiceFlow on LibriTTS: 3.81 / 3.84 / 3.85
  - GradTTS on LJSpeech: 2.98 / 3.97 / 4.03
  - GradTTS on LibriTTS: 2.52 / 3.43 / 3.45

  Table 2: removing ReFlow at 2 steps gives CMOS −0.78 (LJSpeech) and −1.21 (LibriTTS). — [VoiceFlow arXiv 2309.05027](https://arxiv.org/abs/2309.05027)

### Inferences
- Distilled students are typically flat or non-monotonic in step count (CoMoSpeech, FlashSpeech, CM-TTS), unlike their teachers, which degrade sharply at low NFE. This supports DMOSpeech 2's design of giving the student only the last 2 steps, since adding student steps does not reliably buy quality.
- Diversity under distillation is measured with heterogeneous metrics (FD/NDB/JS, pairwise DTW distances, FSD, CV_f0) and never against a switching parameter. A thesis measuring diversity versus t_switch would be reporting a variable no prior TTS paper has measured.
- Schedule-selection methods (EPSS, O3S, TD-DP) are budget-level and global. O3S needs an external metric and repeated synthesis per budget, while TD-DP is metric-free. None of them is per input.

### Gaps
- Venues for CoMoSpeech, FlashSpeech, CM-TTS, ReFlow-TTS, SlimSpeech, E1 TTS, Matcha-TTS, VoiceFlow and DiTTo-TTS were not verified in this session; only arXiv versions are cited.
- The full SlimSpeech Table I row (its own 1-step MOS and RTF) was truncated in extraction. LFMD's tables were not machine-readable, so only numbers stated in its text are given.
- SplitMeanFlow and DSFlow (TTS flow-map methods cited by LFMD) and WaveEx (AAAI 2026, DOI 10.1609/aaai.v40i38.40490, "Accelerating Flow Matching-based Speech Generation via Wavelet-guided Extrapolation") were identified but not read.

## Q6. Industrial and large-scale systems (CosyVoice 2/3, MaskGCT, Seed-TTS, Voicebox, Matcha-TTS, StableTTS, E2 TTS): documented step-count or schedule trade-offs

### Takeaway
Among the large systems, only Voicebox (NFE 4–32 sweep) and MaskGCT (S2A step-schedule ablation) document step trade-offs in their own papers. E2 TTS's step trade-off is documented by third parties (EPSS). CosyVoice 2, CosyVoice 3 and Seed-TTS report single operating points in their main result tables. None mixes models along a trajectory or adapts steps per input.

### Cited Findings
- **Voicebox** Table 9 (ablation models trained 150K updates, zero-shot TTS). FM w/ OT at NFE 4/8/16/32 gives WER 2.4/2.2/2.2/2.1 and Sim-o 0.410/0.481/0.503/0.508. FM w/ diffusion path gives WER 11.5/3.0/2.7/2.6, and score matching gives 94.5/42.3/11.5/5.1. Table B4 gives diverse-sampling FSD (lower is better) for FM w/ OT at 150K updates: 249.7/208.5/198.0/195.3 at NFE 4/8/16/32. — [Voicebox arXiv 2306.15687](https://arxiv.org/abs/2306.15687)
- **MaskGCT** Table 8 (S2A model inference steps per codec layer). Schedule [10,1,1,…] against [40,16,1,…]:
  - Seed-en: SIM-O 0.709 vs 0.728, WER 2.796 vs 2.466, FSD 0.164 vs 0.159
  - Seed-zh: SIM-O 0.766 vs 0.777, WER 2.268 vs 2.183, FSD 0.111 vs 0.101

  The EPSS paper lists MaskGCT at 50 NFE. — [MaskGCT arXiv 2409.00750](https://arxiv.org/abs/2409.00750); [EPSS Table 1](https://arxiv.org/html/2505.19931)
- **E2 TTS**: its own result tables (Tables 3 and 4) report WER and SIM-o without an NFE sweep. Third-party NFE results (32/16/7, and 7 with EPSS) are in EPSS Table 1 (Q4). CosyVoice 2 and 3 list E2 TTS and F5-TTS at "32 NFE". — [E2 TTS arXiv 2406.18009](https://arxiv.org/abs/2406.18009); [CosyVoice 2 Table 6](https://arxiv.org/abs/2412.10117)
- **CosyVoice 2**: EPSS lists it at 10 NFE (flow-matching module). Its main Seed test table reports one operating point (CosyVoice 2: test-zh CER 1.45, test-en WER 2.57, test-hard WER 6.83). IntMeanFlow reports the CosyVoice2 token2mel flow at 32 NFE in its own setting (RTF 0.510), reduced to 1 NFE by IMF (Q5). — [CosyVoice 2](https://arxiv.org/abs/2412.10117); [EPSS Table 1](https://arxiv.org/html/2505.19931); [IntMeanFlow Table 2](https://arxiv.org/abs/2510.07979)
- **CosyVoice 3** and **Seed-TTS**: no step-count ablation tables were found in their arXiv HTML. — [CosyVoice 3](https://arxiv.org/abs/2505.17589); [Seed-TTS](https://arxiv.org/abs/2406.02430)
- **Matcha-TTS**: 2/4/10-step results are given in Q5 (Table 1). — [Matcha-TTS](https://arxiv.org/abs/2309.03199)

### Inferences
- For the F5-TTS / E2 TTS family, the most complete published step trade-off evidence is in F5-TTS Tables 5–6 and EPSS Tables 1 and 3. A thesis using F5-TTS as the guided teacher can cite these to justify teacher grids (16 vs 32, sway s = −1), but not for any diversity claim.

### Gaps
- StableTTS has no paper, and its repository was not checked.
- Seed-TTS's diffusion variant (Seed-TTS_DiT) step settings, and the CosyVoice 2/3 flow-matching NFE ablations, were not found. These may exist in appendices that the table extraction missed, or not at all.
- Voicebox's final-model NFE (FlashSpeech cites Voicebox at NFE 64) and its venue were not verified directly from the Voicebox paper text in this session.
