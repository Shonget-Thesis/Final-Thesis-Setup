# EXP-015: Preliminary test of noise-aware handover methods (SNNG, SCG, SWC, antithetic pair)

**Status:** designed 2026-10-26, not run. Notebook `code/colab/DMOSpeech2_Noise_Guidance_Prelim.ipynb`, runner `code/colab/run_noise_guidance_prelim.py` (Colab T4; same setup cells as EXP-013/014).

**Why:** EXP-013 and EXP-014 showed that most noisy failures at an early handover are speaker-similarity failures (the prompt's noise leaks into the voice) and that no zero-cost change at the handover makes noisy prompts safe. These methods spend a little compute to act on the cause. Requested by a team member as candidate solutions that may come with a new title and objectives.

| Method | Definition | NFE at Step 4 |
|---|---|---:|
| SNNG (`snng05`, `snng10`) | Noise-only prompt = the prompt's quietest 20% of frames, tiled over the prompt length. Each teacher step: v = v_c + γ(v_c − v_u) + η(v_c − v_noise), γ = 2, η = 0.5 or 1.0. Teacher to Step 4 with this velocity, then the usual student. | 18 |
| SNR-Compensated Guidance (`scg`, `scg15`) | Prompt noise shrinks the guidance gap g₄ = ‖v_c − v_u‖/‖v_c‖ (EXP-012: median 0.236 clean, 0.116 at 15 dB; Spearman with SNR 0.84). The CFG scale is raised by the factor the SNR predicts: γ(Ŝ) = 2·clip(0.236 / max(−0.0263 + 0.0078·Ŝ, 0.02), 1, r_max), r_max = 2.5 (`scg`) or 1.5 (`scg15`, γ ≤ 3). The line was fitted to g₄ against SNR on the EXP-012 signals of these same 60 speakers (prompts < 30 dB, R² 0.60); it uses no failure labels. | 13 |
| SNR-Wiener Conditioning (`swc`, `swc05`) | The prompt frames of the teacher's and the student's conditioning are cleaned before use: per mel band, noise floor N_b = mean linear magnitude of the prompt's quietest 20% of frames; gain G = clip(1 − β(Ŝ)·N_b/M, 0.1, 1); β(Ŝ) = β_max·clip((30 − Ŝ)/10, 0, 1), β_max = 1.0 (`swc`) or 0.5 (`swc05`). | 13 |
| Antithetic pair (`anti`) | From the plain Step-4 endpoint, the student runs with ε and with −ε; the output nearer the teacher endpoint (generated frames) is kept. | 16 |

**Data:** the EXP-012 Quick set (60 development speakers × clean, noisy15, Gaussian 18/20/22/25 dB = 360 cases); stored plain Step-4 and Step-8 labels are reused. The smoke test checks that SNNG with η = 0 reproduces the plain teacher endpoint and that the stored Step-4 and Step-8 scores are reproduced.

**Preregistered:** C1 noisy Step-4 failures drop by at least 25% with p < 0.05; C2 clean failures rise by at most 1; C3 the calibrated route (plain Step 4 above the calibrated SNR gate, the method in a calibrated band below it, Step 8 below that) has lower mean NFE than the calibrated SNR gate with no more failures. Secondary, descriptive: average SIM and WER against Step 8 (paired Wilcoxon). Best method: most criteria passed, ties broken by lower route NFE.

**Cost:** about 3.5–4 hours on a T4 (estimate: one plain, two SNNG, two SCG and two SWC teacher passes, nine student runs and up to seven scored outputs per case).

**Prior work (2026-10-26 web search):** a trained triple classifier-free guidance with a separate noise condition exists ([arXiv 2608.03011](https://arxiv.org/html/2608.03011v1)); noisy-prompt robustness is otherwise handled by prompt denoising or noise-augmented training ([arXiv 2505.13830](https://arxiv.org/abs/2505.13830), [arXiv 2406.05699](https://arxiv.org/pdf/2406.05699)). No training-free method was found that builds the negative condition from the prompt's own quiet frames. For SCG (2026-10-26 search): decoupled or selective CFG for style and speaker similarity exists (ReStyle-TTS, ACL Findings 2026; Selective CFG, arXiv 2509.19668, which also reports that image-domain CFG strategies often fail in speech); no method was found that sets the CFG scale from the prompt SNR to undo guidance attenuation. For SWC (2026-10-26 search): noisy prompts are known to degrade zero-shot TTS, and the usual remedies are separate trained enhancement front-ends, trained prompt denoisers or noise-augmented training; no training-free, SNR-scheduled spectral gain on the mel prompt inside the sampler was found. Antithetic variates are classic; their use at a teacher-to-student handover was not found. "Not found" is not proof of absence.

**Caveats:** development evidence only, Gaussian noise only; the methods were chosen after seeing EXP-012 to EXP-014 data.
