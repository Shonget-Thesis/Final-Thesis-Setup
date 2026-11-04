# EXP-016: Same-Hardware (Tesla T4) Confirmation of Noise-Aware Handover Methods (TAS, SWC, PIEC, Prompt Enhancement, SCG, SNNG)

**Status:** Prepared / ready to run on Colab Tesla T4 (task EVAL-008). Notebook: [`code/colab/DMOSpeech2_EXP016_Noise_Guidance_T4.ipynb`](../../../code/colab/DMOSpeech2_EXP016_Noise_Guidance_T4.ipynb); runner: [`code/colab/run_noise_guidance_prelim.py`](../../../code/colab/run_noise_guidance_prelim.py). Outputs flush to dedicated Google Drive directory `MyDrive/dmospeech2_results/exp016_noise_guidance_t4/` and package into `DMOSpeech2_EXP016_Noise_Guidance_T4_Results.zip`.

**Why (EVAL-008):**
EXP-015 tested noise-aware guidance methods on an NVIDIA A100-SXM4-40GB. While the teacher ODE trajectory reproduced identically with 0.0 max absolute difference, PyTorch's pseudorandom noise generation on the GPU differs across GPU architectures (A100 vs. Tesla T4). Consequently, student model synthesis on the A100 drew different noise paths than the stored T4 references, introducing a cross-GPU artifact that inflated clean-prompt failure metrics. EXP-016 performs the exact same evaluation directly on the thesis benchmark hardware (**NVIDIA Tesla T4**), ensuring clean same-hardware comparison and confirming the candidate performance of **SWC** and **TAS**.

| Method | Definition | Cost at Step 4 |
|---|---|---:|
| **SWC** (`swc`, `swc05`) ⭐ | SNR-Wiener Conditioning: quietest 20% prompt frames spectral gain applied before conditioning | **13 NFE** |
| **TAS** (`tas`) ⭐ | Trajectory Anomaly Score: one-sided Mahalanobis distance on teacher drift profile $\log(d_1..d_6)$ | **18.20 NFE** (routed) |
| **PIEC** (`piec4`, `swcpiec4`) | Prompt-Informed Environment Cancellation on generated mel frames before vocoder | **13 NFE** |
| **Prompt Enhancement** (`enh4`, `enh6`, `enhs4`) | Pretrained DNS64 Demucs speech enhancer front-end | **13 / 17 NFE** |
| **SNNG** (`snng05`, `snng10`) | Negative noise guidance steering away from quietest 20% prompt frames | **18 NFE** |
| **SCG** (`scg`, `scg15`) | SNR-Compensated Guidance scale scaling up CFG scale | **13 NFE** |
| **Antithetic pair** (`anti`) | Dual student passes with $\pm\epsilon$ from Step-4 endpoint | **16 NFE** |
| **Controls** (`enh8`, `piec8`) | Same cleaning applied to Step-8 reference for within-cleaned-system benchmarking | 21 NFE |

**Data:** EXP-012 Quick set (60 development speakers × clean, noisy15, Gaussian 18/20/22/25 dB = 360 cases).

**Runtime & Infrastructure:**
- Hardware: NVIDIA Tesla T4 (FP32).
- Estimated duration: ~6.5–7 hours on Tesla T4.
- Execution: Background process (`nohup`) writing live status to `exp016_noise_guidance_t4_log.txt` and persistent results to `MyDrive/dmospeech2_results/exp016_noise_guidance_t4/`. Survives session disconnects and automatically resumes from the last completed case.

**Preregistered Evaluation Criteria:**
- **C1:** Noisy Step-4 failures decrease by $\ge 25\%$ ($p < 0.05$, exact two-sided McNemar).
- **C2:** Clean Step-4 failures increase by at most 1 case.
- **C3:** Pooled out-of-fold route policy achieves lower mean NFE than the calibrated SNR gate with no additional failures.
