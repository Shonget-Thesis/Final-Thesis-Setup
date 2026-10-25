# EXP-002: T4 grid on all 500 split speakers

**Run:** 2026-10-20 on a Google Colab Tesla T4 (FP32). Notebook `code/colab/DMOSpeech2_Live_1000_GPU_Retest.ipynb`, runner `code/colab/run_gpu_expanded_benchmark.py --scope all1000`, from Shonget-DMOSpeech2 (results committed there in `fdd8e9e`; the code commit used in Colab was not recorded).

**What:** every development, calibration and test speaker (500) under clean and noisy15 prompts (white Gaussian noise at 15 dB), synthesized at teacher Steps 4, 5, 6 and 8 from one teacher pass, and scored with Whisper-large-v3 (WER), WavLM-Base+ SV (SIM) and UTMOS. Each candidate step is labelled with the added-failure rule (Definition 3.1) against Step 8. Prompt SNR and d₄–d₆ are stored per case.

**Data:** [research/evidence/benchmarks/gpu_grid_all1000.csv](../../evidence/benchmarks/gpu_grid_all1000.csv) (4,000 rows; SHA-256 in the benchmarks MANIFEST).

## Key results

| Condition | Step-4 failures | Step-5 failures | Step-6 failures |
|---|---:|---:|---:|
| clean (500) | 12 | 9 | 5 |
| noisy15 (500) | 322 | 269 | 247 |

- The 20 dB SNR gate separates the two conditions completely (clean 13.7–50.0 dB with 3 below 20 dB; noisy15 12.4–18.4 dB).
- Fixed-sequence LTT on the calibration split (α = 0.05, δ = 0.10, λ grid 0.20–1.20): every threshold is certified, λ̂ = 1.20, and the policy equals SNR-only. Test split: SNR-only 4/200 speaker failures at 17.04 NFE; SNR + drift at λ = 0.50: 3/200 at 18.15 NFE. Recompute with `python research/audits/2026-10-25/recompute_gpu_grid.py`.
- Reproduced exactly by the EXP-011 pilot (160/160 rows, all differences 0).

## Caveats

- The test split was inspected; test numbers are development evidence.
- The `synthesis_time` and `rtf` columns time only the student completion and vocoder, not the shared teacher pass. Use EXP-010 for runtimes.
- SIM was scored before Shonget-DMOSpeech2 commit `656d86f` (ground-truth sample rate); the ground-truth files checked in EXP-011 are 24 kHz, so this does not change scores.
