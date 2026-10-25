# EXP-010: Full-pipeline timing on the T4

**Run:** 2026-10-20, Colab Tesla T4, session `timing_session_20261020_130952`. Notebook `code/colab/DMOSpeech2_Full_Pipeline_Timing_Benchmark.ipynb`, runner `code/colab/run_gpu_full_pipeline_timing_benchmark.py` (results committed in Shonget-DMOSpeech2 `d00325e`).

**What:** three policies (Step 8, SNR-only, SNR + drift with λ = 0.50 and θ = 20 dB) on all 1,000 cases (500 speakers × clean/noisy15), with real online early stopping. The timed region covers prompt SNR, teacher steps, student completion and vocoder; GPU synchronization before and after; three warm-up runs; policy order rotated across cases.

**Data:** [research/evidence/benchmarks/](../../evidence/benchmarks/MANIFEST.json): `gpu_full_pipeline_timing_3000.csv` / `.parquet` (3,000 runs), `gpu_timing_summary_test.csv`, `gpu_timing_summary_by_condition.csv`, `gpu_timing_cluster_bootstrap_ci.csv`, `GPU_FULL_PIPELINE_TIMING_REPORT.md`.

## Key results (test split, mean core-synthesis time)

| Policy | Clean | Noisy15 |
|---|---:|---:|
| Step 8 | 3.29 s (21 NFE) | 3.37 s |
| SNR-only | 2.06 s (13.08 NFE) | 3.37 s |
| SNR + drift (λ = 0.50) | 2.41 s (15.30 NFE) | 3.37 s |

- Recorded teacher + student forward passes equal the total and match 2(k+1) + 3 for the exit step in 3,000/3,000 rows.

## Caveats

- The report's wording "primary certified threshold λ̂ = 0.50" and "100% complete & verified" overstates: λ = 0.50 is a CPU-label value (see EXP-002 for the T4 calibration).
- Test-split numbers are development evidence (the split was inspected).
