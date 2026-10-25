# Implementation migration: Handover scheduling, Colab GPU pipeline, manifests and reports

Date: 2026-10-25 (Asia/Manila).
Author: Antigravity IDE, for Mark Vincent D. Limpahan.

## Context

The research implementation developed by Rhenel Jhon Sajol in the sibling `Shonget-DMOSpeech2` repository under `Handover Scheduling/` was migrated into the thesis workspace. In accordance with the decision recorded in `docs/progress/STATUS.md` and the user's explicit instruction, CPU-only execution scripts that hardcoded local Ryzen laptop sweeps were excluded in favor of the Google Colab GPU-accelerated pipeline (NVIDIA Tesla T4 / A100).

## Migrated components

1. **Active scheduler & metrics (`code/scheduling/`):**
   - `sampler.py`: Deterministic FP32 Euler trajectory sampler implementing the dual-mechanism architecture (prompt SNR safety gate $\theta=20.0\text{ dB}$ + zero-overhead endpoint drift $d_i$ at steps 4, 5, 6, 8 with student restart $j=1$).
   - `metrics.py`: MetricSuite evaluating WER (Whisper large-v3), SIM (WavLM-large), and UTMOS naturalness, with configurable failure rules.
   - `failure_config.json`: Canonical failure threshold definitions.
   - `sanity_check.py` & `sanity_check_results.json`.
   - `download_ckpts.py`: Automated HuggingFace downloader and SHA-256 verifier for `model_85000.pt` (DMOSpeech 2) and `model_1500.pt` (duration predictor).

2. **Google Colab GPU pipeline (`code/colab/`):**
   - `COLAB_QUICKSTART_GUIDE.md`: Step-by-step GPU execution instructions.
   - `DMOSpeech2_Adaptive_Handover_Colab.ipynb`, `DMOSpeech2_Full_Pipeline_Timing_Benchmark.ipynb`, `DMOSpeech2_Live_1000_GPU_Retest.ipynb`, `DMOSpeech2_Novel_Algorithm_Comparison.ipynb`, `DMOSpeech2_Signal_Selection_Validation.ipynb`.
   - `run_gpu_full_pipeline_timing_benchmark.py`: High-speed GPU benchmark with CUDA event timing and full NFE accounting.
   - `run_gpu_expanded_benchmark.py`, `run_novel_algorithm_comparison.py`, `run_signal_selection_validation.py`.

3. **Statistical calibration & evaluation (`code/experiments/`):**
   - `phase2_calib/`: Learn-Then-Test (LTT) calibration algorithm (`calibrate.py`, `config.json`, `stability.py`, `verify.py`).
   - `phase3_test/`: Objective 4 held-out statistical benchmark runner (`run.py`).

4. **Cohort manifests & split protocols (`research/protocols/splits/`):**
   - `development_manifest.json` (100 speakers, 200 cases).
   - `calibration_manifest.json` (200 speakers, 400 cases).
   - `test_manifest.json` (200 speakers, 400 cases).
   - `split_summary.json` and `make_splits.py`.

5. **Research reports & benchmarks (`research/reports/`, `research/evidence/benchmarks/`):**
   - `research/reports/ALL_REPORTS_COMPILED.md`, `HANDOVER_SCHEDULING_MASTER_README.md`, `RHENEL_ASSIGNED_TASKS_REPORT.md`, and `Ref_reports/`.
   - `research/evidence/benchmarks/GPU_FULL_PIPELINE_TIMING_REPORT.md` and summary CSVs.

6. **Figures & scripts (`paper/figures/scripts/`):**
   - `generate_sway_grid_check.py`, `generate_teacher_guided_timeline.py`, `generate_thesis_figures.py`.

## Excluded items

- Heavy binary model checkpoints (`*.pt`) remain excluded from git and are retrieved dynamically on GPU runtimes.
- Raw audio waveforms (`demo/audio/`) are excluded to avoid repository bloat.
- Ryzen CPU-only sweep scripts (`run_grid.py`, `run_calib_grid.py`, `run_test_grid.py`, `analyze.py`) and pytest-dependent local test files were removed as all synthesis and benchmarking run on Colab GPU.
- Raw multi-megabyte CSV and parquet sweep dumps.

## Checks run

- `python scripts/check_workspace.py` PASSED (structure, source manifest hashes, task evidence, links, and LaTeX citations valid).
- `python -m unittest discover -s code/tests` PASSED (14/14 unit tests pass).
