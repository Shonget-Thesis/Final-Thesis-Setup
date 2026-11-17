# 2026-10-26: EXP-016 same-hardware (Tesla T4) confirmation setup

Tasks: EVAL-008.

## Motivation & Setup

EXP-015 evaluated noise-aware guidance methods on an NVIDIA A100 GPU. While teacher ODE integration and TAS drift features reproduced with 0.0 difference across platforms, PyTorch's pseudorandom normal sampling on GPU differs between A100 and Tesla T4 architectures. Consequently, student model synthesis on the A100 diverged from the stored Tesla T4 reference student trajectories, introducing cross-GPU variance that inflated clean-prompt failures.

To confirm the candidate methods (SWC and TAS) under same-hardware conditions as required by EVAL-008, EXP-016 is set up:
- **Notebook:** [`code/colab/DMOSpeech2_EXP016_Noise_Guidance_T4.ipynb`](../../../code/colab/DMOSpeech2_EXP016_Noise_Guidance_T4.ipynb)
- **Runner:** [`code/colab/run_noise_guidance_prelim.py`](../../../code/colab/run_noise_guidance_prelim.py)
- **Hardware:** Google Colab Tesla T4 GPU (FP32)
- **Drive Destination:** Dedicated output directory `MyDrive/dmospeech2_results/exp016_noise_guidance_t4/` and zip archive `DMOSpeech2_EXP016_Noise_Guidance_T4_Results.zip`
- **Experiment Folder:** [`research/experiments/EXP-016-noise-guidance-t4/`](../../../research/experiments/EXP-016-noise-guidance-t4/README.md)

## Status

Ready to run on Google Colab with T4 runtime. Automatic checkpoint resumption is preserved in case of Colab timeouts or disconnects.
