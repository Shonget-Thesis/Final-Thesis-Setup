# EXP-003: Earlier handover at Steps 2 and 3

**Run:** 2026-10-21 on a Google Colab Tesla T4 (FP32, PyTorch 2.11.0+cu130). Notebook `DMOSpeech2_Early_Handover_Development_GPU.ipynb` and runner `code/colab/run_early_handover_dev_gpu.py` with `code/colab/early_handover_signals.py` (reference copies of Shonget-DMOSpeech2 branch `early-handover-dev`, results commit `cf2ac90`).

**What:** does handing over before Step 4 leave room to save more? For the 100 development speakers × clean/noisy15, Steps 2 (9 NFE) and 3 (11 NFE) were newly synthesized and scored with the study's evaluators; Steps 4/5/6/8 were reused from EXP-002. It also tested three trajectory signals at the early steps: drift trend q_A = d_i / d_{i−1}, local stability q_B and conditioning disagreement q_C.

**Data:** [research/evidence/benchmarks/early_handover/](../../evidence/benchmarks/early_handover/EARLY_HANDOVER_DEVELOPMENT_REPORT.md): `development_expanded_grid.csv` (1,200 rows: 400 newly synthesized, 800 reused), case audit, signal diagnostics, policy comparison, formula search, and the team's report. The 800 reused rows are identical to EXP-002 (checked 2026-10-25).

## Key results (development split)

| Step | NFE | Clean failures (100) | Noisy15 failures (100) |
|---|---:|---:|---:|
| 2 | 9 | 7 | 84 |
| 3 | 11 | 3 | 77 |
| 4 | 13 | 3 | 63 |
| 5 | 15 | 2 | 49 |
| 6 | 17 | 0 | 50 |

- On clean prompts, Step 3 is as safe as Step 4 (3/100 each) at 2 fewer forward passes; Step 2 fails 7/100, mostly small SIM drops.
- With the 20 dB SNR gate, every noisy prompt goes to Step 8, so noisy failures vanish.
- Drift at Step 2 does not predict failure (AUC 0.41; failing cases had *lower* drift). The trend q_A reaches AUC 0.61 at Step 3.

## Caveats

- Development split only (100 speakers), and its policy search is exploratory.
- The candidate set in the proposal stays {4, 5, 6}; Steps 2–3 would be a design change.
