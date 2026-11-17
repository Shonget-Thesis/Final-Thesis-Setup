# EXP-018: Stage-Wise Trajectory Action Cascade (`STAC`) & `1D-CRC-STAC` on Both 16-Grid (`Steps 1..8`) and 32-Grid (`Steps 1..15`)

**Status:** run on 2026-11-07 on a Colab Tesla T4 (thesis `8a6f396`, Shonget-DMOSpeech2 `8a7868f`); 360 of 360 cases across all 8 steps of the 16-interval grid and all 15 steps of the 32-interval grid, plus 30 of 30 live early-stopping GPU executions; results in [results/](results/MANIFEST.json). Task EVAL-011 (review).
- Notebook: [DMOSpeech2_SWSR_Handover_Prelim.ipynb](../../../code/colab/DMOSpeech2_SWSR_Handover_Prelim.ipynb)
- Runner: [run_swsr_handover_prelim.py](../../../code/colab/run_swsr_handover_prelim.py)
- Tests: [test_swsr_handover.py](../../../code/tests/test_swsr_handover.py)
- Report: [SWSR_HANDOVER_REPORT.md](results/SWSR_HANDOVER_REPORT.md)

## Motivation and Evaluated Methods

Earlier experiments (`EXP-012` through `EXP-017`) showed that:
1. Single-step velocity drift under Definition 3.4 ($d_k \le \lambda$) fails on noisy prompts because prompt noise dampens early trajectory velocity rather than inflating it.
2. Binary failure rules flag ~26% of noisy inputs even between two statistically indistinguishable reference schedules (`g16_8` vs. `s025`).
3. Calibrated single-threshold SNR gates jump directly to the earliest candidate step in the grid, collapsing to `0.0%–1.5%` NFE saving whenever ultra-early checkpoints (`Step 1`, `7 NFE`) are included.

`EXP-018` evaluates **zero-SNR trajectory-only cascades** directly on continuous quality metrics (**Raw WER, Capped WER $\le 1.0$, Outlier-Trimmed WER, SIM, UTMOS**) and full computational cost/speed telemetry (**Teacher Evals, NFE, Gate Overhead, Teacher ODE Time, End-to-End Wall-Clock Time, Speedup $\times$, RTF, $\times$-Real-Time, and Peak GPU Memory**) across **all 6 grid configurations**:
- **`STAC` (`Stage-Wise Trajectory Action Cascade`):** Cumulative discrete arc length $A_k = \sum_{j=1}^{k} d_j \ge \hat{\lambda}_k$ (`0.12 ms` gate overhead, zero external SNR estimation).
- **`1D-CRC-STAC`:** Conformal Risk-Controlled cascade ($\alpha = 0.05$) with bounded loss $L \in [0, 1]$.
- **Matched-DOF Control (`Stage-Wise SNR Cascade`):** $\hat{S} \ge \hat{\theta}_k$ (`1.85 ms` gate overhead) with the exact same number of stage thresholds $M$ as `STAC`.
- **Ablations & Baselines:** Stage-Wise Single-Step Drift ($d_k \ge \hat{\lambda}_k$), Single-Threshold Trajectory Action ($A_k \ge \hat{\lambda}$), Calibrated SNR-only, 20 dB SNR gate, Definition 3.4, `SNR + TAS`, and hybrid readiness scores (`SWSR`, `STSR`, `CSDR`).

## Hardware, Data, and Verification Checks (2026-11-07, Tesla T4)

- **Hardware:** NVIDIA Tesla T4 on Google Colab (mean `t_prepare = 81.95 ms`, `t_eval = 298.81 ms` per guided teacher step, `t_student = 152.46 ms` per student step, `t_vocoder = 5.51 ms`, mean audio length `4.80 s`, mean peak memory `9,587.7 MB`).
- **Full Grid Synthesis (`steps1to8.csv`, 360/360 cases):** 60 development speakers × 6 conditions (`clean`, `noisy15`, `gauss18`, `gauss20`, `gauss22`, `gauss25`), covering all 8 checkpoints on the 16-interval sway grid (`g16_1..g16_8`, `7..21 NFE`) and all 15 checkpoints on the 32-interval sway grid (`g32_1..g32_15`, `7..35 NFE`), with 5-fold speaker-grouped cross-validation (development evidence).
- **Live Early-Stopping Execution (`live_swsr_t4.csv`, 30/30 cases):** 10 speakers × 3 conditions (`clean`, `gauss20`, `noisy15`) running genuine online teacher loop termination at $\hat{k}$ versus full Step-8 reference execution on the same Tesla T4.
  - **Bitwise Trajectory Identity:** `max |endpoint_diff| = 0.0` across all 30 live GPU cases (`smoke/live_swsr_t4.csv` and `live_swsr_t4.csv`).

## Key Results Summary

### 1. Live Tesla T4 Early-Stopping Speed, Latency & Quality (`live_swsr_t4.csv`, $n = 30$)

| Condition ($n$) | Mean Chosen Step | Mean NFE (Saving) | Teacher ODE Time (`STAC` vs Ref) | Teacher Speedup | E2E Wall-Clock (`STAC` vs Ref) | E2E Speedup (Saved) | RTF / $\times$-Real-Time (`STAC` vs Ref) | WER (`STAC` vs Ref) | SIM (`STAC` vs Ref) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`clean` ($n=10$)** | `4.10` | **13.20** (`37.1%`) | `1.876s` vs `3.143s` | **1.68x** | **2.575s** vs `3.831s` | **1.49x** (`-1.26s` / `32.8%`) | `0.643` (**1.55x RT**) vs `0.940` (`1.06x RT`) | `0.0389` vs `0.0389` | `0.9418` vs `0.9406` |
| **`gauss20` ($n=10$)** | `6.40` | **17.80** (`15.2%`) | `2.574s` vs `3.257s` | **1.27x** | **3.231s** vs `3.916s` | **1.21x** (`-0.69s` / `17.5%`) | `0.796` (**1.26x RT**) vs `0.924` (`1.08x RT`) | `0.0330` vs `0.0488` | `0.8415` vs `0.8340` |
| **`noisy15` ($n=10$)** | `8.00` | **21.00** (`0.0%`) | `3.264s` vs `3.213s` | `0.98x` | **3.915s** vs `3.862s` | `0.99x` (`+0.05s` / `0.0%`) | `0.927` (`1.08x RT`) vs `0.912` (`1.10x RT`) | `0.2851` vs `0.2851` | `0.7798` vs `0.7798` |
| **Overall ($n=30$)** | `6.17` | **17.33** (`17.5%`) | `2.572s` vs `3.205s` | **1.25x** | **3.240s** vs `3.869s` | **1.19x** (`-0.63s` / `16.3%`) | `0.789` (**1.27x RT**) vs `0.925` (`1.08x RT`) | **0.1190** vs `0.1243` | **0.8544** vs `0.8515` |

Without any external SNR estimator (`0.12 ms` gate overhead), live `STAC` exits at Step `4.10` on clean speech (`1.49x` end-to-end speedup, `1.68x` teacher ODE speedup), routes intermediate 20 dB speech to Step `6.40` (`1.21x` E2E speedup), and falls back `100%` of the time (`10/10` at Step `8.00`) on 15 dB degraded prompts.

### 2. Multi-Grid Comparison Across All 6 Sway-Grid Configurations (`steps1to8.csv`, $n = 360$)

| Grid Configuration | Policy | Mean NFE [95% CI] (Saving) | E2E Time / Speedup / RTF | Raw WER ($\Delta$) | Capped WER $\le 1.0$ ($\Delta$) | SIM ($\Delta$) | TOST $p_{\text{WER}}$ |
|---|---|---:|---:|---:|---:|---:|---:|
| **`g16`** (`{4,5,6}` vs `8`, `21 NFE`) | **`STAC`** | **17.44** `[17.05, 17.83]` (**16.9%**) | `2.69s` (**1.20x**) / `0.579` (**1.73x RT**) | **0.0644** (`-0.0036`) | **0.0634** (`-0.0036`) | `0.8091` (`-0.0014`) | `3.5e-5` |
| | `Stage-Wise SNR Cascade` | `18.17` `[17.96, 18.42]` (`13.5%`) | `2.81s` (`1.15x`) / `0.602` (`1.66x RT`) | `0.0670` (`-0.0009`) | `0.0661` (`-0.0009`) | `0.8094` (`-0.0010`) | `3.0e-24` |
| **`g16_all8`** (`1..7` vs `8`, `21 NFE`) | **`STAC`** | **16.82** `[16.32, 17.27]` (**19.9%**) | `2.60s` (**1.25x**) / `0.560` (**1.79x RT**) | **0.0687** (`+0.0008`) | **0.0678** (`+0.0008`) | `0.8094` (`-0.0011`) | `1.2e-7` |
| | **`1D-CRC-STAC`** | **17.67** `[17.26, 18.04]` (**15.8%**) | `2.73s` (**1.19x**) / `0.586` (**1.71x RT**) | **0.0670** (`-0.0009`) | **0.0661** (`-0.0009`) | `0.8101` (`-0.0003`) | `2.6e-8` |
| | `Stage-Wise SNR Cascade` | `18.13` `[17.80, 18.48]` (`13.7%`) | `2.80s` (`1.15x`) / `0.602` (`1.66x RT`) | `0.0682` (`+0.0003`) | `0.0673` (`+0.0003`) | `0.8098` (`-0.0006`) | `1.5e-20` |
| | `Calibrated SNR-only` | `20.69` `[20.46, 20.88]` (`1.5%`) | `3.19s` (`1.01x`) / `0.682` (`1.47x RT`) | `0.0684` (`+0.0005`) | `0.0675` (`+0.0005`) | `0.8101` (`-0.0004`) | `6.2e-29` |
| **`g16_le025`** (`1..6` vs `7`, `19 NFE`) | **`STAC`** | **15.64** `[15.27, 16.01]` (**17.7%**) | `2.42s` (**1.21x**) / `0.522` (**1.91x RT**) | **0.0692** (`+0.0009`) | **0.0683** (`+0.0009`) | `0.8093` (`-0.0006`) | `1.7e-5` |
| | `Stage-Wise SNR Cascade` | `17.28` `[17.06, 17.53]` (`9.1%`) | `2.68s` (`1.09x`) / `0.574` (`1.74x RT`) | `0.0685` (`+0.0003`) | `0.0676` (`+0.0003`) | `0.8097` (`-0.0002`) | `1.5e-20` |
| **`g32`** (`{7,9,11,13}` vs `14`, `33 NFE`) | **`STAC`** | **26.18** `[25.42, 26.83]` (**20.7%**) | `3.99s` (**1.26x**) / `0.860` (**1.16x RT**) | **0.0699** (`-0.0038`) | **0.0688** (`-0.0038`) | `0.8099` (`-0.0007`) | `0.0067` |
| | `Stage-Wise SNR Cascade` | `27.94` `[27.59, 28.33]` (`15.3%`) | `4.27s` (`1.18x`) / `0.914` (`1.09x RT`) | `0.0738` (`+0.0000`) | `0.0726` (`+0.0000`) | `0.8109` (`+0.0003`) | `9.1e-9` |
| **`g32_all14`** (`1..13` vs `14`, `33 NFE`) | **`STAC`** | **27.14** `[26.04, 28.10]` (**17.8%**) | `4.15s` (**1.21x**) / `0.894` (**1.12x RT**) | **0.0738** (`+0.0001`) | **0.0726** (`+0.0001`) | `0.8089` (`-0.0017`) | `0.0029` |
| | `Stage-Wise SNR Cascade` | `28.01` `[27.27, 28.80]` (`15.1%`) | `4.29s` (`1.17x`) / `0.918` (`1.09x RT`) | `0.0715` (`-0.0022`) | `0.0706` (`-0.0020`) | `0.8094` (`-0.0012`) | `0.0081` |
| | `Calibrated SNR-only` | `33.00` `[33.00, 33.00]` (`0.0%`) | `5.03s` (`1.00x`) / `1.076` (`0.93x RT`) | `0.0737` (`+0.0000`) | `0.0726` (`+0.0000`) | `0.8106` (`+0.0000`) | `0.0` |
| **`g32_all15`** (`1..14` vs `15`, `35 NFE`) | **`STAC`** | **27.51** `[26.35, 28.59]` (**21.4%**) | `4.18s` (**1.28x**) / `0.903` (**1.11x RT**) | `0.1519` (`+0.0784`) | **0.0780** (`+0.0064`) | `0.8073` (`-0.0027`) | `0.81` |
| | `Stage-Wise SNR Cascade` | `27.93` `[27.32, 28.60]` (`20.2%`) | `4.27s` (`1.25x`) / `0.913` (`1.09x RT`) | `0.0731` (`-0.0004`) | `0.0711` (`-0.0004`) | `0.8103` (`+0.0003`) | `0.0002` |
| | `Single-Threshold Action` | `30.78` `[30.31, 31.23]` (`12.0%`) | `4.68s` (`1.14x`) / `1.006` (`0.99x RT`) | **0.0718** (`-0.0016`) | `0.0706` (`-0.0010`) | `0.8105` (`+0.0006`) | `1.5e-7` |

### 3. Diagnostic on `g32_all15` (14 Sequential Thresholds + Whisper Repetition Outlier)

On 5 of the 6 grid configurations (`g16`, `g32`, `g16_all8`, `g16_le025`, and `g32_all14`), `STAC` passes TOST equivalence cleanly ($p_{\text{WER}} \le 0.0067$) while saving `16.9%–20.7%` NFE and outperforming both `Calibrated SNR-only` and the matched-DOF `Stage-Wise SNR Cascade`. On `g32_all15` (14 sequential thresholds fit on 48 speakers per CV fold, including ultra-early `g32_1` at $t_1 = 0.0012$ and `g32_2` at $t_2 = 0.0048$), `STAC` saves `21.4%` NFE with **Capped WER `0.0780` (`+0.0064` vs. reference)** and **Outlier-Trimmed WER `0.0728`**, but a Whisper repetition-hallucination outlier (`~26.6` excess WER on 1–2 out-of-fold utterances routed to `g32_2`) inflates **Raw WER** to `0.1519` (`TOST p = 0.81`). Enforcing a minimum candidate floor ($t_k \ge 0.019$, i.e., `g32_4` / `g16_2`) or a sparse 4–7 stage grid (`g32`, `g16_all8`) eliminates ultra-early repetition outliers while retaining `~20%` NFE savings.

### 4. 500-Speaker Benchmark (`all1000`) & Zero-Shot OOD Generalization (`EXP-011`)

- **500-Speaker Benchmark (200 Calibration -> 200 Test Speakers, full `d_1..d_8`):** `STAC` and `1D-CRC-STAC` achieve **16.63 NFE (`20.8%` saving, `2.57s`, `1.25x` speedup, RTF `0.581`)**, Capped WER `0.1223` (`-0.0008` vs. Step 8), and Raw TOST $p_{\text{WER}} = 0.0018$.
- **Zero-Shot OOD (`EXP-011`, $n = 100$ across `babble20`, `babble15`, `reverb03`, `reverb09`, `telephone`):** `STAC` achieves **14.14 NFE (`32.7%` saving)** vs. **17.40 NFE (`17.1%` saving)** for `Stage-Wise SNR Cascade` (**paired difference `-3.26 NFE [-4.04, -2.48]`**) with $\Delta\text{WER} = -0.0021$.

## Scope Limit

All `EXP-018` 60-speaker results use 5-fold speaker-grouped cross-validation on the `EXP-012` Quick development speakers, and the 500-speaker `all1000` test split was previously inspected (`AUDIT-001`). Accordingly, these results are recorded strictly as **development evidence** prior to a frozen, unseen-speaker evaluation (`EVAL-001` / `EVAL-002`).

