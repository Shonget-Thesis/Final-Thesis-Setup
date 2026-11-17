# Master Handover Methods & Quality/Speed Benchmark Comparison (`EXP-011` to `EXP-018`)

**Updated:** 2026-11-07 | **Hardware:** NVIDIA Tesla T4 (Google Colab; `EXP-015` operators noted with A100 caveat)  
**Thesis Title ([`frontmatter.tex`](../paper/frontmatter.tex)):** *A Risk-Calibrated Adaptive Handover Method for Teacher-Guided Flow-Matching Text-to-Speech Inference*  
**Evidence Status:** Development evidence (5-fold speaker-grouped cross-validation on 60 development speakers for `EXP-012`–`EXP-018`, 200 cal $\to$ 200 test speakers on previously inspected `all1000`, and 20-speaker OOD check on `EXP-011`). Frozen unseen-speaker evaluation (`EVAL-001` / `EVAL-002`) and adviser sign-off remain pending.

---

## 1. Where All Raw Results & Generated Reports Live

| Experiment | Focus | Primary Report | Raw CSV / JSON Data & SHA-256 Manifest |
|---|---|---|---|
| **`EXP-018` (`EVAL-011`)** | **`STAC` & `1D-CRC-STAC`** across **16-Grid (`Steps 1..8`)** & **32-Grid (`Steps 1..15`)** + Live T4 Early Exit (`n=30`) + `all1000` + OOD | [`SWSR_HANDOVER_REPORT.md`](../research/experiments/EXP-018-swsr-handover-prelim/results/SWSR_HANDOVER_REPORT.md) / [`README.md`](../research/experiments/EXP-018-swsr-handover-prelim/README.md) | [`steps1to8.csv`](../research/experiments/EXP-018-swsr-handover-prelim/results/steps1to8.csv), [`live_swsr_t4.csv`](../research/experiments/EXP-018-swsr-handover-prelim/results/live_swsr_t4.csv), [`swsr_results.json`](../research/experiments/EXP-018-swsr-handover-prelim/results/swsr_results.json), [`MANIFEST.json`](../research/experiments/EXP-018-swsr-handover-prelim/results/MANIFEST.json) |
| **`EXP-017` (`EVAL-010`)** | Chapter 3 rehearsal on 16-Grid (`{4,5,6,8}`) & 32-Grid (`s007..s025`), `Teacher32`, `Student4`, `Paper 14+2`, `Alg 2` | [`GRID32_REPORT.md`](../research/experiments/EXP-017-grid32-prelim/results/GRID32_REPORT.md) | [`MANIFEST.json`](../research/experiments/EXP-017-grid32-prelim/results/MANIFEST.json) |
| **`EXP-015` (`EVAL-007`)** | Noise-aware operators & detectors (`TAS`, `SWC`, `SNNG`, `Antithetic`, `SCG`, `PIEC`, `Demucs`) | [`NOISE_GUIDANCE_REPORT.md`](../research/experiments/EXP-015-noise-guidance-prelim/results/NOISE_GUIDANCE_REPORT.md) / [`PROPOSED_CANDIDATES_TAS_SWC_REPORT.md`](../research/reports/Ref_reports/PROPOSED_CANDIDATES_TAS_SWC_REPORT.md) | [`MANIFEST.json`](../research/experiments/EXP-015-noise-guidance-prelim/results/MANIFEST.json) |
| **`EXP-014` (`EVAL-006`)** | Zero-cost Step-4 handover operators (`TCR`, `cDEH`, `SEA`) | [`HANDOVER_OPERATORS_REPORT.md`](../research/experiments/EXP-014-handover-operators-prelim/results/HANDOVER_OPERATORS_REPORT.md) | [`MANIFEST.json`](../research/experiments/EXP-014-handover-operators-prelim/results/MANIFEST.json) |
| **`EXP-013` (`EVAL-005`)** | Drift-Extrapolated Handover (`DEH`, `DEH-half`) | [`EXTRAPOLATION_REPORT.md`](../research/experiments/EXP-013-endpoint-extrapolation-prelim/results/EXTRAPOLATION_REPORT.md) | [`MANIFEST.json`](../research/experiments/EXP-013-endpoint-extrapolation-prelim/results/MANIFEST.json) |
| **`EXP-012` (`EVAL-004`)** | Consistency gates (`SDC`, `Low-Drift Veto`, `Guidance Contrast`, `Definition 3.4`) | [`CONSISTENCY_GATE_REPORT.md`](../research/experiments/EXP-012-consistency-gate-prelim/results/CONSISTENCY_GATE_REPORT.md) | [`MANIFEST.json`](../research/experiments/EXP-012-consistency-gate-prelim/results/MANIFEST.json) |
| **`EXP-011` (`EVAL-003`)** | Seed-noise floor & OOD corruptions (`babble`, `reverb`, `telephone`) | [`README.md`](../research/experiments/EXP-011-followup-t4/README.md) | [`MANIFEST.json`](../research/experiments/EXP-011-followup-t4/results/MANIFEST.json) |

---

## 2. Basis of Comparison: Earlier Rule (`EXP-012`–`017`) vs. Direct Quality/Cost Rule (`EXP-018`)

All preliminary experiments (`EXP-012` through `EXP-018`) evaluate the **same 360 utterances** (`60 development speakers × 6 prompt conditions`: `clean`, `noisy15`, `gauss18`, `gauss20`, `gauss22`, `gauss25`) using **5-fold speaker-grouped cross-validation** (`seed = 2026`).

### 2A. Why Two Evaluation Frameworks Exist in the Repository

| Dimension | Framework A (`EXP-012` to `EXP-017`) | Framework B (`EXP-018`) |
|---|---|---|
| **Quality Rule** | **Binary per-utterance failure** ([`failure_config.json`](../code/scheduling/failure_config.json), Definition 3.1): fails if $\text{WER} > \max(0.125, \text{WER}_{\text{ref}} + 0.01)$ or $\text{SIM} < \text{SIM}_{\text{ref}} - 0.02$. Target: added failures $\le 5\%$ (`<= 18/360`). | **Continuous quality + TOST equivalence** ([`run_swsr_handover_prelim.py`](../code/colab/run_swsr_handover_prelim.py)): paired Raw WER, **Capped WER ($\le 1.0$)**, Outlier-Trimmed WER, SIM, UTMOS, and **TOST $p < 0.05$** within $\pm 0.01$ WER and $\pm 0.02$ SIM. |
| **Calibration** | Fixed-sequence **Learn-Then-Test (LTT)** ($\alpha = 0.05, \delta = 0.10$) on binary failure counts; paired **Exact McNemar** test. | Stage-wise empirical risk (**`STAC`**) and 1-parameter **Conformal Risk Control (`1D-CRC-STAC`, $\alpha = 0.05$)** on bounded continuous loss $L \in [0, 1]$; paired **Wilcoxon & TOST**. |
| **Operator Gates (`EXP-013`–`015`)** | **`C1`**: noisy Step-4 failures drop $\ge 25\%$ ($p < 0.05$); **`C2`**: clean failures rise $\le +1$; **`C3`**: routed policy beats `Calibrated SNR` (`18.71 NFE`) within $5\%$ risk. | Direct comparison against **both** 1-cut `Calibrated SNR` **and** a matched-DOF **`Stage-Wise SNR Cascade`** ($\hat{S} \ge \hat{\theta}_k$) with Speaker-Cluster Bootstrap 95% CIs. |
| **Key Diagnostic Finding** | Comparing the **two equal-quality references** (`32-grid s025` vs `16-grid g16_8`, `0.0000` SIM difference) still flagged **`81/360` (`22.5%`) binary "failures"** purely from stochastic student noise on noisy prompts ([`GRID32_REPORT.md`](../research/experiments/EXP-017-grid32-prelim/results/GRID32_REPORT.md)). | Continuous paired differences ($\Delta\text{WER}, \Delta\text{SIM}$) average out zero-mean student seed noise while Capped WER ($\le 1.0$) isolates single-utterance Whisper repetition hallucinations. |

### 2B. The 5 Objective Criteria Used to Rank All Handover Methods

1. **Criterion 1 — Statistical Quality Equivalence (Hard Gate):** Must pass TOST equivalence ($p_{\text{WER}} < 0.05$ at margin `+0.010`, $p_{\text{SIM}} < 0.05$ at margin `-0.020`) without degrading clean prompts (or keep binary risk $\le 5\%$ under Framework A).
2. **Criterion 2 — Computational Cost & Wall-Clock Speedup (Primary Metric):** Lowest Mean NFE (forward passes), shortest End-to-End synthesis runtime (`s`), highest speedup (`x`), and lowest Real-Time Factor (`RTF`).
3. **Criterion 3 — Full-Grid Granularity (No Step-1 Collapse):** Must harvest early and intermediate steps (`Steps 1..7`) rather than collapsing back to the reference (`0%–4%` saving) when `Step 1` (`7 NFE`) is included.
4. **Criterion 4 — Zero External Sensor Overhead & OOD Robustness:** Using internal flow-matching trajectory states (`0.12 ms` gate overhead) beats external waveform SNR estimation (`1.85–1.97 ms` overhead; brittle under babble, reverb, and bandpass channels).
5. **Criterion 5 — Thesis Scope Integrity ([`01-introduction.tex`](../paper/chapters/01-introduction.tex)):** Changes **only the teacher-to-student switching time**, keeping teacher weights, student weights, duration predictor, student restart level, and input prompt unchanged.

---

## 3. Master Head-to-Head Ranking of All Handover Methods (`EXP-012` → `EXP-018`)

All numbers below are out-of-fold 5-fold speaker CV on the 360-case benchmark (16-grid reference `g16_8` = **21.00 NFE, 3.23s, RTF 0.693, WER 0.0679, SIM 0.8104**; 32-grid reference `s025` = **33.00 NFE, 5.03s, RTF 1.076, WER 0.0737, SIM 0.8106**):

| Rank | Method (Experiment) | Mathematical Rule / Mechanism | Pure Timing? | Needs SNR? | 16-Grid (`Steps 1..8`) NFE / Saved / Time / WER | 32-Grid (`g32`) NFE / Saved / Time / WER | Framework A (`EXP-012`–`017`) Status | Verdict |
|:---:|---|---|:---:|:---:|---|---|---|---|
| **1** | **`STAC`** (`EXP-018`) | $A_k = \sum_{j=1}^k d_j \ge \hat{\lambda}_k$ | **Yes** | **No** (`0.12 ms`) | **`16.82` (`19.9%`)** / **`2.60s` (`1.25x`)** / WER `0.0687` (TOST $p=1.2\text{e-}7$) | **`26.18` (`20.7%`)** / **`3.99s` (`1.26x`)** / WER **`0.0699`** (TOST $p=0.0067$) | — *(continuous calibration)* | **#1 Overall Winner:** Fastest valid pure handover policy across 16-grid, 32-grid, `all1000` (`20.8%`), and OOD (`32.7%`), with zero SNR overhead. |
| **2** | **`1D-CRC-STAC`** (`EXP-018`) | $A_k \ge \hat{\lambda}_k(\hat{\tau})$, $\alpha = 0.05$ | **Yes** | **No** (`0.12 ms`) | **`17.67` (`15.8%`)** / **`2.73s` (`1.19x`)** / WER **`0.0670`** (TOST $p=2.6\text{e-}8$) | **`27.99` (`15.2%`)** / **`4.27s` (`1.18x`)** / WER **`0.0721`** (TOST $p=1.7\text{e-}5$) | Under LTT (`g16`): **`9/360` (`2.5%` $\le 5\%$)**, `18.52` NFE (beats `snr_cal` without SNR!) | **#1 Risk-Certified Winner:** Formal $\alpha=0.05$ risk control + lower WER than full Step-8 reference. |
| **3** | **`Stage-Wise Single-Step Drift`** (`EXP-018`) | $d_k \ge \hat{\lambda}_k$ (inverted sign) | **Yes** | **No** (`0.12 ms`) | `16.91` (`19.5%`) / `2.61s` (`1.24x`) / WER `0.0684` (TOST $p=4.4\text{e-}6$) | `28.35` (`14.1%`) / `4.31s` (`1.17x`) / WER `0.0731` (TOST $p=0.012$) | — | **Strongest Zero-SNR Ablation:** Proves flipping $d_k \le \lambda \to d_k \ge \lambda_k$ works, but loses `6.6%` saving to `STAC` on `g32` due to memoryless single-step jitter. |
| **4** | **`Stage-Wise SNR Cascade`** (`EXP-018`) | $\hat{S} \ge \hat{\theta}_k$ (matched DOF) | **Yes** | Yes (`1.85 ms`) | `18.13` (`13.7%`) / `2.80s` (`1.15x`) / WER `0.0682` (TOST $p=1.5\text{e-}20$) | `27.94` (`15.3%`) / `4.27s` (`1.18x`) / WER `0.0738` (TOST $p=9.1\text{e-}9$) | — | **Best SNR-Only Control:** Loses to `STAC` by `6.2%` on `g16_all8`, `8.6%` on `g16_le025`, `5.4%` on `g32`, and `15.6%` on OOD noise. |
| **5** | **`STSR`** (`EXP-018`) | $\hat{S}\cdot t_k \cdot d_k \ge \hat{\lambda}$ | **Yes** | Yes (`1.97 ms`) | `18.72` (`10.9%`) / `2.89s` (`1.12x`) / WER `0.0692` (TOST $p=4.2\text{e-}8$) | `29.22` (`11.5%`) / `4.45s` (`1.13x`) / WER `0.0725` (TOST $p=8.8\text{e-}8$) | — | **Best 1-Parameter Hybrid:** Safe, but 1 shared threshold leaves `~9%` NFE saving on the table vs `STAC`. |
| **6** | **`SNR + TAS`** (`EXP-015` / `017` / `018`) | $\hat{S} \ge 20\text{ dB} \land A_{\text{Mahal}}(z) \le \hat{\tau}$ | **Yes** | Yes (`1.97 ms`) | `17.38` (`17.2%` on `g16`), **drops to `20.69` (`1.5%`) on `1..8`** | `27.75` (`15.9%` on `g32`), **drops to `33.00` (`0.0%`) on `1..14`** | **`12/360` (`3.3%` $\le 5\%$)**, `18.07–18.20` NFE (`14.0%` saved) | **Best Pre-`EXP-018` Detector:** Beat `SNR-20dB`, but binary `{earliest, latest}` routing collapses when `Step 1` is included. |
| **7** | **`Low-Drift Veto` / `SDC Gate`** (`EXP-012`) | $\hat{S} \ge 20\text{ dB} \land d_4 \ge \hat{\lambda}_{\text{low}}$ | **Yes** | Yes | `18.38` (`12.5%`) / `18.44` (`12.2%`) on `{4, 8}` | — | `9/360` (`2.5%`) & `10/360` (`2.8%`) pooled (`20.29` stratified) | First showed noise *lowers* drift, but tied with `Calibrated SNR` (`18.71` NFE). |
| **8** | **`Calibrated SNR Gate`** (`EXP-012`–`018`) | $\hat{S} \ge \hat{\theta} \to$ earliest step | **Yes** | Yes (`1.85 ms`) | `18.64` (`11.2%` on `g16`), **drops to `20.69` (`1.5%`) on `1..8`** | `29.23` (`11.4%` on `g32`), **drops to `33.00` (`0.0%`) on `1..14`** | `7/360` (`1.9%` $\le 5\%$), `18.71` NFE (`10.9%` saved) | Fails Criterion 3: jumping to `Step 1` forces $\hat{\theta}$ above nearly all prompts. |
| **9** | **`SWSR` / `CSDR` / `Single-Threshold Action`** (`EXP-018`) | 1 shared $\hat{\lambda}$ across all steps | **Yes** | Mixed | `20.15–20.30` (`3.3%–4.0%` saved on `1..8`) | `27.57–31.18` (`5.5%–16.4%`) | — | Proves why **per-step thresholds $\hat{\lambda}_k$** are mandatory (`3.3%` vs `19.9%`). |
| **10** | **`Old Definition 3.4` & `Stabilization-Only`** (`EXP-012` / `017` / `018`) | $d_k \le \hat{\lambda}$ (with/without 20 dB gate) | **Yes** | Mixed | **`21.00` (`0.0%` saved)** / `3.23s` (`1.00x`) | **`32.79–32.82` (`0.6%` saved)** | `0/360` (`0.0%` saved — nothing certified on 16-grid) | **Inert:** Stopping when $d_k \le \lambda$ triggers on noisy prompts first, so calibration shuts it off. |
| **—** | **`SNR + SWC`** (`EXP-015`, *Different Category*) | Spectral Wiener subtraction on prompt mel features | **No** *(alters prompt)* | Yes | **`16.98` (`19.1%`)** heavy-routed (`13.00` unrouted) | — | `11/360` (`3.1%` heavy-routed); **FAILED `C2` (`clean 2->5`) & `C3`**; **A100 vs T4 caveat** | **Prompt Denoiser (Outside Timing Scope):** Raises noisy SIM `+0.126`, but alters conditioning, lacks T4 confirmation (`EVAL-008`), and was removed from proposal (Appendix E). |
| **✗** | **`DEH`, `TCR`, `cDEH`, `SEA`, `SNNG`, `Antithetic`, `SCG`, `PIEC`** (`EXP-013`–`015`) | Endpoint extrapolation / restart / noise guidance | **No** *(alters state)* | Yes | `18.56–20.69` NFE routed | — | **ALL FAILED `C3`** (`DEH` broke clean `2->25`; `TCR/SNNG/PIEC` worsened noisy speech) | **Rejected.** |
| **✗** | **`Uncalibrated 20 dB SNR Gate`** (`EXP-011`–`018`) | $\hat{S} \ge 20\text{ dB} \to$ earliest step | **Yes** | Yes (`1.85 ms`) | `9.45` NFE (`1..8`), **WER `1.5707` (TOST FAIL)**; `14.40` NFE (`g16`), **TOST FAIL** | `21.45` NFE, **WER `0.1198` (TOST FAIL)** | **`86/360` (`23.9%` failures — 4.8x over 5% limit!)** | **Disqualified (Fails Quality Gate).** |

---

## 4. Full Speech Quality & Computational Speed Benchmarks (`EXP-018` & `EXP-017`, Tesla T4, $n = 360$)

Measured Tesla T4 unit latencies across all 360 cases ([`steps1to8.csv`](../research/experiments/EXP-018-swsr-handover-prelim/results/steps1to8.csv)):
- Preparation: `81.95 ms` | Guided Teacher Step (2 passes): `298.81 ms` | Student Step (1 pass): `152.46 ms` | Vocos Vocoder: `5.51 ms` | Mean Audio Length: `4.80 s` | Mean Peak GPU Memory: `9,587.7 MB`.

### 4A. Fixed Baselines & Sway-Grid Checkpoints Quality/Speed Profile ($n = 360$)

| Schedule / Checkpoint | Sway Time $t_k$ | Teacher Evals + Student | Total NFE | Runtime (s) | Speedup vs `g16_8` | RTF ($\times$ Real-Time) | WER (Clean) | WER (Noisy) | WER (All) | Capped WER (All) | SIM (Clean) | SIM (Noisy) | SIM (All) | UTMOS (Clean / Noisy) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`g16_1`** (16-grid Step 1) | `0.0048` | `2 + 3` | **7** | `1.14s` | `2.83x` | `0.245` (`4.08x RT`) | `0.0306` | `2.8017` | `2.3398` | — | `0.9258` | `0.7052` | `0.7419` | `3.059 / 1.282` |
| **`g16_2`** (16-grid Step 2) | `0.0192` | `3 + 3` | **9** | `1.44s` | `2.24x` | `0.309` (`3.23x RT`) | `0.0264` | `0.4344` | `0.3664` | — | `0.9356` | `0.7635` | `0.7922` | `3.855 / 1.382` |
| **`g16_3`** (16-grid Step 3) | `0.0431` | `4 + 3` | **11** | `1.74s` | `1.86x` | `0.373` (`2.68x RT`) | `0.0279` | `0.3569` | `0.3020` | — | `0.9378` | `0.7701` | `0.7981` | `3.867 / 1.456` |
| **`g16_4`** (Code default `0.07`) | `0.0761` | `5 + 3` | **13** | `2.04s` | `1.59x` | `0.437` (`2.29x RT`) | `0.0234` | `0.2469` | `0.2097` | `0.0954` | `0.9357` | `0.7768` | `0.8033` | `3.922 / 1.479` |
| **`g16_5`** (`s = 0.10`) | `0.1181` | `6 + 3` | **15** | `2.34s` | `1.38x` | `0.501` (`2.00x RT`) | `0.0261` | `0.0816` | `0.0723` | `0.0714` | `0.9352` | `0.7812` | `0.8069` | `3.921 / 1.478` |
| **`g16_6`** (`s = 0.15`) | `0.1685` | `7 + 3` | **17** | `2.64s` | `1.23x` | `0.565` (`1.77x RT`) | `0.0255` | `0.0742` | `0.0661` | `0.0652` | `0.9356` | `0.7831` | `0.8085` | `3.923 / 1.475` |
| **`g16_7`** (Last 16-grid $\le 0.25$) | `0.2270` | `8 + 3` | **19** | `2.94s` | `1.10x` | `0.629` (`1.59x RT`) | `0.0272` | `0.0765` | `0.0683` | `0.0673` | `0.9362` | `0.7846` | `0.8099` | `3.932 / 1.466` |
| **`g16_8`** (**16-Grid Reference**) | `0.2929` | `9 + 3` | **21** | **3.23s** | **1.00x** | `0.693` (`1.44x RT`) | `0.0272` | `0.0761` | **0.0679** | **0.0670** | `0.9359` | `0.7854` | **0.8104** | `3.924 / 1.460` |
| **`g32_7`** (`s007` on 32-grid) | `0.0585` | `8 + 3` | **19** | `2.94s` | `1.10x` | `0.629` (`1.59x RT`) | `0.0234` | `0.3663` | `0.3091` | — | `0.9367` | `0.7772` | `0.8038` | `3.927 / 1.479` |
| **`g32_9`** (`s010` on 32-grid) | `0.0960` | `10 + 3` | **23** | `3.53s` | `0.92x` | `0.757` (`1.32x RT`) | `0.0255` | `0.1845` | `0.1580` | — | `0.9355` | `0.7830` | `0.8084` | `3.886 / 1.489` |
| **`g32_11`** (`s015` on 32-grid) | `0.1423` | `12 + 3` | **27** | `4.13s` | `0.78x` | `0.884` (`1.13x RT`) | `0.0248` | `0.1285` | `0.1112` | — | `0.9351` | `0.7854` | `0.8103` | `3.909 / 1.488` |
| **`g32_13`** (`s020` on 32-grid) | `0.1968` | `14 + 3` | **31** | `4.73s` | `0.68x` | `1.012` (`0.99x RT`) | `0.0269` | `0.0748` | `0.0668` | — | `0.9358` | `0.7867` | `0.8116` | `3.907 / 1.478` |
| **`g32_14`** (**32-Grid `s025` Ref**) | `0.2270` | `15 + 3` | **33** | **5.03s** | `0.64x` | `1.076` (`0.93x RT`) | `0.0269` | `0.0831` | **0.0737** | **0.0726** | `0.9365` | `0.7854` | **0.8106** | `3.908 / 1.475` |
| **`g32_15`** (First 32-grid $> 0.25$) | `0.2590` | `16 + 3` | **35** | **5.33s** | `0.61x` | `1.140` (`0.88x RT`) | `0.0269` | `0.0828` | **0.0735** | **0.0716** | `0.9355` | `0.7849` | **0.8100** | `3.904 / 1.473` |
| **`Paper 14+2`** (`j=2`, `EXP-017`) | `0.2270` | `15 + 2` | **32** | `4.87s` | `0.66x` | `1.044` (`0.96x RT`) | `0.0272` | `0.3417` | `0.2893` | — | `0.9333` | `0.7886` | `0.8127` | `3.836 / 1.505` |
| **`Algorithm 2`** (No re-noise, `EXP-017`) | `0.2270` | `14 + 2` | **30** | `4.58s` | `0.71x` | `0.980` (`1.02x RT`) | `0.0263` | `0.4101` | `0.3461` | — | `0.9266` | `0.7748` | `0.8001` | `3.824 / 1.418` |
| **`Teacher-Only 32`** (F5-TTS, `EXP-017`) | `1.0000` | `32 + 0` | **64** | `9.65s` | `0.33x` | `2.065` (`0.48x RT`) | `0.0246` | `0.9736` | `0.8154` | — | `0.9281` | `0.7760` | `0.8014` | `3.269 / 1.367` |
| **`Student-Only 4`** (No teacher, `EXP-017`) | `0.0000` | `0 + 4` | **4** | `0.70s` | `4.61x` | `0.150` (`6.67x RT`) | `0.0215` | `0.0491` | `0.0445` | — | `0.9369` | `0.8624` | `0.8748` | `3.932 / 2.688` |

> **Note on `Student-Only 4`:** While pure 4-step student sampling is fast and noise-tolerant, DMOSpeech 2 uses teacher guidance specifically to recover **prosodic/pitch diversity** (`89.1%` of F5-TTS pitch $\text{CV}_{f_0}$), which `Student-Only` loses.

---

### 4B. Adaptive Policy Comparison Across All 6 Sway-Grid Configurations (`EXP-018`, $n = 360$)

#### Grid 1: `g16_all8` — Full 16-Interval Grid (`Steps 1..7` Candidates vs. `Step 8` Reference = `21 NFE`, `3.23s`)

| Policy | Mean NFE `[95% CI]` (Saving) | Teacher ODE (Speedup) | E2E Time (Saved / Speedup) | RTF (`x-RT`) | Raw WER ($\Delta$) | Capped WER ($\Delta$) | SIM ($\Delta$) | TOST ($p_{\text{WER}} / p_{\text{SIM}}$) | Step Distribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **`STAC` (Zero SNR)** | **16.82** `[16.32, 17.27]` (**19.9%**) | `2.05s` (**1.31x**) | **2.60s** (`-0.64s` / **1.25x**) | `0.560` (**1.79x**) | **0.0687** (`+0.0008`) | **0.0678** (`+0.0008`) | `0.8094` (`-0.0011`) | `1.2e-7 / 4.1e-29` | `1:19, 2:10, 3:56, 4:59, 5:4, 6:2, 7:28, 8:182` |
| **`Single-Step Drift` (`d_k >= lam_k`)** | `16.91` `[16.41, 17.41]` (`19.5%`) | `2.06s` (`1.30x`) | `2.61s` (`-0.63s` / `1.24x`) | `0.563` (`1.78x`) | `0.0684` (`+0.0004`) | `0.0674` (`+0.0004`) | `0.8097` (`-0.0007`) | `4.4e-6 / 1.2e-24` | `1:19, 2:13, 3:42, 4:63, 5:12, 6:11, 7:6, 8:194` |
| **`1D-CRC-STAC` ($\alpha=0.05$)** | **17.67** `[17.26, 18.04]` (**15.8%**) | `2.18s` (**1.23x**) | **2.73s** (`-0.51s` / **1.19x**) | `0.586` (**1.71x**) | **0.0670** (`-0.0009`) | **0.0661** (`-0.0009`) | `0.8101` (`-0.0003`) | `2.6e-8 / 1.5e-40` | `1:19, 2:10, 3:53, 4:34, 6:2, 7:1, 8:241` |
| **`Stage-Wise SNR Cascade`** | `18.13` `[17.80, 18.48]` (`13.7%`) | `2.26s` (`1.19x`) | `2.80s` (`-0.43s` / `1.15x`) | `0.602` (`1.66x`) | `0.0682` (`+0.0003`) | `0.0673` (`+0.0003`) | `0.8098` (`-0.0006`) | `1.5e-20 / 4.5e-35` | `1:14, 3:34, 4:51, 5:7, 6:1, 7:21, 8:232` |
| **`STSR`** | `18.72` `[18.41, 18.99]` (`10.9%`) | `2.35s` (`1.14x`) | `2.89s` (`-0.34s` / `1.12x`) | `0.618` (`1.62x`) | `0.0692` (`+0.0013`) | `0.0682` (`+0.0013`) | `0.8101` (`-0.0003`) | `4.2e-8 / 3.3e-45` | `2:9, 3:33, 4:24, 5:10, 6:22, 7:22, 8:240` |
| **`CSDR` / `SWSR`** | `20.15–20.17` (`3.9%–4.0%`) | `2.56s` (`1.05x`) | `3.10s` (`-0.13s` / `1.04x`) | `0.665` (`1.50x`) | `0.0682–0.0686` | `0.0673–0.0676` | `0.8103–0.8106` | `< 1e-14` | Mostly `g16_8` (`312–331`) |
| **`Single-Threshold Action`** | `20.30` `[19.91, 20.59]` (`3.3%`) | `2.58s` (`1.04x`) | `3.13s` (`-0.10s` / `1.03x`) | `0.670` (`1.49x`) | `0.0670` (`-0.0009`) | `0.0660` (`-0.0009`) | `0.8103` (`-0.0001`) | `2.7e-14 / 5.5e-43` | `3:4, 4:12, 5:10, 6:10, 7:8, 8:316` |
| **`Calibrated SNR` / `SNR+TAS`** | `20.69` `[20.46, 20.88]` (`1.5%`) | `2.64s` (`1.02x`) | `3.19s` (`-0.04s` / `1.01x`) | `0.682` (`1.47x`) | `0.0684–0.0686` | `0.0675–0.0676` | `0.8101–0.8103` | `< 1e-25` | `1:8, 8:352` |
| **`Old Def. 3.4` / `Stab-only`** | `21.00` `[21.00, 21.00]` (`0.0%`) | `2.69s` (`1.00x`) | `3.23s` (`-0.00s` / `1.00x`) | `0.693` (`1.44x`) | `0.0679` (`+0.0000`) | `0.0670` (`+0.0000`) | `0.8104` (`+0.0000`) | `0 / 0` | `8:360` |

#### Grid 2: `g16_le025` — 16-Grid Strictly $\le 0.25$ (`Steps 1..6` vs. `Step 7` Reference = `19 NFE`, `2.94s`)

| Policy | Mean NFE `[95% CI]` (Saving vs `g16_7` / vs `g16_8`) | E2E Time / Speedup / RTF | Raw WER ($\Delta$) | Capped WER ($\Delta$) | SIM ($\Delta$) | TOST $p_{\text{WER}}$ |
|---|---:|---:|---:|---:|---:|---:|
| **`STAC` (Zero SNR)** | **15.64** `[15.27, 16.01]` (**17.7%** / **25.5%**) | **2.42s** (**1.21x**) / `0.522` (**1.91x RT**) | **0.0692** (`+0.0009`) | **0.0683** (`+0.0009`) | `0.8093` (`-0.0006`) | `1.7e-5` |
| **`Single-Step Drift`** | `15.58` `[15.18, 15.94]` (`18.0%` / `25.8%`) | `2.41s` (`1.22x`) / `0.520` (`1.92x RT`) | `0.0664` (`-0.0019`) | `0.0655` (`-0.0019`) | `0.8084` (`-0.0015`) | `1.1e-4` |
| **`1D-CRC-STAC`** | **16.40** `[16.10, 16.69]` (**13.7%** / **21.9%**) | **2.54s** (**1.16x**) / `0.546` (**1.83x RT**) | **0.0672** (`-0.0011`) | **0.0662** (`-0.0011`) | `0.8096` (`-0.0003`) | `4.7e-11` |
| **`Stage-Wise SNR Cascade`** | `17.28` `[17.06, 17.53]` (`9.1%` / `17.7%`) | `2.68s` (`1.09x`) / `0.574` (`1.74x RT`) | `0.0685` (`+0.0003`) | `0.0676` (`+0.0003`) | `0.8097` (`-0.0002`) | `1.5e-20` |
| **`Calibrated SNR` / `SNR+TAS`** | `18.73–18.80` (`1.1%–1.4%` vs `g16_7`) | `2.90s` (`1.01x`) / `0.621` (`1.61x RT`) | `0.0687–0.0689` | `0.0678` (`+0.0005`) | `0.8095–0.8097` | `< 1e-25` |

#### Grid 3 (`g16`), Grid 4 (`g32`), Grid 5 (`g32_all14`), & Grid 6 (`g32_all15`) Summary

| Grid | Policy | Mean NFE `[95% CI]` (Saving) | E2E Time / Speedup / RTF | Raw WER ($\Delta$) | Capped WER ($\Delta$) | SIM ($\Delta$) | TOST $p_{\text{WER}}$ |
|---|---|---:|---:|---:|---:|---:|---:|
| **`g16`** (`{4,5,6}` vs `8`) | **`STAC`** | **17.44** `[17.05, 17.83]` (**16.9%**) | `2.69s` (**1.20x**) / `0.579` (**1.73x RT**) | **0.0644** (`-0.0036`) | **0.0634** (`-0.0036`) | `0.8091` (`-0.0014`) | `3.5e-5` |
| | **`1D-CRC-STAC`** | **18.24** `[17.86, 18.57]` (**13.1%**) | `2.81s` (`1.15x`) / `0.605` (`1.65x RT`) | **0.0665** (`-0.0014`) | **0.0656** (`-0.0014`) | `0.8097` (`-0.0007`) | `6.3e-11` |
| | `Stage-Wise SNR Cascade` | `18.17` `[17.96, 18.42]` (`13.5%`) | `2.81s` (`1.15x`) / `0.602` (`1.66x RT`) | `0.0670` (`-0.0009`) | `0.0661` (`-0.0009`) | `0.8094` (`-0.0010`) | `3.0e-24` |
| **`g32`** (`{7,9,11,13}` vs `14`) | **`STAC`** | **26.18** `[25.42, 26.83]` (**20.7%**) | `3.99s` (**1.26x**) / `0.860` (**1.16x RT**) | **0.0699** (`-0.0038`) | **0.0688** (`-0.0038`) | `0.8099` (`-0.0007`) | `0.0067` |
| | **`1D-CRC-STAC`** | **27.99** `[27.33, 28.57]` (**15.2%**) | `4.27s` (`1.18x`) / `0.918` (`1.09x RT`) | **0.0721** (`-0.0016`) | **0.0709** (`-0.0016`) | `0.8104` (`-0.0002`) | `1.7e-5` |
| | `Stage-Wise SNR Cascade` | `27.94` `[27.59, 28.33]` (`15.3%`) | `4.27s` (`1.18x`) / `0.914` (`1.09x RT`) | `0.0738` (`+0.0000`) | `0.0726` (`+0.0000`) | `0.8109` (`+0.0003`) | `9.1e-9` |
| | `Single-Step Drift` | `28.35` `[27.67, 28.94]` (`14.1%`) | `4.31s` (`1.17x`) / `0.927` (`1.08x RT`) | `0.0731` (`-0.0006`) | `0.0722` (`-0.0004`) | `0.8108` (`+0.0002`) | `0.012` |
| **`g32_all14`** (`1..13` vs `14`) | **`STAC`** | **27.14** `[26.04, 28.10]` (**17.8%**) | `4.15s` (**1.21x**) / `0.894` (**1.12x RT**) | **0.0738** (`+0.0001`) | **0.0726** (`+0.0001`) | `0.8089` (`-0.0017`) | `0.0029` |
| | `Single-Step Drift` | `25.31` `[24.39, 26.23]` (`23.3%`) | `3.87s` (`1.30x`) / `0.833` (`1.20x RT`) | `0.0738` (`+0.0001`) | `0.0729` (`+0.0003`) | `0.8091` (`-0.0014`) | `0.025` |
| | `Stage-Wise SNR Cascade` | `28.01` `[27.27, 28.80]` (`15.1%`) | `4.29s` (`1.17x`) / `0.918` (`1.09x RT`) | `0.0715` (`-0.0022`) | `0.0706` (`-0.0020`) | `0.8094` (`-0.0012`) | `0.0081` |
| | `Calibrated SNR` / `TAS` | `33.00` `[33.00, 33.00]` (`0.0%`) | `5.03s` (`1.00x`) / `1.076` (`0.93x RT`) | `0.0737` (`+0.0000`) | `0.0726` (`+0.0000`) | `0.8106` (`+0.0000`) | `0` |
| **`g32_all15`** (`1..14` vs `15`) | `Stage-Wise SNR Cascade` | `27.93` `[27.32, 28.60]` (`20.2%`) | `4.27s` (`1.25x`) / `0.913` (`1.09x RT`) | **0.0731** (`-0.0004`) | **0.0711** (`-0.0004`) | `0.8103` (`+0.0003`) | `0.0002` |
| | **`STAC`** *(14-stage diag.)* | **27.51** `[26.35, 28.59]` (**21.4%**) | `4.18s` (**1.28x**) / `0.903` (**1.11x RT**) | `0.1519` (`+0.0784`) | **0.0780** (`+0.0064`) | `0.8073` (`-0.0027`) | `0.81` |

> **Diagnostic on `g32_all15`:** Fitting 14 greedy stage thresholds on 48 speakers per fold selected ultra-early `g32_2` ($t_2 = 0.0048$) on 21 cases; **Capped WER ($\le 1.0$) stayed at `0.0780` (`+0.0064` vs reference)** and **Trimmed WER was `0.0728`**, while 1–2 Whisper repetition-hallucination outliers (`~26.6` excess WER) inflated Raw WER to `0.1519`. Setting a minimum candidate floor $t_k \ge 0.019$ (`g32_4` / `g16_2`) or using the 16-grid (`g16_all8` / `g16_le025`) avoids ultra-early repetition outliers completely.

---

## 5. Live Tesla T4 Online Early-Stopping Verification ([`live_swsr_t4.csv`](../research/experiments/EXP-018-swsr-handover-prelim/results/live_swsr_t4.csv), $n = 30$)

Actual live GPU early-stopping execution (`10 speakers × 3 conditions`), terminating the teacher ODE loop online at $\hat{k}$ vs. running full Step 8 (`21 NFE`):
- **Bitwise Trajectory Identity:** `max|endpoint_diff| = 0.0` across all 30 cases.
- **Gate Overhead:** `0.12 ms` | **Peak GPU Memory:** `12,027.0 MB`.

| Condition ($n$) | Mean Chosen Step | Mean NFE (Saved) | Teacher ODE (`STAC` vs Ref) | Teacher Speedup | End-to-End Wall-Clock (`STAC` vs Ref) | E2E Speedup (Saved) | RTF (`x-RT`) `STAC` vs Ref | WER (`STAC` vs Ref) | SIM (`STAC` vs Ref) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`clean` ($n=10$)** | `4.10` | **13.20** (`37.1%`) | `1.876s` vs `3.143s` | **1.68x** | **2.575s** vs `3.831s` | **1.49x** (`-1.26s` / `32.8%`) | `0.643` (**1.55x**) vs `0.940` (`1.06x`) | `0.0389` vs `0.0389` | `0.9418` vs `0.9406` |
| **`gauss20` ($n=10$)** | `6.40` | **17.80** (`15.2%`) | `2.574s` vs `3.257s` | **1.27x** | **3.231s** vs `3.916s` | **1.21x** (`-0.69s` / `17.5%`) | `0.796` (**1.26x**) vs `0.924` (`1.08x`) | **0.0330** vs `0.0488` | **0.8415** vs `0.8340` |
| **`noisy15` ($n=10$)** | `8.00` | **21.00** (`0.0%`) | `3.264s` vs `3.213s` | `0.98x` | **3.915s** vs `3.862s` | `0.99x` (`+0.05s` / `0.0%`) | `0.927` (`1.08x`) vs `0.912` (`1.10x`) | `0.2851` vs `0.2851` | `0.7798` vs `0.7798` |
| **Overall ($n=30$)** | `6.17` | **17.33** (`17.5%`) | `2.572s` vs `3.205s` | **1.25x** | **3.240s** vs `3.869s` | **1.19x** (`-0.63s` / `16.3%`) | `0.789` (**1.27x**) vs `0.925` (`1.08x`) | **0.1190** vs `0.1243` | **0.8544** vs `0.8515` |

---

## 6. 500-Speaker Benchmark (`all1000`) & Zero-Shot OOD Generalization (`EXP-011`)

### 6A. 500-Speaker Tesla T4 Benchmark (`200` Calibration Speakers $\to$ `200` Test Speakers, $n = 400$ Test Utterances, Full `d_1..d_8`)

| Policy | Mean NFE (Saved) | Runtime / Speedup / RTF | Raw WER ($\Delta$) | Capped WER $\le 1.0$ ($\Delta$) | Trimmed WER ($n=393$) | SIM ($\Delta$) | TOST ($p_{\text{WER}} / p_{\text{SIM}}$) | Choices |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **`fixed g16_8` (Ref)** | `21.00` (`0.0%`) | `3.22s` (`1.00x`) / `0.736` | `0.2371` (`+0.0000`) | `0.1231` (`+0.0000`) | `0.1075` | `0.8243` (`+0.0000`) | `0 / 0` | `8:400` |
| **`STAC` & `1D-CRC-STAC`** | **16.63** (**20.8%**) | **2.57s** (**1.25x**) / `0.581` | **0.2378** (`+0.0007`) | **0.1223** (`-0.0008`) | **0.1067** | **0.8236** (`-0.0007`) | **0.0018 / 2.2e-50** | `4:203, 5:6, 6:22, 8:169` |
| **`Single-Step Drift`** | `16.69` (`20.5%`) | `2.58s` (`1.25x`) / `0.583` | `0.2377` (`+0.0005`) | `0.1237` (`+0.0005`) | `0.1081` | `0.8231` (`-0.0012`) | `2e-15 / 2.6e-57` | `4:212, 6:7, 8:181` |
| **`Stage-Wise SNR Cascade`** | `16.73` (`20.3%`) | `2.58s` (`1.25x`) / `0.585` | `0.2344` (`-0.0027`) | `0.1221` (`-0.0010`) | `0.1065` | `0.8242` (`-0.0001`) | `0.012 / 1.2e-84` | `4:200, 5:18, 8:182` |
| **`Calibrated SNR` / `Def. 3.4`** | `17.04` (`18.9%`) | `2.63s` (`1.22x`) / `0.597` | `0.2367` (`-0.0004`) | `0.1227` (`-0.0004`) | `0.1071` | `0.8251` (`+0.0008`) | `2.3e-44 / 4.5e-118` | `4:198, 8:202` |
| **`Single-Threshold Action`** | `16.61` (`20.9%`) | `2.57s` (`1.25x`) / `0.581` | `0.4804` (`+0.2433`) | `0.1208` (`-0.0023`) | `0.1051` | `0.8244` (`+0.0001`) | **0.83 (FAIL)** / `7.9e-51` | `4:195, 5:13, 6:29, 8:163` |

### 6B. Zero-Shot OOD Generalization to 5 Unseen Corruptions (`EXP-011`, $n = 100$ Across 20 Speakers)

| Unseen Corruption | Cases ($n$) | Mean Prompt SNR | **`STAC`** Mean NFE (Saved) | **`SNR Cascade`** Mean NFE (Saved) | Paired $\text{NFE}(\text{STAC}) - \text{NFE}(\text{SNR})$ `[95% CI]` | **`STAC`** $\Delta\text{WER}$ `[95% CI]` |
|---|---:|---:|---:|---:|---:|---:|
| `babble20` | 20 | `23.6 dB` | **13.30** (`36.7%`) | `20.00` (`4.8%`) | **-6.70** `[-7.50, -5.60]` | `+0.0000` `[+0.0000, +0.0000]` |
| `babble15` | 20 | `19.4 dB` | **13.20** (`37.1%`) | `21.00` (`0.0%`) | **-7.80** `[-8.00, -7.20]` | `-0.0050` `[-0.0150, +0.0000]` |
| `reverb03` | 20 | `32.0 dB` | **13.60** (`35.2%`) | `14.20` (`32.4%`) | **-0.60** `[-2.00, +0.80]` | `+0.0028` `[+0.0000, +0.0083]` |
| `reverb09` | 20 | `22.4 dB` | **14.40** (`31.4%`) | `18.80` (`10.5%`) | **-4.40** `[-6.20, -2.60]` | `+0.0000` `[+0.0000, +0.0000]` |
| `telephone` (bandpass) | 20 | `44.2 dB` *(SNR fooled!)* | **16.20** (`22.9%`) | `13.00` (`38.1%`) | **+3.20** `[+2.20, +4.40]` | **-0.0083** `[-0.0250, +0.0000]` |
| **All OOD Pooled** | **100** | `28.3 dB` | **14.14** (**32.7%**) | `17.40` (`17.1%`) | **-3.26** `[-4.04, -2.48]` | **-0.0021** `[-0.0061, +0.0011]` |

> Notice `telephone`: bandpass filtering silences pause frames, fooling the SNR estimator into reporting `44.2 dB` ("ultra-clean") and blindly exiting at Step 4, whereas `STAC` detects the degraded internal trajectory and allocates extra teacher steps, reducing WER by `-0.0083`.

