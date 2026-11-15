# 2026-11-15 — EXP-028 (EVAL-021): Transmissible Slow-Band Bridge (`TBRG`) Results Imported

Task: `EVAL-021` (now `review`). Rhenel Jhon R. Sajol executed [DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb](../../../code/colab/DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb) on Google Colab (Tesla T4, branch `rhnl-3` commit `c21f04b`) and supplied `DMOSpeech2_EXP028_Transmissibility_Bridge_Results.zip` (`sha256: a9872dce323ea821589d45be4fe03e0abe16bfcdae3c476dc912c91985607797`).

## Import and Verification Checks

- **Archive extracted to:** `research/experiments/EXP-028-transmissibility-bridge/results/` (`diversity.csv`, `quality.csv`, `unseen_diversity.csv`, `unseen_quality.csv`, `transmissibility_bridge_results.json`, `TRANSMISSIBILITY_BRIDGE_REPORT.md`, `MANIFEST.json`, `smoke/diversity.csv`, `smoke/quality.csv`, `smoke/smoke_check.json`).
- **Line-ending-normalized SHA-256 verification (`sh.sha256_text`):** All files matched `MANIFEST.json`:
  - `preregistration.json`: `37c7213f35a758b0da3de8dbc224783bdc2835d678a107e74ffbec1674b1f248` (payload `sha256: 5c6c66ab8fee...`)
  - `diversity.csv`: `552879e98a1ec8dc2308d5d737ad166c5c50f5539b62e643606646f80bb7685c` (`20/20` speakers $\times$ `20` seeds, `61–80`)
  - `quality.csv`: `a995eb4d68b0078730d420093e95020dd58b89129f642ed2954d6f4fd280ab23` (`40/40` speakers, `61–100`)
  - `unseen_diversity.csv`: `f80cb9be12985337af7a55eae6a38ec0981e401e38feff464af413bff80aa8c4` (`20/20` disjoint calibration speakers $\times$ `15` seeds)
  - `unseen_quality.csv`: `e79bdd187f260f5a264743f804da60cd0cdd306e29b59a691d05924bd8dc9a05` (`30/30` disjoint calibration speakers)
  - `transmissibility_bridge_results.json`: `fcd4990493b566cbd2dadcae883d0cab40fb33d6b4f7ac067140bc3b31f90086`
  - `TRANSMISSIBILITY_BRIDGE_REPORT.md`: `1ca07181224c9da5ebe169a8c9dee5e594444717a071f57ada4bdef8126200bd`
- **Smoke reproduction check (`smoke_check.json`):** `max_abs_split_diff = 0.0`, `max_d_wer = 0.0`, `max_d_sim = 5.96e-8` against locked `EXP-027` `W4`.
- **Re-analysis check:** Running `run_transmissibility_bridge.py analyze` on a fresh temporary copy of the four CSVs reproduced identical `TRANSMISSIBILITY_BRIDGE_REPORT.md` and identical `transmissibility_bridge_results.json` (excluding `utc` timestamp).

## Results Summary (`supported_direction_only` on Both Cohorts)

### 1. Primary Prospective Cohort (`unseen_cal`: `20 x 15` div, `30` qual; disjoint calibration speakers)

| Arm | NFE | `CVf0` | `MelSpread` | `SlowCorr` | $f_0$ Corr vs `W8R2` (matched / shifted) | `WER` ($\Delta$ vs `W8`) | `SIM` ($\Delta$ vs `W8`) | `UTMOS` ($\Delta$ vs `W8`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `W4` | 13 | `0.0911` | `0.9455` | `0.814` | `0.881 / 0.550` | `0.0037` (`+0.0037`) | `0.9411` (`+0.0006`) | `3.882` (`-0.068`) |
| `W4R2` | 12 | `0.0997` | `1.0346` | `0.928` | `0.895 / 0.525` | `0.0000` (`+0.0000`) | `0.9419` (`+0.0013`) | `3.557` (`-0.393`) |
| `TBRG` | 13 | `0.0987` | `1.0130` | `0.908` | `0.895 / 0.526` | `0.0037` (`+0.0037`) | `0.9435` (`+0.0030`) | `3.756` (`-0.193`) |
| `BLND` | 13 | `0.0930` | `0.9662` | `0.877` | `0.908 / 0.548` | `0.0037` (`+0.0037`) | `0.9427` (`+0.0022`) | `3.880` (`-0.069`) |
| `SHUF` | 13 | `0.1170` | `0.5590` | `0.111` | `0.117 / 0.075` | `1.0004` (`+1.0004`) | `0.8664` (`-0.0741`) | `2.716` (`-1.234`) |
| `W6R2` | 16 | `0.1029` | `0.9450` | `0.842` | `0.963 / 0.539` | `0.0000` (`+0.0000`) | `0.9420` (`+0.0015`) | `3.887` (`-0.063`) |
| `W8R2` | 20 | `0.1048` | `0.9354` | `0.834` | `1.000 / 0.525` | `0.0000` (`+0.0000`) | `0.9418` (`+0.0012`) | `3.844` (`-0.106`) |
| `W8` | 21 | `0.0909` | `0.9328` | `0.789` | `0.874 / 0.533` | `0.0000` | `0.9405` | `3.949` |

- **Co-Primary `H1` / `H2` Contrasts (`unseen_cal`):**
  - `CVf0(TBRG - W4) = +0.0076 [+0.0001, +0.0152]` (`no-top2 = +0.0038`, `16/4`, $p = 0.0118$) — **passes**;
  - `CVf0(TBRG - W8R2) = -0.0061 [-0.0113, -0.0007]` — **fails NI margin `-0.008`**;
  - `UTMOS(TBRG - W4R2) = +0.199 [+0.102, +0.310]` (`no-top2 = +0.159`, `24/6`, $p = 0.00143$) — **passes**;
  - `UTMOS(TBRG - W8R2) = -0.088 [-0.170, -0.008]` — **fails NI margin `-0.05`**.
- **Hierarchical Secondary Controls (`unseen_cal`):**
  - **Causal contour tracking:** `slow_corr(TBRG - SHUF) = +0.7971 [+0.7312, +0.8630]` (`20/0`, $p = 1.91 \times 10^{-6}$); `f0_corr_w8r2(TBRG)` matched minus shifted = **`+0.3690 [+0.3152, +0.4334]`**. (`SHUF` raw `CVf0 = 0.1170` reflects pYIN tracking breakdown on garbled `WER = 1.0004` audio, causing `CVf0(TBRG - SHUF) = -0.0154` with 1 `SHUF` NaN speaker.)
  - **`TBRG` vs `BLND` (`13 NFE`):** `CVf0 +0.0057 [+0.0002, +0.0120]` (`13/7`) vs `UTMOS -0.124 [-0.219, -0.024]` (`7/23`, $p = 0.00522$).
  - **`H3` (`W8` quality):** `dWER +0.0037`, `dSIM +0.0030`, `dUTMOS -0.193 [-0.284, -0.111]` (`< -0.08` bar).
  - **`H4` (`mel_spread` vs `W4R2`):** `-0.0216 [-0.0271, -0.0162]` (`1/19`, $p = 4.01 \times 10^{-5}$).

### 2. Development Cohort (`speakers 61–100`: `20 x 20` div, `40` qual paired vs EXP-027)

- Reproduces the same pattern and verdict (`supported_direction_only`): `CVf0(TBRG - W4) = +0.0087 [+0.0022, +0.0167]`, `UTMOS(TBRG - W4R2) = +0.214 [+0.137, +0.286]` (`3.618 vs 3.404`), `TBRG - STS2 (UTMOS) = +0.046 [+0.000, +0.094]` (`3.618 vs 3.571`), `slow_corr(TBRG - SHUF) = +0.7793 [+0.7511, +0.8079]` (`SHUF WER = 1.1349`), `TBRG - BLND` = `+0.0065 CVf0` / `-0.121 UTMOS`, and `dUTMOS(TBRG - W8) = -0.160 [-0.234, -0.088]` (`3.618 vs 3.777`).
- **`W6R2` (`16 NFE`, Step-6 + `[tau_2, tau_3]`):** Achieves `CVf0 = 0.0840` (`TBRG 0.0828`), `UTMOS = 3.714` (`-0.063 [-0.125, +0.001]` vs `W8`, `+0.096 [+0.011, +0.184]` vs `TBRG`), `WER = 0.0128`, `SIM = 0.9357`.
- **Secondary `STAC-TBRG`:** Routes `3/40` `TBRG` and `37/40` `W8R2` (`19.48` mean NFE, `dUTMOS = -0.129 [-0.199, -0.061]`).

### 3. Pre-Committed Interpretation Matrix Readout

By our pre-committed outcome matrix, **Outcome 3** (`H1` passes, `H2`/`H3` fail non-inferiority) holds cleanly across both prospective and development cohorts: while `TBRG` causally transfers slow prosody at `13 NFE` and eliminates half of `W4R2`'s distortion, its residual `-0.193` / `-0.160` UTMOS cost vs `W8` (`-0.124` / `-0.121` vs `BLND`) means `13-NFE TBRG` trades quality for diversity rather than breaking the Pareto frontier — and the unexpected standout at **`16 NFE` is simple `W6R2`** (`UTMOS 3.887` / `3.714`, `CVf0 0.1029` / `0.0840`, `f0_corr_w8r2 = 0.963`).

## Checks Run

- `python scripts/render_progress.py` — updated `docs/progress/TASKS.md`.
- `python scripts/check_workspace.py` — `Workspace check PASSED`.
- `python -m unittest discover -s code/tests -p "test_transmissibility_bridge.py" -v` — all tests OK.
- `python -m unittest discover -s code/tests` — all tests OK.
- LaTeX build (`latexmk -Werror`): not run (`paper/` unchanged; TeX build unverified for this import).
