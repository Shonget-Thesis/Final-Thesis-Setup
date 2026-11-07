# 2026-11-07: EXP-018 Stage-Wise Trajectory Action Cascade (`STAC`) & `1D-CRC-STAC` Completed on Colab Tesla T4 (`Steps 1..8` & `Steps 1..15`)

Prepared, executed on an NVIDIA Tesla T4 GPU (Google Colab, thesis `8a6f396`, Shonget-DMOSpeech2 `8a7868f`), and imported byte-identical results with SHA-256 verification ([MANIFEST.json](../../../research/experiments/EXP-018-swsr-handover-prelim/results/MANIFEST.json), [README.md](../../../research/experiments/EXP-018-swsr-handover-prelim/README.md), [SWSR_HANDOVER_REPORT.md](../../../research/experiments/EXP-018-swsr-handover-prelim/results/SWSR_HANDOVER_REPORT.md)) for `EXP-018` (`EVAL-011`). All evaluations dispense with binary failure rules and report continuous quality metrics (Raw WER, Capped WER $\le 1.0$, Outlier-Trimmed WER, SIM, UTMOS) alongside full computational cost and wall-clock telemetry (Teacher Evals, NFE, Gate Overhead `ms`, Teacher ODE Time `s`, End-to-End Wall-Clock Time `s`, Speedup `x`, RTF, `x-Real-Time`, and Peak GPU Memory `MB`).

## Method & Controls

- **`STAC` (`Stage-Wise Trajectory Action Cascade`):** Uses cumulative trajectory action $A_k = \sum_{j=1}^{k} d_j \ge \hat{\lambda}_k$ without any external SNR estimator (`0.12 ms` gate overhead vs. `1.85 ms` for SNR-only and `1.97 ms` for hybrid SNR+drift).
- **`1D-CRC-STAC`:** Conformal Risk-Controlled Trajectory Action Cascade ($\alpha = 0.05$) with bounded loss $L \in [0, 1]$.
- **Matched Controls & Ablations:** Stage-Wise SNR Cascade ($\hat{S} \ge \hat{\theta}_k$, matched degrees of freedom), Stage-Wise Single-Step Drift ($d_k \ge \hat{\lambda}_k$), Single-Threshold Trajectory Action ($A_k \ge \hat{\lambda}$), Calibrated SNR-only, Zero-Shot OOD generalization on 5 unseen corruptions from `EXP-011` (`babble20`, `babble15`, `reverb03`, `reverb09`, `telephone`; $n=100$), and Hindsight Per-Utterance Oracle across `Strict`, `Primary`, and `Relaxed` tolerances.

## Empirical Results on Tesla T4 (Development Evidence)

1. **Live Tesla T4 Early-Stopping Verification (`live_swsr_t4.csv`, $n = 30$, 10 speakers × `clean`, `gauss20`, `noisy15`):**
   - **Bitwise Trajectory Identity:** `max |endpoint_diff| = 0.0` across all 30 live GPU runs.
   - **Overall ($n=30$):** Mean Step `6.17` (`17.33` vs `21.00` NFE, `17.5%` NFE saved), Teacher ODE `2.572s` vs `3.205s` (`1.25x` speedup), End-to-End `3.240s` vs `3.869s` (`1.19x` speedup, `-0.63s` / `16.3%` saved), RTF `0.789` (`1.27x RT`) vs `0.925` (`1.08x RT`), WER `0.1190` vs `0.1243`, SIM `0.8544` vs `0.8515`, peak GPU memory `12,027.0 MB`.
   - **Condition-Adaptive Routing Without SNR:** Clean (`n=10`) exits at Step `4.10` (`13.20` NFE, `37.1%` saved, `1.68x` teacher / `1.49x` E2E speedup, RTF `0.643` = `1.55x RT`, identical `0.0389` WER); `gauss20` (`n=10`) exits at Step `6.40` (`17.80` NFE, `15.2%` saved, `1.21x` E2E speedup, WER `0.0330` vs `0.0488`); `noisy15` (`n=10`) routes `10/10` (`100%`) to full Step `8.00` (`21.00` NFE).
2. **Full 16-Interval (`Steps 1..8`) & 32-Interval (`Steps 1..15`) Synthesis (`steps1to8.csv`, $n = 360$, 60 speakers × 6 conditions):**
   - Measured T4 unit costs: `t_prepare = 81.95 ms`, `t_eval = 298.81 ms`, `t_student = 152.46 ms`, `t_vocoder = 5.51 ms`, mean audio `4.80 s`, mean peak memory `9,587.7 MB`.
   - **`g16_all8` (`1..7` vs `8`, `21 NFE`):** `STAC` achieves `16.82` NFE `[16.32, 17.27]` (`19.9%` saved, `2.60s`, `1.25x` E2E speedup, `1.31x` teacher speedup, `1.79x RT`), WER `0.0687` (`+0.0008`, TOST $p = 1.2 \times 10^{-7}$), SIM `0.8094` (`-0.0011`), outperforming matched-DOF `Stage-Wise SNR Cascade` (`18.13` NFE, `13.7%` saved) and `Calibrated SNR-only` (`20.69` NFE, `1.5%` saved). `1D-CRC-STAC` achieves `17.67` NFE (`15.8%` saved) with WER `0.0670` (`-0.0009` vs reference).
   - **`g16_le025` (`1..6` vs `7`, `19 NFE`, strictly $\le 0.25$):** `STAC` achieves `15.64` NFE (`17.7%` saved vs Step 7, `25.5%` vs Step 8; `2.42s`, `1.91x RT`, WER `0.0692`, TOST $p = 1.7 \times 10^{-5}$) vs `17.28` NFE (`9.1%`) for `Stage-Wise SNR Cascade` and `18.80` NFE (`1.1%`) for `Calibrated SNR-only`.
   - **`g32` (`{7,9,11,13}` vs `14`, `33 NFE`) & `g32_all14` (`1..13` vs `14`, `33 NFE`):** `STAC` achieves `26.18` NFE (`20.7%` saved, `3.99s`, `1.26x` speedup, WER `0.0699`, TOST $p = 0.0067$) on `g32` and `27.14` NFE (`17.8%` saved, `4.15s`, `1.21x` speedup, WER `0.0738`, TOST $p = 0.0029$) on `g32_all14`, whereas `Calibrated SNR-only` saves `0.0%` (`33.00` NFE) when `g32_1` is included.
   - **Honest Diagnostic on `g32_all15` (`1..14` vs `15`, `35 NFE`):** Fitting 14 sequential stage thresholds on 48 speakers per fold routes 21 utterances to ultra-early `g32_2` ($t_2 = 0.0048$); **Capped WER ($\le 1.0$) remains `0.0780` (`+0.0064` vs reference)** and **Trimmed WER is `0.0728`** at `27.51` NFE (`21.4%` saved), but 1–2 Whisper repetition hallucinations (`~26.6` excess WER) inflate Raw WER to `0.1519` (TOST $p = 0.81$).
3. **500-Speaker Benchmark & Zero-Shot OOD (`EXP-011`):**
   - On `all1000` (200 cal -> 200 test speakers, full `d_1..d_8`), `STAC` and `1D-CRC-STAC` achieve `16.63` NFE (`20.8%` saved, `2.57s`, `1.25x` speedup), Capped WER `0.1223` (`-0.0008`), Raw TOST $p = 0.0018$.
   - On `EXP-011` OOD ($n = 100$), `STAC` saves `32.7%` (`14.14` NFE) vs `17.1%` (`17.40` NFE) for `Stage-Wise SNR Cascade` (paired diff `-3.26 NFE [-4.04, -2.48]`, $\Delta\text{WER} = -0.0021$).

## Presentation Appendix, LaTeX Manuscript Appendix & Master Report Updates (`PRES-001`, `PAPER-013`)

- Compiled master cross-experiment & speech quality/speed benchmark comparison in [STAC_Handover_Methods_and_Quality_Benchmark_Comparison.md](../../../reports/STAC_Handover_Methods_and_Quality_Benchmark_Comparison.md).
- Updated existing presentation appendix slides (`17-appendix-preliminary-experiments.html`, `18-appendix-signal-diagnostic.html`, `19-appendix-simulation-and-prototype.html`) with verified Tesla T4 `AUDIT-001` and `EXP-011`–`EXP-018` findings (including `live_swsr_t4.csv` online GPU early-exit telemetry), and added two new appendix slides (`20-appendix-stac-architecture.html` and `21-appendix-stac-results.html`) presenting the `STAC` / `1D-CRC-STAC` modular architecture and multi-grid/cohort/OOD benchmarks while keeping main proposal slides `01`–`16` intact.
- Upgraded `presentation/build.py` `--pdf` export to use stdlib RFC 6455 WebSocket over headless Edge CDP (removing external `websockets` dependency), rebuilt `presentation/index.html`, `presentation/SCRIPT.md`, and `presentation/proposal-defense-slides.pdf` (21 pages), and verified zero bounding-box overflow or clipping across slides 17–21.
- Updated the LaTeX manuscript appendices (`PAPER-013`) while keeping Chapters 1–3 unchanged (`\citcproposaltrue` in `paper/main.tex`):
  - Added acronyms (`CRC`, `OOD`, `STAC`, `TOST`) in `paper/frontmatter.tex` and updated header documentation in `paper/main.tex`.
  - Updated `paper/chapters/appendices.tex` Appendices B, C (`Live Online Early-Stopping Verification on the Tesla T4`, Table `tab:live-t4` from `live_swsr_t4.csv`), and D (`D.1`–`D.5` and summary Table `tab:prelim` with `AUDIT-001` and `EXP-011`–`EXP-017` Tesla T4 two-grid oracle, paper `14+2` / Algorithm 2 baselines, `22.5%` binary noise floor, premature stabilization $d_4: 0.61 \to 0.17$, and OOD SNR failures).
  - Added **Appendix E (`Stage-Wise Trajectory Action Cascade and Multi-Grid Benchmarks`, `\label{app:stac}`)** in `paper/chapters/appendices.tex` with TikZ architecture Figure `fig:stac` (`paper/figures/fig_stac.tex`), mathematical formulation of `STAC` and `1D-CRC-STAC`, the full **Fixed `Step 1` through `Step 8` quality/speed profile Table `tab:fixed-steps1to8`**, multi-grid adaptive policy Tables `tab:stac-g16` and `tab:stac-grids`, 500-speaker `all1000` replay, zero-shot OOD Table `tab:stac-ood` (`EXP-011`), and explicit development-evidence scope statement.
  - Built `paper/build/main.pdf` cleanly with `latexmk -interaction=nonstopmode -halt-on-error -Werror` inside `paper/`.

## Checks Run & Remaining Blockers

- `Set-Location paper; latexmk -interaction=nonstopmode -halt-on-error -Werror` — PASSED (`paper/build/main.pdf` built with exit code 0).
- `python scripts/render_progress.py` — updated `docs/progress/TASKS.md`.
- `python scripts/check_workspace.py` — PASSED.
- `python -m unittest discover -s code/tests` — PASSED (including `test_evidence_manifest.py` SHA-256 verification of all 6 imported `EXP-018` files and `test_swsr_handover.py`).
- Remaining blockers: all `EXP-018` evaluations use cross-validation on the 60 development speakers or the previously inspected 500-speaker benchmark split (`AUDIT-001`); an unseen-speaker frozen evaluation (`EVAL-001` / `EVAL-002`) and team/adviser review of whether to adopt `STAC` / `1D-CRC-STAC` into the post-proposal scope remain pending.
