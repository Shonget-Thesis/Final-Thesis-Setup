# 2026-11-18: local risk-calibrated SNR + Invariant Residual (`SNR_R`) prototype update

Task: `PROTO-002` (`review`). Requested by user (`"okay for now udapte the protoy with snr_inariant"`, `"make sure with risk calibration oaky"`).

Nothing here changes the manuscript (`Chapters 1–3`) or the archived experiment scripts (`EXP-034`, `EXP-035`, `EXP-036`). The prototype remains a CPU demonstration and is not adviser-reviewed.

## Changes

- **`code/prototype/snr_invariant.py`:** Standalone model-free `SNR_R` module with fixed-sequence Learn-Then-Test (LTT) risk calibration, matching `code/colab/run_invariant_residual.py` (`EXP-034`) and `code/colab/run_method_comparison.py`:
  - `8`-interval sway grid (`Steps 0..4` ending at `t = 0.2929`, `10` teacher CFG passes), `1`-pass student draft $s_0$ at $\tau = 0.0$, and `2`-pass student completion at $\tau \in \{0.2929, 0.6173\}$ (`13` passes total vs `21` `W8` [`38%` saved] and `30` paper `P25` [`57%` saved]).
  - `quantile_handover`, `quantile_payload` (`CPQH`), `invariant_residual` ($r = \|\hat{x}_1 - Q(\hat{x}_1, s_0)\|_F / (\|\hat{x}_1\|_F + \varepsilon)$), `band_residuals`, `decide`, `evaluate`, exact stdlib binomial CDF `_binom_cdf`, and `calibrate_ltt()` (descending fixed-sequence binomial test $\text{BinomCDF}(k_{\text{wrong}}, n_{\text{clean}}, \alpha) \le \delta$).
  - Four presets verified against `EXP-034` (`30` dev clean rows) and held out on `EXP-035` (`30` disjoint unseen speakers, `240` cases):
    - `ltt_15` *(default)*: $\hat{\lambda}_r = 0.3416$ (`EXP-034` $\alpha=0.15, \delta=0.10$, `1/30` clean error, `p=0.0480`; `EXP-035` unseen: `3/30 = 10.0%` wrong clean flags, `90/90` spectral caught, `2.8885` UTMOS).
    - `ltt_10` *(strict)*: $\hat{\lambda}_r = 0.3816$ (`EXP-034` $\alpha=0.10, \delta=0.10$, `0/30` clean error, `p=0.0424`; `EXP-035` unseen: `1/30 = 3.3%` wrong clean flags, `89/90 = 98.9%` spectral caught, `2.8850` UTMOS).
    - `empirical`: `0.3400` (`EXP-033` empirical `<= 15%` dev anchor).
    - `snr_only`: `10.00` (ablation disabling `r`).
- **`code/prototype/snr_drift.py`:** Preserved untouched for `PROTO-001` reproducibility and shared prompt SNR / 16-grid helpers (`estimate_prompt_snr`, `TEACHER_GRID`, `STUDENT_TAUS`, `REFERENCE_PASSES`).
- **`code/prototype/engine.py`:** Refactored shared `_teacher_step(ctx, state, t_val)` and `_student(ctx, endpoint, taus)` helpers; added `pink25` and `hp25` to `OFFLINE_CONDITIONS` via `run_invariant_residual.degrade`; implemented `SNR_R` in `Engine.synthesize()` tracking live per-step residuals `r_0..r_4` and routing between `CPH8` and `CPQH` at `t = 0.2929` (plus optional `unstabilized` `CPH8` and `21`-pass `W8` comparison audio).
- **`code/prototype/server.py` & `code/prototype/web/index.html`:** Computes `calibrate_ltt` dynamically on `EXP-034` rows for `/api/status["risk_calibration"]`, loads `EXP-034` T4 outcomes (`dev[31:60]`), defaults example selection to `#38` (`common_voice_en_203357-common_voice_en_203358`, `339` frames) and preset to `ltt_15`, preserves exact preset floats when switching chips, displays the active LTT risk certificate in the decision summary, and streams `s_0`, 8-grid Steps `0..4`, `r_0..r_4` vs $\lambda_r$, `CPH8`/`CPQH` routing, cost (`13 / 21`), and multi-branch playback.
- **`code/prototype/cli.py`, `code/prototype/README.md`, `code/prototype/start_prototype.bat`:** Added `--preset` (`ltt_15`, `ltt_10`, `empirical`, `snr_only`), `--lambda-r`, `pink25`/`hp25`, live LTT certificate printout, and `--compare` outputs.
- **Tests:** Added `code/tests/test_prototype_snr_invariant.py` covering `EXP-034` constants/costs, NumPy + PyTorch parity with `run_invariant_residual.py` and `run_method_comparison.py`, monotone invariance of `Q`, binomial CDF and `calibrate_ltt` exact checks against synthetic data and `EXP-034` CSV, `2,500` random policy checks against `ir.choose`, `pink25`/`hp25` degradations, FastAPI server status/`risk_calibration`, and LTT UI tokens; updated `InterfaceTests` in `code/tests/test_prototype_snr_drift.py`.

## Verification

See final check results recorded in `docs/progress/tasks.json` (`PROTO-002`) after final edits:
- `node --check` on extracted `web/index.html` script.
- Prototype suite in `.venv\Scripts\python.exe` (with PyTorch/FastAPI/uvicorn) and full `python -m unittest discover -s code/tests`.
- `python scripts/check_workspace.py` and `python scripts/check_workspace.py --base origin/main`.
- Real local CPU E2E DMOSpeech 2 CLI runs on dev example `common_voice_en_203357-common_voice_en_203358` (`clean -> CPH8` and `gauss25 --compare -> CPQH`, `13/21` passes, `24 kHz` finite WAVs).
