# SNR + Invariant Residual (SNR_R) handover: local prototype

A local CPU demonstration of **risk-calibrated SNR + Invariant Residual (`SNR_R`)** from `EXP-034` and `EXP-035` running DMOSpeech 2 on CPU, with a web page in the style of the proposal presentation. It is a demo, not an evaluation tool.

**What it shows:**
1. **The 1-pass student draft $s_0$ ($\tau = 0.0$):** keeps clean prompt acoustics anchored (`1` pass).
2. **Five guided 8-interval teacher steps ($t_0 \dots t_4 = 0.2929$, `10` CFG passes):** computed step by step on CPU while tracking the invariant residual $r = \|\hat{x}_1 - Q(\hat{x}_1, s_0)\| / (\|\hat{x}_1\| + \varepsilon)$.
3. **Learn-Then-Test (LTT) risk-calibrated hybrid router (`SNR_R`):** if `prompt_snr < 20 dB` or $r > \hat{\lambda}_r$ (`0.3416` default at $\alpha = 0.15, \delta = 0.10$, `p = 0.048` on `EXP-034`), hands over the condition-safe **Quantile Handover (`CPQH`)**; otherwise hands over **`CPH8`** to preserve full clean prosodic variety.
4. **Student restart completion ($\tau \in \{0.2929, 0.6173\}$, `2` passes):** `13` total passes (`38%` saved vs `21`-pass `W8`, `57%` saved vs `30`-pass paper `P25`).

**Optional side-by-side comparison:** also synthesizes unstabilized **`CPH8`** (`12` passes, when flagged to `CPQH`) and the **`21`-pass `W8`** fixed reference from the same latent and student seeds.

## Files

| File | What it is |
| --- | --- |
| `snr_invariant.py` | **The `SNR_R` method only:** `quantile_handover`, `quantile_payload`, `invariant_residual` ($r$), `decide`/`evaluate`, and constants (`theta = 20 dB`, `lambda_r = 0.340`). Model-free and UI-free; tested against `code/colab/run_invariant_residual.py` and `code/colab/run_method_comparison.py` |
| `snr_drift.py` | Shared prompt SNR estimator (`estimate_prompt_snr`), grid constants, and preserved Chapter 3 `SNR + drift` helper |
| `engine.py` | Loads DMOSpeech 2 from vendored `code/vendor/dmospeech2/src` (read-only) and runs `SNR_R` (`draft s_0` -> 8-grid Steps `0..4` -> `CPH8`/`CPQH` -> student completion) |
| `server.py` | Local FastAPI web server: live step streaming, development speaker examples with `EXP-034` T4 records, and audio endpoints |
| `web/index.html` | Interactive UI reusing `presentation/index.html` design tokens |
| `cli.py` | Terminal mode (`--example 38 --condition gauss25 --compare` or `--prompt`) |
| `start_prototype.bat` | Double-click launcher (CRLF preserved) |

## Run it

**Option 1:** double-click `code\prototype\start_prototype.bat`.

**Option 2:** from the repository root:

```powershell
.venv\Scripts\python code\prototype\server.py
```

The page opens at http://127.0.0.1:8765.

**Terminal mode:**

```powershell
.venv\Scripts\python code\prototype\cli.py --example 38 --condition gauss25 --compare
.venv\Scripts\python code\prototype\cli.py --prompt my_voice.wav --prompt-text "What the prompt says." --text "What to say." --out out.wav
```

## Suggested demo (`EXP-034` recorded development examples)

Development examples `#31–#60` carry recorded Colab T4 `EXP-034` metrics in the UI. Example `#38` (`common_voice_en_203357-common_voice_en_203358`, `339` frames) clearly contrasts `SNR_R` against `SNR + drift`:

| Condition | Recorded `SNR` / `r` | `SNR_R` choice (`lambda_r = 0.340`) | `EXP-034` T4 UTMOS (`CPH8 -> CPQH`) |
| --- | --- | --- | --- |
| `clean` | `>= 20 dB`, `r = 0.253 <= 0.340` | **`CPH8`** (keeps full prosody) | `3.85` (stays `CPH8`) |
| `gauss25` | `>= 20 dB`, `r = 0.727 > 0.340` | **`CPQH`** (caught by `r`) | `1.25 -> 4.20` |
| `pink25` | `>= 20 dB`, `r > 0.340` | **`CPQH`** (caught by `r`) | `1.86 -> 3.76` |
| `hp25` | `>= 20 dB`, `r > 0.340` | **`CPQH`** (caught by `r`) | `1.25 -> 4.07` |
| `noisy15` | `< 20 dB` (gate closed) | **`CPQH`** (caught by SNR gate) | Stabilized |

Switching the preset chip to **`Off · SNR only`** (`lambda_r = 10.00`) disables the residual check so `gauss25`, `pink25`, and `hp25` fall through to un-stabilized `CPH8`, reproducing the `EXP-034` failure mode side by side.

## Learn-Then-Test (LTT) Risk Calibration Presets

`snr_invariant.calibrate_ltt()` runs fixed-sequence binomial p-value testing ($\text{BinomCDF}(k_{\text{wrong}}, n_{\text{clean}}, \alpha) \le \delta = 0.10$) on `EXP-034` development speakers `31–60` (`n = 30` clean) and transfers the certified cutoff out-of-sample to `EXP-035` (`30` disjoint unseen speakers):

| Preset (`--preset`) | $\lambda_r$ | Calibration (`EXP-034`, `n=30`) | Held-out (`EXP-035`, `30` unseen) |
| --- | --- | --- | --- |
| `ltt_15` *(default)* | `0.3416` | $\alpha = 0.15, \delta = 0.10$: `1/30` (`p = 0.048`) | `3/30` (`10.0%`) clean flags, **`90/90`** spectral caught, `2.8885` UTMOS |
| `ltt_10` *(strict)* | `0.3816` | $\alpha = 0.10, \delta = 0.10$: `0/30` (`p = 0.042`) | `1/30` (`3.3%`) clean flags, `89/90` (`98.9%`) spectral caught, `2.8850` UTMOS |
| `empirical` | `0.3400` | `EXP-033` empirical `<= 15%` rule | `3/30` (`10.0%`) clean flags, `90/90` spectral caught, `2.8885` UTMOS |
| `snr_only` | `10.00` | Residual disabled (`< 20 dB` gate only) | `0/90` spectral caught, `2.2014` UTMOS |

## Inputs

- **Seed-TTS examples:** `100` development speakers only (never calibration or test), with experiment seeds.
- **Conditions offered:** `clean`, `gauss25`, `pink25`, `hp25`, `noisy15`, `reverb03`, `reverb09`, `telephone` (`babble15` needs DEMAND and is excluded).
- **Upload WAV/FLAC:** 3–12 s voice prompt plus its transcript.

## Setup (checkpoints & environment)

Uses `.venv`, `ckpts/model_85000.pt`, `ckpts/model_1500.pt`, and `research/data/seed-tts-eval/en` (see `code/prototype/requirements.txt`). `.venv/`, `ckpts/`, and `research/data/` are git-ignored.

## Notes

- **CPU only.** Wall-clock times reflect local CPU execution; forward-pass counts (`13` vs `21` `W8` / `30` `P25`) match the T4 runs identically.
- **Development demonstration.** Not adviser-reviewed.
