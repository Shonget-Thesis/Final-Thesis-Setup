# SNR + drift handover: local prototype

A demonstration of the thesis's input-adaptive handover (Chapter 3) running DMOSpeech 2 on this computer's CPU, with a web
page in the style of the proposal presentation. It is a demo, not an evaluation tool.

**What it shows:**
1. **The prompt's SNR:** the gate at 20 dB.
2. **Each teacher step as the CPU computes it:** with its drift d_k.
3. **The handover:** the step the rule chooses.
4. **The student's three steps.**
5. **The passes used:** against the 21 of the fixed reference, and the audio.

**Optional:** it also synthesizes the fixed reference (Step 8) from the same trajectory and seeds, for listening side by side.

## Files

| File | What it is |
|---|---|
| `snr_drift.py` | **The method only:** the prompt SNR estimator, the drift d_k, and the rule. The rule: below 20 dB go to Step 8; otherwise hand over at the first k in {4, 5, 6} with d_k ≤ λ, else Step 8. It has no model and no interface, and NumPy is its only dependency. Tests check it against `code/scheduling/sampler.py` and the experiments' Chapter 3 policy |
| `engine.py` | Loads DMOSpeech 2 from `code/vendor/dmospeech2/src` (read-only) and runs the thesis pipeline, stopping the teacher when `snr_drift.py` says so |
| `server.py` | The local web server: the page, the examples, and jobs with live progress |
| `web/index.html` | The interface. Its design tokens, fonts, panels, tags and buttons come from `presentation/index.html` |
| `cli.py` | Terminal mode: one sentence, the decision printed step by step |
| `start_prototype.bat` | Double-click to start the server and open the page |

## Run it

**Option 1:** double-click `code\prototype\start_prototype.bat`.

**Option 2:** from the repository folder:

```
.venv\Scripts\python code\prototype\server.py
```

The page opens at http://127.0.0.1:8765. The model loads in the background, and the status in the top-right corner turns to "Ready".

**Terminal only:**

```
.venv\Scripts\python code\prototype\cli.py --example 1 --condition noisy15
.venv\Scripts\python code\prototype\cli.py --prompt my_voice.wav --prompt-text "What the prompt says." --text "What to say." --out out.wav
```

**Times on this computer (CPU, 11 threads, 2026-11-16):**
- **Model load:** about 40 s after the first start. The first start took 73 s because it also downloads the vocoder.
- **One teacher step:** about 1.4 s for a short sentence and about 3.8 s for a longer one.
- **One sentence:** about 15–40 s in total.

## Suggested demo

Development examples whose decisions EXP-031 recorded on the T4 (the page marks them "T4 record"). The local runs on 2026-11-16 gave the same decisions.

| Example | Condition | What it shows |
|---|---|---|
| **#5** (default; short, fast) | `clean` | SNR 44.3 dB, d₄ = 0.26 ≤ 0.35: hand over at **Step 4**, 13 of 21 passes (38% saved) |
| #5 | `noisy15` | SNR 18.0 dB: **gate closed → Step 8**. Its drift is already low (d₄ = 0.16), so drift alone would have handed over a noisy start; the gate prevents that |
| #5 | `gauss25` | Recorded: SNR 27.8 dB passes the gate → Step 5. The white noise is not caught (a known limit) |
| #4 | `clean` | Recorded: Step 5 (15 passes) |
| #2 | `clean` | Recorded: Step 6 (17 passes) |
| #1 | `clean` | SNR 28.2 dB, drift stays above 0.35 → Step 8 (no saving). At λ = 1.20 it would be Step 4 |

## Inputs

**Seed-TTS example:**
- **Speakers:** the 100 development speakers only (never calibration or test), with the experiments' per-utterance seeds.
- **Prompt conditions,** generated with the experiments' own functions: `clean`, `gauss25`, `noisy15`, `reverb03`, `reverb09`, `telephone`. Babble needs the DEMAND recording and is not offered.

**Upload WAV:**
- **Your prompt:** your own 3–12 s prompt as WAV or FLAC.
- **Its transcript:** required. Automatic transcription is not used.

**λ presets** (Appendix D):

| λ | Where it comes from | Effect |
|---|---|---|
| 0.35 | Chosen by cross-validation in the development pilot | The drift check is active |
| 1.20 | Certified by Learn-Then-Test on the 200 calibration speakers | The rule acts as SNR only |

Neither is the final calibrated threshold.

**The reference:** Step 8 of the 16-interval grid, the thesis's version of the paper's t_switch = 0.25 (21 passes).

## Set up once (already done on this computer, 2026-11-16)

**1. The environment** (Python 3.12):

```
python -m venv .venv
.venv\Scripts\python -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -r code\prototype\requirements.txt
```

**2. The checkpoints** into `ckpts/`. Both were verified against these SHA-256 hashes on 2026-11-16 with `sha256sum`, the same hashes the Colab notebooks check. The engine does not re-hash them on start.

| File | Size | SHA-256 |
|---|---|---|
| `model_85000.pt` | 4.0 GB | `e143188f7c46...` |
| `model_1500.pt` | 270 MB | `6bd50428f3b5...` |

```
curl -L -C - -o ckpts\model_85000.pt https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_85000.pt
curl -L -C - -o ckpts\model_1500.pt  https://huggingface.co/yl4579/DMOSpeech2/resolve/main/model_1500.pt
```

**3. The Seed-TTS English test set** into `research/data/seed-tts-eval/en` (about 440 MB):

```
.venv\Scripts\python -c "from huggingface_hub import snapshot_download as s; s('zhaochenyang20/seed-tts-eval', repo_type='dataset', local_dir='research/data/seed-tts-eval', allow_patterns=['en/*'])"
```

**4. The vocoder:** Vocos (`charactr/vocos-mel-24khz`, about 50 MB) downloads from Hugging Face on the first start.

**Not committed:** `.venv/`, `ckpts/*.pt` and `research/data/` are all ignored by git.

## Notes

- **CPU only.** This computer has AMD integrated graphics and no CUDA, so every pass runs on the processor. The wall-clock times are this computer's, not the T4 times reported in the experiments. The pass counts are the same.
- **The vocabulary.** The vendored snapshot lacks `data/Emilia_ZH_EN_pinyin/vocab.txt`. The engine loads the identical 2,545-entry vocabulary bundled at `f5_tts/infer/examples/vocab.txt` in memory, and no vendored file is changed.
- **The pipeline.** It matches `code/colab/run_followup_tests.Pipeline.prepare` and `code/scheduling/sampler.py`:
  - prompt SNR on the prompt as given;
  - RMS normalized to 0.3;
  - the duration predictor with the seed;
  - the guided teacher (CFG 2, no stochastic term);
  - the student's restart levels with per-level seeds.
- **Development prototype.** It is not adviser-reviewed.
