# 2026-11-16: local SNR + drift prototype (demo)

Task: PROTO-001 (review). Requested by Rhenel Jhon R. Sajol:
- **The method:** "use SNR + drift and build the prototype", with the SNR + drift code in its own file.
- **The interface:** HTML in the design of `presentation/index.html`.
- **Where it runs:** locally, not on Colab ("our prototype should be local").
- **What it's for:** a demo ("prototype is for demo only").
- **Data:** the model and Seed-TTS downloaded to this computer first.

Nothing here changes the thesis method, the experiments or the manuscript. The prototype is a development demonstration and is not adviser-reviewed.

## Decisions

- **The interface:** an HTML page served by a small local Python server. The terminal only starts it. A terminal mode (`cli.py`) is included for testing.
- **Where it runs:** this computer's CPU. It has AMD Radeon integrated graphics and no CUDA, 13.8 GB RAM and Python 3.12.10. Wall-clock times are this computer's. The pass counts match the thesis.
- **The method file:**
  - **Contents:** `code/prototype/snr_drift.py` holds only the SNR estimator, the drift and the rule (Definition 3.3 and Chapter 3's stopping policy).
  - **Experiment code:** `code/scheduling/sampler.py` and the experiment runners are unchanged; EXP-032 depends on them.
  - **Tests:** they check the file against the sampler's functions (extracted from its source) and against the experiments' Chapter 3 policy function.
- **λ presets** (Appendix D):
  - **0.35** (cross-validated in the development pilot): the default, because it keeps the drift check active.
  - **1.20** (LTT-certified on the calibration speakers): offered as a preset. Appendix D notes that at this value the policy acts as SNR only.
  - **Neither is the final calibrated threshold**, and the page says so.
- **The reference:** Step 8 of the 16-interval grid (21 passes), Chapter 3's reference and the thesis's version of the paper's t_switch = 0.25.
- **Examples:**
  - **Speakers:** development-split speakers only, with the experiments' per-utterance seeds.
  - **Prompt degradations:** the experiments' own functions (`run_followup_tests.degrade`).
  - **Conditions offered:** clean, gauss25, noisy15, reverb03, reverb09 and telephone. Babble needs DEMAND.

## Downloaded and installed (not committed)

- **Checkpoints:** `ckpts/model_85000.pt` (4,019,822,332 bytes) and `ckpts/model_1500.pt` (270,615,794 bytes), from huggingface.co/yl4579/DMOSpeech2. Their SHA-256 hashes are `e143188f7c46...` and `6bd50428f3b5...`, equal to the hashes the Colab notebooks check.
- **Seed-TTS English test set:** `research/data/seed-tts-eval/en`, from the `zhaochenyang20/seed-tts-eval` dataset (English part only, 2,102 files, about 440 MB). Every development, calibration and test item's prompt and ground-truth audio is present (100, 200 and 200 items).
- **Python environment** `.venv`:
  - PyTorch 2.5.1 CPU and torchaudio 2.5.1;
  - the inference packages, pinned in `code/prototype/requirements.txt`;
  - FastAPI and uvicorn for the page.
  - **A warning that doesn't matter:** `transformers` 5.23 disables its PyTorch models because it wants PyTorch 2.6 or later. The prototype does not use them, since automatic transcription is not used.

## Files

**In `code/prototype/`:**
- `snr_drift.py`;
- `engine.py`;
- `server.py`;
- `cli.py`;
- `web/index.html`;
- `README.md`;
- `requirements.txt`;
- `start_prototype.bat`.

**Elsewhere:**
- **Tests:** `code/tests/test_prototype_snr_drift.py`, 12 tests.
- **`.gitattributes`:** `*.bat` files keep CRLF.

**Engine notes:**
- **The vocabulary:** the vendored snapshot lacks `data/Emilia_ZH_EN_pinyin/vocab.txt`, which both `DMOInference` and the guidance model load by name. The engine points the loader, in memory, at the identical 2,545-entry vocabulary bundled in `f5_tts/infer/examples/vocab.txt`. No vendored file is changed.
- **Memory:** after loading, it drops the training-only networks (`fake_unet` and the discriminator) to save RAM.
- **Speed:** the teacher stops as soon as the rule decides, so the skipped passes are not computed.

## First runs on this computer (CPU, 11 threads)

**Terminal, development speaker 1, clean prompt, λ = 0.35.** The local run reproduces EXP-031's T4 record:
- **Prompt SNR:** 28.2 dB (EXP-031: 28.181).
- **Drifts d₁ to d₈:**

  | | d₁ | d₂ | d₃ | d₄ | d₅ | d₆ | d₇ | d₈ |
  |---|---|---|---|---|---|---|---|---|
  | Local CPU | 0.361 | 1.322 | 1.120 | 0.844 | 0.750 | 0.401 | 0.325 | 0.209 |
  | EXP-031 (T4) | 0.3609 | 1.3215 | 1.1202 | 0.8442 | 0.7496 | 0.4007 | 0.3249 | 0.2087 |

- **Decision:** Step 8, as recorded.
- **Time:** model load 73 s (first start, including the vocoder download); 3.6–4.1 s per teacher step; 40.7 s in total for 3.5 s of speech.

**Web server, driven over HTTP the way the page does, development speaker 5:**

| | clean | noisy15 |
|---|---|---|
| Prompt SNR | 44.26 dB | 18.05 dB |
| Decision | d₄ = 0.264 → **Step 4**, 13 of 21 passes (38% saved) | Gate closed → **Step 8** |
| EXP-031's recorded step | 4 | 8 |
| Total time | 17.5 s | 14.6 s |

- **The noisy15 drift:** it was already 0.164 at Step 4, so the gate, not the drift, kept the noisy start away from the student.
- **The fixed reference (Step 8):** synthesized from the same trajectory and seeds when asked. It differs from the Step 4 output (correlation 0.60).
- **The audio:** all files are finite, 24 kHz, with the expected lengths.

**Fixes made from these runs:**
- **Clipping:** generated speech reached full scale and clipped when written as 16-bit WAV, because the pipeline normalizes the prompt to RMS 0.3. Playback files are now scaled down only if they would clip. This is playback only; the experiments score the unscaled float audio.
- **Frames:** the reported "frames" are now the generated frames, as in the experiments' CSVs (329, not the predicted 330).
- **The launcher:** `start_prototype.bat` is now CRLF. Started through PowerShell, it served the page with the model ready in 38 s, and the test server was then stopped.

**Upload path, over HTTP:**
- **Without a transcript:** refused with "Type what the prompt says (its transcript)" (HTTP 400).
- **With a WAV and a new sentence** ("This prototype decides when the teacher can hand over to the student."):
  - SNR 44.3 dB and Step 5, 15 of 21 passes;
  - 5.53 s of speech in 19.3 s;
  - the download works.

**The page:** example #5 is selected by default. Under the condition buttons the page shows EXP-031's recorded T4 decision for development speakers 1–30, and after a run it says whether the local decision matches. The page script passes `node --check`. It was not opened in a browser in this session; the requester should look at it.

## Checks run

- **Unit tests:**
  - `python -m unittest discover -s code/tests -p test_prototype_snr_drift.py`: 12 tests, OK with the system Python (the PyTorch drift comparison skipped).
  - The same with `.venv` (PyTorch present): 12 tests, OK, including an exact match of the drift with the sampler's function.
- **Server:** it imports in `.venv`, its routes register, 100 examples load and no file is missing.
- **End-to-end runs:** the terminal and HTTP runs above.
- **Workspace and full suite:** `python scripts/check_workspace.py` and `python -m unittest discover -s code/tests` (results in STATUS).

## Remaining

- **Look at the page:** the requester should review it in a browser, against `presentation/index.html`.
- **Not committed yet.**
