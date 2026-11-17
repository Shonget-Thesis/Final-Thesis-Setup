# 2026-11-16: An innovation SNR for the handover gate (offline screen, stored T4 scores)

**Correction (2026-11-17):** the formula below survives, but its predictability (Kolmogorov–Szegő) justification was falsified the next day. High-pass hiss, perfectly predictable in theory, did 65% of white noise's harm. The supported explanation is spread: the teacher copies spread-out background texture that the mode-seeking student drops. The statistic is renamed the spread-weighted SNR. See the [spread-weighted SNR log](2026-11-17-spread-weighted-snr.md).

Follow-up to EVAL-026 (EXP-033); no new task ID.
- **The request:** Rhenel Jhon R. Sajol asked: "we need to add something in SNR, what's the flaw of SNR currently that relates to the problem so that we can create one that solves it".
- **Status:** a derivation, an offline screen on stored results (no synthesis) and a targeted literature check. Nothing is adopted.

## The flaw (stored T4 data)

Chapter 3's prompt SNR (Definition 3.3) compares the loudest half of the frames with the quietest 10%. The handover cares whether the teacher hurts the output relative to the student alone. On that question the SNR is wrong in both directions:

| Prompt condition | Cases | SNR reads (median) | 20 dB gate sends to the student | Teacher harm: UTMOS(student) − UTMOS(W8) |
|---|---:|---:|---:|---:|
| White noise at 25 dB (`gauss25`) | 60 | 27.5 dB | 0 of 60 | **+1.27** |
| White noise at 18–22 dB (EXP-018) | 180 | 20.7–24.5 dB | 3 of 180 | (no student arm; W8 UTMOS 1.25–1.45) |
| White noise at 15 dB (`noisy15`) | 60 | 17.9 dB | 60 of 60 | +0.68 |
| Heavy reverberation (`reverb09`) | 30 | 18.3 dB | 20 of 30 | +0.03 |
| Cafeteria babble at 15 dB (`babble15`, EXP-031) | 30 | 19.2 dB | 25 of 30 | +0.05 |
| Telephone band (`telephone`) | 30 | 43.5 dB | 0 of 30 | +0.00 |
| Clean | 60 | 38.0 dB | 0 of 60 | −0.01 |

**What the table shows:**
- **The teacher's harm depends on the kind of background, not its loudness.** White noise is copied by the teacher far more than by the student. Reverberation, babble and band-limiting are copied by both.
- **So the gate is wrong both ways:** it lets through the most harmful prompts (`gauss25`), and it blocks prompts where the student is no better (`reverb09`, `babble15`).

## The changed term

**Hypothesis:**
- **The student:** a mode-seeking distilled student can drop the part of a prompt's background it cannot predict, and keeps the structured part.
- **The teacher:** a guided ODE teacher reproduces both.
- **So:** the teacher's extra harm grows with the power of the background's unpredictable part.

**The formula:** for a stationary background, that power is the one-step prediction-error variance. By the Kolmogorov–Szegő formula, this equals the geometric mean of the power spectrum. The noise term of the SNR changes accordingly:

    SNR     = 10 log10( P_speech / mean energy of the quietest 10% of frames )     (Definition 3.3)
    SNR_inn = 10 log10( P_speech / exp( mean_f log N(f) ) )

- **N(f):** the 10th percentile over frames of the power in frequency bin f, from 60 Hz to the Nyquist frequency. This is a per-band floor, as in minimum-statistics noise estimation.
- **P_speech:** the mean power of the loudest half of the frames.
- **The frames:** the same 25 ms frames with a 10 ms hop as Definition 3.3.
- **The ablation `SNR_floor`:** it keeps the per-band floor but averages it arithmetically. This separates the two changes.

## Offline screen

**Script:** `code/analysis/innovation_snr_screen.py`. Output: `research/experiments/EXP-033-cheaper-teacher-path/offline/innovation_snr_screen.json`. The pass rules are fixed in its docstring before the run.

**Cases:**
- **EXP-031:** development speakers 1–30 × clean, `gauss25`, `noisy15`, `reverb03` and `reverb09`.
- **EXP-030:** development speakers 61–90 × clean, `gauss25`, `noisy15` and `telephone`.
- **Total:** 270 cases.
- **How each case's prompt is rebuilt:** with the experiments' own degradation code and seed.
- **Babble was skipped (30 cases):** the DEMAND recording could not be downloaded, because zenodo.org was unreachable from this computer.
- **Harm:** UTMOS(student) − UTMOS(W8) from the stored T4 scores. A case is harmful when harm > 0.3 (93 cases).

| Pre-set rule | Result |
|---|---|
| 1. Definition 3.3 on the rebuilt prompts equals the stored `prompt_snr` (≤ 0.01 dB) | **Pass** (largest difference 7.6e-6 dB) |
| 2. AUC for harmful cases: `SNR_inn` ≥ `SNR` + 0.05 | **Pass** (0.807 against 0.649; the arithmetic ablation is 0.638) |
| 3. Routing the same 80 cases to the student as the 20 dB gate: mean UTMOS ≥ `SNR`'s + 0.05 | **Pass** (2.546 against 2.438) |

**By condition** (ranges over cases):

| Condition | SNR | SNR_inn | Floor flatness (median) |
|---|---|---|---:|
| `noisy15` | 16.8–18.2 | 25.9–27.2 | 0.92 |
| `gauss25` | 22.0–28.1 | 34.4–36.8 | 0.84 |
| `reverb09` | 13.2–41.3 | 40.5–68.4 | 0.005 |
| `reverb03` | 20.5–50.0 | 45.5–77.3 | 0.011 |
| Clean | 23.3–50.0 | 50.2–80.2 | 0.014 |
| `telephone` | 30.5–50.0 | 86.8–109.7 | 0.00002 |

**Threshold sweep** (mean UTMOS when the cases below the threshold get the student and the rest W8):

| Threshold | Old SNR: cases to the student | Old SNR: mean UTMOS | New SNR_inn: cases to the student | New SNR_inn: mean UTMOS |
|---:|---:|---:|---:|---:|
| 20 dB | 80 | 2.438 | 0 | 2.286 |
| 30 dB | 178 | 2.716 | 60 | 2.436 |
| 40 dB | 218 | 2.713 | **120** | **2.718** |
| 45 dB | 241 | 2.716 | 126 | 2.721 |

Reference points: everything to the student gives 2.714, everything to the teacher 2.286, and a per-case oracle 2.752.

**Reading:**
- **SNR_inn separates the two kinds of background with a wide margin.** Every white-noise prompt reads ≤ 36.8 dB. Every clean, reverberant and telephone prompt reads ≥ 40.5 dB.
- **A gate at 40 dB on SNR_inn** sends exactly the 120 white-noise prompts to the student. It reaches the all-student quality (2.718) while keeping the teacher, and its variety, for the other 150.
- **The old SNR needs 218 student routes for the same quality,** so 98 more prompts would lose the teacher's variety for nothing.
- **What drives the gain:** the change is in recognising the kind of noise. Within white noise alone, neither statistic ranks the individual harmful cases well (AUC 0.21 and 0.24).

## Literature check (targeted, 2026-11-16)

- **Spectral flatness** (the geometric-to-arithmetic ratio of a spectrum) is a classic noise-likeness measure ([overview](https://en.wikipedia.org/wiki/Spectral_flatness)).
- **Non-intrusive SNR estimators** are compared in [Eaton, Brookes and Naylor](https://spiral.imperial.ac.uk/bitstream/10044/1/51055/2/eaton2013b.pdf). Their sensitivity to noise type is a known issue.
- **Distilled TTS students can beat their teachers,** attributed to mode shrinkage ([DMOSpeech](https://arxiv.org/abs/2410.11097v1); [Starodubcev et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Starodubcev_Your_Student_is_Better_Than_Expected_Adaptive_Teacher-Student_Collaboration_for_CVPR_2024_paper.pdf)).
- **Not found:**
  - the claim that the distilled student drops a prompt's *unpredictable* background but keeps structured backgrounds;
  - a teacher–student gate whose noise term is the background's innovation (prediction-error) power.

**Status:** partly novel. The operator (a geometric mean over the spectrum) is classic. What would be new is the mechanism linking the handover's harm to the background's innovation power, and the gate it implies. The search was targeted, not exhaustive.

## Caveats

- **Synthetic degradations only:** white Gaussian noise, synthetic reverberation and a telephone filter.
  - **Colored stationary noise** (fans, hum, pink noise) is untested. The hypothesis predicts it reads higher than white noise of the same power, and that the teacher's harm on it follows its innovation power. That needs synthesis.
- **Babble is untested here** (no DEMAND file). The hypothesis predicts a high reading (speech-shaped), matching its measured lack of harm.
- **The 40 dB threshold was chosen after looking at these data.** It must be calibrated on other speakers (Learn-Then-Test, as in Chapter 3).
- **W8 stands in for the teacher path.** With `CPH8` the teacher path is more fragile on white noise (EXP-033: 7 of 30 runaway outputs at `noisy15`), so catching white noise matters more, not less.

## Next

A T4 test of the gate on new development speakers:
- **Conditions:** white noise at 15–25 dB, pink and brown noise, babble (DEMAND, downloaded on Colab as before), reverberation, telephone and clean.
- **Arms:** the student alone, `CPH8` and W8.
- **Comparison:** `SNR_inn` against Definition 3.3 under one frozen calibration.

## Checks run

- **The screen** as above (exit 0, 270 cases).
- **Tests:** `code/tests/test_innovation_snr_screen.py`, 5 tests. On synthetic signals, a white background gives a flat floor and agreeing terms; a structured background reads higher; ten times louder white noise lowers SNR_inn by 20 dB. Also the AUC orientation and the routing helper.
- **Workspace check:** `python scripts/check_workspace.py` PASSED.
- **Full suite:** `python -m unittest discover -s code/tests` ran 387 tests, OK (1 skipped).
