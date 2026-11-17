# 2026-11-17: The spread principle and a spread-weighted SNR gate (three falsification tests)

Follow-up to EVAL-026 and the [innovation SNR log](2026-11-16-innovation-snr.md); tracked as EVAL-027.
- **The request:** Rhenel Jhon R. Sajol asked for "a brand new in SNR" that the team can name and call its own, novel "just like other existing algorithms were made".
- **What makes a method ownable:** a principle that survives tests built to break it, not a name. So each explanation was put through a test that could falsify it, with the prediction fixed before the run.
- **Status:** screening only (CPU runs, six development sentences per test) plus the offline analysis of 270 stored T4 cases. Nothing is adopted, and no adviser has reviewed it.

## The starting point

The offline screen on 270 stored T4 cases ([innovation SNR log](2026-11-16-innovation-snr.md)) showed two things:
- **Chapter 3's SNR (Definition 3.3) is wrong about where the teacher hurts.** It lets white noise at 25 dB through, where the teacher loses 1.27 UTMOS against the student alone. It sends reverberation and babble to the student, where the student is no better.
- **A statistic fixed it** (AUC 0.65 → 0.81): speech power over the geometric mean, across frequency, of the per-band background floor. It was first justified by predictability (the Kolmogorov–Szegő innovation power).

## The tests

All tests:
- add noise at the same total power, 25 dB SNR, seeded by each sentence's input seed;
- use six development sentences (the shortest clean EXP-033 quality cases);
- compare the student alone (4 passes) with W8 (21 passes), on this computer's CPU.
- **Harm** = UTMOS(student) − UTMOS(W8).

**Test 1, noise colour** (`code/analysis/noise_color_screen.py`). The old SNR reads every condition the same.

| Noise | Old SNR | New SNR | Harm |
|---|---:|---:|---:|
| White | 27.4 | 35.5 | **+1.99** (6 of 6 positive) |
| Pink (1/f) | 27.6 | 39.6 | +0.83 (6 of 6) |
| Speech-shaped | 27.7 | 50.2 | +0.10 |
| Low-pass at 1 kHz | 27.6 | 63.2 | −0.03 |
| Three steady tones | 27.3 | 66.0 | −0.00 |
| Clean | 39.2 | 67.3 | −0.02 |

**Pre-set rules:**
- **Validation:** white harm ≥ 0.5 and clean harm ≤ 0.3. **Pass.**
- **No miss:** no coloured noise reads clean (≥ 40 dB) while harming at least half as much as white. **Pass.** Pink is borderline: it reads 38.1–40.5 dB with harm +0.83.
- **Ordering:** Spearman ≤ −0.6. **Pass** (new SNR −0.9; old SNR +0.1).

**Test 2, high-pass hiss** (empty below 2 kHz; `noise_theory_screen.py`). This separates predictability from masking.
- **The predictions:** the predictability explanation said ≤ 33% of white's harm; masking said ≥ 67%.
- **The result:** 65% (+1.29 against +1.99). That is far above predictability's bound, so **the predictability explanation is rejected.** Masking narrowly missed its bound.
- **The gate:** the new SNR caught all 6 high-pass cases (34.9–38.0 dB).
- **A band-pass 1–4 kHz control:** harm +0.05. The gate let all 6 through (43.2–47.1 dB).

**Test 3, a narrow high band (6–8 kHz)** (`noise_band_screen.py`). This separates spread from masking. The band has half-power edges at 6 and 8 kHz with eighth-order skirts: 70% of its power lies in 6–8 kHz, 95% in 5–9 kHz and under 0.1% below 4 kHz.
- **The predictions:** spread said ≤ 33% of white's harm; masking said ≥ 67% (and the gate would then miss it).
- **The result:** 3% (+0.06 against +1.99), so **the masking explanation is rejected.**
- **The gate:** the new SNR reads it as clean (40.4–47.4 dB), which is correct.

**The statistic comparison** (`noise_statistic_compare.py`; 270 stored T4 cases and 48 local cases):

| Statistic | AUC (T4 stored) | AUC (local) | Best-threshold misroutes (T4 stored) |
|---|---:|---:|---:|
| Definition 3.3 | 0.649 | 0.622 | 86 |
| **Spread-weighted SNR** (below) | **0.807** | **0.907** | **43** |
| Log-average of per-band SNR | 0.790 | 0.778 | 58 |
| Mean per-band noise-to-speech ratio (the masking formula) | 0.720 | 0.672 | 80 |

## The principle that survives

**The teacher copies spread-out background texture that the mode-seeking student drops.** Across eight noise types and the stored T4 conditions, the teacher's extra harm follows how broad the background floor is, not:
- how loud it is (test 1);
- how predictable it is (test 2);
- whether it sits where speech is weak (test 3).

**The evidence, broad to narrow:**
- **Broad floors are copied by the teacher alone:** white, high-pass hiss, pink.
- **Narrow or structured floors are copied by both or neither,** so they do no harm:
  - band-pass at 1–4 kHz;
  - a narrow band at 6–8 kHz;
  - low-pass noise;
  - tones;
  - speech-shaped noise;
  - reverberation;
  - telephone band-limiting.

**Why it fits:** it agrees with the account of distillation in the DMOSpeech papers. A mode-seeking (reverse-KL) student concentrates on high-probability regions ("mode shrinkage"), and broadband noise texture is the highest-entropy part of a prompt.

## The changed term

    Definition 3.3:      SNR    = 10 log10( P_loud / mean energy of the quietest 10% of frames )
    Spread-weighted:     SNR_sw = 10 log10( P_speech / exp( mean_f log N(f) ) )
                                = 10 log10( P_speech / mean_f N(f) )  -  10 log10( SFM_N )

**The terms:**
- **N(f):** the 10th percentile over frames of the power in frequency bin f (60 Hz to Nyquist; the same 25 ms frames and 10 ms hop as Definition 3.3).
- **P_speech:** the mean power of the loudest half of the frames.
- **SFM_N:** the spectral flatness of the floor (its geometric over its arithmetic mean, ≤ 1).

**In words:** the floor's power is counted in proportion to how spread out it is. A flat (white) floor counts in full. A floor concentrated in a few bands counts for little.

**Properties (direct from the formula):**
1. **Reduces to the floor SNR for a perfectly flat floor** (SFM = 1).
2. **Scaling the background by a factor g changes SNR_sw by exactly −20 log10 g dB.** The level enters as in any SNR.
3. **Is never below the floor SNR** (SFM ≤ 1). It can only move a prompt towards the teacher, never towards the student.
4. **Needs no forward pass.** It runs on the prompt before synthesis, as Definition 3.3 does.

**The gate:** below a threshold, the student alone (4 passes); above it, the commit-point handover (`CPH8`, 12 passes).
- **Where the threshold might sit:** on these data it separates at about 38–40 dB, but that is after looking. The threshold is to be certified by Learn-Then-Test on other speakers, as in Chapter 3.

## Novelty, as checked so far

| Part | Status |
|---|---|
| The procedure: an SNR gate between a teacher handover and the student | Known (Chapter 3; the 2026-11-13 offline screen) |
| The building block: spectral flatness (Wiener entropy) of a spectrum | Classic signal processing |
| Non-intrusive SNR estimation, and its sensitivity to noise type | Known |
| Zero-shot TTS copying a noisy prompt's background | Known (for example [arXiv 2406.05699](https://arxiv.org/abs/2406.05699)) |
| Distilled students beating teachers through mode shrinkage | Known as a general effect ([DMOSpeech](https://arxiv.org/abs/2410.11097v1)) |
| **That the teacher, but not the mode-seeking student, copies spread-out background texture, and that the harm follows the floor's spread, not its loudness, predictability or position** | **Not found** in targeted searches |
| **A teacher–student gate whose noise term is the flatness-weighted floor** | **Not found** |

**Where it stands:** an existing procedure with one changed rule, a principle that survived three tests built to break it, and evidence on stored T4 data. That is the same shape as most method contributions. The naming is the team's choice; "innovation SNR" should not be used, because its predictability justification was falsified.

## Caveats

- **Screening only:** six sentences per local test, UTMOS as the only measure, CPU outputs.
- **Synthetic noises:** babble is untested (the DEMAND file could not be downloaded here), as are real recorded noises.
- **Pink noise sits at the threshold** (38–40.5 dB with +0.83 harm), and heavy reverberation reaches down to 40.5 dB with no harm. Any threshold near 40 dB will misroute some of one or the other.
  - **Which side to err on:** sending a harmless prompt to the student costs variety only; sending a harmful prompt to the teacher costs quality. The calibration should err towards the student.

## Next

A preregistered T4 confirmation (EXP-034) on development speakers not used in these screens:
- **Statistics:** the spread-weighted SNR against Definition 3.3, each with its threshold frozen from the stored data.
- **Noises:** white (15–25 dB), pink, high-pass, narrow-band, speech-shaped, low-pass, DEMAND babble (downloaded on Colab, as EXP-031 did), reverberation, telephone and clean.
- **Arms:** the student alone, `CPH8` and W8.
- **Measures:** WER, speaker similarity and UTMOS.

## Checks run

- **The screens:** noise colour (36 cases), noise theory (18), narrow band (6), and the statistic comparison (270 stored + 48 local). All exited 0, and their JSON is in `research/experiments/EXP-033-cheaper-teacher-path/offline/`.
- **Rebuilt prompts:** they reproduce the stored Definition 3.3 values within 0.01 dB.
- **Tests:** `code/tests/test_noise_color_screen.py` (4) and `code/tests/test_spread_snr.py` (4: the floor-SNR bound and the flatness identity, the 20 log10 level scaling, spread noise reading lower than narrow noise of equal power, and the band noise's shape).
- **Workspace check:** `python scripts/check_workspace.py` PASSED.
- **Full suite:** `python -m unittest discover -s code/tests` ran 395 tests, OK (1 skipped).
