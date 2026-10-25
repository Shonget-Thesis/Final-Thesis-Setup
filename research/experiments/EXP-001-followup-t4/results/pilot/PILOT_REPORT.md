# Development pilot

Development pilot (development-split speakers only). Assurance and planning, not evidence for the method.

## Pre-experiment checks (Section 3.3.5)

Overall: **PASSED** on 2 clean development cases (Tesla T4).

| Case | Determinism | Single pass = separate runs | Forced fallback = Step 8 | Gate open, λ = ∞ → Step 4 | Forward passes = 2(k+1)+3 |
|---|---|---|---|---|---|
| common_voice_en_10933825 | yes | yes | yes | yes | k=4: 13/13, k=5: 15/15, k=6: 17/17, k=8: 21/21 |
| 00-common_voice_en_13901 | yes | yes | yes | yes | k=4: 13/13, k=5: 15/15, k=6: 17/17, k=8: 21/21 |

## Appendix A worked example (real intermediate values)

- clean: 459 frames; quietest 46 mean energy 1.747e-05, loudest 230 mean energy 2.043e-02 → Ŝ = 30.68 dB (estimate_prompt_snr 30.68; matches).
  d4 = 116.17/218.01 = 0.533, d5 = 85.92/204.42 = 0.420, d6 = 65.97/197.16 = 0.335 (matches the sampler).
- noisy15: 459 frames; quietest 46 mean energy 3.563e-04, loudest 230 mean energy 2.076e-02 → Ŝ = 17.65 dB (estimate_prompt_snr 17.65; matches).
  d4 = 42.93/295.32 = 0.145, d5 = 37.96/281.35 = 0.135, d6 = 32.68/281.35 = 0.116 (matches the sampler).

## Reproduction of the committed T4 grid (clean + noisy15)

40 cases, 160 rows; same failure label in 160/160 rows. Max |Δ|: prompt SNR 0.0000 dB, d₄ 0.000000, WER 0.0000, SIM 0.0000.

## Conditions

| Condition | n | Prompt SNR range (dB) | Gate fallback | Step-4 failures | SNR-only fails / NFE | SNR + drift 0.50 fails / NFE | Only SNR-only fails | Only SNR + drift fails |
|---|---:|---|---:|---:|---|---|---:|---:|
| clean | 20 | 26.5–50.0 | 0% | 0 | 0 / 13.00 | 0 / 15.10 | 0 | 0 |
| noisy15 | 20 | 17.5–18.2 | 100% | 13 | 0 / 21.00 | 0 / 21.00 | 0 | 0 |
| gauss18 | 20 | 19.9–21.2 | 5% | 10 | 10 / 13.40 | 10 / 13.40 | 0 | 0 |
| gauss20 | 20 | 21.4–23.2 | 0% | 9 | 9 / 13.00 | 9 / 13.00 | 0 | 0 |
| gauss22 | 20 | 22.7–25.1 | 0% | 6 | 6 / 13.00 | 6 / 13.00 | 0 | 0 |
| gauss25 | 20 | 24.2–28.1 | 0% | 2 | 2 / 13.00 | 2 / 13.20 | 0 | 0 |
| babble15 | 20 | 18.5–20.5 | 75% | 1 | 0 / 19.00 | 0 / 19.80 | 0 | 0 |
| babble20 | 20 | 21.4–25.1 | 0% | 1 | 1 / 13.00 | 1 / 15.30 | 1 | 1 |
| reverb03 | 20 | 22.3–50.0 | 0% | 2 | 2 / 13.00 | 2 / 14.60 | 0 | 0 |
| reverb09 | 20 | 14.8–41.3 | 60% | 4 | 0 / 17.80 | 1 / 18.20 | 0 | 1 |
| telephone | 20 | 31.3–50.0 | 0% | 2 | 2 / 13.00 | 2 / 13.00 | 0 | 0 |

Seed-noise floor: Step 8 with seed B fails against seed A for 4/10 speakers.

Ground-truth sample rates: [24000].

Median 15.6 s per case. Estimated full run: fresh grid 1.4 h, degradations 6.5 h, seed floor at most 1.7 h.

Small development sample: use it to check the pipeline and plan, not to estimate risk or test drift.

## Offline analyses on the committed T4 grid (development evidence; no GPU)

- Learn-Then-Test for SNR + drift on the calibration split (n = 200, λ grid 0.20–1.20): λ̂ = 1.2.
- SNR-only rule with θ calibrated by Learn-Then-Test (θ from 30.0 down to 14.0 dB in 0.5 dB steps): θ̂ = 18.5; on the test split 4/200 speaker failures, mean NFE 17.02.
- Calibrate on clean prompts only: λ̂ = 1.2; on test noisy15 prompts 0/200 failures, mean NFE 21.00.

| Calibration size | No threshold certified | Median λ̂ | Mean test risk | Mean test NFE |
|---:|---:|---:|---:|---:|
| 50 | 0% | 0.35 | 1.5% | 19.02 |
| 100 | 0% | 0.35 | 1.4% | 18.71 |
| 200 | 0% | 1.2 | 2.0% | 17.04 |

The 200-speaker test split was inspected before any protocol was frozen, so these are development evidence.
