# EXP-017: Chapter 3 rehearsal on two teacher grids (development speakers)

60 development speakers × 6 conditions (clean, noisy15, gauss18, gauss20, gauss22, gauss25); 360 of 360 cases on Tesla T4; preregistration `7ad1b65fd73697ee…`. Every number is computed from `grid32.csv` by `run_grid32_prelim.py analyze`. Development evidence: schedulers are replayed on recorded outputs and calibrated by cross-validation on the same speakers; the certified evaluation on held-out speakers follows the proposal.

## Outputs

| Output | Teacher evaluations | Student steps | Forward passes | Mean runtime (s) | RTF |
|---|---:|---:|---:|---:|---:|
| 16-interval Step 4 (s = 0.07), code default | 5 | 3 | 13 | 2.04 | 0.437 |
| 16-interval Step 5 (s = 0.10) | 6 | 3 | 15 | 2.34 | 0.501 |
| 16-interval Step 6 (s = 0.15) | 7 | 3 | 17 | 2.64 | 0.565 |
| 16-interval Step 8 (s = 0.25), thesis reference | 9 | 3 | 21 | 3.23 | 0.693 |
| 32-interval s = 0.07 (7 teacher steps) | 8 | 3 | 19 | 2.94 | 0.629 |
| 32-interval s = 0.10 (9 teacher steps) | 10 | 3 | 23 | 3.53 | 0.757 |
| 32-interval s = 0.15 (11 teacher steps) | 12 | 3 | 27 | 4.13 | 0.884 |
| 32-interval s = 0.20 (13 teacher steps) | 14 | 3 | 31 | 4.73 | 1.012 |
| 32-interval s = 0.25 (14 teacher steps), reference on this grid | 15 | 3 | 33 | 5.03 | 1.076 |
| Paper's 14 + 2, released handover (32-interval) | 15 | 2 | 32 | 4.87 | 1.044 |
| Paper's Algorithm 2 as written (32-interval) | 14 | 2 | 30 | 4.58 | 0.980 |
| Teacher only, 32 steps (F5-TTS) | 32 | 0 | 64 | 9.65 | 2.065 |
| Student only, 4 steps (DMOSpeech 2) | 0 | 4 | 4 | 0.70 | 0.150 |

Forward passes: two per guided teacher evaluation plus one per student step (Chapter 3's cost formula). Runtime is synthesis time (preparation, teacher, student, vocoder; not scoring), composed per input from the measured time of one guided evaluation, one student step and one vocoder call; RTF = runtime / audio length.

## Checks

- Fresh 16-interval outputs against the stored T4 labels (max |Δ|, cases compared): wer_g16_4 0.0000 (360); similarity_g16_4 0.0000 (360); wer_g16_5 0.0000 (360); similarity_g16_5 0.0000 (360); wer_g16_6 0.0000 (360); similarity_g16_6 0.0000 (360); wer_g16_8 0.0000 (360); similarity_g16_8 0.0000 (360). Same Step-4 failure label in 360 of 360 cases; max |Δd₄| 0.00e+00.
- Ground-truth sample rates: 24000. Mean per case: 54.7 s (preparation 0.082 s, guided evaluation 0.2988 s, student step 0.1525 s, vocoder 0.0055 s, audio 4.80 s); peak GPU memory 9623 MB.

## Quality of every output

| Output | Prompts | WER | SIM | UTMOS |
|---|---|---:|---:|---:|
| g16_4 | clean | 0.0234 | 0.9357 | 3.922 |
| g16_4 | noisy | 0.2469 | 0.7768 | 1.479 |
| g16_5 | clean | 0.0261 | 0.9352 | 3.921 |
| g16_5 | noisy | 0.0816 | 0.7812 | 1.478 |
| g16_6 | clean | 0.0255 | 0.9356 | 3.923 |
| g16_6 | noisy | 0.0742 | 0.7831 | 1.475 |
| g16_8 | clean | 0.0272 | 0.9359 | 3.924 |
| g16_8 | noisy | 0.0761 | 0.7854 | 1.460 |
| s007 | clean | 0.0234 | 0.9367 | 3.927 |
| s007 | noisy | 0.3663 | 0.7772 | 1.479 |
| s010 | clean | 0.0255 | 0.9355 | 3.886 |
| s010 | noisy | 0.1845 | 0.7830 | 1.489 |
| s015 | clean | 0.0248 | 0.9351 | 3.909 |
| s015 | noisy | 0.1285 | 0.7854 | 1.488 |
| s020 | clean | 0.0269 | 0.9358 | 3.907 |
| s020 | noisy | 0.0748 | 0.7867 | 1.478 |
| s025 | clean | 0.0269 | 0.9365 | 3.908 |
| s025 | noisy | 0.0831 | 0.7854 | 1.475 |
| paper | clean | 0.0272 | 0.9333 | 3.836 |
| paper | noisy | 0.3417 | 0.7886 | 1.505 |
| alg2 | clean | 0.0263 | 0.9266 | 3.824 |
| alg2 | noisy | 0.4101 | 0.7748 | 1.418 |
| teacher32 | clean | 0.0246 | 0.9281 | 3.269 |
| teacher32 | noisy | 0.9736 | 0.7760 | 1.367 |
| student4 | clean | 0.0215 | 0.9369 | 3.932 |
| student4 | noisy | 0.0491 | 0.8624 | 2.688 |

Added failures (Definition 3.1) of each output against the thesis reference (16-interval Step 8), per condition:

| Output | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 | All |
|---|---:|---:|---:|---:|---:|---:|---:|
| g16_4 | 2 | 40 | 33 | 26 | 19 | 8 | 128 |
| g16_5 | 2 | 30 | 27 | 24 | 19 | 8 | 110 |
| g16_6 | 0 | 31 | 19 | 16 | 14 | 5 | 85 |
| s007 | 2 | 47 | 34 | 32 | 25 | 7 | 147 |
| s010 | 1 | 33 | 25 | 25 | 19 | 8 | 111 |
| s015 | 1 | 33 | 25 | 21 | 16 | 8 | 104 |
| s020 | 0 | 25 | 17 | 19 | 12 | 4 | 77 |
| s025 | 0 | 26 | 17 | 19 | 14 | 5 | 81 |
| paper | 7 | 36 | 27 | 27 | 24 | 19 | 140 |
| alg2 | 11 | 44 | 30 | 30 | 31 | 26 | 172 |
| teacher32 | 17 | 49 | 43 | 30 | 28 | 22 | 189 |
| student4 | 8 | 17 | 5 | 5 | 11 | 13 | 59 |

## Objective 1 on the 16-interval grid (released sampler)

### Fixed switching times (Definition 3.1 against g16_8; exact 95% intervals)

| Output | Passes | Clean failures (of 60) | Noisy failures (of 300) | All | 95% interval |
|---|---:|---:|---:|---:|---|
| 16-interval Step 4 (s = 0.07), code default | 13 | 2 | 126 | 128 (35.6%) | [30.6%, 40.7%] |
| 16-interval Step 5 (s = 0.10) | 15 | 2 | 108 | 110 (30.6%) | [25.8%, 35.6%] |
| 16-interval Step 6 (s = 0.15) | 17 | 0 | 85 | 85 (23.6%) | [19.3%, 28.3%] |
| 16-interval Step 8 (s = 0.25), thesis reference | 21 | 0 | 0 | 0 (0.0%) | [0.0%, 1.0%] |

Per condition: g16_4: clean 2, noisy15 40, gauss18 33, gauss20 26, gauss22 19, gauss25 8; g16_5: clean 2, noisy15 30, gauss18 27, gauss20 24, gauss22 19, gauss25 8; g16_6: clean 0, noisy15 31, gauss18 19, gauss20 16, gauss22 14, gauss25 5; g16_8: clean 0, noisy15 0, gauss18 0, gauss20 0, gauss22 0, gauss25 0

### Cheapest safe candidate per input, in hindsight (the oracle)

| Condition | g16_4 | g16_5 | g16_6 | g16_8 | Earlier than the reference |
|---|---:|---:|---:|---:|---:|
| clean | 58 | 0 | 2 | 0 | 100% |
| noisy15 | 20 | 16 | 2 | 22 | 63% |
| gauss18 | 27 | 13 | 8 | 12 | 80% |
| gauss20 | 34 | 9 | 10 | 7 | 88% |
| gauss22 | 41 | 5 | 5 | 9 | 85% |
| gauss25 | 52 | 3 | 2 | 3 | 95% |

### How strongly the signals relate to the oracle choice (Spearman ρ with the oracle's candidate index)

- snr: ρ = -0.450 (p 2.2e-19)
- drift at g16_4: ρ = -0.471 (p 3e-21)
- drift at g16_5: ρ = -0.452 (p 1.6e-19)
- drift at g16_6: ρ = -0.346 (p 1.6e-11)

SNR gate by the Objective 1 rule (lowest SNR above which the earliest candidate fails at most 5%): 28 dB. Plot: `objective1_signals_g16.png`.

## Objective 4 on the 16-interval grid (released sampler) (out of fold; Learn-Then-Test α = 0.05, δ = 0.1; 5-fold speaker cross-validation)

Saving Δ against g16_8 (21 passes) and share of the oracle saving η, with bootstrap 95% intervals over speakers. Within limit = failure rate ≤ 5%.

| Method | Failures | Rate [95% interval] | Within limit | Mean passes | Δ [95%] | η [95%] | Runtime (s) | RTF | Worst condition |
|---|---:|---|---|---:|---|---|---:|---:|---:|
| fixed g16_8 | 0/360 | 0.0% [0.0%, 1.0%] | yes | 21.00 | 0.0% [0.0%, 0.0%] | 0% [0%, 0%] | 3.23 | 0.693 | 0.0% |
| fixed g16_4 | 128/360 | 35.6% [30.6%, 40.7%] | no | 13.00 | 38.1% [38.1%, 38.1%] | 128% [122%, 134%] | 2.04 | 0.437 | 66.7% |
| fixed g16_5 | 110/360 | 30.6% [25.8%, 35.6%] | no | 15.00 | 28.6% [28.6%, 28.6%] | 96% [92%, 101%] | 2.34 | 0.501 | 50.0% |
| fixed g16_6 | 85/360 | 23.6% [19.3%, 28.3%] | no | 17.00 | 19.0% [19.0%, 19.0%] | 64% [61%, 67%] | 2.64 | 0.565 | 51.7% |
| SNR-only, 20 dB gate | 86/360 | 23.9% [19.6%, 28.6%] | no | 14.40 | 31.4% [31.0%, 31.7%] | 106% [101%, 111%] | 2.25 | 0.482 | 51.7% |
| SNR-only rule, gate calibrated (λ = ∞) | 7/360 | 1.9% [0.8%, 4.0%] | yes | 18.71 | 10.9% [9.9%, 11.7%] | 37% [33%, 40%] | 2.90 | 0.620 | 8.3% |
| SNR gate (20 dB), fixed later step 5 | 79/360 | 21.9% [17.8%, 26.6%] | no | 16.05 | 23.6% [23.3%, 23.8%] | 79% [76%, 83%] | 2.49 | 0.535 | 43.3% |
| SNR gate (20 dB), fixed later step 6 | 54/360 | 15.0% [11.5%, 19.1%] | no | 17.70 | 15.7% [15.5%, 15.9%] | 53% [50%, 56%] | 2.74 | 0.587 | 31.7% |
| Stabilization only (θ = −∞) | 0/360 | 0.0% [0.0%, 1.0%] | yes | 21.00 | 0.0% [0.0%, 0.0%] | 0% [0%, 0%] | 3.23 | 0.693 | 0.0% |
| Full policy (Definition 3.4), 20 dB gate | 0/360 | 0.0% [0.0%, 1.0%] | yes | 21.00 | 0.0% [0.0%, 0.0%] | 0% [0%, 0%] | 3.23 | 0.693 | 0.0% |
| Full policy, gate from the Objective 1 rule | 4/360 | 1.1% [0.3%, 2.8%] | yes | 19.53 | 7.0% [6.3%, 7.7%] | 23% [21%, 26%] | 3.02 | 0.644 | 3.3% |
| SNR + TAS (candidate) | 12/360 | 3.3% [1.7%, 5.8%] | yes | 18.07 | 14.0% [12.2%, 15.9%] | 47% [41%, 53%] | 2.79 | 0.600 | 10.0% |
| Cost-matched random rule (p_ref 1.00; mean failures over draws 0.0) | 0/360 | 0.0% [0.0%, 1.0%] | yes | 21.00 | 0.0% n/a | 0% n/a | 3.23 | 0.693 | 0.0% |
| Oracle: cheapest safe candidate in hindsight (diagnostic) | 0/360 | 0.0% [0.0%, 1.0%] | yes | 14.76 | 29.7% [28.3%, 31.1%] | 100% [100%, 100%] | 2.30 | 0.494 | 0.0% |

Choices per input: fixed g16_8: g16_8 360; fixed g16_4: g16_4 360; fixed g16_5: g16_5 360; fixed g16_6: g16_6 360; snr20: g16_4 297, g16_8 63; snr_cal: g16_4 103, g16_8 257; later5: g16_5 297, g16_8 63; later6: g16_6 297, g16_8 63; stab: g16_8 360; snr_drift: g16_8 360; snr_drift_dev: g16_4 66, g16_8 294; snr_tas: g16_4 132, g16_8 228; random: g16_8 360; oracle: g16_4 232, g16_5 46, g16_6 29, g16_8 53

Thresholds per fold (None = nothing certified; every input to the reference): snr20: [20, 20, 20, 20, 20]; snr_cal: [26, 26, 26, 26, 26]; later5: [20, 20, 20, 20, 20]; later6: [20, 20, 20, 20, 20]; stab: [None, None, None, None, None]; snr_drift: [None, None, None, None, None]; snr_drift_dev: [(28, 1.2), (28, 1.2), (28, 1.2), (28, 1.2), (26, 1.2)]; snr_tas: [11.33, 11.39, 10.04, 11.74, 15.53]

η above 100% means a method is cheaper than the oracle, which is possible only by adding failures; η is meaningful for methods within the limit.

### Testing the stabilization signal (exact McNemar on paired failures; fixed = failures of the second method avoided by the first)

| Comparison | Fixed | New | p |
|---|---:|---:|---:|
| snr_drift vs snr20 | 86 | 0 | 2.6e-26 |
| snr_drift vs snr_cal | 7 | 0 | 0.016 |
| snr_drift vs later5 | 79 | 0 | 3.3e-24 |
| snr_drift vs later6 | 54 | 0 | 1.1e-16 |
| snr_drift vs random | 0 | 0 | 1 |
| snr_drift vs stab | 0 | 0 | 1 |
| snr_drift_dev vs snr_drift | 0 | 4 | 0.12 |
| snr_tas vs snr_drift | 0 | 12 | 0.00049 |
| snr_tas vs snr20 | 74 | 0 | 1.1e-22 |

### Paired tests against g16_8 (speaker-level means; Wilcoxon with Holm correction; TOST margins WER ±0.01, SIM ±0.02)

| Method | Δ passes | Δ WER | p (Holm) | TOST WER p | Δ SIM | p (Holm) | TOST SIM p | Δ UTMOS | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed g16_4 | -8.00 | +0.1418 | 0.096 | 0.87 | -0.0071 | 0.14 | 4.5e-05 | +0.016 | 0.16 |
| fixed g16_5 | -6.00 | +0.0044 | 1 | 0.19 | -0.0035 | 0.85 | 1.1e-09 | +0.015 | 0.18 |
| fixed g16_6 | -4.00 | -0.0018 | 1 | 0.077 | -0.0019 | 1 | 1.9e-13 | +0.013 | 0.2 |
| SNR-only, 20 dB gate | -6.60 | +0.0064 | 1 | 0.2 | -0.0047 | 0.26 | 3.6e-09 | +0.015 | 0.2 |
| SNR-only rule, gate calibrated (λ = ∞) | -2.29 | -0.0009 | 1 | 3e-24 | +0.0001 | 1 | 5.1e-46 | +0.011 | 0.32 |
| SNR gate (20 dB), fixed later step 5 | -4.95 | +0.0040 | 1 | 0.081 | -0.0019 | 1 | 1.5e-13 | +0.014 | 0.2 |
| SNR gate (20 dB), fixed later step 6 | -3.30 | +0.0009 | 1 | 0.0025 | -0.0006 | 1 | 3.9e-18 | +0.012 | 0.2 |
| Stabilization only (θ = −∞) | +0.00 | +0.0000 | 1 | 0 | +0.0000 | 1 | 0 | +0.000 | 1 |
| Full policy (Definition 3.4), 20 dB gate | +0.00 | +0.0000 | 1 | 0 | +0.0000 | 1 | 0 | +0.000 | 1 |
| Full policy, gate from the Objective 1 rule | -1.47 | -0.0006 | 1 | 7e-29 | -0.0003 | 1 | 3.6e-63 | +0.003 | 1 |
| SNR + TAS (candidate) | -2.93 | -0.0012 | 1 | 2.6e-11 | -0.0000 | 1 | 8.4e-40 | +0.019 | 0.03 |
| Oracle: cheapest safe candidate in hindsight (diagnostic) | -6.24 | -0.0140 | 0.0012 | 0.82 | +0.0069 | 0.00012 | 5.6e-13 | +0.019 | 0.2 |

TOST p < 0.05 means equivalence within the margins is shown; a non-significant Wilcoxon p alone is not evidence of equal quality.

### Robustness: calibrate on clean prompts, test on noisy prompts (speaker-disjoint folds)

| Method | Failures | Rate [95% interval] | Mean passes | Δ |
|---|---:|---|---:|---:|
| SNR-only rule, gate calibrated (λ = ∞) | 0/300 | 0.0% [0.0%, 1.2%] | 21.00 | 0.0% |
| Stabilization only (θ = −∞) | 126/300 | 42.0% [36.4%, 47.8%] | 13.26 | 36.9% |
| Full policy (Definition 3.4), 20 dB gate | 84/300 | 28.0% [23.0%, 33.4%] | 14.94 | 28.9% |

## Objective 3 rehearsal on the 16-interval grid (released sampler) (100 random 80/20 speaker re-splits; calibration sizes in cases)

| Method | Calibration set | Certified | Mean test risk | 95th percentile | Share of splits with risk > 5% | Mean saving |
|---|---|---:|---:|---:|---:|---:|
| Full policy (Definition 3.4), 20 dB gate | 80% of speakers | 0% | 0.0% | 0.0% | 0% | 0.0% |
| Full policy (Definition 3.4), 20 dB gate | 50 cases | 0% | 0.0% | 0.0% | 0% | 0.0% |
| Full policy (Definition 3.4), 20 dB gate | 100 cases | 0% | 0.0% | 0.0% | 0% | 0.0% |
| Full policy (Definition 3.4), 20 dB gate | 200 cases | 0% | 0.0% | 0.0% | 0% | 0.0% |
| SNR-only rule, gate calibrated (λ = ∞) | 80% of speakers | 100% | 2.1% | 6.9% | 13% | 10.9% |
| SNR-only rule, gate calibrated (λ = ∞) | 50 cases | 90% | 1.2% | 5.6% | 6% | 7.4% |
| SNR-only rule, gate calibrated (λ = ∞) | 100 cases | 100% | 1.3% | 5.6% | 7% | 7.9% |
| SNR-only rule, gate calibrated (λ = ∞) | 200 cases | 100% | 2.1% | 5.6% | 12% | 10.6% |

## Objective 1 on the 32-interval grid (paper's stopping rule)

### Fixed switching times (Definition 3.1 against s025; exact 95% intervals)

| Output | Passes | Clean failures (of 60) | Noisy failures (of 300) | All | 95% interval |
|---|---:|---:|---:|---:|---|
| 32-interval s = 0.07 (7 teacher steps) | 19 | 2 | 129 | 131 (36.4%) | [31.4%, 41.6%] |
| 32-interval s = 0.10 (9 teacher steps) | 23 | 0 | 107 | 107 (29.7%) | [25.0%, 34.7%] |
| 32-interval s = 0.15 (11 teacher steps) | 27 | 0 | 84 | 84 (23.3%) | [19.1%, 28.1%] |
| 32-interval s = 0.20 (13 teacher steps) | 31 | 1 | 42 | 43 (11.9%) | [8.8%, 15.8%] |
| 32-interval s = 0.25 (14 teacher steps), reference on this grid | 33 | 0 | 0 | 0 (0.0%) | [0.0%, 1.0%] |

Per condition: s007: clean 2, noisy15 41, gauss18 30, gauss20 31, gauss22 20, gauss25 7; s010: clean 0, noisy15 32, gauss18 27, gauss20 22, gauss22 21, gauss25 5; s015: clean 0, noisy15 32, gauss18 21, gauss20 13, gauss22 13, gauss25 5; s020: clean 1, noisy15 20, gauss18 8, gauss20 8, gauss22 4, gauss25 2; s025: clean 0, noisy15 0, gauss18 0, gauss20 0, gauss22 0, gauss25 0

### Cheapest safe candidate per input, in hindsight (the oracle)

| Condition | s007 | s010 | s015 | s020 | s025 | Earlier than the reference |
|---|---:|---:|---:|---:|---:|---:|
| clean | 58 | 2 | 0 | 0 | 0 | 100% |
| noisy15 | 19 | 14 | 8 | 11 | 8 | 87% |
| gauss18 | 30 | 10 | 8 | 7 | 5 | 92% |
| gauss20 | 29 | 14 | 9 | 6 | 2 | 97% |
| gauss22 | 40 | 9 | 6 | 4 | 1 | 98% |
| gauss25 | 53 | 5 | 2 | 0 | 0 | 100% |

### How strongly the signals relate to the oracle choice (Spearman ρ with the oracle's candidate index)

- snr: ρ = -0.480 (p 4.1e-22)
- drift at s007: ρ = -0.464 (p 1.4e-20)
- drift at s010: ρ = -0.447 (p 4.4e-19)
- drift at s015: ρ = -0.325 (p 2.7e-10)
- drift at s020: ρ = -0.173 (p 0.00099)

SNR gate by the Objective 1 rule (lowest SNR above which the earliest candidate fails at most 5%): 28 dB. Plot: `objective1_signals_g32.png`.

## Objective 4 on the 32-interval grid (paper's stopping rule) (out of fold; Learn-Then-Test α = 0.05, δ = 0.1; 5-fold speaker cross-validation)

Saving Δ against s025 (33 passes) and share of the oracle saving η, with bootstrap 95% intervals over speakers. Within limit = failure rate ≤ 5%.

| Method | Failures | Rate [95% interval] | Within limit | Mean passes | Δ [95%] | η [95%] | Runtime (s) | RTF | Worst condition |
|---|---:|---|---|---:|---|---|---:|---:|---:|
| fixed s025 | 0/360 | 0.0% [0.0%, 1.0%] | yes | 33.00 | 0.0% [0.0%, 0.0%] | 0% [0%, 0%] | 5.03 | 1.076 | 0.0% |
| fixed s007 | 131/360 | 36.4% [31.4%, 41.6%] | no | 19.00 | 42.4% [42.4%, 42.4%] | 126% [120%, 132%] | 2.94 | 0.629 | 68.3% |
| fixed s010 | 107/360 | 29.7% [25.0%, 34.7%] | no | 23.00 | 30.3% [30.3%, 30.3%] | 90% [86%, 94%] | 3.53 | 0.757 | 53.3% |
| fixed s015 | 84/360 | 23.3% [19.1%, 28.1%] | no | 27.00 | 18.2% [18.2%, 18.2%] | 54% [51%, 57%] | 4.13 | 0.884 | 53.3% |
| fixed s020 | 43/360 | 11.9% [8.8%, 15.8%] | no | 31.00 | 6.1% [6.1%, 6.1%] | 18% [17%, 19%] | 4.73 | 1.012 | 33.3% |
| SNR-only, 20 dB gate | 89/360 | 24.7% [20.4%, 29.5%] | no | 21.45 | 35.0% [34.5%, 35.4%] | 104% [99%, 109%] | 3.30 | 0.707 | 51.7% |
| SNR-only rule, gate calibrated (λ = ∞) | 11/360 | 3.1% [1.5%, 5.4%] | yes | 28.57 | 13.4% [12.4%, 14.4%] | 40% [36%, 43%] | 4.37 | 0.935 | 11.7% |
| SNR gate (20 dB), fixed later step 5 | 73/360 | 20.3% [16.2%, 24.8%] | no | 24.75 | 25.0% [24.7%, 25.3%] | 74% [71%, 78%] | 3.79 | 0.813 | 41.7% |
| SNR gate (20 dB), fixed later step 6 | 52/360 | 14.4% [11.0%, 18.5%] | no | 28.05 | 15.0% [14.8%, 15.2%] | 45% [43%, 47%] | 4.29 | 0.918 | 35.0% |
| Stabilization only (θ = −∞) | 6/360 | 1.7% [0.6%, 3.6%] | yes | 32.47 | 1.6% [0.8%, 2.5%] | 5% [2%, 7%] | 4.95 | 1.059 | 5.0% |
| Full policy (Definition 3.4), 20 dB gate | 8/360 | 2.2% [1.0%, 4.3%] | yes | 32.42 | 1.8% [1.0%, 2.6%] | 5% [3%, 8%] | 4.94 | 1.056 | 5.0% |
| Full policy, gate from the Objective 1 rule | 6/360 | 1.7% [0.6%, 3.6%] | yes | 30.01 | 9.1% [8.0%, 10.0%] | 27% [24%, 30%] | 4.58 | 0.980 | 6.7% |
| SNR + TAS (candidate) | 13/360 | 3.6% [1.9%, 6.1%] | yes | 28.37 | 14.0% [12.4%, 15.7%] | 42% [37%, 47%] | 4.33 | 0.929 | 8.3% |
| Cost-matched random rule (p_ref 0.93; mean failures over draws 6.5) | 6/360 | 1.7% [0.6%, 3.6%] | yes | 32.41 | 1.8% n/a | 5% n/a | 4.95 | 1.057 | 8.3% |
| Oracle: cheapest safe candidate in hindsight (diagnostic) | 0/360 | 0.0% [0.0%, 1.0%] | yes | 21.89 | 33.7% [32.1%, 35.3%] | 100% [100%, 100%] | 3.37 | 0.722 | 0.0% |

Choices per input: fixed s025: s025 360; fixed s007: s007 360; fixed s010: s010 360; fixed s015: s015 360; fixed s020: s020 360; snr20: s007 297, s025 63; snr_cal: s007 114, s025 246; later5: s010 297, s025 63; later6: s015 297, s025 63; stab: s007 4, s010 2, s015 7, s020 36, s025 311; snr_drift: s007 3, s010 3, s015 8, s020 44, s025 302; snr_drift_dev: s007 77, s025 283; snr_tas: s007 119, s025 241; random: s007 7, s010 7, s015 6, s020 5, s025 335; oracle: s007 229, s010 54, s015 33, s020 28, s025 16

Thresholds per fold (None = nothing certified; every input to the reference): snr20: [20, 20, 20, 20, 20]; snr_cal: [25, 25, 25, 26, 26]; later5: [20, 20, 20, 20, 20]; later6: [20, 20, 20, 20, 20]; stab: [0.06787, 0.06787, 0.06787, 0.06787, 0.06787]; snr_drift: [0.06787, 0.06787, 0.06787, 0.06787, 0.0765]; snr_drift_dev: [(27, 1.2), (28, 1.2), (26, 1.2), (28, 1.2), (28, 1.2)]; snr_tas: [12.01, 10.79, 12.9, 17.24, 11.9]

η above 100% means a method is cheaper than the oracle, which is possible only by adding failures; η is meaningful for methods within the limit.

### Testing the stabilization signal (exact McNemar on paired failures; fixed = failures of the second method avoided by the first)

| Comparison | Fixed | New | p |
|---|---:|---:|---:|
| snr_drift vs snr20 | 85 | 4 | 8.3e-21 |
| snr_drift vs snr_cal | 11 | 8 | 0.65 |
| snr_drift vs later5 | 66 | 1 | 9.2e-19 |
| snr_drift vs later6 | 48 | 4 | 1.3e-10 |
| snr_drift vs random | 5 | 7 | 0.77 |
| snr_drift vs stab | 2 | 4 | 0.69 |
| snr_drift_dev vs snr_drift | 8 | 6 | 0.79 |
| snr_tas vs snr_drift | 8 | 13 | 0.38 |
| snr_tas vs snr20 | 76 | 0 | 2.6e-23 |

### Paired tests against s025 (speaker-level means; Wilcoxon with Holm correction; TOST margins WER ±0.01, SIM ±0.02)

| Method | Δ passes | Δ WER | p (Holm) | TOST WER p | Δ SIM | p (Holm) | TOST SIM p | Δ UTMOS | p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed s007 | -14.00 | +0.2354 | 0.0095 | 0.91 | -0.0068 | 1 | 0.00043 | +0.006 | 1 |
| fixed s010 | -10.00 | +0.0843 | 1 | 0.81 | -0.0022 | 1 | 1.6e-09 | +0.008 | 1 |
| fixed s015 | -6.00 | +0.0375 | 1 | 0.74 | -0.0002 | 1 | 9.3e-18 | +0.011 | 0.23 |
| fixed s020 | -2.00 | -0.0069 | 1 | 0.28 | +0.0010 | 1 | 4.5e-28 | +0.003 | 1 |
| SNR-only, 20 dB gate | -11.55 | +0.0461 | 1 | 0.8 | -0.0054 | 0.35 | 4.7e-08 | +0.005 | 1 |
| SNR-only rule, gate calibrated (λ = ∞) | -4.43 | -0.0012 | 1 | 1.1e-09 | -0.0001 | 1 | 2.5e-39 | +0.008 | 1 |
| SNR gate (20 dB), fixed later step 5 | -8.25 | +0.0404 | 1 | 0.76 | -0.0018 | 1 | 3.9e-14 | +0.008 | 1 |
| SNR gate (20 dB), fixed later step 6 | -4.95 | -0.0012 | 1 | 0.0023 | -0.0002 | 1 | 6.3e-24 | +0.011 | 0.26 |
| Stabilization only (θ = −∞) | -0.53 | +0.0037 | 1 | 0.069 | +0.0001 | 1 | 4.4e-46 | +0.000 | 1 |
| Full policy (Definition 3.4), 20 dB gate | -0.58 | +0.0031 | 1 | 0.00038 | +0.0009 | 1 | 7.9e-36 | +0.001 | 1 |
| Full policy, gate from the Objective 1 rule | -2.99 | -0.0012 | 1 | 1.7e-16 | -0.0005 | 1 | 3.1e-53 | +0.003 | 1 |
| SNR + TAS (candidate) | -4.63 | -0.0015 | 1 | 7.5e-08 | -0.0001 | 1 | 3.7e-40 | +0.010 | 1 |
| Oracle: cheapest safe candidate in hindsight (diagnostic) | -11.11 | -0.0203 | 0.0004 | 0.97 | +0.0088 | 6.7e-06 | 2.7e-10 | +0.007 | 1 |

TOST p < 0.05 means equivalence within the margins is shown; a non-significant Wilcoxon p alone is not evidence of equal quality.

### Robustness: calibrate on clean prompts, test on noisy prompts (speaker-disjoint folds)

| Method | Failures | Rate [95% interval] | Mean passes | Δ |
|---|---:|---|---:|---:|
| SNR-only rule, gate calibrated (λ = ∞) | 0/300 | 0.0% [0.0%, 1.2%] | 33.00 | 0.0% |
| Stabilization only (θ = −∞) | 22/300 | 7.3% [4.7%, 10.9%] | 30.23 | 8.4% |
| Full policy (Definition 3.4), 20 dB gate | 15/300 | 5.0% [2.8%, 8.1%] | 30.84 | 6.5% |

## Objective 3 rehearsal on the 32-interval grid (paper's stopping rule) (100 random 80/20 speaker re-splits; calibration sizes in cases)

| Method | Calibration set | Certified | Mean test risk | 95th percentile | Share of splits with risk > 5% | Mean saving |
|---|---|---:|---:|---:|---:|---:|
| Full policy (Definition 3.4), 20 dB gate | 80% of speakers | 100% | 1.6% | 5.6% | 6% | 1.3% |
| Full policy (Definition 3.4), 20 dB gate | 50 cases | 100% | 1.0% | 4.2% | 3% | 0.7% |
| Full policy (Definition 3.4), 20 dB gate | 100 cases | 100% | 1.2% | 4.2% | 3% | 0.8% |
| Full policy (Definition 3.4), 20 dB gate | 200 cases | 100% | 1.6% | 5.6% | 6% | 1.3% |
| SNR-only rule, gate calibrated (λ = ∞) | 80% of speakers | 100% | 2.7% | 6.9% | 18% | 13.3% |
| SNR-only rule, gate calibrated (λ = ∞) | 50 cases | 90% | 1.6% | 5.6% | 8% | 8.9% |
| SNR-only rule, gate calibrated (λ = ∞) | 100 cases | 100% | 1.6% | 6.9% | 8% | 9.5% |
| SNR-only rule, gate calibrated (λ = ∞) | 200 cases | 100% | 2.4% | 6.9% | 18% | 12.4% |

## The paper's settings, teacher-only and student-only against the thesis reference (paired; a − b)

| Comparison | Prompts | n | SIM mean diff | SIM higher | SIM p | WER mean diff | WER p |
|---|---|---:|---:|---:|---:|---:|---:|
| alg2 vs g16_8 | clean | 60 | -0.0093 | 38% | 0.017 | -0.0009 | 1 |
| alg2 vs g16_8 | noisy | 300 | -0.0106 | 42% | 0.0037 | +0.3340 | 0.0024 |
| alg2 vs g16_8 | all | 360 | -0.0104 | 41% | 0.00043 | +0.2782 | 0.003 |
| paper vs g16_8 | clean | 60 | -0.0026 | 50% | 0.82 | +0.0000 | 1 |
| paper vs g16_8 | noisy | 300 | +0.0032 | 52% | 0.36 | +0.2656 | 0.053 |
| paper vs g16_8 | all | 360 | +0.0022 | 52% | 0.5 | +0.2213 | 0.066 |
| alg2 vs paper | clean | 60 | -0.0067 | 37% | 0.00095 | -0.0009 | 0.32 |
| alg2 vs paper | noisy | 300 | -0.0138 | 36% | 1.3e-06 | +0.0684 | 0.28 |
| alg2 vs paper | all | 360 | -0.0126 | 36% | 3.5e-08 | +0.0569 | 0.28 |
| s025 vs g16_8 | clean | 60 | +0.0005 | 55% | 0.71 | -0.0003 | 0.65 |
| s025 vs g16_8 | noisy | 300 | +0.0000 | 53% | 0.97 | +0.0070 | 0.83 |
| s025 vs g16_8 | all | 360 | +0.0001 | 53% | 0.89 | +0.0058 | 0.86 |
| teacher32 vs g16_8 | clean | 60 | -0.0078 | 42% | 0.06 | -0.0026 | 0.29 |
| teacher32 vs g16_8 | noisy | 300 | -0.0093 | 48% | 0.12 | +0.8975 | 7.7e-16 |
| teacher32 vs g16_8 | all | 360 | -0.0091 | 47% | 0.053 | +0.7475 | 1.3e-15 |
| student4 vs g16_8 | clean | 60 | +0.0010 | 60% | 0.52 | -0.0057 | 0.14 |
| student4 vs g16_8 | noisy | 300 | +0.0770 | 83% | 5.7e-35 | -0.0270 | 5e-05 |
| student4 vs g16_8 | all | 360 | +0.0643 | 79% | 4.6e-35 | -0.0235 | 2.2e-05 |
| g16_4 vs g16_8 | clean | 60 | -0.0003 | 52% | 0.73 | -0.0038 | 0.18 |
| g16_4 vs g16_8 | noisy | 300 | -0.0085 | 42% | 0.00067 | +0.1709 | 0.00062 |
| g16_4 vs g16_8 | all | 360 | -0.0071 | 43% | 0.0009 | +0.1418 | 0.00097 |

## Deviations from the Chapter 3 protocol

- Development speakers only, six conditions per speaker; calibration by cross-validation and re-splits on the same 60 speakers, so the cases are not independent and the Learn-Then-Test guarantee is not certified here.
- The 32-interval grid is inferred from the paper; its λ grid (0.01 → 1.20, geometric) replaces Chapter 3's 0.20 → 1.20, because the drift is smaller on a finer grid. Calibration sizes count cases, not speakers.
- `alg2` assumes the student's second time (0.6173); runtime is composed from measured unit times, not timed end to end per method.
- Teacher-only and student-only follow the released code's step counts with the thesis's deterministic teacher (eta = 0).
