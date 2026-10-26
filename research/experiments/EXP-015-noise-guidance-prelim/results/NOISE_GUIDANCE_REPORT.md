# Preliminary test: noise-aware handover methods (PIEC, prompt enhancement, SWC, SCG, SNNG, antithetic pair)

Development speakers only (60 speakers x 6 conditions: clean, noisy15, gauss18, gauss20, gauss22, gauss25); 360 of 360 cases; preregistration `58caa9515cb43384…`. Every number is computed from the CSV files by `run_noise_guidance_prelim.py analyze`.

Drift reproduction: max |Δd₄| = 0.000. Antithetic pair chose −ε in 189 cases.

## Preregistered criteria (C1 noisy failures −25% with p < 0.05; C2 clean +1 at most; C3 the calibrated route beats the calibrated SNR gate)

| Method | C1 | C2 | C3 |
|---|---|---|---|
| SNNG η = 0.5 (18 NFE) | FAIL | FAIL | FAIL |
| SNNG η = 1.0 (18 NFE) | FAIL | FAIL | FAIL |
| Antithetic pair (16 NFE) | FAIL | FAIL | FAIL |
| SNR-Compensated Guidance, r_max 2.5 (13 NFE) | FAIL | FAIL | FAIL |
| SNR-Compensated Guidance, r_max 1.5 (13 NFE) | FAIL | FAIL | FAIL |
| SNR-Wiener Conditioning, β_max 1.0 (13 NFE) | PASS | FAIL | FAIL |
| SNR-Wiener Conditioning, β_max 0.5 (13 NFE) | PASS | FAIL | FAIL |
| Enhanced prompt, Step 4 (13 NFE) | PASS | FAIL | FAIL |
| Enhanced prompt, Step 6 (17 NFE) | PASS | FAIL | FAIL |
| SNR-scheduled enhanced prompt, Step 4 (13 NFE) | PASS | FAIL | FAIL |
| Enhanced prompt, Step 8 (control, 21 NFE) | PASS | FAIL | FAIL |
| PIEC on Step 4 (13 NFE) | FAIL | PASS | FAIL |
| SWC + PIEC, Step 4 (13 NFE) | PASS | FAIL | FAIL |
| PIEC on Step 8 (control, 21 NFE) | FAIL | PASS | FAIL |

## Failures when every input uses the method at Step 4

| Condition | n | plain Step 4 | snng05 | snng10 | anti | scg | scg15 | swc | swc05 | enh4 | enh6 | enhs4 | enh8 | piec4 | swcpiec4 | piec8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 60 | 2 | 10 | 6 | 7 | 7 | 7 | 5 | 6 | 15 | 13 | 4 | 14 | 3 | 5 | 1 |
| noisy15 | 60 | 40 | 46 | 43 | 39 | 36 | 40 | 4 | 4 | 4 | 3 | 4 | 4 | 45 | 4 | 37 |
| gauss18 | 60 | 33 | 35 | 43 | 31 | 32 | 35 | 2 | 6 | 4 | 3 | 3 | 4 | 37 | 3 | 31 |
| gauss20 | 60 | 26 | 34 | 31 | 24 | 28 | 32 | 2 | 5 | 3 | 3 | 3 | 5 | 33 | 4 | 27 |
| gauss22 | 60 | 19 | 24 | 26 | 16 | 28 | 28 | 5 | 7 | 8 | 8 | 6 | 6 | 24 | 6 | 24 |
| gauss25 | 60 | 8 | 21 | 21 | 14 | 18 | 18 | 8 | 11 | 12 | 12 | 11 | 13 | 17 | 9 | 22 |

Exact McNemar (two-sided), unadjusted:

- snng05 vs Step 4, noisy (primary): 126 → 160 of 300 (p 0.0020)
- snng05 vs Step 4, clean: 2 → 10 of 60 (p 0.0215)
- snng10 vs Step 4, noisy (primary): 126 → 164 of 300 (p 0.0003)
- snng10 vs Step 4, clean: 2 → 6 of 60 (p 0.2188)
- anti vs Step 4, noisy (primary): 126 → 124 of 300 (p 0.8919)
- anti vs Step 4, clean: 2 → 7 of 60 (p 0.1250)
- scg vs Step 4, noisy (primary): 126 → 142 of 300 (p 0.1634)
- scg vs Step 4, clean: 2 → 7 of 60 (p 0.1250)
- scg15 vs Step 4, noisy (primary): 126 → 153 of 300 (p 0.0168)
- scg15 vs Step 4, clean: 2 → 7 of 60 (p 0.1250)
- swc vs Step 4, noisy (primary): 126 → 21 of 300 (p 0.0000)
- swc vs Step 4, clean: 2 → 5 of 60 (p 0.3750)
- swc05 vs Step 4, noisy (primary): 126 → 33 of 300 (p 0.0000)
- swc05 vs Step 4, clean: 2 → 6 of 60 (p 0.2188)
- enh4 vs Step 4, noisy (primary): 126 → 31 of 300 (p 0.0000)
- enh4 vs Step 4, clean: 2 → 15 of 60 (p 0.0010)
- enh6 vs Step 4, noisy (primary): 126 → 29 of 300 (p 0.0000)
- enh6 vs Step 4, clean: 2 → 13 of 60 (p 0.0034)
- enhs4 vs Step 4, noisy (primary): 126 → 27 of 300 (p 0.0000)
- enhs4 vs Step 4, clean: 2 → 4 of 60 (p 0.5000)
- enh8 vs Step 4, noisy (primary): 126 → 32 of 300 (p 0.0000)
- enh8 vs Step 4, clean: 2 → 14 of 60 (p 0.0018)
- piec4 vs Step 4, noisy (primary): 126 → 156 of 300 (p 0.0059)
- piec4 vs Step 4, clean: 2 → 3 of 60 (p 1.0000)
- swcpiec4 vs Step 4, noisy (primary): 126 → 26 of 300 (p 0.0000)
- swcpiec4 vs Step 4, clean: 2 → 5 of 60 (p 0.3750)
- piec8 vs Step 4, noisy (primary): 126 → 141 of 300 (p 0.2315)
- piec8 vs Step 4, clean: 2 → 1 of 60 (p 1.0000)

## Against SNR-only (pooled calibration, out of fold; exact McNemar on the same cases)

SNR-only = Step 4 if SNR ≥ 20 dB, else Step 8. Within limit = out-of-fold risk ≤ 5%.

| Policy | Failures | Risk | Within limit | Mean NFE | Fixed vs SNR-only | Broken | p |
|---|---:|---:|---|---:|---:|---:|---:|
| reference | 0/360 | 0.0% | yes | 21.00 | 86 | 0 | 2.6e-26 |
| snr_only | 86/360 | 23.9% | no | 14.40 | 0 | 0 | 1 |
| snr_theta | 7/360 | 1.9% | yes | 18.71 | 79 | 0 | 3.3e-24 |
| d4veto | 13/360 | 3.6% | yes | 18.20 | 73 | 0 | 2.1e-22 |
| tas | 12/360 | 3.3% | yes | 18.20 | 74 | 0 | 1.1e-22 |
| snng05_route | 9/360 | 2.5% | yes | 18.66 | 78 | 1 | 2.6e-22 |
| snng05_heavy | 9/360 | 2.5% | yes | 18.69 | 79 | 2 | 2.7e-21 |
| snng05_tas | 11/360 | 3.1% | yes | 20.53 | 86 | 11 | 1.4e-15 |
| snng05_noisy | 151/360 | 41.9% | no | 16.57 | 34 | 99 | 1.5e-08 |
| snng05_all | 170/360 | 47.2% | no | 18.00 | 36 | 120 | 9.3e-12 |
| snng10_route | 7/360 | 1.9% | yes | 18.71 | 79 | 0 | 3.3e-24 |
| snng10_heavy | 9/360 | 2.5% | yes | 18.69 | 79 | 2 | 2.7e-21 |
| snng10_tas | 12/360 | 3.3% | yes | 20.48 | 84 | 10 | 1e-15 |
| snng10_noisy | 156/360 | 43.3% | no | 16.57 | 25 | 95 | 8.4e-11 |
| snng10_all | 170/360 | 47.2% | no | 18.00 | 27 | 111 | 2.7e-13 |
| anti_route | 11/360 | 3.1% | yes | 18.56 | 76 | 1 | 1e-21 |
| anti_heavy | 9/360 | 2.5% | yes | 18.68 | 79 | 2 | 2.7e-21 |
| anti_tas | 13/360 | 3.6% | yes | 19.76 | 81 | 8 | 2.5e-16 |
| anti_noisy | 121/360 | 33.6% | no | 15.14 | 24 | 59 | 0.00015 |
| anti_all | 131/360 | 36.4% | no | 16.00 | 25 | 70 | 4.3e-06 |
| scg_route | 9/360 | 2.5% | yes | 18.58 | 79 | 2 | 2.7e-21 |
| scg_heavy | 9/360 | 2.5% | yes | 18.67 | 79 | 2 | 2.7e-21 |
| scg_tas | 12/360 | 3.3% | yes | 19.38 | 84 | 10 | 1e-15 |
| scg_noisy | 134/360 | 37.2% | no | 13.00 | 36 | 84 | 1.4e-05 |
| scg_all | 149/360 | 41.4% | no | 13.00 | 38 | 101 | 8.8e-08 |
| scg15_route | 9/360 | 2.5% | yes | 18.58 | 79 | 2 | 2.7e-21 |
| scg15_heavy | 8/360 | 2.2% | yes | 18.67 | 79 | 1 | 1.3e-22 |
| scg15_tas | 13/360 | 3.6% | yes | 19.38 | 84 | 11 | 4.5e-15 |
| scg15_noisy | 145/360 | 40.3% | no | 13.00 | 35 | 94 | 2e-07 |
| scg15_all | 160/360 | 44.4% | no | 13.00 | 37 | 111 | 8.7e-10 |
| swc_route | 14/360 | 3.9% | yes | 17.24 | 76 | 4 | 2.8e-18 |
| swc_heavy | 11/360 | 3.1% | yes | 16.98 | 79 | 4 | 4e-19 |
| swc_tas | 12/360 | 3.3% | yes | 18.13 | 83 | 9 | 3.9e-16 |
| swc_noisy | 20/360 | 5.6% | no | 13.00 | 75 | 9 | 4.3e-14 |
| swc_all | 26/360 | 7.2% | no | 13.00 | 78 | 18 | 4.4e-10 |
| swc05_route | 10/360 | 2.8% | yes | 18.16 | 78 | 2 | 5.4e-21 |
| swc05_heavy | 12/360 | 3.3% | yes | 17.42 | 79 | 5 | 3.4e-18 |
| swc05_tas | 11/360 | 3.1% | yes | 19.02 | 83 | 8 | 7.6e-17 |
| swc05_noisy | 31/360 | 8.6% | no | 13.00 | 71 | 16 | 2e-09 |
| swc05_all | 39/360 | 10.8% | no | 13.00 | 74 | 27 | 3.2e-06 |
| enh4_route | 8/360 | 2.2% | yes | 18.27 | 79 | 1 | 1.3e-22 |
| enh4_heavy | 12/360 | 3.3% | yes | 17.38 | 79 | 5 | 3.4e-18 |
| enh4_tas | 14/360 | 3.9% | yes | 20.00 | 86 | 14 | 8.3e-14 |
| enh4_noisy | 27/360 | 7.5% | no | 13.00 | 73 | 14 | 8.6e-11 |
| enh4_all | 46/360 | 12.8% | no | 13.00 | 78 | 38 | 0.00026 |
| enh6_route | 10/360 | 2.8% | yes | 18.33 | 77 | 1 | 5.2e-22 |
| enh6_heavy | 12/360 | 3.3% | yes | 17.77 | 79 | 5 | 3.4e-18 |
| enh6_tas | 12/360 | 3.3% | yes | 20.42 | 86 | 12 | 5.9e-15 |
| enh6_noisy | 24/360 | 6.7% | no | 15.86 | 74 | 12 | 4.7e-12 |
| enh6_all | 42/360 | 11.7% | no | 17.00 | 79 | 35 | 4.6e-05 |
| enhs4_route | 10/360 | 2.8% | yes | 18.00 | 78 | 2 | 5.4e-21 |
| enhs4_heavy | 12/360 | 3.3% | yes | 17.18 | 79 | 5 | 3.4e-18 |
| enhs4_tas | 10/360 | 2.8% | yes | 18.00 | 83 | 7 | 1.3e-17 |
| enhs4_noisy | 25/360 | 6.9% | no | 13.00 | 74 | 13 | 1.6e-11 |
| enhs4_all | 31/360 | 8.6% | no | 13.00 | 77 | 22 | 2.5e-08 |
| enh8_route | 11/360 | 3.1% | yes | 18.71 | 78 | 3 | 7.3e-20 |
| enh8_heavy | 12/360 | 3.3% | yes | 18.71 | 79 | 5 | 3.4e-18 |
| enh8_tas | 13/360 | 3.6% | yes | 21.00 | 86 | 13 | 2.3e-14 |
| enh8_noisy | 27/360 | 7.5% | no | 18.71 | 73 | 14 | 8.6e-11 |
| enh8_all | 46/360 | 12.8% | no | 21.00 | 78 | 38 | 0.00026 |
| piec4_route | 9/360 | 2.5% | yes | 18.58 | 79 | 2 | 2.7e-21 |
| piec4_heavy | 9/360 | 2.5% | yes | 18.67 | 79 | 2 | 2.7e-21 |
| piec4_tas | 14/360 | 3.9% | yes | 19.24 | 81 | 9 | 1.3e-15 |
| piec4_noisy | 150/360 | 41.7% | no | 13.00 | 31 | 95 | 9.6e-09 |
| piec4_all | 159/360 | 44.2% | no | 13.00 | 32 | 105 | 2.9e-10 |
| swcpiec4_route | 10/360 | 2.8% | yes | 17.84 | 78 | 2 | 5.4e-21 |
| swcpiec4_heavy | 12/360 | 3.3% | yes | 17.22 | 79 | 5 | 3.4e-18 |
| swcpiec4_tas | 14/360 | 3.9% | yes | 18.44 | 83 | 11 | 8e-15 |
| swcpiec4_noisy | 24/360 | 6.7% | no | 13.00 | 74 | 12 | 4.7e-12 |
| swcpiec4_all | 31/360 | 8.6% | no | 13.00 | 78 | 23 | 3.6e-08 |
| piec8_route | 8/360 | 2.2% | yes | 18.71 | 79 | 1 | 1.3e-22 |
| piec8_heavy | 8/360 | 2.2% | yes | 18.71 | 79 | 1 | 1.3e-22 |
| piec8_tas | 13/360 | 3.6% | yes | 21.00 | 84 | 11 | 4.5e-15 |
| piec8_noisy | 130/360 | 36.1% | no | 18.71 | 43 | 87 | 0.00014 |
| piec8_all | 142/360 | 39.4% | no | 21.00 | 47 | 103 | 5.6e-06 |

## Early handover within the cleaned system

The early cleaned output judged against the cleaned Step-8 output (Definition 3.1 thresholds). Low counts mean the early handover is safe within the cleaned system, not only better than the uncleaned reference.

- enh4 judged against enh8: 18 failures (clean 2, noisy15 4, gauss18 3, gauss20 1, gauss22 4, gauss25 4)
- piec4 judged against piec8: 123 failures (clean 2, noisy15 39, gauss18 33, gauss20 28, gauss22 15, gauss25 6)

## Average quality vs the Step-8 reference (paired; SIM higher is better, WER lower is better)

| Method | Prompts | n | SIM mean diff | SIM better | SIM p | WER mean diff | WER p |
|---|---|---:|---:|---:|---:|---:|---:|
| snng05 | noisy | 300 | -0.0209 | 41% | 1.2e-06 | +0.2866 | 0.001 |
| snng05 | clean | 60 | -0.0016 | 52% | 0.71 | +0.0011 | 1 |
| snng10 | noisy | 300 | -0.0243 | 36% | 1.2e-09 | +0.0468 | 9.2e-05 |
| snng10 | clean | 60 | -0.0005 | 52% | 0.79 | -0.0008 | 1 |
| anti | noisy | 300 | -0.0072 | 43% | 0.013 | +0.1870 | 6.4e-05 |
| anti | clean | 60 | -0.0021 | 53% | 0.98 | -0.0028 | 0.35 |
| scg | noisy | 300 | -0.0160 | 41% | 9.1e-05 | +0.0703 | 0.22 |
| scg | clean | 60 | -0.0010 | 42% | 0.51 | +0.0000 | 0.69 |
| scg15 | noisy | 300 | -0.0202 | 38% | 2.4e-06 | +0.0220 | 0.033 |
| scg15 | clean | 60 | -0.0010 | 42% | 0.51 | +0.0000 | 0.69 |
| swc | noisy | 300 | +0.1260 | 90% | 3.6e-44 | -0.0480 | 9.3e-12 |
| swc | clean | 60 | -0.0008 | 43% | 0.64 | -0.0023 | 0.72 |
| swc05 | noisy | 300 | +0.0905 | 84% | 3.4e-36 | -0.0434 | 1.2e-09 |
| swc05 | clean | 60 | -0.0009 | 42% | 0.67 | -0.0023 | 0.72 |
| enh4 | noisy | 300 | +0.1167 | 86% | 6.2e-41 | -0.0515 | 1e-13 |
| enh4 | clean | 60 | -0.0056 | 42% | 0.037 | -0.0024 | 0.5 |
| enh6 | noisy | 300 | +0.1182 | 87% | 1.1e-41 | -0.0519 | 5.9e-14 |
| enh6 | clean | 60 | -0.0042 | 45% | 0.21 | -0.0044 | 0.35 |
| enhs4 | noisy | 300 | +0.1194 | 89% | 1.3e-42 | -0.0484 | 9.4e-11 |
| enhs4 | clean | 60 | -0.0018 | 45% | 0.39 | -0.0053 | 0.11 |
| enh8 | noisy | 300 | +0.1161 | 85% | 3.3e-40 | -0.0516 | 5.9e-14 |
| enh8 | clean | 60 | -0.0047 | 47% | 0.11 | -0.0084 | 0.043 |
| piec4 | noisy | 300 | -0.0178 | 42% | 4.4e-05 | +0.3167 | 3.4e-07 |
| piec4 | clean | 60 | -0.0018 | 45% | 0.45 | -0.0053 | 0.11 |
| swcpiec4 | noisy | 300 | +0.1229 | 91% | 2.7e-43 | -0.0473 | 3.1e-11 |
| swcpiec4 | clean | 60 | -0.0005 | 43% | 0.81 | -0.0023 | 0.72 |
| piec8 | noisy | 300 | -0.0066 | 41% | 0.014 | +0.1131 | 0.23 |
| piec8 | clean | 60 | -0.0010 | 5% | 0.18 | -0.0015 | 0.32 |

## Out-of-fold policies, pooled calibration (5-fold speaker CV, α = 0.05, δ = 0.1)

| Policy | Failures | Mean NFE | Worst-condition risk | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| snr_only | 86/360 | 14.40 | 51.7% | 2 | 0 | 31 | 26 | 19 | 8 |
| snr_theta | 7/360 | 18.71 | 8.3% | 2 | 0 | 0 | 0 | 0 | 5 |
| d4veto | 13/360 | 18.20 | 6.7% | 2 | 0 | 2 | 2 | 4 | 3 |
| tas | 12/360 | 18.20 | 10.0% | 2 | 0 | 1 | 1 | 6 | 2 |
| snng05_route | 9/360 | 18.66 | 8.3% | 2 | 0 | 0 | 0 | 2 | 5 |
| snng05_heavy | 9/360 | 18.69 | 8.3% | 2 | 2 | 0 | 0 | 0 | 5 |
| snng05_tas | 11/360 | 20.53 | 11.7% | 7 | 0 | 0 | 0 | 1 | 3 |
| snng05_noisy | 151/360 | 16.57 | 76.7% | 2 | 46 | 35 | 34 | 24 | 10 |
| snng05_all | 170/360 | 18.00 | 76.7% | 10 | 46 | 35 | 34 | 24 | 21 |
| snng10_route | 7/360 | 18.71 | 8.3% | 2 | 0 | 0 | 0 | 0 | 5 |
| snng10_heavy | 9/360 | 18.69 | 8.3% | 2 | 2 | 0 | 0 | 0 | 5 |
| snng10_tas | 12/360 | 20.48 | 10.0% | 5 | 0 | 0 | 0 | 1 | 6 |
| snng10_noisy | 156/360 | 16.57 | 71.7% | 2 | 43 | 43 | 31 | 26 | 11 |
| snng10_all | 170/360 | 18.00 | 71.7% | 6 | 43 | 43 | 31 | 26 | 21 |
| anti_route | 11/360 | 18.56 | 13.3% | 2 | 0 | 0 | 0 | 1 | 8 |
| anti_heavy | 9/360 | 18.68 | 8.3% | 2 | 2 | 0 | 0 | 0 | 5 |
| anti_tas | 13/360 | 19.76 | 10.0% | 6 | 0 | 0 | 0 | 2 | 5 |
| anti_noisy | 121/360 | 15.14 | 65.0% | 2 | 39 | 31 | 24 | 16 | 9 |
| anti_all | 131/360 | 16.00 | 65.0% | 7 | 39 | 31 | 24 | 16 | 14 |
| scg_route | 9/360 | 18.58 | 10.0% | 2 | 0 | 0 | 0 | 1 | 6 |
| scg_heavy | 9/360 | 18.67 | 8.3% | 2 | 2 | 0 | 0 | 0 | 5 |
| scg_tas | 12/360 | 19.38 | 10.0% | 6 | 0 | 0 | 0 | 1 | 5 |
| scg_noisy | 134/360 | 13.00 | 60.0% | 2 | 36 | 32 | 28 | 28 | 8 |
| scg_all | 149/360 | 13.00 | 60.0% | 7 | 36 | 32 | 28 | 28 | 18 |
| scg15_route | 9/360 | 18.58 | 10.0% | 2 | 0 | 0 | 0 | 1 | 6 |
| scg15_heavy | 8/360 | 18.67 | 8.3% | 2 | 1 | 0 | 0 | 0 | 5 |
| scg15_tas | 13/360 | 19.38 | 10.0% | 6 | 0 | 0 | 1 | 1 | 5 |
| scg15_noisy | 145/360 | 13.00 | 66.7% | 2 | 40 | 35 | 32 | 28 | 8 |
| scg15_all | 160/360 | 13.00 | 66.7% | 7 | 40 | 35 | 32 | 28 | 18 |
| swc_route | 14/360 | 17.24 | 8.3% | 2 | 0 | 0 | 2 | 5 | 5 |
| swc_heavy | 11/360 | 16.98 | 8.3% | 2 | 4 | 0 | 0 | 0 | 5 |
| swc_tas | 12/360 | 18.13 | 8.3% | 5 | 0 | 0 | 1 | 2 | 4 |
| swc_noisy | 20/360 | 13.00 | 8.3% | 2 | 4 | 2 | 2 | 5 | 5 |
| swc_all | 26/360 | 13.00 | 13.3% | 5 | 4 | 2 | 2 | 5 | 8 |
| swc05_route | 10/360 | 18.16 | 10.0% | 2 | 0 | 0 | 0 | 2 | 6 |
| swc05_heavy | 12/360 | 17.42 | 8.3% | 2 | 4 | 1 | 0 | 0 | 5 |
| swc05_tas | 11/360 | 19.02 | 10.0% | 6 | 0 | 0 | 0 | 0 | 5 |
| swc05_noisy | 31/360 | 13.00 | 11.7% | 2 | 4 | 6 | 5 | 7 | 7 |
| swc05_all | 39/360 | 13.00 | 18.3% | 6 | 4 | 6 | 5 | 7 | 11 |
| enh4_route | 8/360 | 18.27 | 8.3% | 2 | 0 | 0 | 0 | 1 | 5 |
| enh4_heavy | 12/360 | 17.38 | 8.3% | 2 | 4 | 1 | 0 | 0 | 5 |
| enh4_tas | 14/360 | 20.00 | 18.3% | 11 | 0 | 0 | 0 | 0 | 3 |
| enh4_noisy | 27/360 | 13.00 | 13.3% | 3 | 4 | 4 | 3 | 8 | 5 |
| enh4_all | 46/360 | 13.00 | 25.0% | 15 | 4 | 4 | 3 | 8 | 12 |
| enh6_route | 10/360 | 18.33 | 8.3% | 2 | 0 | 0 | 0 | 3 | 5 |
| enh6_heavy | 12/360 | 17.77 | 8.3% | 2 | 3 | 1 | 1 | 0 | 5 |
| enh6_tas | 12/360 | 20.42 | 15.0% | 9 | 0 | 0 | 0 | 0 | 3 |
| enh6_noisy | 24/360 | 15.86 | 13.3% | 2 | 3 | 3 | 3 | 8 | 5 |
| enh6_all | 42/360 | 17.00 | 21.7% | 13 | 3 | 3 | 3 | 8 | 12 |
| enhs4_route | 10/360 | 18.00 | 10.0% | 2 | 0 | 0 | 0 | 2 | 6 |
| enhs4_heavy | 12/360 | 17.18 | 8.3% | 2 | 4 | 1 | 0 | 0 | 5 |
| enhs4_tas | 10/360 | 18.00 | 11.7% | 3 | 0 | 0 | 0 | 0 | 7 |
| enhs4_noisy | 25/360 | 13.00 | 11.7% | 2 | 4 | 3 | 3 | 6 | 7 |
| enhs4_all | 31/360 | 13.00 | 18.3% | 4 | 4 | 3 | 3 | 6 | 11 |
| enh8_route | 11/360 | 18.71 | 8.3% | 3 | 0 | 0 | 0 | 3 | 5 |
| enh8_heavy | 12/360 | 18.71 | 8.3% | 2 | 4 | 1 | 0 | 0 | 5 |
| enh8_tas | 13/360 | 21.00 | 16.7% | 10 | 0 | 0 | 0 | 0 | 3 |
| enh8_noisy | 27/360 | 18.71 | 10.0% | 3 | 4 | 4 | 5 | 6 | 5 |
| enh8_all | 46/360 | 21.00 | 23.3% | 14 | 4 | 4 | 5 | 6 | 13 |
| piec4_route | 9/360 | 18.58 | 10.0% | 2 | 0 | 0 | 0 | 1 | 6 |
| piec4_heavy | 9/360 | 18.67 | 8.3% | 2 | 2 | 0 | 0 | 0 | 5 |
| piec4_tas | 14/360 | 19.24 | 11.7% | 3 | 0 | 0 | 2 | 2 | 7 |
| piec4_noisy | 150/360 | 13.00 | 75.0% | 2 | 45 | 37 | 33 | 24 | 9 |
| piec4_all | 159/360 | 13.00 | 75.0% | 3 | 45 | 37 | 33 | 24 | 17 |
| swcpiec4_route | 10/360 | 17.84 | 8.3% | 2 | 0 | 0 | 0 | 3 | 5 |
| swcpiec4_heavy | 12/360 | 17.22 | 8.3% | 2 | 4 | 1 | 0 | 0 | 5 |
| swcpiec4_tas | 14/360 | 18.44 | 10.0% | 5 | 0 | 0 | 1 | 2 | 6 |
| swcpiec4_noisy | 24/360 | 13.00 | 10.0% | 2 | 4 | 3 | 4 | 6 | 5 |
| swcpiec4_all | 31/360 | 13.00 | 15.0% | 5 | 4 | 3 | 4 | 6 | 9 |
| piec8_route | 8/360 | 18.71 | 10.0% | 2 | 0 | 0 | 0 | 0 | 6 |
| piec8_heavy | 8/360 | 18.71 | 8.3% | 2 | 1 | 0 | 0 | 0 | 5 |
| piec8_tas | 13/360 | 21.00 | 15.0% | 1 | 0 | 0 | 1 | 2 | 9 |
| piec8_noisy | 130/360 | 18.71 | 61.7% | 2 | 37 | 31 | 27 | 24 | 9 |
| piec8_all | 142/360 | 21.00 | 61.7% | 1 | 37 | 31 | 27 | 24 | 22 |

Chosen per fold: snr_theta: [26.0, 26.0, 26.0, 26.0, 26.0]; d4veto: [-0.2957540452480316, -0.2880088984966278, -0.2885846495628357, -0.2977767884731293, -0.2880088984966278]; tas: [14.5583314986581, 13.15591819506043, 14.029510932291117, 15.551913438228445, 18.609797680116795]; snng05_route: [(26.0, None), (26.0, None), (26.0, None), (26.0, None), (26.0, 25.0)]; snng05_heavy: [(26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0)]; snng05_tas: [4.527651865156822, 5.678021609814424, 6.036469393072731, 5.254562310840941, 6.865475711895615]; snng05_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; snng10_route: [(26.0, None), (26.0, None), (26.0, None), (26.0, None), (26.0, None)]; snng10_heavy: [(26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0)]; snng10_tas: [7.351493779901263, 6.069456940728465, 6.416715664966307, 6.277395893941147, 9.33915004821024]; snng10_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; anti_route: [(26.0, None), (26.0, 25.0), (26.0, None), (26.0, None), (26.0, 25.0)]; anti_heavy: [(26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0)]; anti_tas: [9.664435617904752, 9.624552082524115, 10.907254971808849, 10.539299808073492, 9.961663799582723]; anti_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; scg_route: [(26.0, None), (26.0, None), (26.0, None), (26.0, None), (26.0, 25.0)]; scg_heavy: [(26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0)]; scg_tas: [7.568974833101611, 8.89452597156824, 6.416715664966307, 8.314390968624023, 9.961663799582723]; scg_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; scg15_route: [(26.0, None), (26.0, None), (26.0, None), (26.0, None), (26.0, 25.0)]; scg15_heavy: [(26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0)]; scg15_tas: [7.568974833101611, 8.667187587959452, 6.416715664966307, 8.314390968624023, 9.961663799582723]; scg15_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; swc_route: [(26.0, 23.0), (26.0, 22.0), (26.0, 25.0), (26.0, 25.0), (26.0, 23.0)]; swc_heavy: [(26.0, 21.0), (26.0, 18.0), (26.0, 20.0), (26.0, 20.0), (26.0, 21.0)]; swc_tas: [15.217274890620496, 15.634176842800747, 14.03843925057886, 15.324188445682024, 16.587726669672197]; swc_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; swc05_route: [(26.0, 25.0), (26.0, 25.0), (26.0, 25.0), (26.0, 25.0), (26.0, 24.0)]; swc05_heavy: [(26.0, 20.0), (26.0, 18.0), (26.0, 19.0), (26.0, 19.0), (26.0, 20.0)]; swc05_tas: [10.068236075234427, 8.89452597156824, 9.614169446022483, 11.783664968784526, 12.775297230877749]; swc05_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; enh4_route: [(26.0, 25.0), (26.0, 25.0), (26.0, 25.0), (26.0, 25.0), (26.0, 25.0)]; enh4_heavy: [(26.0, 20.0), (26.0, 19.0), (26.0, 18.0), (26.0, 19.0), (26.0, 20.0)]; enh4_tas: [3.3535028895183236, 5.006742503663594, 4.831661682921066, 3.4389153993531125, 4.801284802342482]; enh4_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; enh6_route: [(26.0, 23.0), (26.0, 25.0), (26.0, 25.0), (26.0, 25.0), (26.0, 25.0)]; enh6_heavy: [(26.0, 22.0), (26.0, 19.0), (26.0, 19.0), (26.0, 20.0), (26.0, 21.0)]; enh6_tas: [4.369126583468424, 7.13420032929871, 6.318501043778361, 4.301639621044108, 5.147738071594188]; enh6_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; enhs4_route: [(26.0, 25.0), (26.0, 25.0), (26.0, 25.0), (26.0, 25.0), (26.0, 23.0)]; enhs4_heavy: [(26.0, 21.0), (26.0, 19.0), (26.0, 18.0), (26.0, 19.0), (26.0, 20.0)]; enhs4_tas: [18.797365310784897, 14.3523887451412, 14.583876343988113, 16.451044856389018, 18.191983444578142]; enhs4_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; enh8_route: [(26.0, 24.0), (26.0, 25.0), (26.0, 25.0), (26.0, 25.0), (26.0, 23.0)]; enh8_heavy: [(26.0, 20.0), (26.0, 18.0), (26.0, 18.0), (26.0, 19.0), (26.0, 20.0)]; enh8_tas: [4.118892990513526, 5.006742503663594, 5.359713099928667, 4.301639621044108, 4.998896355327897]; enh8_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; piec4_route: [(26.0, None), (26.0, None), (26.0, None), (26.0, None), (26.0, 25.0)]; piec4_heavy: [(26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0)]; piec4_tas: [7.439644779599937, 10.293237946795179, 9.120990460307521, 8.469173664132379, 9.961663799582723]; piec4_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; swcpiec4_route: [(26.0, 23.0), (26.0, 25.0), (26.0, 25.0), (26.0, 25.0), (26.0, 24.0)]; swcpiec4_heavy: [(26.0, 21.0), (26.0, 18.0), (26.0, 19.0), (26.0, 19.0), (26.0, 20.0)]; swcpiec4_tas: [14.741537826264588, 14.805599360561876, 9.812005620202102, 13.93149500099815, 13.36410156490936]; swcpiec4_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]; piec8_route: [(26.0, None), (26.0, None), (26.0, None), (26.0, None), (26.0, 25.0)]; piec8_heavy: [(26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0), (26.0, 17.0)]; piec8_tas: [6.5254850390457015, 8.89452597156824, 9.522512516811611, 8.314390968624023, 9.33915004821024]; piec8_noisy: [(26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below'), (26.0, 'all below')]

## Out-of-fold policies, stratified calibration (5-fold speaker CV, α = 0.05, δ = 0.1)

| Policy | Failures | Mean NFE | Worst-condition risk | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| snr_only | 86/360 | 14.40 | 51.7% | 2 | 0 | 31 | 26 | 19 | 8 |
| snr_theta | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| d4veto | 1/360 | 20.29 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| tas | 1/360 | 20.82 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| snng05_route | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| snng05_heavy | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| snng05_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| snng05_noisy | 170/360 | 17.90 | 76.7% | 10 | 46 | 35 | 34 | 24 | 21 |
| snng05_all | 170/360 | 18.00 | 76.7% | 10 | 46 | 35 | 34 | 24 | 21 |
| snng10_route | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| snng10_heavy | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| snng10_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| snng10_noisy | 170/360 | 17.90 | 71.7% | 6 | 43 | 43 | 31 | 26 | 21 |
| snng10_all | 170/360 | 18.00 | 71.7% | 6 | 43 | 43 | 31 | 26 | 21 |
| anti_route | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| anti_heavy | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| anti_tas | 1/360 | 20.54 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| anti_noisy | 131/360 | 15.94 | 65.0% | 7 | 39 | 31 | 24 | 16 | 14 |
| anti_all | 131/360 | 16.00 | 65.0% | 7 | 39 | 31 | 24 | 16 | 14 |
| scg_route | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| scg_heavy | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| scg_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| scg_noisy | 149/360 | 13.00 | 60.0% | 7 | 36 | 32 | 28 | 28 | 18 |
| scg_all | 149/360 | 13.00 | 60.0% | 7 | 36 | 32 | 28 | 28 | 18 |
| scg15_route | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| scg15_heavy | 2/360 | 20.82 | 1.7% | 1 | 1 | 0 | 0 | 0 | 0 |
| scg15_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| scg15_noisy | 160/360 | 13.00 | 66.7% | 7 | 40 | 35 | 32 | 28 | 18 |
| scg15_all | 160/360 | 13.00 | 66.7% | 7 | 40 | 35 | 32 | 28 | 18 |
| swc_route | 2/360 | 20.76 | 3.3% | 2 | 0 | 0 | 0 | 0 | 0 |
| swc_heavy | 1/360 | 20.80 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| swc_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| swc_noisy | 26/360 | 13.00 | 13.3% | 5 | 4 | 2 | 2 | 5 | 8 |
| swc_all | 26/360 | 13.00 | 13.3% | 5 | 4 | 2 | 2 | 5 | 8 |
| swc05_route | 2/360 | 20.76 | 3.3% | 2 | 0 | 0 | 0 | 0 | 0 |
| swc05_heavy | 1/360 | 20.80 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| swc05_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| swc05_noisy | 39/360 | 13.00 | 18.3% | 6 | 4 | 6 | 5 | 7 | 11 |
| swc05_all | 39/360 | 13.00 | 18.3% | 6 | 4 | 6 | 5 | 7 | 11 |
| enh4_route | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| enh4_heavy | 1/360 | 20.80 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| enh4_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| enh4_noisy | 46/360 | 13.00 | 25.0% | 15 | 4 | 4 | 3 | 8 | 12 |
| enh4_all | 46/360 | 13.00 | 25.0% | 15 | 4 | 4 | 3 | 8 | 12 |
| enh6_route | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| enh6_heavy | 1/360 | 20.82 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| enh6_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| enh6_noisy | 42/360 | 16.92 | 21.7% | 13 | 3 | 3 | 3 | 8 | 12 |
| enh6_all | 42/360 | 17.00 | 21.7% | 13 | 3 | 3 | 3 | 8 | 12 |
| enhs4_route | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| enhs4_heavy | 1/360 | 20.80 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| enhs4_tas | 2/360 | 20.82 | 3.3% | 2 | 0 | 0 | 0 | 0 | 0 |
| enhs4_noisy | 31/360 | 13.00 | 18.3% | 4 | 4 | 3 | 3 | 6 | 11 |
| enhs4_all | 31/360 | 13.00 | 18.3% | 4 | 4 | 3 | 3 | 6 | 11 |
| enh8_route | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| enh8_heavy | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| enh8_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| enh8_noisy | 46/360 | 20.84 | 23.3% | 14 | 4 | 4 | 5 | 6 | 13 |
| enh8_all | 46/360 | 21.00 | 23.3% | 14 | 4 | 4 | 5 | 6 | 13 |
| piec4_route | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| piec4_heavy | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| piec4_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| piec4_noisy | 159/360 | 13.00 | 75.0% | 3 | 45 | 37 | 33 | 24 | 17 |
| piec4_all | 159/360 | 13.00 | 75.0% | 3 | 45 | 37 | 33 | 24 | 17 |
| swcpiec4_route | 2/360 | 20.76 | 3.3% | 2 | 0 | 0 | 0 | 0 | 0 |
| swcpiec4_heavy | 1/360 | 20.80 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| swcpiec4_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| swcpiec4_noisy | 31/360 | 13.00 | 15.0% | 5 | 4 | 3 | 4 | 6 | 9 |
| swcpiec4_all | 31/360 | 13.00 | 15.0% | 5 | 4 | 3 | 4 | 6 | 9 |
| piec8_route | 3/360 | 20.84 | 3.3% | 2 | 0 | 0 | 0 | 0 | 1 |
| piec8_heavy | 2/360 | 20.84 | 1.7% | 1 | 1 | 0 | 0 | 0 | 0 |
| piec8_tas | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| piec8_noisy | 143/360 | 20.84 | 61.7% | 2 | 37 | 31 | 27 | 24 | 22 |
| piec8_all | 142/360 | 21.00 | 61.7% | 1 | 37 | 31 | 27 | 24 | 22 |

Chosen per fold: snr_theta: [34.0, None, None, None, None]; d4veto: [-0.555221438407898, -0.6119210124015808, -0.5893959999084473, -0.5893959999084473, -0.5893959999084473]; tas: [None, None, None, None, 4.998896355327897]; snng05_route: [(34.0, 33.0), (None, None), (None, None), (None, None), (None, None)]; snng05_heavy: [(34.0, 16.0), (None, 16.0), (None, 16.0), (None, 16.0), (None, 16.0)]; snng05_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; snng10_route: [(34.0, 33.0), (None, None), (None, None), (None, None), (None, None)]; snng10_heavy: [(34.0, 16.0), (None, 16.0), (None, 16.0), (None, 16.0), (None, 16.0)]; snng10_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; anti_route: [(34.0, 33.0), (None, None), (None, None), (None, None), (None, None)]; anti_heavy: [(34.0, 16.0), (None, 16.0), (None, 16.0), (None, 16.0), (None, 16.0)]; anti_tas: [2.490913668008525, 1.97890911025122, 2.3748563409797607, 2.8260303186435447, 2.666421551027987]; anti_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; scg_route: [(34.0, 33.0), (None, None), (None, None), (None, None), (None, None)]; scg_heavy: [(34.0, 16.0), (None, 16.0), (None, 16.0), (None, 16.0), (None, 16.0)]; scg_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; scg15_route: [(34.0, 33.0), (None, None), (None, None), (None, None), (None, None)]; scg15_heavy: [(34.0, 16.0), (None, 16.0), (None, 16.0), (None, 17.0), (None, 16.0)]; scg15_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; swc_route: [(34.0, 30.0), (None, None), (None, None), (None, None), (None, None)]; swc_heavy: [(34.0, 17.0), (None, 17.0), (None, 17.0), (None, 17.0), (None, 17.0)]; swc_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; swc05_route: [(34.0, 30.0), (None, None), (None, None), (None, None), (None, None)]; swc05_heavy: [(34.0, 17.0), (None, 17.0), (None, 17.0), (None, 17.0), (None, 17.0)]; swc05_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; enh4_route: [(34.0, 33.0), (None, None), (None, None), (None, None), (None, None)]; enh4_heavy: [(34.0, 17.0), (None, 17.0), (None, 17.0), (None, 17.0), (None, 17.0)]; enh4_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; enh6_route: [(34.0, 33.0), (None, None), (None, None), (None, None), (None, None)]; enh6_heavy: [(34.0, 17.0), (None, 17.0), (None, 17.0), (None, 17.0), (None, 17.0)]; enh6_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; enhs4_route: [(34.0, None), (None, None), (None, None), (None, None), (None, None)]; enhs4_heavy: [(34.0, 17.0), (None, 17.0), (None, 17.0), (None, 17.0), (None, 17.0)]; enhs4_tas: [None, None, None, None, 4.998896355327897]; enhs4_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; enh8_route: [(34.0, 33.0), (None, None), (None, None), (None, None), (None, None)]; enh8_heavy: [(34.0, 17.0), (None, 17.0), (None, 17.0), (None, 17.0), (None, 17.0)]; enh8_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; piec4_route: [(34.0, None), (None, None), (None, None), (None, None), (None, None)]; piec4_heavy: [(34.0, 16.0), (None, 16.0), (None, 16.0), (None, 16.0), (None, 16.0)]; piec4_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; swcpiec4_route: [(34.0, 30.0), (None, None), (None, None), (None, None), (None, None)]; swcpiec4_heavy: [(34.0, 17.0), (None, 17.0), (None, 17.0), (None, 17.0), (None, 17.0)]; swcpiec4_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]; piec8_route: [(34.0, 30.0), (None, 30.0), (None, 30.0), (None, 30.0), (None, 28.0)]; piec8_heavy: [(34.0, 16.0), (None, 16.0), (None, 16.0), (None, 16.0), (None, 17.0)]; piec8_noisy: [(34.0, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below'), (None, 'all below')]

Policies: snr_theta = plain Step 4 if SNR ≥ θ (calibrated), else Step 8. <m>_route = plain Step 4 if SNR ≥ θ_hi (calibrated as snr_theta), the method if θ_lo ≤ SNR < θ_hi (θ_lo calibrated next on the same fold), else Step 8. <m>_heavy = plain Step 4 if SNR ≥ θ_hi, the method if SNR < θ_e (heavy noise; θ_e calibrated next), else Step 8. d4veto / tas = plain Step 4 if SNR ≥ 20 dB and the detector (−d₄, or the Trajectory Anomaly Score of log d₁..d₆ fitted on clean training prompts) is below its certified threshold, else Step 8; <m>_tas = the same gate with the method at the handover. <m>_noisy = the method for every input below θ_hi. <m>_all = the method for every input.

Development evidence only; a claim needs a held-out test with the protocol frozen first.
