# Preliminary test: Drift-Extrapolated Handover (DEH)

Development speakers only (60 speakers x 6 conditions: clean, noisy15, gauss18, gauss20, gauss22, gauss25); 360 of 360 cases; preregistration `a53f1ebb6cfa8b50…`. Every number is computed from the CSV files by `run_endpoint_extrapolation_prelim.py analyze`.

Drift reproduction (this teacher pass vs stored labels): max |Δd₄| = 0.000.

## Preregistered criteria

- PASS: C1 noisy Step-4 failures drop by at least 25% with exact McNemar p < 0.05
- FAIL: C2 clean Step-4 failures rise by at most 1
- FAIL: C3 pooled deh_theta: lower mean NFE than snr_theta with no more failures

## Step-4 failures: plain endpoint vs DEH (13 NFE each)

| Condition | n | Step 4 | DEH | DEH-half |
|---|---:|---:|---:|---:|
| clean | 60 | 2 | 25 | 5 |
| noisy15 | 60 | 40 | 31 | 39 |
| gauss18 | 60 | 33 | 20 | 26 |
| gauss20 | 60 | 26 | 12 | 23 |
| gauss22 | 60 | 19 | 15 | 17 |
| gauss25 | 60 | 8 | 12 | 11 |

Exact McNemar (two-sided), unadjusted; the noisy pooled test is the primary one:

- deh vs Step 4, noisy (primary): 126 → 90 failures of 300 (p 0.0005)
- deh vs Step 4, clean: 2 → 25 failures of 60 (p 0.0000)
- half vs Step 4, noisy (primary): 126 → 116 failures of 300 (p 0.2888)
- half vs Step 4, clean: 2 → 5 failures of 60 (p 0.4531)
- deh vs Step 4, noisy15: 40 → 31 failures of 60 (p 0.0636)
- deh vs Step 4, gauss18: 33 → 20 failures of 60 (p 0.0241)
- deh vs Step 4, gauss20: 26 → 12 failures of 60 (p 0.0094)
- deh vs Step 4, gauss22: 19 → 15 failures of 60 (p 0.4545)
- deh vs Step 4, gauss25: 8 → 12 failures of 60 (p 0.4240)

## Endpoint diagnostics (medians; no scoring needed)

Gain = 1 − ‖T − x̂₁(t_H)‖ / ‖x̂₁(t₄) − x̂₁(t_H)‖: 1 means the extrapolated endpoint equals the later one, 0 no better than Step 4.

| Condition | ρ | S | cos(Δ₄, Δ₃) | cos(Δ₄, rest) | gain DEH | gain half | gain to Step 6 | gain linear-in-t | share gain DEH > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.79 | 2.30 | -0.17 | -0.34 | -2.31 | -0.99 | -1.28 | -6.94 | 0% |
| noisy15 | 0.78 | 2.22 | 0.60 | 0.23 | -0.35 | -0.03 | -0.07 | -2.35 | 13% |
| gauss18 | 0.80 | 2.36 | 0.47 | 0.06 | -0.55 | -0.14 | -0.19 | -3.14 | 3% |
| gauss20 | 0.80 | 2.39 | 0.43 | -0.08 | -0.95 | -0.32 | -0.37 | -3.79 | 0% |
| gauss22 | 0.77 | 2.18 | 0.38 | -0.20 | -1.19 | -0.46 | -0.59 | -4.40 | 0% |
| gauss25 | 0.76 | 2.08 | 0.27 | -0.28 | -1.31 | -0.51 | -0.69 | -5.15 | 0% |

## Out-of-fold policies, pooled calibration (5-fold speaker CV, α = 0.05, δ = 0.1)

| Policy | Failures | Mean NFE | Worst-condition risk | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| snr_only | 86/360 | 14.40 | 51.7% | 2 | 0 | 31 | 26 | 19 | 8 |
| snr_theta | 7/360 | 18.71 | 8.3% | 2 | 0 | 0 | 0 | 0 | 5 |
| drift_window | 9/360 | 18.38 | 5.0% | 2 | 0 | 0 | 1 | 3 | 3 |
| deh_snr_only | 84/360 | 14.40 | 41.7% | 25 | 0 | 20 | 12 | 15 | 12 |
| deh_theta | 12/360 | 20.42 | 20.0% | 12 | 0 | 0 | 0 | 0 | 0 |
| deh_window | 6/360 | 20.69 | 10.0% | 6 | 0 | 0 | 0 | 0 | 0 |
| deh_all | 115/360 | 13.00 | 51.7% | 25 | 31 | 20 | 12 | 15 | 12 |
| half_theta | 11/360 | 18.84 | 10.0% | 5 | 0 | 0 | 0 | 0 | 6 |
| half_all | 121/360 | 13.00 | 65.0% | 5 | 39 | 26 | 23 | 17 | 11 |

Chosen parameter per fold: snr_theta: [26.0, 26.0, 26.0, 26.0, 26.0]; drift_window: [0.3, 0.3, 0.3, 0.3, 0.3]; deh_theta: [38.0, 37.0, 39.0, 36.0, 37.0]; deh_window: [0.6, None, None, None, 0.6]; half_theta: [27.0, 27.0, 26.0, 26.0, 27.0]

## Out-of-fold policies, stratified calibration (5-fold speaker CV, α = 0.05, δ = 0.1)

| Policy | Failures | Mean NFE | Worst-condition risk | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| snr_only | 86/360 | 14.40 | 51.7% | 2 | 0 | 31 | 26 | 19 | 8 |
| snr_theta | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| drift_window | 0/360 | 20.29 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| deh_snr_only | 84/360 | 14.40 | 41.7% | 25 | 0 | 20 | 12 | 15 | 12 |
| deh_theta | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| deh_window | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| deh_all | 115/360 | 13.00 | 51.7% | 25 | 31 | 20 | 12 | 15 | 12 |
| half_theta | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| half_all | 121/360 | 13.00 | 65.0% | 5 | 39 | 26 | 23 | 17 | 11 |

Chosen parameter per fold: snr_theta: [34.0, None, None, None, None]; drift_window: [0.6, 0.6, 0.6, 0.6, 0.6]; deh_theta: [None, None, None, None, None]; deh_window: [None, None, None, None, None]; half_theta: [None, None, None, None, None]

Policies: Step 4 = plain endpoint; deh/half = DEH endpoint handed over at Step 4 (13 NFE); theta = calibrated SNR gate; window = SNR ≥ 20 dB and d₄ ≥ λ_low; *_all = no gate. Parameterless policies are reported as is.

Development evidence only; a claim needs a held-out test with the protocol frozen first.
