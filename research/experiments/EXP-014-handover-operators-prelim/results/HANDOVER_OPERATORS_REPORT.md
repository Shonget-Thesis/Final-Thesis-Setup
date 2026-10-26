# Preliminary test: zero-cost handover operators at Step 4

Development speakers only (60 speakers x 6 conditions: clean, noisy15, gauss18, gauss20, gauss22, gauss25); 360 of 360 cases; preregistration `2a5d8ca27633b3dc…`. Every number is computed from the CSV files by `run_handover_operators_prelim.py analyze`.

Drift reproduction: max |Δd₄| = 0.000.

## Preregistered criteria (C1 noisy failures −25% with p < 0.05; C2 clean +1 at most; C3 calibrated gate beats the calibrated SNR gate)

| Operator | C1 | C2 | C3 |
|---|---|---|---|
| Teacher-Coupled Restart (`tcr`) | FAIL | FAIL | FAIL |
| Coherence-gated DEH (`cdeh`) | PASS | PASS | FAIL |
| Speech-Envelope Anchoring (`sea`) | FAIL | PASS | FAIL |

## Step-4 failures (13 NFE each)

| Condition | n | plain | tcr | cdeh | sea |
|---|---:|---:|---:|---:|---:|
| clean | 60 | 2 | 7 | 3 | 1 |
| noisy15 | 60 | 40 | 56 | 31 | 37 |
| gauss18 | 60 | 33 | 50 | 20 | 30 |
| gauss20 | 60 | 26 | 49 | 12 | 34 |
| gauss22 | 60 | 19 | 44 | 17 | 24 |
| gauss25 | 60 | 8 | 29 | 12 | 15 |

Median SIM change vs the plain Step-4 output: tcr: clean -0.0015, noisy15 -0.0103, gauss18 -0.0195, gauss20 -0.0242, gauss22 -0.0366, gauss25 -0.0180; cdeh: clean +0.0000, noisy15 +0.0169, gauss18 +0.0330, gauss20 +0.0120, gauss22 +0.0062, gauss25 +0.0000; sea: clean +0.0010, noisy15 +0.0014, gauss18 +0.0059, gauss20 -0.0033, gauss22 -0.0059, gauss25 -0.0041

Exact McNemar (two-sided), unadjusted:

- tcr vs Step 4, noisy (primary): 126 → 228 of 300 (p 0.0000)
- tcr vs Step 4, clean: 2 → 7 of 60 (p 0.0625)
- cdeh vs Step 4, noisy (primary): 126 → 92 of 300 (p 0.0006)
- cdeh vs Step 4, clean: 2 → 3 of 60 (p 1.0000)
- sea vs Step 4, noisy (primary): 126 → 140 of 300 (p 0.0436)
- sea vs Step 4, clean: 2 → 1 of 60 (p 1.0000)

## Out-of-fold policies, pooled calibration (5-fold speaker CV, α = 0.05, δ = 0.1)

| Policy | Failures | Mean NFE | Worst-condition risk | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| snr_only | 86/360 | 14.40 | 51.7% | 2 | 0 | 31 | 26 | 19 | 8 |
| snr_theta | 7/360 | 18.71 | 8.3% | 2 | 0 | 0 | 0 | 0 | 5 |
| tcr_theta | 8/360 | 19.64 | 10.0% | 6 | 0 | 0 | 0 | 0 | 2 |
| tcr_all | 235/360 | 13.00 | 93.3% | 7 | 56 | 50 | 49 | 44 | 29 |
| cdeh_theta | 10/360 | 19.09 | 11.7% | 3 | 0 | 0 | 0 | 0 | 7 |
| cdeh_all | 95/360 | 13.00 | 51.7% | 3 | 31 | 20 | 12 | 17 | 12 |
| sea_theta | 11/360 | 19.13 | 16.7% | 1 | 0 | 0 | 0 | 0 | 10 |
| sea_all | 141/360 | 13.00 | 61.7% | 1 | 37 | 30 | 34 | 24 | 15 |

Chosen θ per fold: snr_theta: [26.0, 26.0, 26.0, 26.0, 26.0]; tcr_theta: [28.0, 28.0, 28.0, 28.0, 28.0]; cdeh_theta: [26.0, 28.0, 28.0, 26.0, 26.0]; sea_theta: [28.0, 28.0, 26.0, 26.0, 26.0]

## Out-of-fold policies, stratified calibration (5-fold speaker CV, α = 0.05, δ = 0.1)

| Policy | Failures | Mean NFE | Worst-condition risk | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| snr_only | 86/360 | 14.40 | 51.7% | 2 | 0 | 31 | 26 | 19 | 8 |
| snr_theta | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| tcr_theta | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| tcr_all | 235/360 | 13.00 | 93.3% | 7 | 56 | 50 | 49 | 44 | 29 |
| cdeh_theta | 1/360 | 20.84 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| cdeh_all | 95/360 | 13.00 | 51.7% | 3 | 31 | 20 | 12 | 17 | 12 |
| sea_theta | 1/360 | 20.02 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| sea_all | 141/360 | 13.00 | 61.7% | 1 | 37 | 30 | 34 | 24 | 15 |

Chosen θ per fold: snr_theta: [34.0, None, None, None, None]; tcr_theta: [None, None, None, None, None]; cdeh_theta: [34.0, None, None, None, None]; sea_theta: [33.0, 33.0, 28.0, 33.0, 33.0]

*_theta = the operator's output for inputs with SNR ≥ θ (calibrated), Step 8 otherwise; *_all = the operator for every input.

Development evidence only; a claim needs a held-out test with the protocol frozen first.
