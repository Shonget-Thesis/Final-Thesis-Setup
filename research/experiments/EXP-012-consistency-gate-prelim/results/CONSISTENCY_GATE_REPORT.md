# Preliminary test: proposed gates vs SNR-only and SNR + drift

Development speakers only (60 speakers x 6 conditions: clean, noisy15, gauss18, gauss20, gauss22, gauss25); 360 of 360 cases; preregistration `7cc969344e46fed4…`. Every number is computed from the CSV files by `run_consistency_gate_prelim.py analyze`.

Drift reproduction (teacher-only pass vs stored labels): max |Δd| = 0.0.

## Which signal flags gate-passed Step-4 failures? (AUC, all conditions pooled)

| Signal | AUC | Failures / passes |
|---|---:|---|
| low SNR (-S) | 0.752 | 86 / 211 |
| low drift (-d_4) | 0.773 | 86 / 211 |
| SDC residual (-r) | 0.775 | 86 / 211 |
| low guidance (-g_4) | 0.759 | 86 / 211 |
| high guidance (+g_4) | 0.241 | 86 / 211 |

## Out-of-fold results, pooled calibration (5-fold speaker CV, α = 0.05, δ = 0.10)

| Policy | Failures (all) | Mean NFE | Worst-condition risk | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| snr_only | 86/360 | 14.40 | 51.7% | 2 | 0 | 31 | 26 | 19 | 8 |
| snr_drift | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| low_drift_veto | 9/360 | 18.38 | 5.0% | 2 | 0 | 0 | 1 | 3 | 3 |
| sdc_gate | 10/360 | 18.44 | 5.0% | 2 | 0 | 1 | 1 | 3 | 3 |
| guidance_gate | 4/360 | 19.38 | 3.3% | 2 | 0 | 0 | 0 | 1 | 1 |
| sdc_guidance | 4/360 | 19.38 | 3.3% | 2 | 0 | 0 | 0 | 1 | 1 |

Chosen parameter per fold: low_drift_veto: [0.3, 0.3, 0.3, 0.3, 0.3]; sdc_gate: [0.7, 0.7, 0.7, 0.7, 0.7]; guidance_gate: [0.8, 0.8, 0.8, 0.8, 0.8]; sdc_guidance: [1.5, 1.5, 1.5, 1.5, 1.5]

McNemar vs SNR-only (per condition, Holm-adjusted p; only conditions with discordant pairs):

- snr_drift: clean: 2 vs 0 (p 1.000); gauss18: 31 vs 0 (p 0.000); gauss20: 26 vs 0 (p 0.000); gauss22: 19 vs 0 (p 0.000); gauss25: 8 vs 0 (p 0.023)
- low_drift_veto: gauss18: 31 vs 0 (p 0.000); gauss20: 25 vs 0 (p 0.000); gauss22: 16 vs 0 (p 0.000); gauss25: 5 vs 0 (p 0.188)
- sdc_gate: gauss18: 30 vs 0 (p 0.000); gauss20: 25 vs 0 (p 0.000); gauss22: 16 vs 0 (p 0.000); gauss25: 5 vs 0 (p 0.188)
- guidance_gate: gauss18: 31 vs 0 (p 0.000); gauss20: 26 vs 0 (p 0.000); gauss22: 18 vs 0 (p 0.000); gauss25: 7 vs 0 (p 0.047)
- sdc_guidance: gauss18: 31 vs 0 (p 0.000); gauss20: 26 vs 0 (p 0.000); gauss22: 18 vs 0 (p 0.000); gauss25: 7 vs 0 (p 0.047)

## Out-of-fold results, stratified calibration (5-fold speaker CV, α = 0.05, δ = 0.10)

| Policy | Failures (all) | Mean NFE | Worst-condition risk | clean | noisy15 | gauss18 | gauss20 | gauss22 | gauss25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reference | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| snr_only | 86/360 | 14.40 | 51.7% | 2 | 0 | 31 | 26 | 19 | 8 |
| snr_drift | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| low_drift_veto | 0/360 | 20.29 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| sdc_gate | 1/360 | 20.82 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |
| guidance_gate | 0/360 | 21.00 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| sdc_guidance | 1/360 | 20.82 | 1.7% | 1 | 0 | 0 | 0 | 0 | 0 |

Chosen parameter per fold: low_drift_veto: [0.6, 0.6, 0.6, 0.6, 0.6]; sdc_gate: [0.0, None, None, None, None]; sdc_guidance: [0.0, None, None, None, None]

McNemar vs SNR-only (per condition, Holm-adjusted p; only conditions with discordant pairs):

- snr_drift: clean: 2 vs 0 (p 1.000); gauss18: 31 vs 0 (p 0.000); gauss20: 26 vs 0 (p 0.000); gauss22: 19 vs 0 (p 0.000); gauss25: 8 vs 0 (p 0.023)
- low_drift_veto: clean: 2 vs 0 (p 1.000); gauss18: 31 vs 0 (p 0.000); gauss20: 26 vs 0 (p 0.000); gauss22: 19 vs 0 (p 0.000); gauss25: 8 vs 0 (p 0.023)
- sdc_gate: clean: 1 vs 0 (p 1.000); gauss18: 31 vs 0 (p 0.000); gauss20: 26 vs 0 (p 0.000); gauss22: 19 vs 0 (p 0.000); gauss25: 8 vs 0 (p 0.023)
- guidance_gate: clean: 2 vs 0 (p 1.000); gauss18: 31 vs 0 (p 0.000); gauss20: 26 vs 0 (p 0.000); gauss22: 19 vs 0 (p 0.000); gauss25: 8 vs 0 (p 0.023)
- sdc_guidance: clean: 1 vs 0 (p 1.000); gauss18: 31 vs 0 (p 0.000); gauss20: 26 vs 0 (p 0.000); gauss22: 19 vs 0 (p 0.000); gauss25: 8 vs 0 (p 0.023)

Development evidence only (100 speakers); the parameters were chosen on training folds. A claim needs a held-out test with the protocol frozen first.
