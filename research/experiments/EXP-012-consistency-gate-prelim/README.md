# EXP-012: Preliminary test of the proposed gates

**Status:** designed 2026-10-25, not run. Notebook `code/colab/DMOSpeech2_Consistency_Gate_Prelim.ipynb`, runner `code/colab/run_consistency_gate_prelim.py` (Colab T4; same setup cells as EXP-011).

**Why:** EXP-011's pilot found that the 20 dB SNR gate misses 18–22 dB white noise, and that noise flattens the teacher's trajectory (low d₄). This experiment tests proposed fixes on development data before any is written into the method.

**Modes:** **Quick** (the notebook default, `--quick`): the first 60 development speakers by utterance ID (they include the 20 pilot speakers) × clean, noisy15 and Gaussian 18/20/22/25 dB; 40 speakers × 4 levels = 160 new cases are synthesized and 360 cases get the teacher-only pass. It covers the failure EXP-011 found, the 18–22 dB band around the gate. **Full**: as described below. Each mode writes to its own folder; `preregistration.json` records the speaker count and conditions, and a restart with other settings in the same folder is refused.

**What (Full):** the 100 development speakers under all eleven prompt conditions (clean, noisy15, Gaussian 18/20/22/25 dB, DEMAND babble 15/20 dB, reverberation 0.3/0.9 s, telephone). Labels are reused where they exist (EXP-002 for clean/noisy15; EXP-011's pilot for 20 speakers × 11 conditions); the missing 80 speakers × 9 conditions are synthesized and scored. A teacher-only pass records prompt SNR, d₁–d₆ and the guidance contrast g₀–g₆ for all 1,100 cases.

| Policy | Rule (candidates Steps 4/5/6, reference Step 8, SNR gate 20 dB) |
|---|---|
| reference | Step 8 |
| snr_only | Step 4 if Ŝ ≥ 20 dB (current gate) |
| snr_drift | Definition 3.4, first candidate with d_k ≤ λ (the proposal as written) |
| low_drift_veto | fall back if d₄ < λ_low (two-sided stabilization window) |
| sdc_gate | SNR–Drift Consistency: fall back if log d₄ − (b₀ + b₁Ŝ) < −κ, the line fitted on clean training prompts |
| guidance_gate | fall back on g₄ = ‖v_c − v_u‖ / ‖v_c‖, direction set on training folds |
| sdc_guidance | fall back if either fires |

Parameters are chosen on training folds (5-fold speaker cross-validation, seed 2026) by fixed-sequence Learn-Then-Test (α = 0.05, δ = 0.10), **pooled** over conditions or **stratified** (certified only if every condition's binomial p-value ≤ δ, an intersection–union test). Reported: out-of-fold failures and NFE per condition, the worst-condition risk, AUCs for gate-passed Step-4 failures, and McNemar against SNR-only (Holm-adjusted). Settings are written to `preregistration.json` before any synthesis.

**Cost:** Quick about 75 minutes, Full about 3–4 hours on a T4 (estimates from EXP-011's 15.6 s per case).

**Caveats:** development evidence only. The low-drift threshold region and the SDC idea came from looking at development data (EXP-011), so a positive result here motivates a held-out test, not a claim. Novelty of the SDC gate and the guidance-contrast handover signal still needs a literature check.
