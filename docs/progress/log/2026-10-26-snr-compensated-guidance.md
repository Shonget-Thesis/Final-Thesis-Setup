# 2026-10-26: SNR-Compensated Guidance added to EXP-015

Task: EVAL-007. Requested by a team member: a computational method in which the prompt SNR is the main contributor, tested in the same run as SNNG.

## Basis (exploratory, from committed EXP-012 data)

- The guidance gap g₄ = ‖v_c − v_u‖/‖v_c‖ falls with prompt noise: median 0.236 (clean), 0.179 (+25 dB), 0.144 (+20 dB), 0.116 (+15 dB). Spearman correlation with SNR 0.84; with d₄ 0.89 (360 cases).
- Below 30 dB, g₄ ≈ −0.0263 + 0.0078·SNR (R² 0.60, n = 311).
- The endpoint change is Δᵢ = (1 − tᵢ)(vᵢ − vᵢ₋₁) with v = v_c + γ(v_c − v_u), so a weaker guidance gap also lowers drift.
- An SNR-scheduled endpoint extrapolation (zero cost) was scored offline on EXP-013/014 outcomes and gave no gain over the calibrated SNR gate (pooled 11/360 at 18.47 NFE for DEH vs 7/360 at 18.71); not pursued.

## Changed

- `code/colab/run_noise_guidance_prelim.py`: new outputs `scg` (r_max 2.5) and `scg15` (r_max 1.5) with γ(Ŝ) = 2·clip(0.236 / max(−0.0263 + 0.0078·Ŝ, 0.02), 1, r_max), 13 NFE; the teacher loop is generalized to (γ, η) specs; the smoke check now compares the generalized loop (γ = 2, η = 0) with the plain teacher endpoint. Analysis, criteria and route policies apply to the new outputs unchanged.
- `code/colab/DMOSpeech2_Noise_Guidance_Prelim.ipynb`: title, method table, time estimate (2.5–3 h) and smoke columns.
- `code/tests/test_noise_guidance.py`: SCG scale, cap and floor.
- EXP-015 README and the EVAL-007 title.

## Checks run

- python -m unittest discover -s code/tests and python scripts/check_workspace.py: see below. torch is not installed locally; the Colab smoke test exercises the torch path. Not run on a GPU.

## Follow-up: SNR-Wiener Conditioning (same day)

Requested by a team member: more computational candidates in the same run, in case SCG and SNNG fail. SNR-Wiener Conditioning removes the prompt's noise from the conditioning before the teacher and student see it, with a strength set by the prompt SNR: per mel band, gain G = clip(1 − β(Ŝ)·N_b/M, 0.1, 1), N_b from the quietest 20% of prompt frames, β(Ŝ) = β_max·clip((30 − Ŝ)/10, 0, 1), β_max 1.0 and 0.5; 13 NFE. Added to `run_noise_guidance_prelim.py` (outputs `swc`, `swc05`), the notebook (estimate now 3.5–4 h), tests (gain floor, speech frames nearly unchanged, generated frames untouched, β = 0 identity) and the EXP-015 README with a prior-work note. 51 tests pass; workspace check passes. Not run on a GPU.
