# 2026-10-26: SNR-routed prompt enhancement added to EXP-015

Task: EVAL-007. Requested by a team member: a solution that can beat the calibrated SNR gate, tested in the same (not yet run) EXP-015 notebook.

## Analysis (offline, committed T4 data; exploratory)

The EXP-015 README records the numbers: beating the calibrated SNR gate needs a mean SIM gain of about 0.08 on noisy Step-4 outputs; the tested in-sampler methods reached at most +0.024; the clean-versus-noisy gap at Step 8 is 0.06–0.24; a perfect prompt-noise remover (the clean-prompt output of the same speakers and seeds, judged against the noisy-prompt Step-8 reference) would reach 14.64 NFE at 11/360 failures out of fold, against 18.71 NFE at 7/360 for the calibrated SNR gate.

## Changed

- `code/colab/run_followup_tests.py`: `Pipeline.prepare` takes an optional `prompt_transform(prompt, sr)`, applied after the prompt SNR is measured (default None; existing runners are unaffected).
- `code/colab/run_noise_guidance_prelim.py`: DNS64 Demucs enhancer (pip `denoiser` 0.1.5, installed with --no-deps); outputs `enh4`, `enh6`, `enhs4`; enhancer time recorded (`enh_ms`); a new policy family `<m>_heavy` for every method; C3 now passes if `<m>_route` or `<m>_heavy` beats the calibrated SNR gate (fixed before any run).
- Notebook: installs the enhancer in the smoke and run cells; method table and time estimate (5–5.5 h).
- Tests: heavy-noise routing and the enhancement schedule.

## Checks run

- python -m unittest discover -s code/tests (52 tests) and python scripts/check_workspace.py: pass. torch is not installed locally; the Colab smoke test exercises the enhancer and torch path. Not run on a GPU.
- Scope note: the enhancer is an extra pretrained network, so it departs from the manuscript's "no extra network evaluations" statement; any manuscript use is a team and adviser decision.

## Follow-up: PIEC, controls and the SNR-only comparison (same day)

- A team member asked for the comparison against SNR-only (fixed 20 dB gate) only. Computed exactly from committed outputs on the 360-case prelim set: SNR-only 86/360 failures at 14.40 NFE; SNR-only + cDEH 64/360 at 14.40 (McNemar 48 vs 26, p 0.014, five operators compared); SNR-only + low-drift veto 9/360 at 18.38 (p ≈ 1e-23). A per-case interpolation predicts that an enhancer removing about 10% of the noise damage beats SNR-only under the same routing, and about 25% without any SNR routing.
- Added Prompt-Informed Environment Cancellation (`piec4`, `swcpiec4`; zero network cost) and controls (`enh8`, `piec8`) with a within-cleaned-system check; the plain teacher pass now runs to Step 8 in one pass; TF32 is disabled so a faster GPU gives the same comparison (no effect on a T4).
- Checks: 53 tests pass; workspace check passes. Not run on a GPU. Estimate: 2.5–3 h on an A100, 6.5–7 h on a T4.

## Follow-up: Trajectory Anomaly Score (same day)

Added TAS gating (`tas`, `d4veto`, `<m>_tas`) and an "Against SNR-only" table with exact McNemar to the EXP-015 analysis. On the committed data the integrated analysis reproduces the offline result exactly: `tas` 12/360 at 18.20 NFE, `d4veto` 13/360 at 18.20, SNR-only 86/360 at 14.40. 54 tests pass; workspace check passes. Analysis-only change: the notebook and the synthesis are unchanged.
