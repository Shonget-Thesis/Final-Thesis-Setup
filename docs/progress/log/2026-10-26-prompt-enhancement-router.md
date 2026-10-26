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
