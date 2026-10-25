# EXP-001: Follow-up Colab T4 tests

Status: designed 2026-10-25, not run. Recommended by the [October 25 audit](../../audits/2026-10-25/FINDINGS.md).

Notebook: [code/colab/DMOSpeech2_Followup_Tests.ipynb](../../../code/colab/DMOSpeech2_Followup_Tests.ipynb). Runner: [code/colab/run_followup_tests.py](../../../code/colab/run_followup_tests.py). Colab GPU only.

| Test | Question | Data |
|---|---|---|
| H1, fresh held-out grid | Does the policy calibrated on the T4 calibration split keep risk ≤ 5% on speakers nobody has inspected? | The 166 Seed-TTS test-en prompt speakers in no split; clean + noisy15; Steps 4, 5, 6, 8 |
| H2, degradations | Does the stabilization score catch failures the SNR gate misses? SNR-only vs SNR + drift (λ = 0.50), exact McNemar, Holm across conditions | Same speakers; Gaussian 18/20/22/25 dB, DEMAND cafeteria babble 15/20 dB, synthetic reverberation (RT60 0.3/0.9 s), telephone band |
| H3, seed-noise floor | How often does Step 8 fail against Step 8 with another seed? | 200 test speakers, clean + noisy15 |

Secondary: seed-A Step-8 reproducibility against the committed T4 grid, and two development-evidence analyses on that grid (calibrate on clean and test on noisy15; calibration size 50/100/200).

## Procedure

1. Commit and push the runner, so the Colab clone has it.
2. Run the smoke test (two development speakers).
3. Run `freeze`. Commit its `preregistration.json` and `fresh_manifest.json` here before running any test. After freezing, the runner refuses to run if the preregistration, the split manifests, the committed T4 grid or the runner itself change.
4. Run the seed-floor, fresh-grid and degradation tests, then `analyze`.
5. Commit the CSV files, `analysis_results.json`, `FOLLOWUP_REPORT.md`, `run_info_*.json` and `demand_sha256.txt` to `results/` here. Keep the audio in `listening/` on Drive for the listening check of every failure; do not commit audio.

## Comparability note

The committed T4 grid scored SIM with `core/metrics.py` before commit `656d86f`, which read the ground-truth array as 24 kHz regardless of its rate. The runner uses the fixed version, records each ground-truth sample rate (`gt_sr`), and the report flags the comparison if any rate is not 24 kHz. The seed-A reproducibility check measures the effect directly.

## Not included

The listening test itself (the team listens to `listening/` and the 50 packaged pairs) and anything run on CPU.
