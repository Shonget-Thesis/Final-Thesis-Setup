# EXP-000: Historical timing replay

Status: implemented; validation results are recorded in the setup work log.
Task: INFRA-001. Date: 2026-10-09.

Purpose: check that the imported fixed and adaptive tables reproduce the descriptive arithmetic in the October 5 audit. This is an analysis of rounded stored data, not an inference run, listening test, or held-out evaluation.

~~~powershell
python code/analysis/replay_timing.py --output research/experiments/EXP-000-timing-replay/results.json
python -m unittest discover -s code/tests
~~~

Inputs: fixed_handover_timing_proof.json and calibrated_hybrid_scheduler_telemetry.json under research/evidence/legacy/metrics_and_telemetry/. Source hashes and the evaluator-config hash are included in results.json. Configuration: code/configs/evaluation_v1.json; WER is in percentage points. The upstream revision is recorded in docs/SOURCE_MANIFEST.json; the new maintained replay belongs to the workspace setup change and is not yet committed.

The SNR-only rule reuses the recorded three-teacher-step result when estimated SNR is at least 18 dB, otherwise the four-teacher-step result. Historical t labels are retained as lookup keys and must not be interpreted as actual flow times. The output reports both relative and strict failures and exact hybrid/fixed metric matches.

Environment: standard-library Python; no GPU, model checkpoint, new waveform, new random seed, or statistical significance test is involved. The independently recorded audit numbers provide regression expectations. Collisions and p-values are outside this replay's scope.
