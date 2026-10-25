# EXP-001: Problem validation on the T4 (does one fixed switching time suit every input?)

**Run:** 2026-10-25, offline (no GPU) on the T4 grid of EXP-002. Script `code/analysis/problem_validation_t4.py` (standard library); output [results.json](results.json).

**What:** the T4 counterpart of the team's first experiment, the CPU "empirical proof of the fixed handover-timing problem" (12 speakers × 4 conditions = 48 instances, Whisper-base; Appendix D.2; kept as legacy evidence in `research/evidence/legacy/reports/FIXED_HANDOVER_TIMING_EMPIRICAL_PROOF.md`, `research/evidence/legacy/metrics_and_telemetry/fixed_handover_timing_proof.json` and `code/experiments/legacy/run_proving_fixed_timing_problem.py`). Here: all 500 split speakers × clean/noisy15 (1,000 cases), the study's evaluators, and the added-failure rule of Definition 3.1 against Step 8 (s = 0.25).

## Key results

**1. A fixed switching time fails more the earlier it is.**

| Fixed switch | Step | NFE | Clean | Noisy15 | All 1,000 |
|---|---|---:|---:|---:|---:|
| s = 0.07 | 4 | 13 | 12 (2.4%) | 322 (64.4%) | 334 (33.4%) |
| s = 0.10 | 5 | 15 | 9 (1.8%) | 269 (53.8%) | 278 (27.8%) |
| s = 0.15 | 6 | 17 | 5 (1.0%) | 247 (49.4%) | 252 (25.2%) |
| s = 0.25 (reference) | 8 | 21 | 0 | 0 | 0 |

The same ordering holds on each split (Step 4 / 5 / 6 failures: development 66/51/50 of 200, calibration 130/111/96 of 400, test 138/116/106 of 400).

**2. The cheapest switching time that does not fail differs between inputs.**

| Prompts | Step 4 | Step 5 | Step 6 | Step 8 | Oracle mean NFE |
|---|---:|---:|---:|---:|---:|
| clean (500) | 488 | 9 | 3 | 0 | 13.06 |
| noisy15 (500) | 178 | 91 | 52 | 179 | 16.64 |
| all (1,000) | 666 | 100 | 55 | 179 | 14.85 |

No fixed switching time fits all inputs: for 82.1% of cases an earlier switch than the reference would have been acceptable, but which one varies, and 179 noisy cases need the reference. A per-input oracle would use 14.85 forward passes on average instead of 21 (−29%), the headroom an adaptive policy can aim for.

## Caveats

- The oracle is hindsight and not deployable; it bounds what any per-input rule could save (Proposition 3.5).
- The test split was inspected in earlier analyses, so all numbers here are development evidence.
- Only two prompt conditions (clean, noisy15); the CPU proof also varied target duration (±15%).
