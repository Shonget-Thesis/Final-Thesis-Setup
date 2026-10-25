# Reports and results audit — 25 October 2026

Checkout reviewed: `a849432` (main). Scope: the reports in `research/reports/`, the T4 evidence in `research/evidence/benchmarks/`, `docs/progress/STATUS.md`, and how they relate to the proposal in `paper/`. Audit run by Claude Code at the request of a team member; it is not adviser review.

All synthesis runs on Google Colab (Tesla T4) by cloning the sibling `Shonget-DMOSpeech2` repository, so the scripts in `code/` are a reference copy and are not expected to run from this workspace. This audit therefore checks the records and the committed results, not code execution.

Reproduce the numbers below with `python research/audits/2026-10-25/recompute_gpu_grid.py` (standard library only). Input: `research/evidence/benchmarks/gpu_grid_all1000.csv`, SHA-256 `249191ec8b8454d10451185760b3b834c9f673bcfa866ce9b116770a8b4febd3`, 4,000 rows; every utterance maps to exactly one split manifest. Failure labels are taken from the grid's `is_failure` column; they were not re-scored.

## Findings

### 1. The test split has already been used (critical)

The proposal (Chapter 3) describes calibration and the held-out test as future work and says the test split stays locked; the drift-framing log (2026-10-25) says the same. The reports, however, contain completed calibration and test-split results (Chapter 3 master report Sections 4.2–4.10; signal-selection, novel-algorithm and proposal TO-DO reports). The signal-selection and novel-algorithm reports state that the test split "had been inspected in earlier analyses". The SNR gate (20 dB), the λ grid, the candidate set and the failure margins were all fixed after those analyses.

Consequence: the 200 test speakers cannot serve as an independent held-out set for the thesis. 166 of the 666 Seed-TTS test-en prompt speakers are in no split (666 total, 500 assigned, per `research/protocols/splits/split_summary.json`) and can serve as a fresh test set once the protocol is frozen (EVAL-001).

### 2. On the T4 labels, calibration certifies the whole grid (critical)

The reports' threshold λ̂ = 0.50 (p = 0.0264) comes from the CPU calibration grid. The thesis now states that all synthesis and scoring are on the T4, and the committed T4 grid gives a different result. Fixed-sequence LTT (α = 0.05, δ = 0.10, θ = 20 dB) on the 200 calibration speakers, recomputed from the T4 grid:

| λ | 0.20–0.30 | 0.35 | 0.40–0.45 | 0.50–0.55 | 0.60–0.65 | 0.70 | 0.75 | 0.80–0.90 | 0.95–1.20 |
|---|---|---|---|---|---|---|---|---|---|
| Speaker failures /200 | 0 | 2 | 3 | 2 | 5 | 4 | 3 | 4 | 5 |
| p | <0.0001 | 0.0023 | 0.0090 | 0.0023 | 0.0623 | 0.0264 | 0.0090 | 0.0264 | 0.0623 |

Every p-value is at most 0.0623 ≤ δ, so the procedure never stops and returns λ̂ = 1.20, the end of the grid. Test-split outcomes from the same file:

| Policy | Speaker failures /200 | Mean NFE per case |
|---|---:|---:|
| Stabilization + SNR, λ = 0.50 (reports' threshold) | 3 | 18.15 |
| Stabilization + SNR, λ̂ = 1.20 (T4 calibration) | 4 | 17.04 |
| SNR-only (λ = ∞) | 4 | 17.04 |
| Fixed Step 4 | 135 | 13.00 |
| Reference Step 8 | 0 | 21.00 |

With λ̂ = 1.20 the stabilization condition no longer binds and the policy reduces to SNR-only. The λ = 0.50 row reproduces the reports' T4 test figures (3/200, 18.15 NFE), so the data agree; the difference is which labels the threshold was calibrated on. Either way, SNR-only is the cheapest safe rule in the current evidence.

### 3. The stabilization score does not discriminate on clean prompts

Recomputed from the T4 grid, development split, clean condition: 3 of 100 Step-4 handovers fail, and the AUC of d₄ for those failures is 0.474. This agrees with the development drift report and with Appendix D. With three failures the estimate is very imprecise.

### 4. The noisy condition is trivially separable

Estimated prompt SNR over all 500 speakers: clean 13.7–50.0 dB (3 below 20 dB), noisy15 12.4–18.4 dB (all below 20 dB). The 20 dB gate therefore sends every synthetic-noise prompt to the reference, and noisy15 contributes no failures to any gated policy. The "adaptive" decision currently amounts to detecting the added white noise; Appendix D step 3 (reverberant and telephone-band prompts on unused speakers) is the test that would show otherwise.

### 5. Report wording that overstates the evidence

- "Certified" λ̂ = 0.50: a CPU-label result that the T4 labels do not reproduce (Finding 2).
- "Held-out" and "unbiased held-out benchmark": the test split was inspected before the protocol was fixed (Finding 1).
- "100% COMPLETE" and "ANALYSIS COMPLETE & VERIFIED": the reports are development evidence; no adviser review or fresh held-out test exists.
- Execution target "AMD Ryzen 7 5700U CPU" (master README): the thesis now uses the T4 only; the CPU grids remain historical development evidence.
- Speakers are Seed-TTS prompt identities, not verified distinct Common Voice speakers; "new speakers" should read "new prompt identities".

### 6. STATUS.md contradicted the records

STATUS said "no new synthesis or held-out evaluation has been performed" and "inference has not been smoke-tested". The T4 grid (4,000 rows), the T4 timing run, the test-split analyses and `code/scheduling/sanity_check_results.json` show otherwise. Corrected on 2026-10-25.

### 7. Provenance gaps

- The CPU grids that produced the reports' λ̂ = 0.50 (`phase1_dev_grid_100spk`, the calibration and test grids) and their scripts were not migrated; only the T4 CSVs are in this repository.
- The Colab notebooks run code cloned from `Shonget-DMOSpeech2`; the reports record commits `38ac387` and `381ec71` for two runs, but the T4 grid and timing runs do not record the commit they used.
- The reports cite `VERDICT_VALIDATE_DRIFT_BEFORE_REDESIGN.md`, a note that was never committed.

### 8. Sibling-repository experiments (addendum, same day)

The sibling working tree `Shonget-DMOSpeech2/Handover Scheduling/` (HEAD `8a7868f`, plus branch `SNRgated`) was read at a team member's request. It holds the CPU phase grids, the T4 grid and timing runs, the signal-selection and novel-algorithm runs (each with a preregistration file written before its analysis), a development-only three-scheduler revision study (steps 2–8, 1,200 records), and an 18-test "comprehensive evaluation suite" (commit `20a37bb` on `SNRgated`, results zip in `colab/DMOSpeech2_Comprehensive_Test_Results.zip`).

In `colab/run_comprehensive_evaluation_suite.py` at `20a37bb`:

- Test 12 (SNR sweep) and Test 17 (latency) write constants typed into the function; they read no data.
- Test 1 (seed-noise floor) and Test 15 (DEMAND babble) read `seed_noise_control_audit.csv` and `demand_real_noise_snr_sweep.csv` if present and otherwise write typed-in values (5/200; 100%/62% fallback). Neither file exists in the working tree or in any commit, so the published outputs are the typed-in values. The seed-noise runner in the earlier commit `3580b6d` synthesizes speech, but its output was never committed.
- Test 7 falls back to typed-in AUCs (0.69 and similar) when development data are missing; the executive summary quotes 0.69.
- The executive summary of `CHAPTER3_MASTER_EVALUATION_REPORT.md` is a fixed string, not computed from the results, and contradicts the suite's own JSON files: worst-20% AUC 0.52 (summary 0.69), noise-detection AUC 0.0 (summary 0.94), oracle headroom captured 37% (summary 57%), re-split coverage 100% (summary 97%).
- `master_evaluation_cases_log.csv` sets prompt duration (3.0 s), target frames (250) and bottom energy (0.0001) to the same constant for all 400 rows.
- Test 18 only packages 50 audio pairs; no listening test was run.

Consequences for this repository: the manuscript cites none of these tests. The Chapter 3 master, signal-selection and novel-algorithm reports cite the "2.5% seed-noise floor" (Chapter 3 report §4.9.1) to explain clean-prompt failures; no committed output supports that number, so it is unverified until the seed-perturbed Step-8 run is repeated and its per-speaker output committed. The same holds for the DEMAND and SNR-sweep claims, which the Chapter 3 report already lists as "not run" (§8). Tests 2–11, 13, 14 and 16 compute from the committed grids and are development evidence subject to Findings 1–2.

## What still holds

- The committed T4 grid is internally consistent with the reports' T4 numbers at λ = 0.50.
- T4 timing run (`gpu_full_pipeline_timing_3000.csv`, one session, all rows Tesla T4): teacher + student NFE equals the recorded total and matches 2(k+1)+3 for the exit step in 3,000/3,000 rows. Test-split clean-prompt mean core-synthesis time: SNR-only 2.06 s, SNR + drift 2.41 s, Step 8 3.29 s; noisy prompts 3.37 s for all three policies.
- Signal-selection run (T4, commit `38ac387`): its preregistration file was written at 10:48 UTC, before the 11:28 summary; the results folder and the Colab_Output zip are byte-identical. `Colab_Output/drift_features_all1000.csv` matches the grid's d₄ exactly in all 1,000 cases.
- The comprehensive-suite case log (`master_evaluation_cases_log.csv`) is built from the CPU test grid (`backend = CPU` in all 400 rows), so that suite is not T4 evidence; its Tests 1, 12, 15 and 17 have identical typed-in values in both zipped runs.
- The calibration code (`code/experiments/phase2_calib/calibrate.py`) implements fixed-sequence LTT as Chapter 3 states.
- The negative finding on the stabilization score is reported openly in the drift report and in Appendix D.

## Recommended actions

1. Freeze the protocol (EVAL-001): failure rule, θ, λ grid, candidate set, α, δ, and the choice of T4 labels as primary.
2. Run the final held-out test on the 166 unused prompt speakers and report the earlier inspection of the 200-speaker test split.
3. Recalibrate on the T4 calibration labels and report λ̂ = 1.20 (or the frozen grid's result) instead of the CPU λ̂ = 0.50.
4. Decide whether the contribution is the calibrated per-utterance framework (which holds with SNR-only) or the stabilization score.
5. Record the `Shonget-DMOSpeech2` commit and the grid SHA-256 for every Colab run.
