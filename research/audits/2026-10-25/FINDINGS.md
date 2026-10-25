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
4. Decide whether the contribution is the calibrated per-utterance framework (which holds with SNR-only) or the stabilization score. *Decided by the team on 2026-10-25: the method stays SNR + drift as presented at the concept defense. Learn-Then-Test sets how strongly drift acts (λ̂); whether drift adds value beyond the SNR gate is tested on degraded prompts (EXP-001, H2) and reported either way. A calibrated threshold must not be replaced by a chosen one to make drift active.*
5. Record the `Shonget-DMOSpeech2` commit and the grid SHA-256 for every Colab run.

## Pending actions

Status on 2026-10-25. Owners are unassigned unless named. The follow-up tests (EXP-001, about 5–7 hours on a T4 by estimate, not measured) are deferred until after the proposal defense.

### Done

- Audit notes added to every report in `research/reports/`; STATUS.md corrected.
- Verified Colab T4 results imported with hashes (`research/evidence/benchmarks/MANIFEST.json`, REPRO-003); CPU runs and the comprehensive evaluation suite excluded.
- Follow-up runner and notebook written and tested locally (`code/colab/run_followup_tests.py`, `DMOSpeech2_Followup_Tests.ipynb`, EVAL-003); the notebook clones branch `rhenelAudit`. Not yet run on a GPU.

### Before the proposal defense (no new results needed)

1. **Chapter 3 edits**, in the proposal's future tense; the title, objectives and SNR + drift method stay as presented:
   - Data and splits: add the 166-speaker fresh held-out set; state that the 200-speaker test split was inspected and is development evidence; name the T4 results as primary.
   - Robustness: describe the degraded-prompt tests (Gaussian 18–25 dB, DEMAND babble 15/20 dB, reverberation RT60 0.3/0.9 s, telephone band) and the seed-noise floor as planned work.
   - Calibration: replace "the grid extends well beyond 0.60 … so that the data rather than the grid will limit the calibrated threshold" (Section 3.6.1); on the T4 labels the end of the grid binds (Finding 2).
   - Drift: describe drift as the calibrated stopping signal whose strength λ̂ sets, with its added value tested on degraded prompts.
   - SNR-only baseline: define how its gate θ is calibrated (grid and order).
   - *2026-10-25:* an edit basing Appendices A–C on the T4 development data (PAPER-012) was made and reverted at the team's request; these items are open again.
   - Small fixes: Appendix A cites Algorithm 3.3 "line 7" for continuing the teacher (it is line 6 with `[noend]`; use `\label`/`\ref`); Appendix B names `sim/ltt_sim.py` (actual path `code/analysis/ltt_sim.py`); the repeated-split check in Section 3.6.4 compares empirical test risk with 1 − δ, which the guarantee does not imply, so present it as descriptive.
2. **Smoke test and development pilot, run once** (notebook Steps 1–6b; the pilot takes about 1.5–2.5 hours by estimate). Twenty development speakers under all eleven prompt conditions, nothing frozen. `PILOT_REPORT.md` gives everything the Chapter 3 edits need: the Section 3.3.5 checks (determinism, forced fallback reproduces Step 8, forward-pass counts), reproduction of the committed T4 grid, per-condition gate fallback and drift discordance (including where the gate switches between 18 and 25 dB), a 10-speaker seed-floor sample, the measured time per case, and offline T4-grid analyses (λ̂, the SNR-only θ calibrated by Learn-Then-Test, calibrate-on-clean / test-on-noisy, calibration size). Use it to check the pipeline and plan Chapter 3, not as evidence of risk or of drift's value.
3. **Keep the 166 unused speakers untouched** until the protocol is frozen in Step 7; inspecting them early would end their use as a held-out set.
4. Do not cite the "2.5% seed-noise floor", λ̂ = 0.50 as the certified threshold, or the comprehensive suite's Tests 1, 12, 15 and 17 in the proposal.

### After the proposal defense

5. Run the notebook: smoke test, freeze (Step 7), commit `preregistration.json` and `fresh_manifest.json` to `research/experiments/EXP-001-followup-t4/` **before** Step 8, then the full run and the analysis.
6. Commit the results zip contents to `research/experiments/EXP-001-followup-t4/results/`; keep the failure audio in `listening/` on Drive.
7. Listening check of every failure (Chapter 3 promises it) and, if planned, of the 50 packaged pairs.
8. Write Chapter 4 from `FOLLOWUP_REPORT.md`, and finalize the drift wording in Chapters 1 and 3 from H2: if drift prevents failures the SNR gate misses, report that as the main result; if not, report that the calibrated method reduces to the SNR gate in practice and the guarantee holds either way.
9. If the ground-truth sample rates are not all 24 kHz (flagged in the report), recalibrate on newly scored labels before using H1.
10. After merging `rhenelAudit` into `main`, set `THESIS_BRANCH = 'main'` in Step 2 of the notebook.

### Open provenance and team decisions

11. The CPU grids and scripts behind the reports' λ̂ = 0.50, and `run_drift_features_benchmark.py`, were never committed; record them or note them as unrecoverable.
12. REPRO-001 is marked done, but its acceptance criterion (generating scripts, tested environment export) is not met for the CPU grids; the team decides its status.
13. Adviser review of the drift framing and of these findings.
