# Experiments

Every experiment the team ran on the Google Colab GPU (Tesla T4), plus one replay. CPU runs are not listed; the thesis uses T4 results only. All experiments use Seed-TTS test-en (Common Voice prompts). The splits in `research/protocols/splits/` hold 500 speakers (development 100, calibration 200, test 200; clean and noisy15 prompts); EXP-006 used the remaining 166.

**Order:** the IDs follow the research logic, not the run dates (renumbered on 2026-10-25): the problem, then the switching-time space, the decision signals, alternative policies, calibration, cost, and finally verification. The run dates are in the table.

Large result files live in [research/evidence/benchmarks/](../evidence/benchmarks/MANIFEST.json) or in an experiment's `results/` folder, with SHA-256 hashes checked by `code/tests/test_evidence_manifest.py`. Each folder has a README with what was run, the code and commit, the data, the key results and the caveats.

| ID | Stage | Experiment | Run (2026) | Key result |
|---|---|---|---|---|
| [EXP-001](EXP-001-problem-validation-t4/README.md) | Problem | **Problem validation:** does one fixed switching time suit every input? (offline on EXP-002) | 25 Oct | Fixed-switch failures rise as the switch moves earlier (25% → 33% of 1,000 cases); the cheapest safe step varies across inputs; a per-input oracle needs 14.85 instead of 21 NFE. |
| [EXP-002](EXP-002-t4-grid-500-speakers/README.md) | Switching-time space (Objective 1) | T4 grid: 500 speakers × clean/noisy15 × Steps 4/5/6/8 | 20 Oct | Noisy15 prompts fail at early steps (322/500 at Step 4), clean rarely (12/500). LTT on calibration: λ̂ = 1.20, policy = SNR-only. |
| [EXP-003](EXP-003-early-handover-steps-2-3/README.md) | Switching-time space | Earlier handover at Steps 2 and 3 (100 development speakers) | 21 Oct | On clean prompts Step 3 is as safe as Step 4 (3/100) at 11 instead of 13 NFE; Step 2 fails 7/100. |
| [EXP-004](EXP-004-signal-selection/README.md) | Decision signals (Objective 2) | Signal selection: SNR vs other prompt signals, drift vs other trajectory signals, LTT vs alternatives | 23 Oct | SNR gate matched by spectral flatness; drift matched by a fixed Step 4; LTT justified. |
| [EXP-005](EXP-005-drift-features/README.md) | Decision signals | Richer zero-cost drift features ("Phase 13"; exploration) | 22–23 Oct | Best feature development AUC 0.69 (chance maximum 0.63); confirmation not run. |
| [EXP-006](EXP-006-drift-under-reverb-telephone/README.md) | Decision signals | Drift under reverberation and telephone prompts ("Phase 12", 166 speakers outside the splits) | 22 Oct | Drift adds nothing over SNR (McNemar 0 vs 0). ⚠️ Result files never committed; report only. |
| [EXP-007](EXP-007-three-scheduler-revisions/README.md) | Alternative policies | Three scheduler revisions over Steps 2–8 (offline on EXP-003) | 21 Oct | No revision beats SNR-gated Step 3 (16.00 NFE, 3/100). |
| [EXP-008](EXP-008-novel-algorithms/README.md) | Alternative policies | Novel algorithms: student probe, SNR-band schedule vs the proposal | 23–24 Oct | No candidate beats SNR-only (4/200, 17.04 NFE). |
| [EXP-009](EXP-009-cv-stabilization-t4/README.md) | Calibration (Objective 3) | Cross-validated λ for the stabilization policy (Appendix D.4; offline on EXP-002) | 25 Oct | λ = 0.35–0.50 per fold; 1/200 out-of-fold failures; 18.88 NFE. |
| [EXP-010](EXP-010-t4-full-pipeline-timing/README.md) | Cost (Objective 4) | Full-pipeline timing: 3 policies × 1,000 cases | 20 Oct | Test clean: SNR-only 2.06 s, SNR + drift (λ = 0.50) 2.41 s, Step 8 3.29 s; noisy 3.37 s for all. |
| [EXP-011](EXP-011-followup-t4/README.md) | Verification and robustness | Follow-up tests: smoke test and development pilot (20 development speakers × 11 prompt conditions); full run pending | 25 Oct | Checks pass; EXP-002 reproduced exactly; the 20 dB gate misses 18–22 dB noise; low drift flags noisy prompts (pilot failures 32 → 5 with a low-drift safeguard). |
| [EXP-012](EXP-012-consistency-gate-prelim/README.md) | Proposed solution | Preliminary test of the proposed gates (SNR–Drift Consistency, low-drift veto, guidance contrast, stratified calibration) vs SNR-only and SNR + drift; Quick 60 development speakers × 6 conditions, Full 100 × 11 | 25 Oct (Quick) | No gate beats a calibrated SNR threshold (7/360 failures, 18.71 NFE, exploratory); Definition 3.4 certifies nothing because noise lowers drift; only the low-drift veto passes stratified calibration. |
| [EXP-013](EXP-013-endpoint-extrapolation-prelim/README.md) | Proposed solution | **Drift-Extrapolated Handover:** predict the Step-8 endpoint from the drift's contraction rate and hand it over at Step 4 (EXP-012 Quick set) | not yet run | — |
| [EXP-000](EXP-000-timing-replay/README.md) | Legacy | Replay of the timing arithmetic of the earliest (pre-T4) telemetry | 9 Oct | Arithmetic check only; not a GPU run. |

**Common caveats.**
- The 200-speaker test split was inspected in EXP-002 to EXP-010, so their test-split numbers are development evidence, not an independent held-out test.
- EXP-006 used the 166 speakers outside the splits, so no Seed-TTS test-en speaker remains unseen. A clean held-out test needs new data (for example LibriSpeech test-clean), or must reuse those 166 speakers with that history stated.

**Not included.**
- The "comprehensive evaluation suite" (Shonget-DMOSpeech2 branch `SNRgated`, commits `3580b6d` and `20a37bb`): its Tests 1, 12, 15 and 17 write typed-in values, and its case log is built from the CPU grid (audit Finding 8).
- The CPU phase grids and the 48-instance CPU pilots, because the thesis uses T4 results only. The original CPU problem-validation proof (12 speakers, Appendix D.2) stays as legacy evidence in `research/evidence/legacy/`; EXP-001 repeats its analysis on the T4 data.
