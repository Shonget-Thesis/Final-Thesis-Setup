# 2026-10-25: Colab T4 results imported; follow-up test notebook

Tasks: REPRO-003 (import), EVAL-003 (follow-up tests, not yet run). Requested by a team member: keep only the good Colab GPU tests, disregard CPU runs, and add a notebook for further testing.

## Imported

From Shonget-DMOSpeech2 `Handover Scheduling/colab/` (HEAD 8a7868f), byte-identical, listed with source commit and SHA-256 in [research/evidence/benchmarks/MANIFEST.json](../../../research/evidence/benchmarks/MANIFEST.json):

- signal_selection/ (run at 38ac387 on a Tesla T4; preregistration written before the summary)
- novel_algorithms/ (run at 381ec71 on a Tesla T4)
- drift_features/drift_features_all1000.csv (d₄ matches the T4 grid in all 1,000 cases; its producer script was never committed)
- gpu_full_pipeline_timing_3000.parquet (same run as the CSV already imported)

Already present and unchanged: the T4 grid, the T4 timing CSV and its summaries and report. The migrated code in code/colab/ and code/scheduling/ already matched the source byte for byte.

Excluded, with reasons in the manifest: the comprehensive evaluation suite (typed-in values, CPU case log), the CPU phase grids and pilot, the three-scheduler revision study (run on CPU), and a duplicate zip.

## Follow-up tests (designed, not run)

- [code/colab/run_followup_tests.py](../../../code/colab/run_followup_tests.py) and [DMOSpeech2_Followup_Tests.ipynb](../../../code/colab/DMOSpeech2_Followup_Tests.ipynb); protocol in [EXP-011](../../../research/experiments/EXP-011-followup-t4/README.md).
- Tests: seed-noise floor on the test split; fresh held-out grid on the 166 unused prompt speakers; nine degraded-prompt conditions on those speakers; analysis computed from the CSV files only, with no fallback values.
- The runner reuses core/sampler.py and core/metrics.py from Shonget-DMOSpeech2 pinned at 8a7868f, refuses to run without CUDA, and refuses to run after freezing if the preregistration, split manifests, T4 grid or runner change.
- Found while pinning: the T4 grid was scored before Shonget-DMOSpeech2 commit 656d86f, when SIM read the ground-truth array as 24 kHz regardless of its rate. Mean SIM agrees with the CPU grid (0.8197 vs 0.8201 over 1,600 matched rows), which suggests the ground-truth files are 24 kHz, but this is not confirmed. The runner records each rate and the report flags any that is not 24 kHz.

## Notebook aligned with the team's Colab notebooks

Same steps, dependency pins, checkpoint and Seed-TTS download cells as the novel-algorithm and signal-selection notebooks; results in `MyDrive/dmospeech2_results/followup_t4/`; self-contained setup in every later cell; the full run starts in the background with `nohup`, logs to Drive and prints one `[n/N]` line per minute; Step 8b checks progress without starting a second job. The runner flushes each case to Drive, repairs a half-written last line and re-runs a case cut off mid-write (tested). `.gitattributes` now keeps research/evidence/benchmarks/ byte-exact (the imported files have CRLF line endings), so the recorded hashes hold on CI and in Colab.

## Checks run

- python -m unittest discover -s code/tests: all tests pass (see STATUS for the count); the follow-up tests reproduce the audit's λ̂ = 1.20 and the test-split outcomes from the committed grid, and exercise freeze, tamper detection, the degradations and an end-to-end analysis on synthetic data.
- python scripts/check_workspace.py: passed.
- The notebook and runner have not been executed on a GPU. The DEMAND download URL and archive layout are unverified until the notebook's step 7 runs.

## Pending actions recorded

The team decided to keep SNR + drift as presented at the concept defense and to run EXP-011 after the proposal defense. Both decisions, and the full list of pending actions (Chapter 3 edits before the proposal, the run and Chapter 4 after it, open provenance items), are in the "Pending actions" section of the [October 25 audit](../../../research/audits/2026-10-25/FINDINGS.md).

## Open

- Commit and push before running: the notebook clones the thesis repository from GitHub.
- Run EVAL-003 and commit the preregistration before the tests and the results after them.

## Correction (later on 2026-10-25)

This log describes the 166 Seed-TTS speakers outside the splits as unused or fresh. They were already used in the team's Phase 12 run (22 Oct; research/experiments/EXP-006). No Seed-TTS test-en speaker is unseen; see the correction under Finding 1 of research/audits/2026-10-25/FINDINGS.md.
