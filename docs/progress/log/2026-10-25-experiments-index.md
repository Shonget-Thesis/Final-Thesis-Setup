# 2026-10-25: Experiment index, three more T4 experiments, and a held-out correction

Requested by a team member: list every Colab GPU experiment in research/experiments/, ordered so the sequence makes sense, after checking the Shonget-DMOSpeech2 history (113 commits, all branches).

## Changes

- **Index and ordering.** [research/experiments/README.md](../../../research/experiments/README.md) lists EXP-000 to EXP-011 in research-logic order: problem validation, switching-time space, decision signals, alternative policies, calibration, cost, verification. Run dates are in the table. The existing folders were renumbered (the follow-up pilot folder by `git mv`); every reference was updated.
- **New experiments found in branches `early-handover-dev` and `SNRgated`:**
  - EXP-003, earlier handover at Steps 2–3 (T4, 21 Oct): results imported from `cf2ac90` into research/evidence/benchmarks/early_handover/. Its 800 reused rows match the T4 grid exactly.
  - EXP-007, three scheduler revisions (offline on EXP-003, 21 Oct): 16 files imported into research/evidence/benchmarks/three_scheduler_revisions/, identical to the run's archive (earlier mislabelled a CPU experiment; it is an offline analysis of GPU data).
  - EXP-006, "Phase 12" drift under reverberation and telephone prompts (T4, 22 Oct): its result files were never committed, so only the report summary is cited.
  - Reference copies of their runners in code/colab/ (`run_early_handover_dev_gpu.py`, `early_handover_signals.py`, `run_three_scheduler_revisions_dev.py`, `run_drift_justification_benchmark.py`), checked for typed-in results (none).
- **Offline analyses:** EXP-001 problem validation (`code/analysis/problem_validation_t4.py`) and EXP-009 cross-validation for Appendix D.4 (`code/analysis/cv_stabilization_t4.py`).
- **Evidence manifest:** 23 new files listed with SHA-256; the test covers them. The drift-features producer note was corrected (its script is on branch `SNRgated`, `e15cd55`).

## Correction

The October 25 audit, report Part 6, STATUS, EXP-011 and the follow-up notebook described the 166 Seed-TTS speakers outside the splits as unused, a fresh held-out set. Phase 12 already used them (the team's Chapter 3 master report says so in §4.10.2). All those places are corrected, and the two earlier logs carry a correction note. No Seed-TTS test-en speaker is unseen; the held-out data (new data such as LibriSpeech test-clean, or the 166 with their history stated) is a team and adviser decision before the protocol is frozen.

## Checks run

- python scripts/check_workspace.py (including every local link) and python -m unittest discover -s code/tests: see the final run in this session.
- Imported files compared with their sources: early handover 7 files from `cf2ac90`; three scheduler 16/16 identical to the archive.
- The follow-up notebook was rebuilt from its builder; the builder reproduced the previous notebook exactly before the text change.

Nothing in paper/ changed in this step.
