# 2026-10-25: Reports and results audit

Task: AUDIT-001. Requested by a team member, who said the code in code/ is not run locally because all synthesis runs on Google Colab, and asked for the corrections to go into the reports and results.

## Changes

- New audit: [research/audits/2026-10-25/FINDINGS.md](../../../research/audits/2026-10-25/FINDINGS.md) with a standard-library recompute script, `recompute_gpu_grid.py`, over the committed T4 grid (`research/evidence/benchmarks/gpu_grid_all1000.csv`, SHA-256 249191ec…febd3).
- Every report in research/reports/ (eight files) now opens with a dated audit note listing the corrections that apply to it. The original text below each note is unchanged. ALL_REPORTS_COMPILED.md was not recompiled; it carries its own note.
- STATUS.md: corrected the statements that no synthesis or held-out evaluation had been run and that inference was not smoke-tested; added the audit's main results and blockers.
- research/README.md links the new audit.

## Main findings

1. The 200-speaker test split was inspected before the protocol was fixed; its results are development evidence. 166 unused prompt speakers remain for a fresh test.
2. On the T4 labels, fixed-sequence LTT over the manuscript's grid certifies every threshold (λ̂ = 1.20), and the policy then equals SNR-only (4/200 test failures, 17.04 NFE). The reports' λ̂ = 0.50 is a CPU-label result.
3. d₄ does not separate Step-4 failures on clean development prompts (AUC 0.474, three failures).
4. The 20 dB gate separates the synthetic noisy15 prompts from clean ones completely.

## Not changed

- The manuscript. Chapter 3 and Appendix D do not quote the test-split results or λ̂ = 0.50; the protocol decisions in the audit's recommendations (EVAL-001) need the team and adviser.
- Task REPRO-001 is marked done, but its acceptance criterion (generating scripts, tested environment export) is not met for the CPU grids. Its status is left for the team to decide.
- The implementation migration (code/scheduling/, code/colab/, research/reports/ and the related STATUS, tasks and checker edits) was uncommitted when this audit started and is left as found.

## Checks run

- Addendum (Finding 8): read Shonget-DMOSpeech2/Handover Scheduling at 8a7868f and branch SNRgated; the comprehensive suite at 20a37bb writes typed-in values for Tests 1, 12, 15 and 17. Seed-noise-floor note added to the three reports that cite it.
- python research/audits/2026-10-25/recompute_gpu_grid.py: reproduces the reports' T4 test figures at λ = 0.50; λ̂ = 1.20 on calibration.
- python scripts/render_progress.py, then python scripts/check_workspace.py: passed. The checker skips links inside research/reports/; the audit-note links were resolved separately.
- python -m unittest discover -s code/tests: 14 tests passed.
- latexmk -Werror in paper/: exit 0 (manuscript unchanged).

## Correction (later on 2026-10-25)

This log describes the 166 Seed-TTS speakers outside the splits as unused or fresh. They were already used in the team's Phase 12 run (22 Oct; research/experiments/EXP-006). No Seed-TTS test-en speaker is unseen; see the correction under Finding 1 of research/audits/2026-10-25/FINDINGS.md.
