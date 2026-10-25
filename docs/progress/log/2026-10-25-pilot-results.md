# 2026-10-25: Development pilot results (EXP-001, Steps 1–6b)

Task: EVAL-003. The team ran the follow-up notebook's smoke test and development pilot on a Colab Tesla T4 and downloaded the results zip.

## Recorded

- Results without audio, byte-identical to the zip, in [research/experiments/EXP-001-followup-t4/results/](../../../research/experiments/EXP-001-followup-t4/results/MANIFEST.json): `pilot/`, `smoke/`, and `superseded_freeze_20261025/`. The manifest lists every SHA-256; `code/tests/test_evidence_manifest.py` now checks this folder too, and `.gitattributes` keeps it byte-exact.
- Findings added as Part 6 of [research/reports/ALL_REPORTS_COMPILED.md](../../../research/reports/ALL_REPORTS_COMPILED.md). The paper is not changed.

## Main findings (development evidence, 20 speakers per condition)

1. The Section 3.3.5 checks pass, and the pilot reproduces the committed T4 grid exactly (160/160 failure labels, all differences 0).
2. The 20 dB SNR gate misses 18–22 dB white noise: the estimate reads 2–3 dB high, and 30–50% of those prompts fail at Step 4.
3. Drift as a stopping signal adds nothing (1 failure prevented, 2 caused); as a low-drift safeguard (d₄ < 0.30 → no early handover) it cuts pilot failures from 32 to 5, while sending 1/20 clean pilot prompts and 1/100 and 5/200 clean T4 development and calibration prompts back to Step 8. Exploratory; to be fixed on development data and tested once on held-out speakers.
4. The seed-noise sample gives 4/10 speakers, not 2.5%.
5. 15.6 s per case on the T4: the full run is about 9.6 hours.

## Incident

Step 7 (freeze) and the first case of Step 8 also ran (one held-out speaker, `common_voice_en_10119832-common_voice_en_10119847`, clean). That freeze is void; its files are kept as a record. On Google Drive, move `preregistration.json`, `fresh_manifest.json` and `fresh_grid.csv` out of `followup_t4/` before freezing again, and exclude that speaker from the held-out set.

## Checks run

- Imported files compared with the zip entries: 12/12 identical.
- python scripts/check_workspace.py and python -m unittest discover -s code/tests: see the commit.

## Open

- Add the moderate-noise conditions to calibration or revise the gate, and add the low-drift safeguard policy to the runner, before the real freeze.
- Chapter 3 and appendix edits from these findings (not made; the team will decide after review).
