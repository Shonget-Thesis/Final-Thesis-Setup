# EXP-006: Drift under reverberation and telephone-band prompts ("Phase 12")

**Run:** 2026-10-22, Google Colab Tesla T4, preregistered. Runner `code/colab/run_drift_justification_benchmark.py` (reference copy of Shonget-DMOSpeech2 branch `SNRgated`, commit `d72e84f`).

**What:** does drift add value over the SNR gate on corruptions that an energy-based SNR estimate may miss? **Data: the 166 Seed-TTS test-en speakers that are in no split.** Conditions: clean, reverberation (RT60 0.6 s), and 8 kHz telephone with 8-bit μ-law. Policies frozen before synthesis, all handing over at Step 5: SNR-only (Ŝ ≥ 20 dB), drift-only (d₄ > 0.30), and SNR AND drift. Primary hypothesis: SNR AND drift has fewer failures than SNR-only on an eligible condition (McNemar, Holm-corrected).

**Data:** ⚠️ **the result files were never committed.** Only the summary in Shonget-DMOSpeech2 `Handover Scheduling/reports/COLAB_RESULTS_CONSOLIDATED_REPORT.md` (commit `6a7917f`, Section 9) exists. The outputs may still be on the team's Google Drive.

## Key results (as reported; not recomputed here)

| Condition | Mean Ŝ | Mean d₄ | Step-5 failures | SNR-only risk / NFE | Drift-only | SNR + drift |
|---|---:|---:|---:|---|---|---|
| clean | 37.1 dB | 0.601 | 0.6% | 0.6% / 15.18 | 0.6% / 15.11 | 0.6% / 15.29 |
| reverb 0.6 s | 24.4 dB | 0.529 | 8.4% | 6.0% / 16.95 | 8.4% / 15.22 | 6.0% / 17.10 |
| telephone | 40.9 dB | 0.480 | 4.8% | 4.8% / 15.00 | 4.8% / 15.14 | 4.8% / 15.14 |

- Verdict: **not supported.** Adding drift to SNR changed no decision on any failing utterance (McNemar 0 vs 0 on every condition).
- Reverberation does degrade early handover (14/166 fail at Step 5, 13 of them on SIM), but neither SNR nor any drift step predicts which ones (AUC 0.42–0.59).

## Caveats

- **This experiment used the 166 speakers outside the splits.** They are therefore not unseen and cannot serve as an untouched held-out test set (see EXP-011).
- Numbers are from the team's report; without the result files they cannot be checked here.
- Its preregistration used 0.025 as the "seed-noise floor", a value no committed output supports (audit Finding 8).
