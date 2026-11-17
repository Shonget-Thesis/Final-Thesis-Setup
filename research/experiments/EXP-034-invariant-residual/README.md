# EXP-034: the invariant residual against SNR only and Chapter 3's drift

**Status:** prepared and preregistered on 2026-11-17 (Asia/Manila), before any GPU synthesis; freeze `eb1a38d3225a...` after Amendment 1 (first freeze `7c2ec8fca37b...`). Not yet run.

Task EVAL-029 (in progress). Files: [notebook](../../../code/colab/DMOSpeech2_EXP034_Invariant_Residual.ipynb), [runner](../../../code/colab/run_invariant_residual.py), [tests](../../../code/tests/test_invariant_residual_run.py), [preregistration](preregistration.json), [design log](../../../docs/progress/log/2026-11-17-exp034-invariant-residual-setup.md).

- **A development test:** the speakers are development speakers. It is not held-out evidence.
- **Nothing is adopted:** the thesis method is unchanged. Adopting the residual is a team decision.
- **Novelty, as checked:** no published use of the residual was found in three targeted searches ([invariant residual log](../../../docs/progress/log/2026-11-17-invariant-residual.md)). It is built from known mathematics.

## Why this test

**The question** (requested by Rhenel Jhon R. Sajol on 2026-11-17): is the proposed decision rule better than the SNR alone, and better than Chapter 3's SNR + drift?

**What the CPU screens found** (12 development sentences from positions 1–30; screening only):
- **Detection:** the invariant residual spotted a breaking teacher handover with AUC 0.915. Drift reached 0.624 and Chapter 3's SNR 0.649.
- **Policies:** on a held-out half, SNR + residual reached mean UTMOS 3.02, against 2.24 for SNR alone and 2.51 for SNR + drift.
- **Where the difference came from:** the residual caught all 72 hiss prompts at 25 dB that Chapter 3's SNR lets through.

## What is compared

**The two payloads,** both handed over at the commit point t = 0.2929:
- **`CPH8`:** the teacher's estimate, after 4 Euler steps on the 8-interval sway grid. 12 passes.
- **`CPQH`:** its noise-safe version Q(x̂, s₀), the student draft's values in the teacher's time order per mel band, with the draft's prompt frames. 13 passes.

In both, the student restarts at 0.2929 and 0.6173.

**The decision rules** differ only in the signal. The thresholds were fixed before this run, from the CPU screen.

| Rule | Hands over `CPQH` when | Passes |
|---|---|---:|
| `SNR_ONLY` | Chapter 3's SNR (Definition 3.3) < 20 dB | 12 or 13 |
| `SNR_DRIFT` | SNR < 20 dB, or drift d₄ < 0.2252 (low drift flags noise, the screen's direction) | 12 or 13 |
| **`SNR_R`** (proposed) | SNR < 20 dB, or the invariant residual r > 0.3400 | 13 |
| `R_ONLY` | r > 0.3400 | 13 |

**The residual:**

    r = ‖x̂ − Q(x̂, s₀)‖_M / ‖x̂‖_M

It is Chapter 3's drift formula with the previous estimate replaced by the estimate's noise-safe version.

**Reference arms on the same cases:**

| Arm | What it is | Passes |
|---|---|---:|
| `STU` | The released student alone | 4 |
| `STUR2` | The student's draft restarted at 0.2929 and 0.6173 | 3 |
| `W8` | Chapter 3's reference: 16-interval Step 8, student from 0.0761 | 21 |
| `P25` | The paper's setting; variety stage and timing only | 30 |

| Stage | Cases | Measures |
|---|---|---|
| Quality | Development positions 31–60 × 8 prompt conditions = 240 | WER, speaker similarity, UTMOS; the SNR, the spread-weighted SNR, r, drift and the roughness share per case |
| Variety | Positions 41–60 × 12 seeds × clean and `gauss25` | CVf0, aligned CVf0, voiced share, timing spread, melody agreement with `P25`, per arm and per rule |
| Cost | Positions 31–40, clean, each arm and rule on its own | Forward passes counted on both networks; wall-clock seconds |

**Prompt conditions:**
- clean;
- white noise at 15 dB (`noisy15`) and 25 dB (`gauss25`);
- pink noise and high-pass hiss at 25 dB (`pink25`, `hp25`, the screens' noises);
- heavy reverberation (`reverb09`);
- telephone;
- DEMAND babble at 15 dB (`babble15`).

**The speakers:**
- **Not used:** positions 1–30, which hold the CPU screens' sentences.
- **New to every T4 quality run:** positions 41–60.
- **Positions 31–40:** used only in EXP-026.

## Smoke test

A run stops before synthesis if any of these fails, on development speaker 1:
- **Identities:**
  - the 8-interval teacher loop against the shared run (≤ 1e-4);
  - the noise-safe version built from the teacher's own estimate returns it, with r = 0, and takes its prompt frames from the draft;
  - the split student (≤ 1e-5).
- **Each arm and rule run on its own:** it equals the shared run, and its counted forward passes equal the cost table.
- **Reproductions (≤ 0.01):**
  - `CPH8`, `STU` and `W8` against EXP-033's stored scores, clean and `noisy15`;
  - `STU` and `W8` against EXP-031's: clean, `noisy15`, `gauss25`, `reverb09` and `babble15`.

## Hypotheses and verdict (frozen)

| | Rule |
|---|---|
| H1 | `SNR_R` − `SNR_ONLY`: mean ΔUTMOS > 0 with the bootstrap lower bound > 0, ΔWER upper bound ≤ +0.01, Δsimilarity lower bound ≥ −0.01. Per-case differences are averaged per speaker over conditions and bootstrapped over speakers |
| H2 | `SNR_R` − `SNR_DRIFT` under the same conditions |
| H3 | AUC(r) for "`CPQH` needed" (UTMOS(`CPQH`) − UTMOS(`CPH8`) > 0.30) ≥ 0.80, and ≥ drift's AUC in its better direction + 0.10 |
| H4 | `SNR_R`'s wrong clean flags ≤ 15% of clean cases. A wrong flag is a clean prompt sent to `CPQH` where it is not needed. Catching a teacher that breaks on a clean prompt is not counted |
| H5 | On clean prompts, `SNR_R` keeps ≥ 90% of `CPH8`'s variety gain over `STU` on CVf0 and aligned CVf0 (point estimates) |

**Verdict:**
- **`snr_r_better`:** H1–H5 all hold.
- **`snr_r_better_quality`:** H1 holds and any other fails.
- **`snr_only_not_beaten`:** H1 fails.
- **`partial`:** a stage is incomplete.

**Secondary, reported but not part of the verdict:**
- **Re-chosen thresholds:** chosen on positions 31–45 and applied to 46–60.
- **EVAL-027's spread-weighted SNR gate:** below 40 dB, descriptive.
- **The per-case best of the two payloads:** the oracle.

## Amendments

**Amendment 1 (2026-11-17, before any main synthesis):** the runner now loads DEMAND's cafeteria track and records its SHA-256 in `demand_sha256.txt`, as EXP-031 did. The notebook downloads the track, caching it on Drive, and passes its folder to the runner.
- **Why:** the first smoke test on the T4 stopped at `babble15`, because no track was loaded.
- **What that smoke test had already shown:**
  - every identity check passed with a difference of 0.0;
  - every arm and rule run on its own matched the shared run;
  - every counted pass matched the cost table;
  - 7 of the 8 conditions ran on development speaker 1.
- **What did not change:** no arm, threshold, cohort, estimand or hypothesis.
- **The freeze:** re-frozen as `eb1a38d3225a...`; it was `7c2ec8fca37b...`.

## How to run

1. Open the notebook on Colab with a T4 runtime.
2. Run all cells. The notebook clones this repository's branch `rhnl-3`, so the runner, the preregistration and the screen output it hashes must be pushed first.
3. Send back `DMOSpeech2_EXP034_Invariant_Residual_Results.zip`, which is saved to Drive and downloaded.

## Results

Not yet run.
