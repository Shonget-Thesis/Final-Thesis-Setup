# 2026-11-17: The invariant residual, a replacement for Chapter 3's drift (local screen)

Follow-up to the [CP-QH log](2026-11-17-commit-point-quantile-handover.md); tracked under EVAL-028.
- **The request:** Rhenel Jhon R. Sajol asked for a novel algorithm that keeps the SNR (modified or not) and replaces drift with something new, then combines the two.
- **Status:** a local CPU screen with rules fixed before the run. Screening only: nothing is adopted, and no adviser has reviewed it.

## The changed term

**Chapter 3's stabilization score** (Definition 3.4, "drift"):

    d_i = ‖ x̂(t_i) − x̂(t_{i−1}) ‖_M / ( ‖ x̂(t_i) ‖_M + ε )

It asks whether the teacher's estimate has stopped changing.

**The problems found in this project:**
- **Noise makes the estimate settle early (EXP-012).** So d says "settled" exactly when the teacher is copying the prompt's noise.
- **It is the solver's step error.** Up to a factor, d is the teacher's local truncation error ([adaptive teacher log](2026-11-16-adaptive-teacher-screen.md)), a signal adaptive solvers already use.
- **Its stopping job is now done by the commit point,** t = 0.2929 (EXP-033).

**The replacement, the invariant residual:** the previous estimate is replaced by the current estimate's condition-safe version, Q(x̂, s₀). That is the quantile handover: in each band, the student draft's values in the teacher's time order.

    r = ‖ x̂ − Q(x̂, s₀) ‖_M / ( ‖ x̂ ‖_M + ε )

**What it measures:** the share of the teacher's estimate that lies outside its condition-safe part, meaning its levels against the student's own at the same ranks.
- **It rises with what only the teacher copies,** such as spread-out background, which the mode-seeking student drops.
- **It ignores what both models copy,** such as reverberation or band-limiting. That part cancels.
- **It ignores melody and timing.** They live in the time order, which Q keeps. A test checks this: r is zero when the draft's values are any time-rearrangement of the teacher's.
- **The known math underneath:** r² × ‖x̂‖² is the sum, over bands, of the squared Wasserstein-2 distances between the teacher's and the draft's value distributions.

**The algorithm with the SNR:** Chapter 3's structure, one threshold per signal, the same for every prompt.
1. The teacher runs to t = 0.2929 on the 8-interval sway grid: 10 passes.
2. The student makes its one-pass draft s₀: 1 pass.
3. If Ŝ ≥ θ and r ≤ λ, hand over x̂ (`CPH8`).
4. Otherwise hand over Q(x̂, s₀) with the draft's prompt frames (CP-QH).
5. The student restarts at 0.2929 and 0.6173: 2 passes.

**Cost:** 13 passes, against 21 for Chapter 3's reference (W8) and 30 for the paper's setting.

## The screen (`code/analysis/invariant_residual_screen.py`)

**Cases:** 168.
- **Sentences:** the 12 shortest clean EXP-033 quality sentences. The 6 shortest form the calibration half; the 7th to 12th, the test half.
- **Conditions:** 7 prompt conditions, two seeds each:
  - clean;
  - white noise at 15 and 25 dB;
  - pink noise at 25 dB;
  - high-pass hiss at 25 dB;
  - heavy reverberation;
  - telephone.

**Per case:**
- Chapter 3's SNR (Definition 3.3) and the spread-weighted SNR;
- drift at the commit point, from the estimates at steps 3 and 4;
- r, and the earlier roughness share;
- UTMOS of both payloads.

"QH needed" means CP-QH beats `CPH8` by more than 0.30 UTMOS. 103 of 168 cases qualify.

| Pre-set rule | Result |
|---|---|
| R1. AUC(r) ≥ 0.80 and ≥ drift's AUC + 0.10, with drift in its better direction | **Pass:** 0.915 against 0.624 |
| R2. AUC(r) ≥ 0.75 among cases Chapter 3's SNR lets through (≥ 20 dB) | **Pass:** 0.935 (80 needed, 48 not) |
| R3. With λ from the calibration half, SNR + r on the test half gains ≥ 0.05 UTMOS over SNR alone, sending ≤ 15% of clean cases to CP-QH | **Fail:** +0.78 UTMOS, but 2 of 12 clean cases (16.7%) |

**Verdict:** fails R3 as written.

**The two clean cases behind the failure:**
- **Both come from one sentence where the teacher breaks on a clean prompt:** `CPH8` scored 2.48 and 1.39 UTMOS there, against 4.40 and 3.64 for CP-QH.
- **The same holds for the calibration half:** its one clean flag was also a case where CP-QH was better.
- **So all three clean flags were correct catches.** That is the "risk from the model itself" Chapter 3 meant the stabilization score to catch. The rule counted them as a variety cost.
- **The verdict is not changed after the fact.** A later test should count only wrong clean flags, and that rule must be set before the run.

### Detection (all 168 cases; AUC for "QH needed")

| Signal | AUC |
|---|---:|
| **Invariant residual r** | **0.915** |
| Spread-weighted SNR (low reads as noisy) | 0.866 |
| Roughness share (earlier probe) | 0.734 |
| Chapter 3's SNR (low reads as noisy) | 0.649 |
| Drift, better direction (low drift reads as noisy) | 0.624 |

### The policy on the test half (6 sentences, 84 cases; λ = 0.340 from the calibration half)

| Policy | Mean UTMOS | Clean cases sent to CP-QH |
|---|---:|---:|
| `CPH8` for every prompt | 2.12 | 0% |
| SNR only: CP-QH below 20 dB | 2.24 | 0% |
| SNR + drift, calibrated the same way (flags low drift) | 2.51 | 0% |
| **SNR + r** | **3.02** | 16.7% (both correct) |
| r alone | 3.02 | 16.7% |
| CP-QH for every prompt | 3.03 | 100% |
| Per-case best of the two (oracle) | 3.05 | — |

**Harmful cases missed** (CP-QH needed, `CPH8` handed over):

| Half | SNR alone | SNR + r |
|---|---:|---:|
| Calibration | 41 of 55 | 5 of 55 |
| Test | 39 of 48 | 2 of 48 |

### By condition (both halves)

| Condition | Chapter 3's SNR, median | r, median | Flagged by r at λ | QH needed | UTMOS `CPH8` → CP-QH |
|---|---:|---:|---:|---:|---|
| Clean | 34.8 dB | 0.23 | 3 of 24 | 6 of 24 | 3.74 → 3.92 |
| White noise, 15 dB | 17.9 dB | 0.62 | 24 of 24 | 19 of 24 | 1.27 → 2.43 |
| White noise, 25 dB | 27.1 dB | 0.58 | 24 of 24 | 24 of 24 | 2.01 → 3.89 |
| Pink noise, 25 dB | 27.2 dB | 0.48 | 24 of 24 | 21 of 24 | 2.56 → 3.98 |
| High-pass hiss, 25 dB | 27.1 dB | 0.54 | 24 of 24 | 24 of 24 | 2.20 → 3.78 |
| Heavy reverberation | 18.6 dB | 0.31 | 6 of 24 | 5 of 24 | 1.52 → 1.62 |
| Telephone | 39.1 dB | 0.19 | 2 of 24 | 4 of 24 | 1.53 → 1.66 |

**Reading:**
- **What the SNR misses, r catches:** Chapter 3's SNR reads all three 25 dB hiss conditions as about 27 dB and lets them through. r flags all 72, and the teacher handover fails on them.
- **r leaves alone conditions both models copy:** reverberation and telephone.
- **Drift does not separate the conditions.**
- **The SNR adds nothing here:** r alone does as well as SNR + r.

## Novelty, as checked

| Part | Status |
|---|---|
| Chapter 3's two-signal structure, thresholds and calibration | Chapter 3 |
| The building blocks: quantile mapping and the Wasserstein-2 distance between one-dimensional distributions | Known mathematics |
| A teacher–student gap as a routing signal | Tried in this project as EXP-023's agreement routing (the whole Step-0 gap; did not advance) |
| **r: the teacher's distance from its own condition-safe version** (timing removed, only the levels that a condition can change kept) **as the handover signal** | **Not found** in three targeted searches (2026-11-17): per-band teacher–student distribution distances as a routing or stopping signal; residuals after quantile mapping as a detector; condition-leakage scores for a handover |

**In the requested framing:** Chapter 3's formula with one changed term. The previous estimate becomes the condition-safe projection, derived from the invariance theory note. It changes what the score measures, from convergence to condition leakage.

## Caveats

- **CPU screening only:** twelve development sentences, two seeds per condition, and UTMOS as the only quality measure.
- **Not measured:** pitch variety per case. The cost of sending a clean prompt to CP-QH is variety (51–75% of `CPH8`'s gain), which per-case UTMOS does not show.
- **Synthetic noises only:** babble is untested here.
- **Several unneeded flags:** 17 of the test half's 36 not-needed cases were flagged, mostly white noise at 15 dB and reverberation, where both payloads fail or tie. They cost no quality.
- **r reuses the draft and the QH payload,** so it costs no extra pass beyond CP-QH's one.

## Next

A preregistered T4 run on new development speakers, with the clean-flag rule counting only wrong flags, set before synthesis.
- **Policies:** SNR + r against SNR + drift (Chapter 3), SNR only, and CP-QH everywhere.
- **Conditions:** the seven above plus DEMAND babble.
- **Measures:** WER, speaker similarity, UTMOS and pitch variety.

## Checks run

- **The screen:** 168 cases, exit 0. Output: `research/experiments/EXP-033-cheaper-teacher-path/offline/invariant_residual_screen.json`.
- **Tests:** `code/tests/test_invariant_residual.py`, 6 tests:
  - r is zero for identical estimates and for any time-rearrangement of the draft's values;
  - an added floor raises r monotonically;
  - the stabilization score;
  - the AUC;
  - the calibration's clean budget.
- **Workspace check:** `python scripts/check_workspace.py` PASSED.
- **Full suite:** `python -m unittest discover -s code/tests` ran 419 tests, OK.
- **Manuscript:** not changed, so not rebuilt.
