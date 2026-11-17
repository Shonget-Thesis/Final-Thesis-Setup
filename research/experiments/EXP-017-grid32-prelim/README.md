# EXP-017: Chapter 3 rehearsal on two teacher grids (preliminary)

**Status:** run on 2026-11-02 on a Colab Tesla T4 (thesis `c3f6cbe`, Shonget-DMOSpeech2 `8a7868f`); 360 of 360 cases; results in [results/](results/MANIFEST.json). Task EVAL-010 (review).
- Notebook: [DMOSpeech2_Grid32_Prelim.ipynb](../../../code/colab/DMOSpeech2_Grid32_Prelim.ipynb)
- Runner: [run_grid32_prelim.py](../../../code/colab/run_grid32_prelim.py)
- Tests: [test_grid32.py](../../../code/tests/test_grid32.py)

One run that rehearses Objectives 1, 3 and 4 of Chapter 3 on the development speakers, so that no other test is needed before the proposal (requested by the team, 2026-11-02).

## Why

- **What the paper says.** The DMOSpeech 2 paper (arXiv:2507.14988v1) reports a single teacher-guided setting: "tswitch = 0.25, with teacher handling 14 steps and student handling 2 steps, for a total of 16 steps" (Section 4.1, p. 9). Its Algorithm 2 (p. 8) runs the teacher to the last sway-grid point at or below t_switch.
  - Only a 31- or 32-interval grid gives 14 steps at 0.25, and the teacher's full schedule is 32 steps. The paper does not state the grid size.
  - Its teacher-guided evaluation code is not public. The official repository has one branch and no tags, and its pull requests, issues and model repository contain none (checked 2026-11-02).
- **What the thesis runs.** The released sampler on its default 16-interval grid, which stops at the first point above s: 0.25 becomes Step 8, 21 forward passes. The released code's default is 0.07, which is Step 4.
- **Team decision (2026-11-02).** Follow the published paper where it is specified, and test both grids.

## Outputs (13 per input, all synthesized and scored in the same run)

| Output | What it is | Teacher evaluations + student steps | Forward passes |
|---|---|---|---:|
| `g16_4`, `g16_5`, `g16_6` | 16-interval Steps 4, 5, 6 (Step 4 is the code default) | 5 / 6 / 7 + 3 | 13 / 15 / 17 |
| `g16_8` | 16-interval Step 8, the thesis reference | 9 + 3 | 21 |
| `s007` … `s020` | 32-interval, s = 0.07 / 0.10 / 0.15 / 0.20, the paper's rule | 8 / 10 / 12 / 14 + 3 | 19 / 23 / 27 / 31 |
| `s025` | 32-interval, s = 0.25 (14 teacher steps), this grid's reference | 15 + 3 | 33 |
| `paper` | The paper's 14 + 2, released handover (student restart j = 2) | 15 + 2 | 32 |
| `alg2` | Algorithm 2 as written: the student continues from the teacher's state at 0.2270, then steps at 0.6173 (assumed; the paper does not state it) | 14 + 2 | 30 |
| `teacher32` | Teacher only, 32 steps (F5-TTS) | 32 + 0 | 64 |
| `student4` | Student only, 4 steps from noise (DMOSpeech 2's released student path) | 0 + 4 | 4 |

- **Fixed throughout:** guidance γ = 2 with no stochastic term, and the same seeds; the student is as in the thesis unless stated.
- **Recorded per input:** the prompt SNR, the drift on both grids (d₁..d₈ and d₁..d₁₄), and the measured unit times (preparation, one guided evaluation, one student step, the vocoder, audio length, peak GPU memory).
- **Failures:** Definition 3.1 (`code/scheduling/failure_config.json`).

## Data, hardware and checks

- **Data:** the EXP-012 Quick set: 60 development speakers × clean, noisy15 and Gaussian noise at 18, 20, 22 and 25 dB = 360 cases.
- **Hardware:** Colab Tesla T4 only. The runner stops on another GPU, because the student's random draws differ between GPU models.
- **Checks:**
  - the smoke test checks that the grid-agnostic teacher loop gives the sampler's endpoints on the 16-interval grid;
  - the analysis compares the fresh 16-interval scores with the stored T4 labels for every case.
- **Preregistration:** `preregistration.json`, with every setting and analysis, is written before any synthesis.

## Analysis (from the CSV only; schedulers are replayed on the recorded outputs)

**On each grid:**

- **Objective 1:**
  - WER, SIM and UTMOS, and the added-failure rate of every fixed switching time with Clopper–Pearson 95% intervals, per condition;
  - the oracle (the cheapest non-failing candidate) per condition and its share earlier than the reference;
  - Spearman correlation of the prompt SNR and of the drift at each candidate with the oracle choice;
  - the SNR gate by Chapter 3's rule (the lowest SNR above which the earliest candidate fails at most 5%);
  - signal plots.
- **Objective 4:** all compared methods of Chapter 3. These are the fixed reference and alternatives, SNR-only (20 dB and calibrated), the SNR gate with a fixed later step (5 and 6), the full policy (Definition 3.4; 20 dB gate and the Objective 1 gate), stabilization only, a cost-matched random rule and the oracle, plus SNR + TAS.
  - **Calibration:** fixed-sequence Learn-Then-Test (α = 0.05, δ = 0.10). The λ grid is fixed in advance: 0.20 → 1.20 on the 16-interval grid (Chapter 3), and 0.01 → 1.20 geometric on the 32-interval grid. Validation is 5-fold speaker-grouped cross-validation (seed 2026), out of fold.
  - **Reported:**
    - failures with exact intervals, mean forward passes, the saving Δ and the share of the oracle saving η, with speaker-bootstrap intervals;
    - runtime and real-time factor, and the worst condition;
    - exact McNemar for the drift signal: the full policy against SNR-only, both fixed-later-step controls, stabilization only and the random rule;
    - Wilcoxon on speaker-level differences with Holm correction, and TOST equivalence with margins of 1 WER point and 0.02 SIM;
    - robustness: calibrate on clean prompts, test on noisy prompts.
- **Objective 3, rehearsed:** 100 random 80/20 speaker re-splits, and calibration sizes of 50, 100 and 200 cases. The report gives the certified share, the test risk and the saving.

**Also:** the paper's settings, teacher-only and student-only against the thesis reference (DMOSpeech's own quality, as the panel asked at the concept presentation, 60:24).

## What this test does not do

- **Certify the guarantee.** The calibration and test splits are reserved for the full study after the proposal. Here, cross-validation and re-splits reuse the same 60 speakers, and each speaker contributes six correlated cases.
- **Fix the paper's unstated details.** The 32-interval grid and `alg2`'s second student time are inferred or assumed.
- **Time each method end to end.** Runtime is composed from measured unit times.
- **Use new speakers or real-world noise.**

## Cost

About 5 hours for the full run on a T4, and about 5.5 hours with setup and the smoke test. This is an estimate: the smoke test prints the measured seconds per case and the projected total.

## Relation to other tasks

EVAL-009 (j = 1 versus j = 2 on the 16-interval grid) is a separate check. The `paper` output here adds one j = 2 point on the 32-interval grid.

## Results (360 of 360 cases, 2026-11-02, Tesla T4)

From [GRID32_REPORT.md](results/GRID32_REPORT.md). The preregistration (`7ad1b65f…`) was written at 05:10 UTC, before the first case at 05:11. The run took 5 h 27 min, 54.7 s per case. Development evidence.

**Checks.**
- The fresh 16-interval Step 4, 5, 6 and 8 scores equal the stored T4 labels in all 360 cases (max |Δ| 0), and d₄ is identical.
- The grid-agnostic teacher loop matches the sampler exactly (16 smoke checks, max |diff| 0).
- Re-running the analysis from the CSV reproduces the report; it is identical apart from line endings.

**The problem holds on both grids.** Fixed early switching fails mainly on noisy prompts:
- 16-interval Step 4 fails 2 of 60 clean inputs against 126 of 300 noisy ones;
- 32-interval s = 0.07 fails 2 of 60 against 129 of 300.

The cheapest safe step varies per input. A perfect per-input choice would cost 14.76 passes instead of 21 (−29.7%) on the 16-interval grid, and 21.89 instead of 33 (−33.7%) on the 32-interval grid.

**The failure rule's noise floor on noisy prompts.** The two references (16-interval Step 8 and 32-interval s = 0.25) have equal mean quality on noisy prompts (SIM +0.0000, WER +0.007; not significant). Yet they flag each other on 81 and 78 of 300 noisy inputs, against 0 and 2 of 60 clean ones. About a quarter of any noisy-prompt failure count therefore reflects output variability rather than damage.

**Schedulers** (out of fold; Learn-Then-Test with α = 0.05, δ = 0.10):

| Grid | Full policy (Definition 3.4, 20 dB) | SNR-only, calibrated gate | Full policy, Objective 1 gate | SNR + TAS | Oracle (hindsight) |
|---|---|---|---|---|---|
| 16-interval (vs 21 passes) | 0 failures, 0.0% saving (nothing certified in any fold) | 7 (1.9%), 10.9% | 4 (1.1%), 7.0% | 12 (3.3%), 14.0% | 0, 29.7% |
| 32-interval (vs 33 passes) | 8 (2.2%), 1.8% | 11 (3.1%), 13.4% | 6 (1.7%), 9.1% | 13 (3.6%), 14.0% | 0, 33.7% |

- **The drift signal (Definition 3.4) adds little.**
  - On the 16-interval grid no threshold is certified, as in EXP-012.
  - On the 32-interval grid it saves 1.8% and does not differ from a cost-matched random rule (exact McNemar p = 0.77).
  - The calibrated SNR gate does most of the work. SNR + TAS saves the most of any method within the 5% limit, on both grids.
- **Robustness:** calibrated on clean prompts only, the full policy fails on 28.0% (16-interval) and 5.0% (32-interval) of noisy prompts, and stabilization only on 42.0% and 7.3%.
- **Objective 3 rehearsal** (100 re-splits): on the 16-interval grid the full policy is never certified. The calibrated SNR gate is always certified, and its test risk exceeds 5% in 13% of splits.

**The paper's settings are worse than the thesis reference:**
- `paper` (14 + 2, 32 passes) adds 140 failures and `alg2` (Algorithm 2 as written, 30 passes) adds 172; on clean prompts that is 7 and 11.
- Noisy WER is 0.34 and 0.41, against 0.076 for the thesis reference.
- `alg2` has lower speaker similarity than `paper` (−0.013, p = 3.5 × 10⁻⁸), consistent with its student starting at a level it was not trained on.
- The paper's teacher schedule with the thesis handover (`s025`, 33 passes) matches the thesis reference in quality (no significant difference) but uses 57% more passes.

**Teacher-only and student-only:**
- Teacher-only (64 passes) has the worst quality: noisy WER 0.97, and clean UTMOS 3.27 against 3.92.
- Student-only (4 passes, 0.70 s) matches the reference on clean prompts and is better on noisy ones: SIM +0.077, WER −0.027, UTMOS 2.69 against 1.46. Under Definition 3.1 it still adds 59 failures, 8 of them on clean prompts.
- The study does not measure prosodic diversity, which is the reason DMOSpeech 2 uses the teacher.

**Runtime** (composed from measured unit times on the T4):

| Output | Time | RTF |
|---|---:|---:|
| 16-interval Step 8 | 3.23 s | 0.69 |
| 16-interval Step 4 | 2.04 s | 0.44 |
| Teacher-only | 9.65 s | 2.07 |
| Student-only | 0.70 s | 0.15 |
