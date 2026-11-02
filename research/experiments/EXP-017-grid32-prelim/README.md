# EXP-017: The switching time on a 32-interval teacher grid (preliminary)

**Status:** designed 2026-11-02, not run. Task EVAL-010. Notebook: [DMOSpeech2_Grid32_Prelim.ipynb](../../../code/colab/DMOSpeech2_Grid32_Prelim.ipynb); runner: [run_grid32_prelim.py](../../../code/colab/run_grid32_prelim.py); tests: [test_grid32.py](../../../code/tests/test_grid32.py).

## Why

- **What the paper says.** The DMOSpeech 2 paper (arXiv:2507.14988v1) reports a single teacher-guided setting: "tswitch = 0.25, with teacher handling 14 steps and student handling 2 steps, for a total of 16 steps" (Section 4.1, p. 9). It reports no other switching time. Its Algorithm 2 (p. 8) runs the teacher to the last sway-grid point at or below t_switch. Fourteen steps at 0.25 correspond to a 32-interval sway grid (t₁₄ = 0.2270); the paper does not state the grid size.
- **What the thesis runs.** The released sampler on its default 16-interval grid, which stops at the first point above s: 0.25 becomes Step 8 (t₈ = 0.2929), 21 forward passes. All earlier experiments use this grid.
- **Context.** The released code's `generate()` defaults to `teacher_stopping_time=0.07` (8 steps, 13 passes; github.com/yl4579/DMOSpeech2, `src/infer.py`). The official demo runs the paper's 16-step setting with `teacher_steps=24, teacher_stopping_time=0.3, student_start_step=2`. On 2026-11-02 the team decided to anchor its claims on the paper. This test checks whether the problem-validation pattern holds on the paper's grid, and it measures the paper's own 14 + 2 setting.

## Design

The grid and the stopping rule change for every 32-interval output. Guidance (γ = 2, no stochastic term), seeds and the student stay as in the thesis. Failures follow Definition 3.1 (`code/scheduling/failure_config.json`).

`alg2` follows the paper where it is specified. On 2026-11-02 the team decided to follow the published paper where the released code differs and the evaluation code is not public. Algorithm 2 hands the teacher's state, not its endpoint estimate, to the student at s₁ = t_k*, and re-noises only between student steps. The paper does not give the student's second time, so `alg2` uses 0.6173, the released student's last trained level; this is the one assumption.

| Output | Grid and stop | Teacher steps | Student steps | Forward passes |
|---|---|---:|---:|---:|
| `g16_4`: code default | 16 intervals, first point above 0.07 | 4 | 3 | 13 |
| `g16_8`: thesis reference | 16 intervals, first point above 0.25 | 8 | 3 | 21 |
| `s007`, `s010`, `s015`, `s020` | 32 intervals, last point ≤ 0.07 / 0.10 / 0.15 / 0.20 | 7 / 9 / 11 / 13 | 3 | 19 / 23 / 27 / 31 |
| `s025`: reference on this grid | 32 intervals, last point ≤ 0.25 | 14 | 3 | 33 |
| `paper`: the paper's 14 + 2, released handover | 32 intervals, last point ≤ 0.25; student restart j = 2 | 14 | 2 | 32 |
| `alg2`: the paper's Algorithm 2 as written | 32 intervals, last point ≤ 0.25; the student continues from the teacher's state at 0.2270, then steps at 0.6173 | 14 | 2 | 30 |

- **Data:** the EXP-012 Quick set: 60 development speakers × clean, noisy15 and Gaussian noise at 18, 20, 22 and 25 dB = 360 cases. All 8 outputs are synthesized and scored in the same run.
- **Hardware:** Colab Tesla T4 only. The runner stops on another GPU, because the student's random draws differ between GPU models (EXP-015).
- **Reproduction checks:**
  - the new grid-agnostic teacher loop, run on the released 16-interval grid, must give the sampler's Step-4 and Step-8 endpoints (smoke test);
  - the fresh `g16_4` and `g16_8` scores are compared with the stored T4 labels for all 360 cases.
- **Preregistration:** `preregistration.json` is written before any synthesis.

## Questions (descriptive; no pass/fail criteria)

1. On the 32-interval grid, do fixed early switching times fail mainly on noisy prompts and rarely on clean ones, as on the 16-interval grid (Appendix D)? Failures per output and condition are counted against `s025`.
2. Does the cheapest safe switching time vary between inputs? The report gives its distribution in hindsight and the mean cost against 33 passes.
3. How do the paper's 14 + 2 setting, in both versions (`alg2`, `paper`), and the code default (`g16_4`) compare in quality with the thesis reference (`g16_8`)? The report gives mean WER, SIM and UTMOS, paired Wilcoxon tests, and added failures. `alg2` against `paper` isolates the handover mechanism.
4. How do the adaptive schedulers compare on the 32-interval grid, out of fold? The candidates are:
   - SNR-only with a 20 dB gate;
   - SNR-only with a gate calibrated by Learn-Then-Test;
   - stabilization only;
   - **SNR + drift as in Definition 3.4 (the thesis method)**;
   - SNR + TAS (one-sided Mahalanobis on log d₁..d₇, fitted on clean training prompts).

   Each picks, per input, one of `s007` to `s020` or the `s025` reference. The thresholds are certified by fixed-sequence Learn-Then-Test (α = 0.05, δ = 0.10) on the training speakers of a 5-fold speaker-grouped cross-validation (seed 2026). The report gives out-of-fold failures against `s025`, mean forward passes, the worst-condition rate and exact McNemar against SNR-only at 20 dB, beside the reference and the per-input best in hindsight. This is a replay from the recorded SNR, drift and candidate scores, not a live run, so it adds no GPU time.

## What this test does not reproduce

- **The student's second time in Algorithm 2.** The paper does not state it; `alg2` assumes 0.6173. The released student was distilled at four fixed levels, so its first step at 0.2270 is off its training levels; if that hurts quality, it is a finding about the paper's description.
- **A stated grid size.** The 32-interval grid is inferred: only 31 or 32 intervals give 14 steps at 0.25 under Algorithm 2's rule, and the teacher's full schedule is 32 steps. The paper does not state it.
- **The authors' evaluation.** No public branch, pull request, issue or model repository contains the paper's teacher-guided evaluation code (checked 2026-11-02).
- **New speakers.** The 60 speakers are development speakers already used in EXP-012 to EXP-016, so the results are development evidence.

## Cost

About 3.5 hours for the full run on a T4, and about 4 hours with setup and the smoke test. This is an estimate from the 10–13 s per case measured in EXP-013 and EXP-014 for 2–3 scored outputs; this run scores 9 outputs per case. The smoke test prints the measured seconds per case and the projected total.

## Relation to other tasks

EVAL-009 (j = 1 versus j = 2 on the 16-interval grid) is a separate check. The `paper` output here adds one j = 2 point on the 32-interval grid and does not replace it.

## Results

Not run yet. Import the results zip into `results/` with a SHA-256 manifest, as for EXP-015.
