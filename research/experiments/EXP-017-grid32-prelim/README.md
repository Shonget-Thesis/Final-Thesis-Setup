# EXP-017: The switching time on a 32-interval teacher grid (preliminary)

**Status:** designed 2026-11-02, not run. Task EVAL-010. Notebook: [DMOSpeech2_Grid32_Prelim.ipynb](../../../code/colab/DMOSpeech2_Grid32_Prelim.ipynb); runner: [run_grid32_prelim.py](../../../code/colab/run_grid32_prelim.py); tests: [test_grid32.py](../../../code/tests/test_grid32.py).

## Why

- **What the paper says.** The DMOSpeech 2 paper (arXiv:2507.14988v1) reports a single teacher-guided setting: "tswitch = 0.25, with teacher handling 14 steps and student handling 2 steps, for a total of 16 steps" (Section 4.1, p. 9). It reports no other switching time. Its Algorithm 2 (p. 8) runs the teacher to the last sway-grid point at or below t_switch. Fourteen steps at 0.25 correspond to a 32-interval sway grid (t₁₄ = 0.2270); the paper does not state the grid size.
- **What the thesis runs.** The released sampler on its default 16-interval grid, which stops at the first point above s: 0.25 becomes Step 8 (t₈ = 0.2929), 21 forward passes. All earlier experiments use this grid.
- **Context.** The released code's `generate()` defaults to `teacher_stopping_time=0.07` (8 steps, 13 passes; github.com/yl4579/DMOSpeech2, `src/infer.py`). The official demo runs the paper's 16-step setting with `teacher_steps=24, teacher_stopping_time=0.3, student_start_step=2`. On 2026-11-02 the team decided to anchor its claims on the paper. This test checks whether the problem-validation pattern holds on the paper's grid, and it measures the paper's own 14 + 2 setting.

## Design

Only the grid and the stopping rule change. Guidance (γ = 2, no stochastic term), seeds, the student and its restart level stay as in the thesis. Failures follow Definition 3.1 (`code/scheduling/failure_config.json`).

| Output | Grid and stop | Teacher steps | Student steps | Forward passes |
|---|---|---:|---:|---:|
| `g16_4`: code default | 16 intervals, first point above 0.07 | 4 | 3 | 13 |
| `g16_8`: thesis reference | 16 intervals, first point above 0.25 | 8 | 3 | 21 |
| `s007`, `s010`, `s015`, `s020` | 32 intervals, last point ≤ 0.07 / 0.10 / 0.15 / 0.20 | 7 / 9 / 11 / 13 | 3 | 19 / 23 / 27 / 31 |
| `s025`: reference on this grid | 32 intervals, last point ≤ 0.25 | 14 | 3 | 33 |
| `paper`: the paper's 14 + 2 | 32 intervals, last point ≤ 0.25; student restart j = 2 | 14 | 2 | 32 |

- **Data:** the EXP-012 Quick set: 60 development speakers × clean, noisy15 and Gaussian noise at 18, 20, 22 and 25 dB = 360 cases. All 8 outputs are synthesized and scored in the same run.
- **Hardware:** Colab Tesla T4 only. The runner stops on another GPU, because the student's random draws differ between GPU models (EXP-015).
- **Reproduction checks:**
  - the new grid-agnostic teacher loop, run on the released 16-interval grid, must give the sampler's Step-4 and Step-8 endpoints (smoke test);
  - the fresh `g16_4` and `g16_8` scores are compared with the stored T4 labels for all 360 cases.
- **Preregistration:** `preregistration.json` is written before any synthesis.

## Questions (descriptive; no pass/fail criteria)

1. On the 32-interval grid, do fixed early switching times fail mainly on noisy prompts and rarely on clean ones, as on the 16-interval grid (Appendix D)? Failures per output and condition are counted against `s025`.
2. Does the cheapest safe switching time vary between inputs? The report gives its distribution in hindsight and the mean cost against 33 passes.
3. How do the paper's 14 + 2 setting (`paper`) and the code default (`g16_4`) compare in quality with the thesis reference (`g16_8`)? The report gives mean WER, SIM and UTMOS, paired Wilcoxon tests, and added failures.

## What this test does not reproduce

- **The paper's handover itself.** Algorithm 2 starts the student from the teacher's state at the switch point and does not state the student's later times. This test keeps the released handover instead: the endpoint estimate is re-noised at the student's restart levels. `paper` uses the student's last two levels (j = 2), as the official demo does for the 16-step setting.
- **A stated grid size.** The 32-interval grid is inferred from "14 steps at 0.25"; the paper does not state it.
- **New speakers.** The 60 speakers are development speakers already used in EXP-012 to EXP-016, so the results are development evidence.

## Cost

About 3–3.5 hours on a T4. This is an estimate from the 10–13 s per case measured in EXP-013 and EXP-014 for 2–3 scored outputs; this run scores 8 outputs per case and records its own time per case.

## Relation to other tasks

EVAL-009 (j = 1 versus j = 2 on the 16-interval grid) is a separate check. The `paper` output here adds one j = 2 point on the 32-interval grid and does not replace it.

## Results

Not run yet. Import the results zip into `results/` with a SHA-256 manifest, as for EXP-015.
