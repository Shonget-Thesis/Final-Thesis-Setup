# 2026-11-16: Matching the paper's 0.25 variety for fewer passes: re-reading the stored arms (no new data)

No new task ID; a follow-up to EVAL-025 (EXP-032).
- **The question,** from Rhenel Jhon R. Sajol: "how can [we] match 0.25 variety and save compute costs in the handover".
- **What this is:** post hoc and exploratory. It uses committed results only, and nothing is adopted.

## Starting point

- **EXP-032:** the paper's own 0.25 setting (`P25`) costs 30 passes. The thesis's `W8` (21 passes, student restart j = 1) keeps about a third of its aligned variety gain.
- **Its join with EXP-026:** `W8R2` (Step 8, student restarted at 0.2929, 20 passes) matches `P25`:
  - in UTMOS on 30 shared clean cases;
  - roughly in CVf0.
- **So:** `W8R2` serves here as the stand-in for the paper's 0.25.
- **What the earlier experiments did:** EXP-027, 028 and 029 ran cheaper arms next to `W8R2`, but judged them against `W8`, whose UTMOS is high partly because it has less variety.

**Script:** `code/analysis/paper_anchor_rescreen.py`. Output: `research/experiments/EXP-032-variety-per-pass/offline/paper_anchor_rescreen.json`.
- **Pairing:** every comparison is within one experiment, on the same speakers, seeds and cases, with the runners' bootstrap.

## Cheaper arms against `W8R2` (clean prompts)

"Share" is the arm's CVf0 gain over `W4` divided by `W8R2`'s gain over `W4`.

| Arm | Passes | EXP-027 (61–100) | EXP-028 calibration-split | EXP-028 development | EXP-029 (61–90) | UTMOS − `W8R2` (range over cohorts) |
|---|---:|---|---|---|---|---|
| `W4` (j = 1) | 13 | 0% | 0% | 0% | 0% | +0.04 to +0.10 |
| `W4R2` | 12 | 73% | 63% | | | −0.27 to −0.29 |
| `TBRG` | 13 | | 55% | 61% | 99% (wide interval) | −0.05 to −0.09 |
| `W5R2` | 14 | | | | 59% | +0.01 |
| `W6R2` | 16 | | 86% | 69% | 72% | +0.03 to +0.04 |
| `W8R2` | 20 | 100% | 100% | 100% | 100% | 0 |
| `W8` (j = 1) | 21 | 1% | −2% | 3% | −12% | +0.10 to +0.13 |

**Reading the table:**
- **W6R2 is the closest cheap arm:** at 16 passes it keeps 69–86% of `W8R2`'s gain over `W4`, at equal or slightly higher UTMOS. Its CVf0 − `W8R2` intervals:
  - calibration-split: −0.0019 [−0.0055, +0.0025];
  - development: −0.0044 [−0.0089, +0.0000];
  - EXP-029: −0.0034 [−0.0066, +0.0007].
- **Against the student alone,** `W6R2` would be at roughly 90–96% of `W8R2`'s gain. This is a rough cross-cohort estimate:
  - EXP-027 puts `W4` at 68% of `W8R2`'s gain over the student;
  - the aligned measure was not computed in these experiments.
- **TBRG** (13 passes) keeps 55–99% at 0.05–0.09 lower UTMOS. It lies on the same trade-off as the `R2` stops.
- **W4R2** loses 0.27 UTMOS.
- **Nothing below 20 passes is shown to keep all of the paper's variety.**

**Noisy prompts (EXP-029):** every arm copies the noise (UTMOS about 1.25). The j = 2 restarts produce more runaway outputs on noisy15 than `W8`:

| | `W5R2` | `W8R2` | `W6R2` | `TBRG` | `W8` |
|---|---:|---:|---:|---:|---:|
| Runaway outputs on noisy15 | 5 | 6 | 3 | 3 | 0 |

## Can drift choose who stops early under j = 2?

- **The test:** EXP-029 logged d₁–d₈ on each clean quality case. The question is whether a speaker's drift at the stop predicts how much variety that early stop loses.
- **The result: no.**

| Early stop | Spearman(d_k, CVf0 − `W8R2`) | Speakers within 0.005 of `W8R2` |
|---|---:|---:|
| `W5R2` (d₅) | +0.06 | 11 of 20 |
| `W6R2` (d₆) | −0.01 | 13 of 20 |

- **Caveats:**
  - drift was logged on one seed, and variety is measured over 12;
  - 20 speakers.
- **But it matches E2:** drift has not added information beyond SNR in any experiment so far.
- **So:** an SNR + drift policy with the j = 2 restart would choose its early stops at random with respect to variety. Its saving would only be the cost mix along the same trade-off.

## What remains untested (checked against all experiments and logs)

All three would make the teacher's path to t ≈ 0.29 cheaper, which is where the melody forms (bank screen: Steps 3–8). None is novel by itself.

| Lever | What it would do | Status here |
|---|---|---|
| **A coarser teacher grid** | 4 sway steps on an 8-interval grid reach t = 0.2929, then j = 2: 12 passes against `W8R2`'s 20. Fewer, larger steps are a known technique | Never run. The 32-interval (`P25`) to 16-interval (`W8R2`) step lost nothing measurable |
| **Guidance only on the early steps** | Steps 5–8 conditional-only: 16 passes. This is the guidance interval of Kynkäänniemi et al. (2024) | Never run. EXP-023 found a fully unguided teacher (Steps 0–3) loses pitch variety, which argues for keeping the early guidance |
| **Analytic unconditional branch** (EXP-024) | The unconditional pass replaced by its closed form where that is accurate | Preregistered (`993c306d13cf...`), never run. Set aside on 2026-11-14 as a principled variant of a practised idea |

## Answer to the question

1. **Already available, and not new:** the paper's 0.25 variety and quality at 20 passes instead of 30 (−33%). Use the released 16-interval grid with the paper's two-step student restart (`W8R2`).
   - **For the thesis:** this means changing Chapter 3's fixed j = 1. That is a team decision (EVAL-009, unassigned), and `W8R2` would become the reference.
2. **Below 20 passes with stored settings:** `W6R2` (16 passes, −47% against the paper's 30) keeps most but not all of the variety, at the same quality.
3. **Adaptivity by drift does not help** on these data.
   - **SNR still matters on noisy prompts:** the student alone is safer there than any teacher handover (E4, E7).
4. **To keep all of the variety below 20 passes,** the teacher's own path must get cheaper. The candidates are in the table above. Each needs one T4 run, and any novelty claim would need a new part beyond those known levers.

## Checks run

- `python code/analysis/paper_anchor_rescreen.py`: ran and wrote its JSON.
- **Tests:** `code/tests/test_paper_anchor_rescreen.py`, 5 tests: the helpers, the pass counts, and `W8R2` above `W8` in every cohort.
- **Workspace check:** `python scripts/check_workspace.py` PASSED.
- **Full suite:** `python -m unittest discover -s code/tests` ran 352 tests, OK (1 skipped).
