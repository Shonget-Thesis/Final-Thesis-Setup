# 2026-11-15: EXP-029 final results imported (both stages)

Task: EVAL-022 (now review). Rhenel Jhon R. Sajol ran the diversity stage on Colab (Tesla T4) with the notebook defaulting to `STAGE="diversity"`.
- **The zip:** `DMOSpeech2_EXP029_Damped_Transmissibility_STAC_Results (3).zip`, sha256 `e7bbd72ac6bbdf52...`, 103,639 bytes. It was downloaded from the Drive copy, because the Colab tab had lost its connection to the runtime.
- **Timing:** the run finished at 07:05 UTC and the analysis at 07:06 UTC.
- **Earlier zips:** the quality-stage import is in the [quality-stage log](2026-11-15-exp029-quality-stage-results.md).

## Import and checks

- **Preregistration:**
  - The manifest records preregistration file `98859522...`, which equals the committed file (payload `e58f3856ad20...`).
  - The runner's `run` command stops unless `check_prereg()` passes.
- **Smoke test (06:06 UTC, run again before the diversity stage):** the same results as the first smoke test, apart from its timestamp.
  - TB(α=0) = S1, TB(α=1) = TBRG and the split student matched exactly.
  - EXP-027's W4 was reproduced (ΔWER 0, ΔSIM 6e-8).
- **Rows:** 90/90 quality and 20/20 diversity. The final `quality.csv` is byte-identical to the one imported from the quality stage.
- **Hashes:** all four data and analysis files match the runner's `MANIFEST.json` once line endings are normalised. The runner hashes with `sh.sha256_text`, and Python's csv module writes CRLF. Earlier imports noted the same thing.
- **Import:**
  - Byte-for-byte copies of the zip's files went into `research/experiments/EXP-029-damped-transmissibility-stac/results/`, checked with `cmp`.
  - New: `diversity.csv`. Replaced: the report, the results JSON, `MANIFEST.json` and `smoke/smoke_check.json`. Unchanged: `quality.csv` and the smoke CSV files.
  - The zip's `diagnostics.json` was only the committed copy carried through the Colab clone, so it was regenerated locally.
- **Analysis recomputed:** `python -I code/colab/run_damped_transmissibility_stac.py analyze` (strict, no `--allow_partial`) on scratch copies of the two CSV files gave:
  - the same results JSON apart from its timestamp;
  - the same report.
- **Diagnostics:** `code/analysis/exp029_diagnostics.py` now also writes a frontier section (each arm's ΔUTMOS against W8 beside its ΔCVf0 against W4). `results/diagnostics.json` sha256 is `59270ddf9693d229...`. It is post hoc and descriptive.

## Results

**Verdict (frozen rule): `monotone_tradeoff_only`.** No arm is both inside the −0.08 UTMOS bar and adds pitch variety over W4 (lower bound above zero), and no teacher-feature router passed.

**Clean prompts:**

| Arm | NFE | ΔUTMOS vs W8 | ΔCVf0 vs W4 |
|---|---:|---|---|
| W4 | 13 | −0.026 | |
| TB25 | 13 | −0.065 | −0.0003 [−0.0032, +0.0023] |
| TB50 | 13 | −0.052 | −0.0008 [−0.0040, +0.0023] |
| BLND | 13 | −0.062 | +0.0023 [−0.0006, +0.0058] |
| TB75 | 13 | −0.091 | +0.0036 [−0.0010, +0.0088] |
| TBRG | 13 | −0.190 | +0.0120 [+0.0022, +0.0266] |
| W5R2 | 14 | −0.114 | +0.0072 [+0.0025, +0.0119] |
| W6R2 | 16 | −0.095 | +0.0088 [+0.0026, +0.0144] |
| W8R2 | 20 | −0.125 | +0.0122 [+0.0069, +0.0177] |
| W8 | 21 | | −0.0015 [−0.0041, +0.0004] |

- **The trade-off is tight:** across the eight cheap arms, ΔUTMOS and ΔCVf0 have Spearman −0.93 (post hoc).
- **W4 matches the reference's pitch variety:** W8 − W4 is −0.0015, which repeats EXP-026's +0.0007.
- **Damping gives nothing until α ≥ 0.75,** and that is where UTMOS drops.

**Degraded prompts and routing:** unchanged from the quality-stage log.
- Every arm copies the noise (UTMOS about 1.25), and only W8 has no runaway output.
- The teacher features measure prompt noise (AUC ≥ 0.998).
- The prompt-SNR routers met the routing bars with TB25 and BLND. They are excluded as the existing `STAC`.

## How to read it

- **For the compute problem:** on clean prompts, nothing tested since EXP-026 improves on plain W4 for matching W8. The family covers restart level, slow-band injection, damping and blending. Each operator only moves along a variety-against-UTMOS trade-off, and W4 already has W8's variety.
- **The transmissibility line is closed as a source of a compute-saving method.**
- **What is left for an adaptive method** is which cases can hand over early and which need W8, chiefly on degraded prompts. There, only W8 avoided runaway outputs, and the zero-extra-NFE teacher features added nothing beyond prompt SNR.
- **These outcomes match the preregistration's `monotone_tradeoff_only` branch.** EXP-029 was preregistered as a development diagnostic before any held-out confirmation (EXP-030). With no arm passing, there is nothing to carry into EXP-030 from this family.

**Limits:**
- **Speakers:** development speakers only. The diversity speakers (61–80) overlap EXP-028's 61–100 cohort.
- **Run:** one run.
- **Diagnostics:** post hoc.
- **Evidence status:** no held-out evidence; not adopted; not reviewed by an adviser.

## Checks run

- `cmp` of every imported file against the extracted zip: identical.
- Line-ending-normalised SHA-256 of `quality.csv`, `diversity.csv`, the results JSON and the report: equal to `MANIFEST.json`.
- `python -I code/colab/run_damped_transmissibility_stac.py analyze --out_dir <scratch>`: same results as the zip (above).
- `python -I code/analysis/exp029_diagnostics.py`: wrote `results/diagnostics.json`.
- `check_prereg()`: accepts `e58f3856ad20...`.
- `python scripts/render_progress.py`, `python scripts/check_workspace.py` and `python -m unittest discover -s code/tests`: results are in the handoff.
- LaTeX build: not run. The manuscript did not change.

## Remaining blockers

- **Algorithm:** the agreed next step is to create the algorithm, now that EXP-029 is complete.
  - It needs a mechanism that is new under the team's novelty rubric, checked against the rejected-idea logs and a family search.
  - It needs an offline screen on stored data before any GPU run.
- **Adviser review** of the EXP-026–029 arc is not recorded.
