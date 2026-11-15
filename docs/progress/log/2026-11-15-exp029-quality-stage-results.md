# 2026-11-15: EXP-029 quality-stage results imported (diversity stage still to run)

Task: EVAL-022 (now in_progress). Rhenel Jhon R. Sajol ran the EXP-029 notebook on Colab (Tesla T4) with its default `STAGE='quality'`, and supplied `DMOSpeech2_EXP029_Damped_Transmissibility_STAC_Results.zip` (sha256 `8b7ffa0706afce0e...`, 83,204 bytes). The 20 × 12 pitch-diversity stage has not run.

## Import and checks

- **Preregistration:**
  - The zip's manifest records preregistration file hash `98859522...`. That equals the committed file with LF line endings (its stored payload hash is `e58f3856ad20...`).
  - The runner's `run` command calls `check_prereg()` before synthesis and stops on any code or input hash mismatch. So the runner and its eight dependencies matched the frozen hashes.
  - The zip does not record the thesis commit or the notebook version.
- **Smoke test (03:15 UTC):**
  - TB(α=0) = S1, TB(α=1) = TBRG and the split student all matched exactly (max difference 0.0).
  - EXP-027's W4 scores were reproduced (ΔWER 0, ΔSIM 6e-8).
- **Rows:** 90 of 90 quality rows (30 speakers × clean, gauss18, noisy15), all on a Tesla T4. There is no `diversity.csv`.
- **Import:**
  - All seven files were copied byte for byte into `research/experiments/EXP-029-damped-transmissibility-stac/results/` and checked with `cmp`. These are the report, the results JSON, `quality.csv`, the runner's `MANIFEST.json`, and three smoke files.
  - `.gitattributes` marks that folder `-text`, as for EXP-028.
- **Analysis recomputed:** `python -I code/colab/run_damped_transmissibility_stac.py analyze --allow_partial` on a scratch copy of `quality.csv` gave:
  - the same results JSON apart from its timestamp;
  - the same report once Windows line endings were normalised.
- **New script:** `code/analysis/exp029_diagnostics.py` writes `results/diagnostics.json` (sha256 `f21aca5e04f40ce3...`). It was written after the results were seen, so it is descriptive and cannot change the verdict.

## Results

**Frozen rule's output: `clean_pareto_found_no_stac_headroom`, partial.**
- **TB25, TB50 and BLND are listed as passed, but on quality only.** `determine_verdict` treats the CVf0 condition as met when `diversity.csv` is missing.
- **No router using teacher features met the routing bars.** This part is final, because the diversity stage can only remove arms from the passed list.
- **The final verdict** will be this one or `monotone_tradeoff_only`.

**Clean prompts (30 speakers, ΔUTMOS with 95% bootstrap intervals):**
- **Against W8:**
  - W4 −0.026 [−0.093, +0.038];
  - TB25 −0.065, TB50 −0.052, BLND −0.062;
  - TB75 −0.091, TBRG −0.190;
  - W5R2 −0.114, W6R2 −0.095, W8R2 −0.125.
- **Against W4 at the same 13 NFE:** every bridge's point estimate is lower:
  - TB50 −0.026 [−0.105, +0.051];
  - TB25 −0.039;
  - BLND −0.035;
  - TBRG −0.164 [−0.249, −0.076].
- **WER:** within 0.007 of W8 for every arm.

**Degraded prompts (gauss18 and noisy15, 30 cases each):**
- **UTMOS is about 1.25 for every arm, W8 included.** All arms copy the prompt noise, as in EXP-018.
- **W8 has no runaway output (WER > 1). Every other arm has at least one:** up to 6 for W8R2 on noisy15.
- **Under Definition 3.1 against W8, 11 to 23 of 30 cases fail for every arm.**
  - The 13-NFE arms fail mostly on similarity.
  - The τ = 0.2929 restarts on noisy15 fail mostly on WER: 18 for W5R2 and 17 for W8R2, against 9 for W4.
- **The τ = 0.2929 restarts sound closer to the speaker's clean recording:** +0.012 to +0.027 similarity against W8.

**Routing:**
- **The teacher features measure prompt noise.** `A_4`, `A_4_norm` and `gc_0` separate clean from degraded prompts as well as prompt SNR does (AUC ≥ 0.998), and track it (|Spearman| 0.71–0.76).
- **Why their routers fail:**
  - They send every clean case and 10–53% of degraded cases to the 13-NFE arm.
  - They miss the pooled-similarity lower bound (−0.011 to −0.015 against −0.01).
  - Some also miss the pooled-WER upper bound.
- **Prompt-SNR routers** are excluded by the rule as the existing `STAC`. They met every bar with TB25 (16% saving) and BLND (13% saving).

## How to read it

- **For the compute problem:** on clean prompts the plain Step-4 handover W4 is the best 13-NFE arm on UTMOS. The transmissibility-motivated bridges have no quality advantage over it. A bridge is worth anything only if it adds pitch variety over W4, which is the diversity stage's question.
- **For an adaptive method:** the zero-extra-NFE teacher features add nothing beyond prompt SNR here. On degraded prompts, only W8 avoided runaway outputs.
- **What did not replicate:** EXP-028's "Pareto-strong W6R2" did not hold up on these speakers (−0.095 against −0.063).

**Limits:**
- **Data:** 30 development speakers, one run, intervals about ±0.07 UTMOS.
- **Diagnostics:** post hoc.
- **Evidence status:** no held-out evidence; not adopted; not reviewed by an adviser.

## Checks run

- `cmp` of every imported file against the extracted zip: identical.
- `python -I code/colab/run_damped_transmissibility_stac.py analyze --out_dir <scratch> --allow_partial`: same results as the zip (above).
- `python -I code/analysis/exp029_diagnostics.py`: wrote `results/diagnostics.json`.
- `python scripts/render_progress.py`, `python scripts/check_workspace.py` and `python -m unittest discover -s code/tests`: results are in the handoff below.
- LaTeX build: not run. The manuscript did not change.

## Remaining blockers

- **Diversity stage:** the notebook now defaults to it (see "Notebook default changed" below), so Run all is enough.
  - Step 3 restores `quality.csv` from Drive, and the analysis then runs the full rule.
  - About 45 minutes on a T4.
- **Algorithm:** the agreed next step (create the algorithm after EXP-029) waits on that stage. Whether any 13-NFE bridge adds pitch variety over W4 decides which direction is open.

## Notebook default changed (later on 2026-11-15)

**What happened:**
- A second zip (`...Results (1).zip`, sha256 `5ca827daa3a4a50c...`) was a rerun of the quality stage. It had no `diversity.csv`.
- Its quality file, report, smoke files and diagnostics are byte-identical to the imported ones. Its results JSON differs only in its timestamp. It was not imported.
- **Cause:** `STAGE` was still `"quality"`. The edit had been made in the Step 5 cell as `globals().get("STAGE", "Diversity")`, which keeps the setup cell's value, and the capital D would have been rejected anyway.

**What changed:** notebook only. The runner and preregistration are unchanged, and `check_prereg()` still accepts `e58f3856ad20...`.
- The setup cell now sets `STAGE = "diversity"`.
- The Step 5 cell lowercases and checks `STAGE` and prints it before the run.
- The title cell says the quality stage is complete.

**Checks run:**
- All notebook code cells compile.
- `python -m unittest discover -s code/tests -p "test_damped_transmissibility_stac.py"`: 5 tests, OK.
- `python scripts/check_workspace.py`: PASSED.
