# EXP-029: Damped Slow-Band Transmissibility Bridge & Mixed-Condition `STAC` Diagnostic

- **Task ID:** `EVAL-022` (`in_progress`; prepared `2026-11-15`; quality stage run on 2026-11-15 on a Colab Tesla T4, diversity stage not run yet; development diagnostic only). See [Results](#results-quality-stage-2026-11-15).
- **Preregistration:** [preregistration.json](preregistration.json) (`sha256: e58f3856ad20...`)
- **Runner:** [run_damped_transmissibility_stac.py](../../../code/colab/run_damped_transmissibility_stac.py)
- **Notebook:** [DMOSpeech2_EXP029_Damped_Transmissibility_STAC.ipynb](../../../code/colab/DMOSpeech2_EXP029_Damped_Transmissibility_STAC.ipynb)
- **Unit Tests:** [test_damped_transmissibility_stac.py](../../../code/tests/test_damped_transmissibility_stac.py)

## Scientific Motivation (Following `EXP-028` Outcome 3)

In the thesis's **`STAC` (`EXP-018`) → Transmissibility Theory (`EXP-026`/`027`) → Upgraded `STAC` (`EXP-028`)** progression, `EXP-028` showed two fixable gaps:
1. Full slow-band replacement at Step 4 (`TBRG`, $\alpha = 1.0$, `13 NFE`) lost `-0.193` / `-0.160` UTMOS vs `W8`, whereas `BLND` (`0.5 S1 + 0.5 T4`, `13 NFE`) preserved UTMOS (`-0.069` / `-0.038`) and `W6R2` (`16 NFE`) achieved both high `CVf0` (`0.1029`) and `UTMOS` (`3.887`).
2. Secondary `STAC-TBRG` was tested on **clean prompts only** with teacher-only `A_4`, which had near-zero correlation with clean `dUTMOS(TBRG)` (`|r| <= 0.11`) and never tested whether $\tau_2$ restarts or slow-band bridges break on degraded prompts (`EXP-018`'s core regime).

`EXP-029` closes both gaps in one focused development diagnostic before any final held-out confirmation (`EXP-030`):

## Protocol Summary

- **Disjoint Development Cohorts (keeps remaining `170` `calibration` speakers and all `200` `test` speakers untouched):**
  - **Mixed Quality (`90` cases):** `30` development speakers (`cg.dev_items()[60:90]`) $\times$ `3` conditions (`clean`, `gauss18`, `noisy15`).
  - **Clean Diversity (`20` $\times$ `12` = `240` seed trajectories):** `20` development speakers (`cg.dev_items()[60:80]`) $\times$ `12` seeds.
- **10 Paired Arms (One Shared Teacher Trajectory `0..8` per Seed/Case):**
  - **Energy-Preserving Slow-Band Damping Sweep (`13 NFE`):**
    $$X_{\text{TB}(\alpha)}[0, \text{plen}:, :] = v_S + \frac{\|s_S\|_F}{\|(1-\alpha) s_S + \alpha \tilde{s}_T\|_F + \epsilon}\bigl((1-\alpha) s_S + \alpha \tilde{s}_T\bigr) + f_S$$
    for $\alpha \in \{0.25, 0.50, 0.75, 1.00\}$ (`TB25`, `TB50`, `TB75`, `TBRG`), with algebraic endpoints $\alpha=0 \equiv W4$ (`S1`) and $\alpha=1 \equiv \text{TBRG}$.
  - **Baselines & Step Sweep:** `W4` (`13` NFE), `BLND` (`13` NFE), `W5R2` (`14` NFE), `W6R2` (`16` NFE), `W8R2` (`20` NFE), `W8` (`21` NFE).
- **Zero-Extra-NFE Telemetry & 5-Fold CV Routing:** Logs `gc_0`, `d_1..d_8`, `A_4..A_8`, `A_4_norm = A_4 / gc_0`, `prompt_snr`, and post-`S1` discrepancy signals (`slow_cos_4`, `slow_gap_4`, `voice_gap_4`, `fine_gap_4`, `anchor_gap_4`), and evaluates 5-fold speaker-grouped CV routing across mixed `clean` + `gauss18` + `noisy15` cohorts against `W8` and `W8R2`.
- **Execution Note:** Colab notebook defaults `STAGE='quality'` (`~25–30 min` on T4) to reveal clean Pareto quality, degraded-prompt stability, and mixed `STAC` routing first, before running `STAGE='diversity'` (`~45 min`) or `STAGE='all'`.

## Status & Results

- Prepared and preregistered locally on `2026-11-15`.
- **Quality stage:** run on 2026-11-15 (Colab Tesla T4). Results are below.
- **Diversity stage:** the `20 x 12` pitch-diversity stage (`STAGE='diversity'`) has not run yet. The preregistered verdict needs it.

## Results: quality stage (2026-11-15)

Files: [report](results/DAMPED_TRANSMISSIBILITY_REPORT.md), [results](results/damped_transmissibility_results.json), [quality rows](results/quality.csv), [manifest](results/MANIFEST.json), [smoke check](results/smoke/smoke_check.json), and [diagnostics](results/diagnostics.json), which are post hoc and descriptive ([script](../../../code/analysis/exp029_diagnostics.py)).

**Checks before reading:**
- **Preregistration:** the run used the committed preregistration (file `98859522...`, payload `e58f3856ad20...`). The runner's `run` command refuses to start unless `check_prereg()` passes, so the runner and its eight dependencies matched the frozen hashes. The zip does not record the thesis commit.
- **Smoke test:**
  - TB(α=0) equals S1, TB(α=1) equals TBRG, and the split student matches (max difference 0.0 for all three).
  - EXP-027's W4 scores were reproduced (ΔWER 0, ΔSIM 6e-8).
- **Run:** all 90 rows ran on a Tesla T4.
- **Recomputation:** running `analyze --allow_partial` locally on `quality.csv` gave the same results JSON (apart from its timestamp) and the same report (line endings normalised).

**The frozen rule's output: `clean_pareto_found_no_stac_headroom`, marked partial.**
- **The "passed" arms passed on quality only.** TB25, TB50 and BLND are listed as passed. With no `diversity.csv`, the rule treats the CVf0 condition (lower bound of CVf0 − CVf0(W4) > 0) as met. The diversity stage decides whether any of them really passes.
- **"No `STAC` headroom" is already final.** No router using teacher features (`A_4`, `A_4_norm`) met the routing bars. The diversity stage can only remove arms from the passed list, so this cannot change.
- **The final verdict** will be either this one or `monotone_tradeoff_only`.

**Clean prompts (30 speakers; per-speaker paired differences with 95% bootstrap intervals):**

| Arm | NFE | ΔUTMOS vs W8 | ΔUTMOS vs W4 |
|---|---:|---|---|
| W4 | 13 | −0.026 [−0.093, +0.038] | |
| TB25 | 13 | −0.065 [−0.138, +0.006] | −0.039 [−0.114, +0.032] |
| TB50 | 13 | −0.052 [−0.123, +0.021] | −0.026 [−0.105, +0.051] |
| TB75 | 13 | −0.091 [−0.164, −0.019] | −0.065 [−0.142, +0.011] |
| TBRG | 13 | −0.190 [−0.275, −0.103] | −0.164 [−0.249, −0.076] |
| BLND | 13 | −0.062 [−0.131, +0.008] | −0.035 [−0.106, +0.035] |
| W5R2 | 14 | −0.114 [−0.201, −0.028] | −0.088 [−0.197, +0.023] |
| W6R2 | 16 | −0.095 [−0.161, −0.030] | −0.068 [−0.161, +0.028] |
| W8R2 | 20 | −0.125 [−0.192, −0.056] | −0.098 [−0.189, −0.005] |
| W8 (reference) | 21 | UTMOS 3.764 | |

The vs-W8 column is the frozen report's. The vs-W4 column is from the post hoc diagnostics. On clean prompts, WER is within 0.007 of W8 for every arm, and no case has WER above 0.13.

**What it shows:**
- **Damping recovers most of TBRG's loss but does not beat W4.**
  - TB25, TB50 and BLND lose 0.05–0.07 UTMOS against W8, against TBRG's 0.19.
  - Every bridge's point estimate is below plain W4 at the same 13 NFE.
  - A damped bridge can only be worth having if it adds pitch variety over W4, and that is what the diversity stage measures.
- **The Step 5 and Step 6 restarts at τ = 0.2929 do not hold quality.** W5R2 (−0.114) and W6R2 (−0.095) miss the −0.08 bar. EXP-028's W6R2 lost −0.063 on its unseen calibration speakers.
- **Degraded prompts: every arm copies the noise.**
  - UTMOS is about 1.25 for every arm, W8 included, as in EXP-018.
  - W8 is the only arm with no runaway output (WER > 1). W5R2 has 5 and W8R2 has 6 on `noisy15`.
  - Under the thesis's failure rule (Definition 3.1, against W8), 11 to 23 of the 30 cases fail for every arm in both degraded conditions.
    - The 13-NFE arms fail mostly on similarity: on `gauss18`, 14 to 19 similarity failures against 4 to 8 WER failures.
    - The τ = 0.2929 restarts fail mostly on WER on `noisy15`: 17 and 18 WER failures for W8R2 and W5R2, against 9 for W4.
  - The τ = 0.2929 restarts are closer to the speaker's clean recording: +0.012 to +0.027 similarity against W8.
  - The mean WERs in the report are dominated by a few runaway cases. The medians and failure counts are in the diagnostics.
- **The teacher features act as a noise detector.**
  - `A_4`, `A_4_norm` and `gc_0` separate clean from degraded prompts as well as prompt SNR does (AUC ≥ 0.998), and they track prompt SNR across the 90 cases (|Spearman| 0.71–0.76).
  - Routers using them send every clean case and 10–53% of degraded cases to the 13-NFE arm. They miss the pooled-similarity bound (lower bound −0.011 to −0.015 against −0.01).
  - For comparison: routers on prompt SNR, which the rule excludes because SNR routing is the existing `STAC`, met every routing bar with TB25 (16% saving) and BLND (13% saving).

**Limits:**
- **Data:** 30 development speakers, one run. The intervals are about ±0.07 UTMOS wide.
- **Diagnostics:** chosen after the results were seen.
- **Evidence status:** no held-out evidence; not adopted; not reviewed by an adviser.
