# EXP-029: Damped Slow-Band Transmissibility Bridge & Mixed-Condition `STAC` Diagnostic

- **Task ID:** `EVAL-022` (`review`; prepared `2026-11-15`; both stages run on 2026-11-15 on a Colab Tesla T4; development diagnostic only). **Verdict (frozen rule): `monotone_tradeoff_only`.** See [Results](#results-2026-11-15).
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
- **Execution Note:** the notebook first defaulted to `STAGE='quality'` (`~25–30 min` on T4) to show clean Pareto quality, degraded-prompt stability, and mixed `STAC` routing before `STAGE='diversity'` (`~45 min`) or `STAGE='all'`. After the quality stage finished on 2026-11-15, its default became `STAGE='diversity'`.

## Status & Results

- Prepared and preregistered locally on `2026-11-15`.
- Both stages ran on 2026-11-15 on a Colab Tesla T4:
  - **Quality stage:** 90/90 cases, 03:15–04:21 UTC.
  - **Diversity stage:** 20/20 speakers × 12 seeds, finished 07:05 UTC.

## Results (2026-11-15)

Files: [report](results/DAMPED_TRANSMISSIBILITY_REPORT.md), [results](results/damped_transmissibility_results.json), [quality rows](results/quality.csv), [diversity rows](results/diversity.csv), [manifest](results/MANIFEST.json), [smoke check](results/smoke/smoke_check.json), and [diagnostics](results/diagnostics.json), which are post hoc and descriptive ([script](../../../code/analysis/exp029_diagnostics.py)).

**Checks before reading:**
- **Preregistration:** the run used the committed preregistration (file `98859522...`, payload `e58f3856ad20...`). The runner's `run` command refuses to start unless `check_prereg()` passes, so the runner and its eight dependencies matched the frozen hashes. The zip does not record the thesis commit.
- **Smoke test (run again before the diversity stage):**
  - TB(α=0) equals S1, TB(α=1) equals TBRG, and the split student matches (max difference 0.0 for all three).
  - EXP-027's W4 scores were reproduced (ΔWER 0, ΔSIM 6e-8).
- **Integrity:**
  - All rows ran on a Tesla T4.
  - The final zip's `quality.csv` is byte-identical to the quality-stage zip's.
  - Every file matches the runner's manifest. The runner hashes with line endings normalised (`sh.sha256_text`), and its CSV files are written with CRLF.
- **Recomputation:** running `analyze` locally on the two CSV files gave the same results JSON (apart from its timestamp) and the same report.

**Verdict (frozen rule): `monotone_tradeoff_only`.**
- **Clean Pareto:** no 13–16-NFE arm both stays within the −0.08 UTMOS bar against W8 and raises pitch variety over W4. Every arm that adds variety loses more than 0.08 UTMOS.
- **Routing:** no router on teacher features met the routing bars.
- **Frontier:** across the eight cheap arms, quality loss and variety gain are strongly linked (Spearman −0.93, post hoc).

**Clean prompts: quality (30 speakers) against pitch variety (20 speakers × 12 seeds):**

| Arm | NFE | ΔUTMOS vs W8 | CVf0 | ΔCVf0 vs W4 | Speakers above zero (of 20) |
|---|---:|---|---:|---|---:|
| W4 | 13 | −0.026 [−0.093, +0.038] | 0.0735 | | |
| TB25 | 13 | −0.065 [−0.138, +0.006] | 0.0732 | −0.0003 [−0.0032, +0.0023] | 11 |
| TB50 | 13 | −0.052 [−0.123, +0.021] | 0.0727 | −0.0008 [−0.0040, +0.0023] | 11 |
| BLND | 13 | −0.062 [−0.131, +0.008] | 0.0757 | +0.0023 [−0.0006, +0.0058] | 12 |
| TB75 | 13 | −0.091 [−0.164, −0.019] | 0.0770 | +0.0036 [−0.0010, +0.0088] | 13 |
| TBRG | 13 | −0.190 [−0.275, −0.103] | 0.0855 | +0.0120 [+0.0022, +0.0266] | 15 |
| W5R2 | 14 | −0.114 [−0.201, −0.028] | 0.0807 | +0.0072 [+0.0025, +0.0119] | 17 |
| W6R2 | 16 | −0.095 [−0.161, −0.030] | 0.0822 | +0.0088 [+0.0026, +0.0144] | 17 |
| W8R2 | 20 | −0.125 [−0.192, −0.056] | 0.0857 | +0.0122 [+0.0069, +0.0177] | 17 |
| W8 (reference) | 21 | UTMOS 3.764 | 0.0720 | −0.0015 [−0.0041, +0.0004] | 9 |

On clean prompts, WER is within 0.007 of W8 for every arm, and no case has WER above 0.13.

**What it shows:**
- **W4 already matches the reference.**
  - Plain W4 at 13 NFE has the reference's pitch variety (W8 − W4: −0.0015 CVf0) and is within −0.026 UTMOS of it.
  - This repeats EXP-026's finding (W8 − W4: +0.0007) on other speakers.
  - Any variety gained over W4 is variety beyond the reference.
- **Damping finds no sweet spot.**
  - CVf0 stays at W4's level up to α = 0.5 (TB25, TB50). It rises only at α = 0.75 and 1.0, where UTMOS drops (TB75 −0.091, TBRG −0.190).
  - BLND behaves like the low-α bridges.
- **The τ = 0.2929 restarts trade the same way.**
  - W5R2, W6R2 and W8R2 add 0.007–0.012 CVf0 over W4 (17 of 20 speakers above zero) and lose 0.095–0.125 UTMOS.
  - TBRG reaches W8R2's variety at 13 NFE (−0.0002 [−0.0122, +0.0176]) but loses more UTMOS.
  - EXP-028's "Pareto-strong W6R2" did not hold up here.
- **Degraded prompts: every arm copies the noise.**
  - UTMOS is about 1.25 for every arm, W8 included, as in EXP-018.
  - W8 is the only arm with no runaway output (WER > 1). W5R2 has 5 and W8R2 has 6 on `noisy15`.
  - Under the thesis's failure rule (Definition 3.1, against W8), 11 to 23 of the 30 cases fail for every arm in both degraded conditions.
    - The 13-NFE arms fail mostly on similarity: on `gauss18`, 14 to 19 similarity failures against 4 to 8 WER failures.
    - The τ = 0.2929 restarts fail mostly on WER on `noisy15`: 18 and 17 for W5R2 and W8R2, against 9 for W4.
  - The τ = 0.2929 restarts are closer to the speaker's clean recording: +0.012 to +0.027 similarity against W8.
- **The teacher features act as a noise detector.**
  - `A_4`, `A_4_norm` and `gc_0` separate clean from degraded prompts as well as prompt SNR does (AUC ≥ 0.998), and they track it (|Spearman| 0.71–0.76).
  - Routers using them send every clean case and 10–53% of degraded cases to the 13-NFE arm, and miss the pooled-similarity bound (lower bound −0.011 to −0.015 against −0.01).
  - Prompt-SNR routers, which the rule excludes as the existing `STAC`, met every routing bar with TB25 (16% saving) and BLND (13% saving).

**What it means for the thesis:**
- **On clean prompts,** the handover operators tested since EXP-026 do not improve on W4 for matching the W8 reference. These are the restart level, slow-band injection, damping and blending. Each one only moves along a variety-against-UTMOS trade-off.
- **As a source of a compute-saving method,** the transmissibility line is closed.
- **What is left** is which cases can hand over early and which need W8, chiefly on degraded prompts. There, only W8 avoided runaway outputs, and the zero-extra-NFE teacher features added nothing beyond prompt SNR.

**Limits:**
- **Speakers:** development speakers only.
  - The diversity speakers (61–80) overlap EXP-028's 61–100 cohort.
  - The quality cohort is 30 speakers.
- **Run:** one run, with intervals about ±0.07 UTMOS and ±0.003–0.012 CVf0.
- **Diagnostics:** chosen after the results were seen.
- **Evidence status:** no held-out evidence; not adopted; not reviewed by an adviser.
