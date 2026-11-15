# EXP-029: Damped Slow-Band Transmissibility Bridge & Mixed-Condition `STAC` Diagnostic

- **Task ID:** `EVAL-022` (prepared `2026-11-15`, awaiting Colab Tesla T4 run; development diagnostic only)
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

- Prepared and preregistered locally on `2026-11-15`; awaiting Colab Tesla T4 execution (`EVAL-022` `todo`).
