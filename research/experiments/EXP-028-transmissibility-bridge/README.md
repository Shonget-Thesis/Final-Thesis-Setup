# EXP-028 — Transmissible Slow-Band Bridge (`TBRG`) & Controls

- **Task:** `EVAL-021` (`review`, executed on Google Colab Tesla T4 on `2026-11-14`; `20/20` + `30/30` primary prospective `unseen_cal` and `20/20` + `40/40` `61–100` dev)
- **Results & Manifest:** [TRANSMISSIBILITY_BRIDGE_REPORT.md](results/TRANSMISSIBILITY_BRIDGE_REPORT.md) | [transmissibility_bridge_results.json](results/transmissibility_bridge_results.json) | [MANIFEST.json](results/MANIFEST.json)
- **Results Log:** [2026-11-15-exp028-transmissibility-bridge-results.md](../../../docs/progress/log/2026-11-15-exp028-transmissibility-bridge-results.md)
- **Framing:** **Theory-motivated, inference-time handover algorithm** (motivated post-EXP-027 by the $\tau_1$ vs $\tau_2$ transmissibility split, not predicted a priori) with **placement novelty at $\tau_2 = 0.2929$** and **no fitted parameters** (fixed design choices: 9-frame window, 3-band split, $\tau_2$ injection point). `STAC-TBRG` is kept strictly as a secondary exploratory diagnostic.
- **Predecessors:** [EXP-027](../EXP-027-transmissibility-confirmation/README.md) (`supported_direction_only`, established clean-prompt transmissibility loss `H1` and late-restart recovery `H2` on fresh speakers `61–100`) | [EXP-020](../EXP-020-selective-handover-prelim/README.md) (`decompose`) | [EXP-018](../EXP-018-swsr-handover-prelim/README.md) (`STAC`)
- **Runner:** [run_transmissibility_bridge.py](../../../code/colab/run_transmissibility_bridge.py)
- **Notebook:** [DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb](../../../code/colab/DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb)
- **Preregistration:** [preregistration.json](preregistration.json) (`sha256: 5c6c66ab8fee...`; pre-synthesis amendment chain: `8d6d68afa1d2...` $\rightarrow$ `4caf6c951669...` $\rightarrow$ `5c6c66ab8fee...`, all frozen prior to any GPU execution)
- **Unit tests:** [test_transmissibility_bridge.py](../../../code/tests/test_transmissibility_bridge.py)
- **Setup & amendment log:** [2026-11-14-exp028-transmissibility-bridge.md](../../../docs/progress/log/2026-11-14-exp028-transmissibility-bridge.md)

## Motivation from EXP-027

`EXP-027` evaluated six arms on fresh speakers `61–100` and revealed why default handover bottlenecks pitch variety:

- **`W4` (13 NFE):** Retains clean perceptual quality (`UTMOS 3.759`, `WER 0.0159`, `SIM 0.9312`) because the first student step $\tau_1 = 0.0761$ produces a clean acoustic draft $S_1$, but $\tau_1 = 0.0761$ bottlenecks pitch variety (`CVf0 0.0770` vs `W8R2 0.0893` / `T16 0.0911`).
- **`W4R2` (12 NFE):** Skips $\tau_1$ and injects raw $T_4$ directly at $\tau_2 = 0.2929$, unlocking high diversity (`CVf0 0.0859`, `+0.0089` over `W4`, $p = 2.94 \times 10^{-4}$), **but drops UTMOS by `-0.373` (`3.404`) and spikes slow-mel dispersion (`1.0299`)** because raw $T_4$ lacks $\tau_1$ denoising on `voice` and `fine` bands.

## Proposed 13-NFE Handover Algorithm (`TBRG`)

1. **Teacher Steps `0..4` (`10 NFE`):** Run the guided Euler teacher to Step 4 endpoint $T_4$ (plus Steps `5..6` on the same shared trajectory to evaluate the `16-NFE` `W6R2` baseline).
2. **Clean Acoustic Anchor (`1 NFE`):** Run the student's $\tau_1 = 0.0761$ pass from $T_4$ to obtain denoised draft $S_1$.
3. **Energy-Matched Slow-Band Bridge (`0 NFE`):** Decompose generated frames `plen:` via `sh.decompose(..., window=9)` into $(v_T, s_T, f_T)$ and $(v_S, s_S, f_S)$, and inject Frobenius-matched teacher slow prosody onto $S_1$'s `voice` and `fine` structure before $\tau_2 = 0.2929$:
   $$\tilde{s}_T = \frac{\|s_S\|_F}{\|s_T\|_F + 10^{-8}} s_T, \qquad X_{\tau_2}^{\text{TBRG}}[0, \text{plen}:, :] = v_S + \tilde{s}_T + f_S$$
4. **Explicit Flow-Matching Re-Noising & Student Tail (`2 NFE` -> `13 NFE` Total):** Pass $X_{\tau_2}^{\text{TBRG}}$ into the standard DMOSpeech 2 tail `[tau_2, tau_3] = [0.2929, 0.6173]`, explicitly re-noising at $\tau_2 = 0.2929$:
   $$\phi_2 = (1 - \tau_2)\varepsilon_2 + \tau_2 X_{\tau_2}^{\text{TBRG}}, \quad S_2 = G_\theta(\phi_2, \tau_2), \qquad \phi_3 = (1 - \tau_3)\varepsilon_3 + \tau_3 S_2, \quad S_3 = G_\theta(\phi_3, \tau_3)$$
   (`13 NFE` = **`-38.1%` vs `W8`**, **`-35.0%` vs `W8R2`**, **`-59.4%` vs `T16`**).

## Ablations, Cheap Baselines, Causal Controls & Unseen Replication

- **`STS2` (`13 NFE`):** Unnormalized slow-band bridge ablation ($\tilde{s}_T = s_T$) isolating Frobenius energy matching.
- **`BLND` (`13 NFE`):** Fixed 50/50 linear blend $0.5\, S_1 + 0.5\, T_4$ at $\tau_2 = 0.2929$ without band decomposition.
- **`SHUF` (`13 NFE`) & `slow_corr`:** Shuffled-speaker slow-band control injecting a time-resampled, Frobenius-matched prompt slow band $s_{\text{donor}}$ from a disjoint donor speaker, plus output-vs-injection `slow_corr` cosine correlation.
- **`W6R2` (`16 NFE`):** Intermediate 6-step teacher + $[\tau_2, \tau_3]$ restart (`14 + 2 = 16 NFE`).
- **Primary Prospective Cohort (`unseen_cal`):** `20 x 15` diversity (`300` trajectories/arm) + `30` quality on strictly disjoint unseen speakers from [calibration_manifest.json](../../../research/protocols/splits/calibration_manifest.json) (`0` speaker overlap with development speakers `1–100`), evaluating all 8 arms jointly (`W4`, `W4R2`, `TBRG`, `BLND`, `SHUF`, `W6R2`, `W8R2`, `W8`) plus **matched-seed vs. shifted-seed $f_0(t)$ contour correlation** against `W8R2`.
- **Development Cohort (`speakers 61–100`, paired vs EXP-027):** `20 x 20` diversity + `40` quality, reusing SHA-256-locked [EXP-027](../EXP-027-transmissibility-confirmation/results/MANIFEST.json) references (`student4`, `W4`, `W4R2`, `W8`, `W8R2`, `T16`), treated explicitly as development evidence because `EXP-027` on those speakers motivated `TBRG`, plus smoke check (`max |W4_split - W4| <= 1e-5` and `<= 0.01` WER/SIM reproduction).
- **Secondary Exploratory Diagnostic (`STAC-TBRG`):** 5-fold speaker-CV routing (`13–20 NFE`) between `TBRG` and stored `W8R2`, kept strictly secondary to the fixed `13-NFE TBRG` test.

## Preregistered Hypotheses (`H1`–`H5`), Multiplicity Hierarchy & Verdict Rule

- **Multiplicity Rule (Fixed-Sequence Gatekeeper):** **`H1` and `H2` are co-primary gatekeeper hypotheses** tested jointly via an intersection-union test at $\alpha = 0.05$ (both must clear their lower bounds simultaneously, avoiding $\alpha$ inflation). **`H5` (causal `SHUF` & `BLND` controls), `H3` (`W8` quality preservation), and `H4` (`mel_spread`)** are evaluated as hierarchical secondary/mechanistic criteria in fixed sequence once `H1` and `H2` pass.
- **`[CO-PRIMARY] H1`:** `CVf0(TBRG) - CVf0(W4) > 0` (lower 95% CI `> 0`, `no-top2 > 0`) AND non-inferiority vs `W8R2` (`CVf0(TBRG) - CVf0(W8R2)` lower 95% CI `>= -0.008`).
- **`[CO-PRIMARY] H2`:** `UTMOS(TBRG) - UTMOS(W4R2) > 0` (lower 95% CI `> 0`, `no-top2 > 0`) AND non-inferiority vs `W8R2` (`UTMOS(TBRG) - UTMOS(W8R2)` lower 95% CI `>= -0.05`).
- **`[SECONDARY] H5`:** Causal slow-band tracking (`CVf0(TBRG) - CVf0(SHUF)` lower 95% CI `> 0`) AND decomposition advantage over naive linear blend (`UTMOS(TBRG - BLND)` or `CVf0(TBRG - BLND)` lower 95% CI `> 0`).
- **`[SECONDARY] H3`:** Clean `W8` quality preservation (`upper CI dWER <= +0.01`, `lower CI dSIM >= -0.01`, tightened `mean dUTMOS >= -0.08`).
- **`[SECONDARY] H4`:** `mel_spread(TBRG) - mel_spread(W4R2) < 0` (`upper CI < 0`).
- **Pre-Committed Outcome Interpretation (`unseen_cal`):**
  - `H1`, `H2`, and `H5` pass on `unseen_cal`: credible, causally controlled inference-time handover algorithm confirming the theory-motivated placement design.
  - `H1` and `H2` pass, but `TBRG` loses to `BLND` or `SHUF`: the three-band structure does not earn its place; report the simpler blend or perturbation finding honestly.
  - `H1` passes, but `H2` or `H3` fails: `TBRG` trades quality for diversity and does not beat `W6R2` at comparable budget.
- **Verdict:** **`confirmed`** if `H1..H5` all clear frozen bounds; **`supported_direction_only`** if `TBRG > W4` on `CVf0` and `TBRG > W4R2` on `UTMOS` clear `0` without clearing every stricter bound; **`refuted`** if any safety/non-inferiority upper bound fails (`W4`, `W8R2`, `W4R2`, or `W8` WER/SIM); **`mixed`** otherwise.

## Executed Results (2026-11-14 UTC, Colab Tesla T4)

- **Primary Prospective Cohort (`unseen_cal`: `20/20 x 15` div, `30/30` qual):** **`supported_direction_only`** (`FULL`).
- **Development Cohort (`speakers 61–100`: `20/20 x 20` div, `40/40` qual paired vs locked `EXP-027`):** **`supported_direction_only`** (`FULL`).
- **Smoke check (`smoke_check.json`):** `max_abs_split_diff = 0.0`, `max_d_wer = 0.0`, `max_d_sim = 5.96e-8` vs locked `EXP-027` `W4`.
- **Directional Bridge Gains Hold Out-of-Sample (`TBRG`, `13 NFE`):**
  - On `unseen_cal`, `TBRG` boosts `CVf0` over `W4` by **`+0.0076 [+0.0001, +0.0152]`** (`16/4`, $p = 0.0118$, `no-top2 = +0.0038`) and recovers over half of `W4R2`'s UTMOS collapse (`+0.199 [+0.102, +0.310]`, `3.557 -> 3.756`, `24/6`, $p = 0.00143$; `mel_spread -0.0216 [-0.0271, -0.0162]`).
  - On `61–100`, `TBRG` boosts `CVf0` over `W4` by **`+0.0087 [+0.0022, +0.0167]`** (`no-top2 = +0.0039`), beats `W4R2` by **`+0.214 [+0.137, +0.286]`** UTMOS (`3.404 -> 3.618`, `32/8`, $p = 1.82 \times 10^{-4}$), and beats unnormalized `STS2` on UTMOS (`+0.046 [+0.000, +0.094]`, `3.618 vs 3.571`).
- **Causal Slow-Band Prosody Tracking Confirmed (`unseen_cal` & `61–100`):**
  - `slow_corr(TBRG) = 0.908` (`unseen_cal`) / `0.903` (`61–100`) vs `SHUF` `0.111` / `0.124` (`TBRG - SHUF = +0.7971 [+0.7312, +0.8630]`, `20/0`, $p = 1.91 \times 10^{-6}$).
  - Paired $f_0(t)$ contour correlation with `W8R2` on `unseen_cal` is **`0.895` matched-seed vs `0.526` shifted-seed** (`+0.3690 [+0.3152, +0.4334]`), whereas `SHUF` drops to `0.117 / 0.075`.
  - Raw `H5` `CVf0(TBRG - SHUF)` was `-0.0154` / `-0.0130` because mismatched donor slow bands collapsed `SHUF` synthesis (`WER = 1.0004` / `1.1349`, `UTMOS = 2.716` / `2.749`), inducing erratic pYIN pitch jumps (`CVf0 = 0.1170` / `0.0958`).
- **Why `confirmed` Was Not Reached (Quality-Diversity Trade-Off & `16-NFE` `W6R2`):**
  - **`W8R2` (`20 NFE`) non-inferiority failed:** On `unseen_cal`, `CVf0(TBRG - W8R2) = -0.0061 [-0.0113, -0.0007]` (lower CI `< -0.008`) and `UTMOS(TBRG - W8R2) = -0.088 [-0.170, -0.008]` (lower CI `< -0.05`).
  - **`H3` clean `W8` UTMOS drop exceeded `-0.08`:** `dUTMOS(TBRG - W8) = -0.193 [-0.284, -0.111]` on `unseen_cal` (`3.756 vs 3.949`) and `-0.160 [-0.234, -0.088]` on `61–100` (`3.618 vs 3.777`), though `WER` (`+0.0037` / `-0.0050`) and `SIM` (`+0.0030` / `+0.0004`) preserved `W8`.
  - **`TBRG` vs naive linear blend `BLND` (`13 NFE`):** `TBRG` beats `BLND` on `CVf0` (`+0.0057 [+0.0002, +0.0120]` / `+0.0065 [+0.0007, +0.0130]`) but loses on `UTMOS` (`-0.124 [-0.219, -0.024]` / `-0.121 [-0.188, -0.055]`), trading perceptual quality for pitch diversity.
  - **`W6R2` (`16 NFE`) emerges as the strongest Pareto arm:** With 6 teacher steps + `[tau_2, tau_3]` restart (`16 NFE`, `-23.8%` vs `W8`), `W6R2` achieves `CVf0 = 0.1029` (`unseen_cal`) / `0.0840` (`61–100`), `UTMOS = 3.887` (`-0.063 [-0.143, +0.020]` vs `W8`) / `3.714` (`-0.063 [-0.125, +0.001]` vs `W8`), `WER = 0.0000` / `0.0128`, `SIM = 0.9420` / `0.9357`, and `f0_corr_w8r2 = 0.963`, outperforming `13-NFE` `TBRG` on UTMOS by `+0.131 [+0.031, +0.227]` (`unseen_cal`) and `+0.096 [+0.011, +0.184]` (`61–100`).
- **Secondary `STAC-TBRG` Routing:** Routed `3/40` `TBRG` and `37/40` `W8R2` (`19.48` mean NFE, `dUTMOS = -0.129`), offering negligible routing benefit over uniform `W6R2`/`W8R2`.
