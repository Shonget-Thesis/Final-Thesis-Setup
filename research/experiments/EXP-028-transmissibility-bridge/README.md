# EXP-028 — Transmissible Slow-Band Bridge (`TBRG`) & `STAC-TBRG` Handover

- **Task:** `EVAL-021` (`todo`, prepared and preregistered on `2026-11-14`; ready for fast ~25–30 min Colab Tesla T4 execution)
- **Predecessors:** [EXP-027](../EXP-027-transmissibility-confirmation/README.md) (`supported_direction_only`, established clean-prompt transmissibility loss `H1` and late-restart recovery `H2` on 40 fresh speakers `61–100`) | [EXP-018](../EXP-018-swsr-handover-prelim/README.md) (`STAC`) | [EXP-020](../EXP-020-selective-handover-prelim/README.md) (`decompose`)
- **Runner:** [run_transmissibility_bridge.py](../../../code/colab/run_transmissibility_bridge.py)
- **Notebook:** [DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb](../../../code/colab/DMOSpeech2_EXP028_Transmissibility_Bridge.ipynb)
- **Preregistration:** [preregistration.json](preregistration.json) (`sha256: 8d6d68afa1d2...`, frozen before synthesis)
- **Unit tests:** [test_transmissibility_bridge.py](../../../code/tests/test_transmissibility_bridge.py)
- **Setup log:** [2026-11-14-exp028-transmissibility-bridge.md](../../../docs/progress/log/2026-11-14-exp028-transmissibility-bridge.md)

## Motivation from EXP-027

`EXP-027` evaluated six arms on fresh speakers `61–100` (`40` quality, `40 x 20` diversity) and revealed a clean structural split at **12–13 NFE**:

- **`W4` (13 NFE):** Retains clean perceptual quality (`UTMOS 3.759`, `WER 0.0159`, `SIM 0.9312`) because the first student step $\tau_1 = 0.0761$ produces a clean acoustic draft $S_1$, but $\tau_1 = 0.0761$ bottlenecks pitch variety (`CVf0 0.0770` vs `W8R2 0.0893` / `T16 0.0911`).
- **`W4R2` (12 NFE):** Skips $\tau_1$ and injects raw $T_4$ directly at $\tau_2 = 0.2929$, unlocking high diversity (`CVf0 0.0859`, `+0.0089` over `W4`, $p = 2.94 \times 10^{-4}$), **but drops UTMOS by `-0.373` (`3.404`) and spikes slow-mel dispersion (`1.0299`)** because raw $T_4$ lacks $\tau_1$ denoising on `voice` and `fine` components.

## Proposed 13-NFE Solution (`TBRG`) & `STAC-TBRG`

1. **Teacher Steps `0..4` (`10 NFE`):** Run the guided Euler teacher to Step 4 endpoint $T_4$, recording drifts $d_1 \dots d_4$, cumulative STAC trajectory action $A_4 = \sum_{j=1}^4 d_j$, and contraction $\rho_4 = d_4 / (d_3 + 10^{-8})$ at `0` extra NFE.
2. **Clean Acoustic Anchor (`1 NFE`):** Run the student's $\tau_1 = 0.0761$ pass from $T_4$ to obtain $S_1$.
3. **Energy-Matched Transmissible Slow-Band Bridge (`TBRG`, `0 NFE`):** Decompose generated frames `plen:` via `sh.decompose(..., window=9)` into $(v_T, s_T, f_T)$ and $(v_S, s_S, f_S)$, and inject Frobenius-matched teacher slow prosody onto $S_1$'s `voice` and `fine` structure before the $\tau_2 = 0.2929$ step:
   $$\tilde{s}_T = \frac{\|s_S\|_F}{\|s_T\|_F + 10^{-8}} s_T, \qquad X_{\tau_2}^{\text{TBRG}}[0, \text{plen}:, :] = v_S + \tilde{s}_T + f_S$$
4. **Student Tail (`2 NFE` -> `13 NFE` Total):** Complete `[tau_2, tau_3] = [0.2929, 0.6173]` (**`-38.1%` NFE vs `W8`**, **`-35.0%` vs `W8R2`**, **`-59.4%` vs `T16`**).
5. **Ablation (`STS2`, `13 NFE`):** Identical pipeline without slow-band Frobenius normalization ($\tilde{s}_T = s_T$) to isolate energy matching.
6. **Adaptive Routing (`STAC-TBRG`, `13–20 NFE`):** Out-of-fold 5-fold speaker cross-validation routing between `TBRG` (`13 NFE`) and stored `W8R2` (`20 NFE`) via Step-4 relative action velocity $r_4 = d_4 / (A_4 + 10^{-8})$.

## Fast Fresh-Speaker Design (`61–100`)

- **Reference Reuse (Verified by SHA-256):** Reuses locked [EXP-027](../../../research/experiments/EXP-027-transmissibility-confirmation/results/MANIFEST.json) outputs for `student4`, `W4`, `W4R2`, `W8`, `W8R2`, `T16` on the exact same fresh speakers (`61–100`) and seeds (`DIV_SEED_BASE = 1000`).
- **Smoke Gate (`2` fresh speakers):** Confirms exact split-student identity (`max |W4_split - W4| <= 1e-5`) and reproduces stored `EXP-027` `W4` `WER` and `SIM` within `0.01`.
- **Diversity:** First `20` fresh speakers (`61–80`) $\times$ `20` seeds -> `CVf0`, voiced share, and `slow_mel_spread` for `TBRG` and `STS2`.
- **Quality:** All `40` fresh speakers (`61–100`) -> `wer`, `similarity`, `utmos`, and `d_1..d_4, A_4, rho_4` paired against `W8`, `W4R2`, and `W8R2`.
- **Preregistered Hypotheses & Verdict:**
  - **`H1`:** `CVf0(TBRG) - CVf0(W4) > 0`
  - **`H2`:** `UTMOS(TBRG) - UTMOS(W4R2) > 0`
  - **`H3`:** Clean `W8` non-inferiority (`upper CI dWER <= +0.01`, `lower CI dSIM >= -0.01`, `mean dUTMOS >= -0.12`)
  - **`H4`:** `mel_spread(TBRG) - mel_spread(W4R2) < 0`
  - **`refuted`** if `H1` upper CI `< 0`, `H2` upper CI `< 0`, or `H3` fails `WER`/`SIM` non-inferiority; **`confirmed`** if `H1` & `H2` clear `0` (plus `no-top2 > 0`), `H3` passes non-inferiority + `mean dUTMOS >= -0.12`, and `H4` upper CI `< 0`; **`supported_direction_only`** if `H1` & `H2` clear `0` without full confirmation; **`mixed`** otherwise.
