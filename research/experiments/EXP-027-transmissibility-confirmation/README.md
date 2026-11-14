# EXP-027 — Transmissibility Confirmation & Step-by-Restart Interaction

- **Task:** `EVAL-020` (prepared on `2026-11-14`; awaiting Colab Tesla T4 execution)
- **Predecessor:** [EXP-026](../EXP-026-transmissibility-test/README.md) (`supported` under the frozen rule, but four blockers left the $\Phi((A - \beta)/B)$ ceiling formula unseparated from a simpler additive-erasure view)
- **Runner:** [run_transmissibility_confirm.py](../../../code/colab/run_transmissibility_confirm.py)
- **Notebook:** [DMOSpeech2_EXP027_Transmissibility_Confirmation.ipynb](../../../code/colab/DMOSpeech2_EXP027_Transmissibility_Confirmation.ipynb)
- **Preregistration:** [preregistration.json](preregistration.json) (`sha256: dd7c90581af6...`, frozen before synthesis)
- **Unit tests:** [test_transmissibility_confirm.py](../../../code/tests/test_transmissibility_confirm.py)
- **Setup log:** [2026-11-14-exp027-transmissibility-confirmation.md](../../../docs/progress/log/2026-11-14-exp027-transmissibility-confirmation.md)

## Why EXP-027 Is Needed

EXP-026 found that pure teacher (`T16`) exceeds default Step-8 handover (`W8`) in pitch variety (`+0.0226` CVf0) and restarting at $\tau_2 = 0.2929$ (`W8R2`) recovers `+0.0223` CVf0, while `W8 - W4` was flat (`+0.0007`). Four blockers prevented confirming the $\Phi((A - \beta)/B)$ ceiling law:

1. **Stored-data reuse:** `W4` and `W8` diversity in EXP-026 matched stored EXP-020 trajectories rather than fresh data.
2. **Ceiling vs simpler erasure:** `H1` and `H2` alone follow both from a bounded transmissibility ceiling $\Phi((A-\beta)/B)$ and from the simpler hypothesis that starting the student at a larger $\tau$ preserves more of any teacher output. The specific ceiling signature is the **Step-by-Restart interaction**: at $\tau_1 = 0.0761$, retention saturates near a low ceiling regardless of teacher step (`W8 - W4 ≈ 0`), whereas at $\tau_2 = 0.2929$ the extra teacher commitment from Step 8 vs Step 4 emerges (`W8R2 - W4R2 > W8 - W4`).
3. **Outlier concentration:** One development speaker carried roughly half of EXP-026's mean effect.
4. **Pitch-tracker voiced-frame confound:** `T16` and `W8R2` voiced a smaller frame share than `W8` in EXP-026 (`0.520` and `0.619` vs `0.630`).

## Fresh Design (`61–100`, All 6 Arms Fresh)

- **Speakers:** 40 fresh development speakers (`dev_items()[60:100]`), strictly disjoint from the 60 development speakers (`0:60`) used across EXP-012 through EXP-026. Clean prompts.
- **Diversity (`diversity.csv`):** 40 speakers $\times$ 20 seeds per speaker (`DIV_SEED_BASE = 1000`). Each seed runs one shared 16-step guided teacher trajectory and derives 6 arms from it:
  - `student4` (4 student passes from $\tau_0 = 0$, independent noise)
  - `W4` (13 passes: Step 4 + student $\tau_1 = 0.0761, \tau_2, \tau_3$)
  - `W8` (21 passes: Step 8 + student $\tau_1, \tau_2, \tau_3$)
  - `W4R2` (12 passes: Step 4 + late restart $\tau_2 = 0.2929, \tau_3$)
  - `W8R2` (20 passes: Step 8 + late restart $\tau_2 = 0.2929, \tau_3$)
  - `T16` (32 passes: pure 16-step guided teacher)
- **Tracker-confound controls logged per speaker:**
  - `cvf0_common_<arm>`: `CVf0` evaluated strictly on the intersection of frames voiced ($\ge 50\%$ of seeds) in all 6 arms (`voiced_common`).
  - `mel_spread_<arm>`: pitch-tracker-free pairwise relative Frobenius spread of generated slow-mel frames (`sh.decompose(..., window=9)[1]`).
- **Quality (`quality.csv`):** All 40 fresh speakers $\times$ 6 arms scored on `wer`, `similarity`, and `utmos` against fresh `W8` from the same teacher trajectory.

## Preregistered Hypotheses & Frozen Verdict Rule

- **`H1` (`T16 - W8 > 0`):** Pure teacher exceeds default Step-8 handover in pitch variety.
- **`H2` (`W8R2 - W8 > 0`):** Restarting at $\tau_2 = 0.2929$ recovers pitch variety from Step 8.
- **`H3` (`|W8 - W4| < 0.01`):** Default handover plateau at $\tau_1 = 0.0761$ holds on fresh data.
- **`H4` (`(W8R2 - W4R2) - (W8 - W4) > 0`):** Step-by-Restart interaction predicted by $\Phi((A - \beta)/B)$.
- **Exploratory `H5`:** Speaker-level Spearman $\rho(\mathrm{T16} - \mathrm{W8}, \mathrm{W8R2} - \mathrm{W8}) > 0$.

Frozen verdict (95% bootstrap CIs with $B = 2000$):

- **`refuted`** if any of `H1`, `H2`, or `H4` has 95% upper CI `< 0` in `cvf0` or `cvf0_common`.
- **`confirmed`** if `H1`, `H2`, `H4` have 95% lower CI `> 0` in both `cvf0` and `cvf0_common`, `|mean(W8 - W4)| < 0.01` in `cvf0`, slow-mel spread means of `H1`, `H2`, `H4` are `> 0`, and `mean_without_top2_abs` of `H1`, `H2`, `H4` in `cvf0` are `> 0`.
- **`supported_direction_only`** if `H1`, `H2` have 95% lower CI `> 0` and `|mean(W8 - W4)| < 0.01` in `cvf0`, without meeting all `confirmed` criteria.
- **`mixed`** otherwise.
