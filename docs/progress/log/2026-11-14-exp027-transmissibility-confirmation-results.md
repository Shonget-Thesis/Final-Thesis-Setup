# 2026-11-14 — EXP-027 (EVAL-020): Transmissibility Confirmation Results Imported

Task: `EVAL-020` (now `review`). Rhenel Jhon R. Sajol executed [DMOSpeech2_EXP027_Transmissibility_Confirmation.ipynb](../../../code/colab/DMOSpeech2_EXP027_Transmissibility_Confirmation.ipynb) on Google Colab (Tesla T4, branch `rhnl-3` commit `2112059`) and supplied `DMOSpeech2_EXP027_Transmissibility_Confirmation_Results (1).zip` (`sha256: 1a9f832c698d84f611f5cfc1fa0428c9226252e76442024329fba71192ee36d0`).

## Import and Verification Checks

- **Archive extracted to:** `research/experiments/EXP-027-transmissibility-confirmation/results/` (`diversity.csv`, `quality.csv`, `transmissibility_confirm_results.json`, `TRANSMISSIBILITY_CONFIRM_REPORT.md`, `MANIFEST.json`, `smoke/diversity.csv`, `smoke/quality.csv`, `smoke/smoke_check.json`).
- **Line-ending-normalized SHA-256 verification (`sh.sha256_text`):** All files matched `MANIFEST.json`:
  - `preregistration.json`: `323ca9540594...` (payload `sha256: dd7c90581af6...`)
  - `diversity.csv`: `b38c8d8c293d...` (`40/40` fresh speakers $\times$ `20` seeds)
  - `quality.csv`: `fadc58e2d546...` (`40/40` fresh speakers $\times$ `6` arms)
  - `transmissibility_confirm_results.json`: `9d8578ffa940...`
  - `TRANSMISSIBILITY_CONFIRM_REPORT.md`: `82af9dfe9ab0...`
- **Smoke reproduction check (`smoke_check.json`):** `max_d_wer = 0.0`, `max_d_sim = 0.0` against stored EXP-018 `W4` and `W8`.
- **Re-analysis check:** Running `run_transmissibility_confirm.py analyze` on a fresh temporary copy of `diversity.csv` and `quality.csv` reproduced identical `TRANSMISSIBILITY_CONFIRM_REPORT.md` and identical `transmissibility_confirm_results.json` (excluding `utc` timestamp).

## Results Summary (`supported_direction_only`)

| Arm | Passes | `CVf0` | Voiced Share | Common-Frame `CVf0` (`0.389` share) | Slow-Mel Spread | Fresh `WER` ($\Delta$ vs `W8`) | Fresh `SIM` ($\Delta$) | Fresh `UTMOS` ($\Delta$) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `student4` | 4 | `0.0506` | `0.561` | `0.0491` | `0.5299` | `+0.0000` | `-0.0010` | `-0.038` |
| `W4` | 13 | `0.0770` | `0.602` | `0.0749` | `0.9436` | `+0.0000` | `-0.0010` | `-0.018` |
| `W8` | 21 | `0.0772` | `0.609` | `0.0754` | `0.9343` | `0.0159` | `0.9322` | `3.777` |
| `W4R2` | 12 | `0.0859` | `0.562` | `0.0853` | `1.0299` | `+0.0009` | `-0.0029` | `-0.373` |
| `W8R2` | 20 | `0.0893` | `0.604` | `0.0862` | `0.9351` | `+0.0000` | `+0.0012` | `-0.105` |
| `T16` | 32 | `0.0911` | `0.526` | `0.0874` | `0.9244` | `-0.0023` | `-0.0049` | `-0.548` |

Note: One of the 40 fresh speakers produced `NaN` for `cvf0_W4` (`< 50%` voiced seeds), so contrasts involving `W4` or common-frame masks across all 6 arms have `n = 39` (sign-test non-zero `n = 38` on common-frame `H1..H4`).

- **`H1` (`T16 - W8 > 0`):** `+0.0139 [+0.0057, +0.0232]` (`33/7`, $p = 4.23 \times 10^{-5}$, `no-top2 = +0.0124`); `cvf0_common = +0.0120 [+0.0044, +0.0188]` (`32/6`, $p = 2.43 \times 10^{-5}$).
- **`H2` (`W8R2 - W8 > 0`):** `+0.0121 [+0.0081, +0.0159]` (`34/6`, $p = 8.36 \times 10^{-6}$, `no-top2 = +0.0106`, max share `0.093`); `cvf0_common = +0.0107 [+0.0049, +0.0158]` (`34/4`, $p = 6.04 \times 10^{-7}$).
- **`H3` (`|W8 - W4| < 0.01`):** `+0.0008 [-0.0008, +0.0023]` (`cvf0_common = +0.0005 [-0.0008, +0.0019]`).
- **`H4` (`(W8R2 - W4R2) - (W8 - W4) > 0`):** `+0.0030 [-0.0019, +0.0078]` (`24/15`, $p = 0.200$, `no-top2 = +0.0032`); `cvf0_common = +0.0003 [-0.0095, +0.0076]` (mean lowered by one negative common-frame speaker; median `+0.0031`, `no-top2 = +0.0032`, sign test `27/11`, $p = 0.0139$).
- **Exploratory `H5`:** Spearman $\rho(\mathrm{T16}-\mathrm{W8}, \mathrm{W8R2}-\mathrm{W8}) = +0.539$.
- **Clean-prompt interpretation:** Blockers #1 (fresh synthesis, no stored reuse), #3 (broad across `33/7` and `34/6` speakers rather than 1 outlier), and #4 (`cvf0_common` matches `cvf0` closely) are definitively resolved. On clean prompts, `W4` has nearly reached commitment (`W4R2 - W4 = +0.0089 [+0.0051, +0.0129]`), leaving only `+0.0038` between `W8R2` and `W4R2` (`27/12`, $p = 0.0237$) so `H4` is small (`+0.0030`) and statistically overlaps zero. Slow-mel Frobenius spread measures total un-converged slow-envelope variance across seeds (`W4R2` highest at `1.0299`, consistent with its `-0.373` UTMOS penalty) rather than pitch-contour diversity, causing `mel_spread` `H1`/`H4` to turn negative.

## Checks Run

- `python scripts/render_progress.py` — updated `docs/progress/TASKS.md`.
- `python scripts/check_workspace.py` — `Workspace check PASSED: structure, task evidence, source hashes, links, and manuscript references.`
- `python -m unittest discover -s code/tests -p "test_transmissibility_confirm.py" -v` — 14 tests OK (`7.403s`).
- `python -m unittest discover -s code/tests -p "test_evidence_manifest.py" -v` — 2 tests OK (`0.039s`).
- `python -m unittest discover -s code/tests` — 265 tests OK (`119.803s`).
- LaTeX build: not run (`paper/` unchanged; TeX build unverified for this import).
