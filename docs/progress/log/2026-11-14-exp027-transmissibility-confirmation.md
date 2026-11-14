# 2026-11-14 — EXP-027 (EVAL-020): Transmissibility Confirmation Setup & Preregistration

## Summary

Prepared and froze **EXP-027** ([README](../../../research/experiments/EXP-027-transmissibility-confirmation/README.md), task `EVAL-020`, `status: todo`) to test the transmissibility theory of teacher-to-student handover on **40 fresh disjoint speakers** (`dev_items()[60:100]`, speakers 61-100, clean prompts) and address all four blockers identified after [EXP-026](2026-11-14-exp026-transmissibility-results.md). No GPU synthesis has been executed yet.

## Changes Made

- **Runner:** Added [run_transmissibility_confirm.py](../../../code/colab/run_transmissibility_confirm.py) implementing:
  - Strict fresh-speaker isolation (`DEV_N = 60`, `FRESH_N = 40`) with disjointness check.
  - Shared-trajectory synthesis of all 6 arms (`student4` [4 NFE], `W4R2` [12 NFE], `W4` [13 NFE], `W8R2` [20 NFE], `W8` [21 NFE], `T16` [32 NFE]) on 40 speakers x 20 seeds (`DIV_SEED_BASE = 1000`) plus clean quality scoring (`wer`, `similarity`, `utmos` vs fresh `W8`) across all 40 speakers.
  - Pitch-tracker confound controls: intersection-frame `cvf0_common_<arm>` across frames voiced by `>= 50%` of seeds in all 6 arms, plus tracker-free `mel_spread_<arm>` via `sh.decompose(..., window=9)[1]`.
  - Step-by-Restart interaction hypothesis `H4 = (W8R2 - W4R2) - (W8 - W4) > 0` separating the bounded $\Phi((A-\beta)/B)$ ceiling from linear erasure, alongside outlier-robust summaries (`median`, `mean_without_top2_abs`, `trimmed_mean_10`, `max_abs_share`, exact two-sided sign test) and the frozen four-level verdict rule (`confirmed`, `supported_direction_only`, `refuted`, `mixed`).
  - Preregistration freeze/check (`freeze`, `check_prereg`), 2-speaker dev reproduction check (`smoke`), resumable `run`, and `analyze` (`--allow_partial`, writes `MANIFEST.json`).
- **Preregistration:** Froze [preregistration.json](../../../research/experiments/EXP-027-transmissibility-confirmation/preregistration.json) (`sha256: dd7c90581af6...`) before any synthesis and excluded `research/experiments/EXP-027-transmissibility-confirmation/results/**` text normalization in `.gitattributes`.
- **Colab Notebook:** Created the 17-cell T4 notebook [DMOSpeech2_EXP027_Transmissibility_Confirmation.ipynb](../../../code/colab/DMOSpeech2_EXP027_Transmissibility_Confirmation.ipynb) targeting branch `rhnl-3` and `Shonget-DMOSpeech2` commit `8a7868fc...`, with Drive caching, periodic Drive backup, smoke test, full/partial analysis, and ZIP export.
- **Unit Tests:** Added [test_transmissibility_confirm.py](../../../code/tests/test_transmissibility_confirm.py) (14 tests covering fresh disjointness, pass costs matching EXP-026, `common_cvf0` frame-dropout resistance, monotonic `slow_mel_spread`, mocked `outputs`/`run_diversity`/`run_quality`/`smoke` loops, all four `analyze` verdicts including `--allow_partial`, preregistration/code-hash tampering rejection, and notebook AST syntax).
- **Progress Tracking:** Registered `EVAL-020` in `docs/progress/tasks.json`, re-rendered `docs/progress/TASKS.md`, updated `research/experiments/README.md` and `docs/progress/STATUS.md`.

## Verification

- `python code/colab/run_transmissibility_confirm.py freeze --force` succeeded (`sha256=dd7c90581af6...`).
- `python -m unittest discover -s code/tests -p "test_transmissibility_confirm.py" -v`: 14 tests passed (`OK`).
- Full test discovery (`python -m unittest discover -s code/tests`) and workspace check (`python scripts/check_workspace.py`) run prior to handoff.

## Remaining Blockers

- **Colab Tesla T4 execution:** Needs execution of `smoke` and `run` on Google Colab T4 using `DMOSpeech2_EXP027_Transmissibility_Confirmation.ipynb` (`EVAL-020` remains `todo` until GPU results are committed).
- **LaTeX build:** Not run/required for this experimental setup (`paper/` files unchanged; TeX build unverified in this turn).
