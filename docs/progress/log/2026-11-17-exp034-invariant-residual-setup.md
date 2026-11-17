# 2026-11-17: EXP-034 prepared: the invariant residual against SNR only and Chapter 3's drift

Task EVAL-029 (new). Follow-up to the [invariant residual log](2026-11-17-invariant-residual.md).
- **The request:** Rhenel Jhon R. Sajol asked for a notebook that compares the proposed method with the SNR alone, to see which is better.
- **Status:** prepared and preregistered before any GPU synthesis. The freeze is now `eb1a38d3225a...`, after Amendment 1 below; the first freeze was `7c2ec8fca37b...`. The main run has not started. The notebook clones branch `rhnl-3`, so the amended files must be pushed before it can run again.

## The design

**Isolating the decision signal:** every decision rule chooses between the same two payloads at the commit point t = 0.2929:
- **`CPH8`:** the teacher's estimate, 12 passes;
- **`CPQH`:** its noise-safe version, 13 passes.

Both payloads are synthesized and scored for every case, so each rule's outcome is read off the same outputs, and the comparison isolates the decision signal.

| Rule | Hands over `CPQH` when |
|---|---|
| `SNR_ONLY` | Definition 3.3 < 20 dB |
| `SNR_DRIFT` | SNR < 20 dB or drift d₄ < 0.2252 |
| `SNR_R` (proposed) | SNR < 20 dB or r > 0.3400 |
| `R_ONLY` | r > 0.3400 |

**Where the thresholds come from:** the CPU screen's calibration half (`invariant_residual_screen.json`, hashed in the preregistration).
- **Drift's direction:** fixed as the screen found it. Low drift flags noise, as EXP-012 reported.

**References:**
- `STU`, the released student;
- `STUR2`, its draft restarted at 0.2929;
- `W8`, Chapter 3's reference;
- `P25`, the paper's setting, in the variety and timing stages.

**Stages:**

| Stage | Cases | Notes |
|---|---|---|
| Quality | Development positions 31–60 × 8 conditions = 240 | Clean, `noisy15`, `gauss25`, `pink25`, `hp25`, `reverb09`, telephone, `babble15` |
| Variety | Positions 41–60 × 12 seeds × clean and `gauss25` | |
| Cost | Positions 31–40, clean | Each arm and rule on its own, passes counted on both networks |

**Speakers:**
- **The screens' sentences are excluded:** all come from positions 1–30, which this run does not use. A unit test checks this.
- **Positions 41–60** have not been used in any T4 quality run.
- **Positions 31–40** were used only in EXP-026.

**New prompt conditions:**
- **`pink25` and `hp25`:** the screens' pink noise and high-pass hiss at 25 dB, added to the shared pipeline at run time. A unit test checks that the prompts are identical to the screens' prompts.
- **The released conditions** pass through unchanged.

**Frozen hypotheses** (full text in the [README](../../../research/experiments/EXP-034-invariant-residual/README.md) and the preregistration):

| | Hypothesis |
|---|---|
| H1 | `SNR_R` beats `SNR_ONLY`: ΔUTMOS lower bound > 0, with WER and similarity not worse by more than 0.01 |
| H2 | `SNR_R` beats `SNR_DRIFT` in the same way |
| H3 | AUC(r) ≥ 0.80 and ≥ drift's AUC + 0.10 |
| H4 | Wrong clean flags ≤ 15%; correct catches of a broken teacher do not count, as recommended after the CPU screen |
| H5 | Clean variety ≥ 90% of `CPH8`'s gain |

**Verdict labels:** `snr_r_better`, `snr_r_better_quality` and `snr_only_not_beaten`.

**Estimated T4 time:** about 3.5 hours. The notebook resumes from Drive.

## Files

- **Runner:** `code/colab/run_invariant_residual.py`, with the subcommands `freeze`, `smoke`, `run` and `analyze`. It reuses EXP-033's teacher loops, pass counter and student, and EXP-032's bootstrap and reproduction helpers.
- **Notebook:** `code/colab/DMOSpeech2_EXP034_Invariant_Residual.ipynb`, generated from EXP-033's notebook with EXP-034's paths, row targets and descriptions.
- **Preregistration:** `research/experiments/EXP-034-invariant-residual/preregistration.json`. It holds 15 code hashes and 3 input hashes: EXP-031's and EXP-033's quality CSVs and the screen output.
- **Tests:** `code/tests/test_invariant_residual_run.py`, 10 tests:
  - the thresholds equal the screen's;
  - the shaped-noise prompts equal the screens' prompts;
  - the released conditions pass through, and installing the new conditions is reversible;
  - r and the spread-weighted SNR equal the screened versions;
  - the rules and the pass accounting;
  - the cohorts are disjoint from the screens';
  - freeze, check and tamper detection;
  - a full analysis on synthetic results.

## Amendment 1 (same day, before any main synthesis)

**What happened:** Rhenel Jhon R. Sajol ran the notebook on a Colab T4. The smoke test stopped at `babble15`:

    Babble condition requested but no DEMAND track loaded (--demand_dir).

**What had already passed** on development speaker 1:
- every identity check, with a difference of 0.0;
- every arm and rule run on its own against the shared run;
- every pass count against the cost table: STU 0 + 4, STUR2 0 + 3, CPH8 10 + 2, CPQH 10 + 3, W8 18 + 3, P25 28 + 2, SNR_ONLY 10 + 2, SNR_R 10 + 3;
- 7 of the 8 quality conditions.

| Condition | SNR | r | Rule's choice |
|---|---:|---:|---|
| Clean | 28.2 dB | 0.369 | CPQH |
| `noisy15` | 17.5 dB | 0.604 | CPQH |
| `gauss25` | 25.0 dB | 0.621 | CPQH |
| `pink25` | 25.2 dB | 0.581 | CPQH |
| `hp25` | 25.0 dB | 0.405 | CPQH |
| `reverb09` | 16.6 dB | 0.326 | CPQH, by the SNR |
| Telephone | 31.3 dB | 0.264 | CPH8 |

**The fix,** following EXP-031:
- **Runner:** it requires `--demand_dir`, loads `PCAFETER/ch01.wav` when the pipeline is built, and writes `demand_sha256.txt`, which the manifest hashes.
- **Notebook:** it gains EXP-031's DEMAND download cell, from Zenodo with a Drive cache, and passes the folder to the smoke and run commands.
- **Preregistration:** it records the amendment and was re-frozen as `eb1a38d3225a...`; it was `7c2ec8fca37b...`.

No arm, threshold, cohort, estimand or hypothesis changed.

## Checks run

- **Unit tests:** `test_invariant_residual_run.py`, 10 tests, OK. The synthetic analysis renders the report and reaches the expected verdict.
- **Preregistration:** `freeze`, then `check_prereg`, passed.
- **Not run here:** the smoke test and synthesis need the Colab T4.
- **Workspace check:** `python scripts/check_workspace.py` PASSED.
- **Full suite:** `python -m unittest discover -s code/tests` ran 429 tests, OK.
- **Preregistration after all edits:** `check_prereg` still passes.
