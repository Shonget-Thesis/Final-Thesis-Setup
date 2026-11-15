# 2026-11-15: EXP-031 prepared: SNR + TOH against the previous handover methods

Task: EVAL-024 (todo). Requested by Rhenel Jhon R. Sajol after EXP-030: "do exp 31 so we can compare all prev exp". This follows their plan to combine TOH with the SNR gate so the method adapts to the input. Nothing has been synthesized, and nothing here is adopted.

## Design decisions

- **The method under test, `SNR_TOH`:**
  - **The gate:** Chapter 3's SNR gate (θ = 20 dB), unchanged.
  - **Below it:** the student alone.
  - **At or above it:** TOH3.
  - **Why:** this keeps one known part and one new part, the bar the requester set on 2026-11-15.
- **The previous methods, with their earlier thresholds:**
  - W4 and W8;
  - STAC, with EXP-018's `DEFAULT_STAC_LAMS`;
  - Chapter 3's policy, with λ = 0.35 (Appendix D's four-fold CV) and with λ = 1.20 (Appendix D's LTT certificate);
  - SNR + W4, which isolates what TOH adds.

  STAC and the Chapter 3 policy are read per case from W4, W5, W6 and W8, using the recorded prompt SNR and teacher drifts. The unit tests check:
  - the copied thresholds against their sources;
  - the Chapter 3 policy against `scheduling/sampler.evaluate_handover_policy`, extracted from its source code and run on 6,000 random cases.
- **A teacher-free control (`RND`), replacing EXP-030's mirrored map:**
  - **What it is:** a smooth random monotone map that re-times the student's draft with no teacher, at 4 passes.
  - **Size:** fixed from EXP-030's TOH3 maps on clean prompts (mean RMS 0.0270 of the length, largest displacement 1.98 × RMS).
  - **Why these settings:** a simulation over 250–750 frames chose an RMS target of 0.029 and smoothing of 0.10 of the length, giving 0.0261 and 1.90. A first construction that clipped slopes after scaling broke the limits, so the map now shrinks uniformly instead, and the limits hold by construction.
- **Conditions, chosen from stored EXP-011 data** (speakers 1–20, W4 and W8, before anything was built):

  | Condition | Median prompt SNR | W4 UTMOS (clean 3.86) | Why included |
  |---|---|---|---|
  | `reverb03` | 28.5 dB, read as clean | 2.71 | The case where TOH's safety net could matter |
  | `reverb09` | 19.0 dB | 1.47 | Around the gate |
  | `babble15` | 19.4 dB | 3.34 | Around the gate |
  | `gauss25`, `noisy15` | — | — | EXP-030's white-noise conditions |
  | `clean` | — | — | Baseline |

  Telephone prompts were dropped: no arm differed on them in EXP-030.
- **Speakers:**
  - **Quality:** development speakers 1–30; **diversity:** 1–20 × 12 seeds.
  - **New to TOH:** EXP-030 used 61–90. The calibration and test splits stay untouched.
  - **Reproduction:** speakers 1–20 are EXP-011's pilot speakers with the same seeds, so prompt SNR and W4 and W8 can be checked against EXP-011.
  - **A bias:** STAC's and CH3's thresholds were set on development data that include these speakers, which favours them.
- **Yardstick:**
  - **Quality** is judged on UTMOS, WER against the text, and similarity against the clean ground truth, not by agreement with W8, which copies the prompt's condition.
  - **Definition 3.1's failures against W8** are still reported for continuity. Whether Chapter 3 should change its yardstick is a team decision.
- **A new variety measure:**
  - **Aligned CVf0:** each seed is aligned to the first in time with TOH's DTW before CVf0 is computed.
  - **Timing spread:** reported alongside, in frames.
  - **Why:** together they separate pitch variety from timing variety. A unit test confirms that timing-only differences disappear after alignment while frame-by-frame CVf0 stays high.

## What was created

- **Runner:** `code/colab/run_method_comparison.py`:
  - the random map;
  - the policies;
  - aligned CVf0;
  - synthesis of 7 quality arms and 5 diversity arms;
  - the smoke test;
  - freeze and check;
  - analysis and report.
- **Preregistration:** `research/experiments/EXP-031-snr-toh-method-comparison/preregistration.json`, frozen before any synthesis:
  - **Hashes:** payload sha256 `ec06ed0568b4...`, file `3ce09fe26e3b...`, 11:56 UTC.
  - **What it locks:**
    - the runner and its 10 dependencies;
    - EXP-027's `quality.csv` (smoke reproduction);
    - EXP-011's `pilot_grid.csv` (reproduction);
    - EXP-030's `quality.csv` (the random map's size);
    - `failure_config.json`.
- **Notebook:** `code/colab/DMOSpeech2_EXP031_SNR_TOH_Method_Comparison.ipynb`, 16 cells, built from EXP-030's. It adds a DEMAND cafeteria-babble download step, cached on Drive, and `--demand_dir`.
- **Tests:** `code/tests/test_method_comparison.py`, 16 tests:
  - **Random map:** the limits hold, it is deterministic per seed, its size matches EXP-030's maps, and short inputs give the identity.
  - **Policies:** the constants match their sources, CH3 matches the released policy function, STAC's semantics, and the routing.
  - **Aligned CVf0:** timing differences vanish after alignment; identical seeds give zero.
  - **Design:**
    - costs;
    - cohorts: new to TOH, outside the calibration and test splits, and equal to EXP-011's pilot;
    - conditions;
    - dependencies.
  - **Verdicts:** every branch, on synthetic data.
  - **Tampering detection, and the notebook.**
- **README:** `research/experiments/EXP-031-snr-toh-method-comparison/README.md`.
- **Tracking:**
  - EVAL-024 in `tasks.json` and `TASKS.md`;
  - STATUS;
  - the experiments index;
  - `.gitattributes`, so result bytes stay unchanged.

## Checks run

- `python -m unittest discover -s code/tests -p "test_method_comparison.py"`: 16 tests, OK.
- `check_prereg()`: accepts `ec06ed0568b4...`.
- A synthetic analysis was rendered and read, to check the report layout.
- `python scripts/check_workspace.py`: PASSED.
- `python -m unittest discover -s code/tests`: 311 tests, OK.
- LaTeX build: not run. The manuscript did not change.

**Not testable here:** the PyTorch path of the runner (this PC has no PyTorch). The smoke test checks it on the T4 before any synthesis.

## Remaining blockers

- **Push before running:** the notebook clones `rhnl-3`, and the preregistration locks EXP-030's committed `quality.csv`. EXP-030's import and EXP-031 must both be pushed first.
- **The GPU run:** about 2 hours 10 minutes on a T4, plus the DEMAND download (cached on Drive after the first time).
- **Before any thesis claim:**
  - a full novelty audit of TOH (family and neighbouring-field searches);
  - the team's and adviser's decisions on replacing drift in Chapter 3 and on its failure yardstick.

## Amendment 1 (2026-11-15, before any synthesis)

**Why:** the requester judged the first design's method, SNR + TOH, to be "just a policy" (an if-else around the method) and asked for handover formulas instead. Two new candidates came out of the same day's work:
- the Quantile Handover (QH), from the [theory note](2026-11-15-handover-invariance-theory.md);
- the Blind-Teacher Handover (BTH), from the [design log](2026-11-15-blind-teacher-handover-design.md).

The requester asked for both to be added.

**What changed:**
- **New arms:** `QH3` and `BTH3`, in both stages. Each costs 12 passes.
  - **QH3:** in each band, the student's draft values placed in the time order of the teacher's Step-3 estimate.
  - **BTH3:** TOH's map, taken from a teacher that receives only the target text over the target's frames, with the audio condition dropped in both guidance branches, onto the student's draft.
- **New primary methods:** the three single-formula handovers TOH3, QH3 and BTH3, applied to every input. Each is judged by:
  - **T1 and T2:** against the random re-timing;
  - **C:** reverberation and babble, against W4;
  - **M:** against W4, W8, STAC, CH3 and CH3_LTT.
- **Demoted to exploratory:** SNR_TOH and the first design's S1.
- **Smoke test:** new exact checks:
  - QH of the draft onto itself equals the draft;
  - QH's values equal the draft's values in every band;
  - the blind teacher gives the same payload twice;
  - BTH's map passes the map checks.
- **The QH code on the GPU** uses only out-of-place operations (a gather with the inverse permutation, checked equal to the reference formula), because the student returns inference-mode tensors.
- **Run time:** about 3 hours on a T4. The notebook text was updated.
- **Unchanged:** speakers, seeds, conditions, the routing constants, the random map, the bars and the reproduction checks.

**Re-freeze:**
- **First freeze:** `ec06ed0568b4...`. It never governed any result: nothing has been synthesized.
- **New payload:** `df26da1412bf...`; file `72322b205c95...`; 13:14 UTC.
- **Runner hash:** `a0f434381f08...`.

**Checks:**
- `python -m unittest discover -s code/tests -p "test_method_comparison.py"`: 21 tests, OK.
- `check_prereg()`: accepts `df26da1412bf...`.
- `python scripts/check_workspace.py`: PASSED.
- `python -m unittest discover -s code/tests`: 316 tests, OK.
- **Not testable here:** the PyTorch paths of QH and BTH, because this PC has no PyTorch. The smoke test checks them on the T4 before any synthesis.
