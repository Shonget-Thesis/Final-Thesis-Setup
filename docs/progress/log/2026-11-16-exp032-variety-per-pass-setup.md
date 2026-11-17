# 2026-11-16: EXP-032 prepared (variety per teacher pass against the paper's t_switch = 0.25)

Task: EVAL-025 (todo). Requested by Rhenel Jhon R. Sajol after the EXP-031 import, in three steps:
1. Read the DMOSpeech 2 paper on why it uses a teacher.
2. Measure against the paper's t_switch = 0.25, not the released code's default ("their paper is 0.25, code don't matter").
3. Run it on clean and noisy prompts "so we can study, then create another novel handover solution".

Nothing has been synthesized. No handover method is tested, and nothing is adopted.

## Reasoning

**What the paper says** (arXiv 2507.14988v1, read on 2026-11-16):
- **Why the teacher:** distillation makes the student focus on high-probability regions ("mode shrinkage"). Its outputs vary less across seeds, mainly in prosody and timing (§3.3.1).
- **What the teacher does:** its early steps set "phoneme durations, pauses, pitch contours" (§3.3.1).
- **The numbers (Table 1):**
  - CV_f0 is 0.6659 for the teacher, 0.4640 for the student and 0.5932 for teacher-guided sampling at 0.25, which keeps 89.1% of the teacher's (§4.3.3).
  - **English listening (CMOS):** teacher-guided sampling equals the student (+0.01).
  - **Chinese listening:** it is preferred, with naturalness +0.45 and similarity +0.3 against the student.
  - The caption's convention is that a negative value means the student is better.
- **When to use it:** the paper recommends the teacher for "applications where diversity is critical".

**What EXP-031 found:**
- **Quality:** the student alone matched or beat every teacher-based method.
- **Variety:** the teacher's measurable contribution was pitch variety beyond timing.
- **So:** the handover's cost is the price of variety.

**What is missing:** the paper's setting (14 teacher steps on a 32-interval grid, then 2 student steps) has never been measured for variety. EXP-017 ran it for quality only. The thesis's W8 restarts the student with mostly fresh noise, so it may not keep the paper's variety.

## Design

**Arms:**
- **`P25`, the 100% reference:** the paper's Algorithm 2 at 0.25, 30 passes. This is EXP-017's `alg2` code (`run_grid32_prelim.student_from_state`).
- **`P25R`:** the same teacher steps with the released handover (32 passes).
- **The released 16-interval sampler:** `W1`, `W2`, `W3`, `W4`, `W8` (7–21 passes).
- **`STU`:** the student alone (4 passes).
- **`TCH`:** the teacher alone (64 passes, quality stage only).

**Stages:**
- **Quality:** development speakers 1–30 × clean and noisy15 (60 cases).
- **Variety:** speakers 1–20 × 12 seeds × both conditions.
- **Tensor bank:** for speakers 1–5, so new handover designs can be screened offline. This follows the standing rule to screen offline before a GPU run.

**Why these choices:**
- **Speakers and seeds:** they are EXP-031's, so `STU`, `W4` and `W8` must reproduce its scores. The paper's arms can be compared with EXP-017's stored scores on the same speakers.
- **No W5, no W6, and the teacher alone only in quality:** EXP-017's measured times (0.30 s per guided teacher evaluation and 0.15 s per student step on a T4) put the full variety stage with the teacher alone at about 3.8 hours. Without them it takes about 3 hours.
- **noisy15:** the thesis's noisy condition. The requester asked for "clean and noisy". gauss25 and the reverberation and babble conditions are not run.
- **The 90% bar:** the paper's own teacher-guided setting kept 89.1% of its teacher's CV_f0.

**Frozen labels:**
- `cheapest_step_point`;
- `cheapest_step_confirmed`;
- `w8_matches_paper`;
- `noisy_cheapest_step_point`;
- `noisy_quality_safe_arms`.

The rules are in the [README](../../../research/experiments/EXP-032-variety-per-pass/README.md).

## Files

- **Runner:** `code/colab/run_variety_per_pass.py` (sha256 `fcc3808694da...`). It reuses EXP-017's 32-interval teacher and Algorithm 2 student, EXP-031's aligned CVf0, and the released student loop.
- **Tests:** `code/tests/test_variety_per_pass.py`, 12 tests:
  - the costs and the paper's grid (t₁₄ ≤ 0.25 < t₁₅, and t₁₄ equals Step 7 of the 16-interval grid);
  - the cohorts against EXP-017 and EXP-031;
  - the arm wiring, with every model call replaced by a recorder;
  - the tensor bank's round trip;
  - the share and 90% bootstrap;
  - the labels on synthetic profiles;
  - preregistration tampering;
  - the notebook.
- **Notebook:** `code/colab/DMOSpeech2_EXP032_Variety_Per_Pass.ipynb`, built from EXP-030's notebook, with these changes:
  - no DEMAND step;
  - a Drive sync that skips unchanged files;
  - restore of the tensor bank;
  - a separate tensor-bank zip saved to Drive only.
- **Preregistration:** `research/experiments/EXP-032-variety-per-pass/preregistration.json`.
  - **Payload:** `c952fe555d49...` (file `ce5d97b4a6dd...`, created 2026-11-15 17:32 UTC).
  - **Code locked:** the runner and its 12 dependencies.
  - **Inputs locked:** EXP-017's `grid32.csv`, EXP-031's two CSVs and EXP-027's quality CSV. Because EXP-031's results are inputs, they must be committed with this preregistration.
- **Repository:**
  - `.gitattributes`: EXP-032's results are stored as `-text`.
  - `.gitignore`: ignores `*.npz` (no `.npz` was tracked).

## Checks run

- **Unit tests:** `python -m unittest discover -s code/tests -p test_variety_per_pass.py`: 12 tests, OK (after the notebook was built).
- **Report:** rendered on synthetic data (scratchpad) to check its layout.
- **Workspace check:** `python scripts/check_workspace.py` PASSED.
- **Full suite:** `python -m unittest discover -s code/tests` ran 329 tests, OK.
- **LaTeX build:** not run. The manuscript did not change.
- **Not checked here:** the GPU path itself (the 32-interval teacher in a single run with the 16-interval trajectory, the tensor bank on real tensors). The smoke test checks it on the T4.

## Remaining blockers

- **The run:** a Colab T4 run by the requester, after the branch is pushed.
- **The next handover design:** waits for this study's results and an offline screen on the tensor bank, with a novelty audit before any build.
