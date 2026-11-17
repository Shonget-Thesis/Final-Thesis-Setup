# 2026-11-14: EXP-026 transmissibility test results imported

Task: EVAL-019 (now review). Rhenel Jhon R. Sajol ran the notebook on Colab (Tesla T4, thesis commit `9998709`) and supplied `DMOSpeech2_EXP026_Transmissibility_Test_Results.zip` (sha256 `7315a078...`).

## Import and checks

- **Run integrity:**
  - The preregistration in the zip matches the frozen one, line endings normalised (`78f3432d...`).
  - The zip's seven runner modules match `code/colab/`.
  - Smoke test: the stored W4 and W8 scores were reproduced exactly, and the teacher loop matched the released teacher (max difference 0.0).
- **Import:**
  - The results were copied byte for byte into `research/experiments/EXP-026-transmissibility-test/results/` with a SHA-256 `MANIFEST.json`.
  - The zip's `code/` snapshot was not copied.
- **Analysis recomputed:** `python code/colab/run_transmissibility_test.py analyze` on a copy of the two CSV files gave identical `transmissibility_results.json` and report.
- **New script:** `code/analysis/exp026_diagnostics.py` writes `results/diagnostics.json`. It was written after the results were seen, so it is descriptive and cannot change the verdict.

## Results

**Verdict under the frozen rule: supported.**
- **H1, the reference loses variety:** CVf0(T16) − CVf0(W8) = +0.0226 [+0.0089, +0.0447].
- **H2, a Step-8 restart at τ = 0.2929 recovers it:** W8R2 − W8 = +0.0223 [+0.0084, +0.0419]. W8R2 is closer to T16 than W8 is (mean gap 0.0148 against 0.0249).
- **H3, saturation:** W8 − W4 = +0.0007.
- **H4 (exploratory):** Spearman +0.60.

**Pitch variety:**
- **The reference keeps about 81% of the pure teacher's mean CVf0** (0.0948 against 0.1175).
- **W8R2 recovers essentially all of it** (0.1171).
- **W4R2 does not reliably recover it** (+0.0094 [−0.0041, +0.0259] over W4; 11 of 20 speakers above zero).

**Quality against the stored reference (40 speakers):**
- **The recovered variety comes with lower UTMOS:**
  - W8R2: −0.169 [−0.310, −0.042];
  - W4R2: −0.427;
  - T16: −0.594.
- **W8R2's WER:** +0.0018 [+0.0000, +0.0054].
- **Similarity:** lower for T16 (−0.0131) and W4R2 (−0.0078); W8R2 is within ±0.0065.

## How to read it

- **Supported means the predicted directions held.** The specific ceiling formula is not confirmed. H1 and H2 also follow from the simpler view that a later restart lets the student overwrite less of the teacher's sample.
- **H3 is not new evidence.** The run's W4 and W8 CVf0 equal EXP-020's stored values, which had already informed the power figures.
- **The pattern the ceiling predicts most specifically** is that the handover step matters at τ = 0.2929 but not at τ = 0.0761. That pattern did appear (W4R2 < W8R2, W4 ≈ W8), but it was not preregistered.
- **The effect is broad but skewed:**
  - 18 of 20 speakers are above zero for H1 and H2 (sign test p = 0.0004).
  - One speaker has the largest difference in both, and without the two largest speakers the means roughly halve (+0.0120 and +0.0107).
- **The pitch tracker is a possible confound:**
  - T16 keeps fewer frames voiced in at least half the seeds (0.52 against 0.63), and the per-speaker gain is unrelated to that change (Spearman +0.02).
  - For W8R2 the gain is moderately related to its voiced-share change (−0.46).
  - No audio was stored, so a tracker contribution cannot be excluded.
- **For the compute problem:** the variety the reference loses is lost at the handover, not from too few teacher steps. Raising the restart level buys it back at a UTMOS cost. No compute-saving method follows directly; W8R2 saves only 1 pass.

**Limits:** development speakers, clean prompts, one run. This is not evidence for a method, not adopted, and not reviewed by an adviser. The theory's novelty is still not certified.

## Checks run

- `python code/colab/run_transmissibility_test.py analyze` on a copy of the CSV files: identical outputs.
- `python code/analysis/exp026_diagnostics.py`: wrote `results/diagnostics.json`.
- `python scripts/render_progress.py`: TASKS.md updated.
- `python scripts/check_workspace.py`: PASSED.
- `python -m unittest discover -s code/tests`: 251 tests, OK. There is no unit test for the descriptive diagnostics script.
- LaTeX build: not run. The manuscript did not change, so it is unverified for this change.

## Remaining blockers

- **Ceiling formula:** a preregistered test of the step-by-restart interaction on new speakers would be needed.
- **Pitch tracker:** an audio-level check would be needed to rule out a tracker contribution.
- **Method:** any method aimed at the ceiling needs its own novelty audit and preregistration, and it has to deal with the UTMOS cost.
