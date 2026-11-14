# 2026-11-14: EXP-026 transmissibility test prepared

Task: EVAL-019 (todo). Requested by Rhenel Jhon R. Sajol ("make ipynb and push"), after the requester's strict novelty rubric.

## Why

- **Under the rubric, about 65 candidate mechanisms that cut compute were rejected.** Each was found in published work:
  - **Lite-CFG:** the analytic unconditional pass (EXP-024) is a variant of it.
  - **Deep Image Prior self-validation:** the prompt probe (EXP-025) is a variant of it.
  - **Speculative sampling with reflection coupling** ([arXiv 2501.05370](https://www.alphaxiv.org/abs/2501.05370)): it already does noise reflection along the model-disagreement direction.
- **One candidate survived:** a mathematical model, the transmissibility theory of teacher-to-student handover.

## The theory

**Model.** One prosodic direction, with two modes at ±m and teacher weight π on the dominant one.
- **The teacher's belief.** Under flow matching (x_t = (1−t)ε + t·x₁), its log-odds for the dominant mode after Step k are ℓ_k = log(π/(1−π)) + 2m·t·x_t/(1−t)². Its estimate along that direction is m·tanh(ℓ_k/2).
- **The handover.** y = τ·m·tanh(ℓ_k/2) + (1−τ)·ε′, with fresh noise ε′.
- **The student.** It is modelled as a mode-picker with biased prior log-odds β greater than log(π/(1−π)), the mode-seeking of distribution-matching distillation. It keeps the dominant mode iff β + A·tanh(ℓ_k/2) + B·ε′ > 0, with A = 2m²τ²/(1−τ)² and B = 2mτ/(1−τ).

**Results.**
- A minority mode that the teacher has committed to by Step k survives with probability R_k = Φ((A·|tanh(ℓ_k/2)| − β)/B).
- Its ceiling under full commitment is **R\* = Φ((A − β)/B)**.

**Predictions.**
1. The reference can lose pitch variety against the pure teacher even at Step 8.
2. A later restart level raises the ceiling: τ = 0.2929 instead of 0.0761 multiplies A by about 25 and B by about 5.
3. At a fixed restart level, retention saturates once the teacher commits.

**What it is and isn't.** The theory explains why the Step-4 and Step-8 handovers agree on clean prompts and why early handovers lose variety. It is a theory, not a method. On its own it saves no compute beyond what a Step-4 handover already gives.

## Novelty audit (2026-11-14)

- **Speciation and symmetry breaking:** single-model commitment theory (Raya and Ambrogioni; Biroli, Mézard and colleagues).
- **Empirical work:** first-step diversity restoration (Gandikota and Bau; arXiv 2610.37110).
- **Empirical switch points:** FlowBlending, model scheduling for masked diffusion language models, T-Stitch.
- **SDEdit theory:** treats noise level as a realism-versus-faithfulness knob, with no mode probabilities.
- **Mode-seeking distillation fixes:** DP-DMD, DUET, DistillAlign.
- **Not found:** a two-model analysis, or a retention ceiling for a biased student after re-noising.
- **What is textbook:** the two-mode posterior and the tanh estimate.
- **Status:** a candidate, not certified. It needs this test and expert review.

## What changed

- **`code/colab/run_transmissibility_test.py`:**
  - **One trajectory per seed:** one teacher run to t = 1 gives all five arms. W4 and W8 restart the student at τ = 0.0761; W4R2 and W8R2 restart it at τ = 0.2929; T16 is the teacher alone.
  - **Pitch variety:** CVf0 on 20 development speakers × 20 seeds (EXP-020's seeds).
  - **Quality:** the new arms on 40 speakers against the stored reference.
  - **Reported results:**
    - H1–H3 and an exploratory H4;
    - the verdict;
    - reproduction of EXP-020's stored W4 and W8 CVf0.
- **`research/experiments/EXP-026-transmissibility-test/`:** the README, and `preregistration.json` (sha256 `78f3432d54ebe994...`), frozen before any synthesis.
- **Power figures in the preregistration:** computed from EXP-020's stored CVf0. The 95% half-width of a difference is 0.0016 between handovers from the same trajectory, 0.0083 for Voice-Swap against the reference, and 0.0215 for the student alone against the reference.
- **`code/tests/test_transmissibility.py`:** 14 tests, covering:
  - pass accounting;
  - the teacher run to t = 1 against the released steps;
  - the arms' restart levels;
  - the loops on a fake model;
  - the verdict rule (supported, refuted, mixed, partial);
  - preregistration tampering.
- **`code/colab/DMOSpeech2_EXP026_Transmissibility_Test.ipynb`:** 17 cells, from the EXP-023 layout.
- **Other files:**
  - the EVAL-019 task, with TASKS.md re-rendered;
  - a row in `research/experiments/README.md`;
  - STATUS.md;
  - `.gitattributes`.

## Checks run

- `python -m unittest test_transmissibility` (from `code/tests`): 14 tests, OK, against the frozen preregistration.
- The notebook builder: all 8 code cells compile.
- `python scripts/render_progress.py`: TASKS.md updated.
- `python scripts/check_workspace.py`: PASSED (structure, task evidence, source hashes, links, manuscript references).
- `python -m unittest discover -s code/tests`: 251 tests, OK.
- LaTeX build: not run. The manuscript did not change, so it is unverified for this change.

## Remaining blockers

- **GPU run:** it needs a Colab Tesla T4. Nothing has been synthesized.
- **Test scope:** this tests the theory's predictions. A method aimed at the ceiling would be a separate experiment.
