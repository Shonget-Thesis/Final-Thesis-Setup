# EXP-026: Transmissibility test (a theory's predictions)

**Status:** prepared, not yet run. Preregistration frozen 2026-11-14 before any synthesis (`78f3432d54ebe994...`). Task EVAL-019 (todo). The run needs a Colab Tesla T4: [notebook](../../../code/colab/DMOSpeech2_EXP026_Transmissibility_Test.ipynb), [runner](../../../code/colab/run_transmissibility_test.py), [tests](../../../code/tests/test_transmissibility.py).

- **A test of a theory:** it checks predictions. It is not evidence for a method.
- **Not adopted:** nothing here changes the thesis method, and no adviser has reviewed it.

## The theory being tested

The model takes one prosodic direction with two modes at ±m, with the teacher putting weight π on the dominant one.
- **The teacher:** after Step k, its belief in the dominant mode has log-odds ℓ_k = log(π/(1−π)) + 2m·t·x_t/(1−t)², and its estimate is m·tanh(ℓ_k/2).
- **The handover:** y = τ·estimate + (1−τ)·fresh noise.
- **The student:** it is mode-seeking, with prior log-odds β greater than log(π/(1−π)). It keeps the dominant mode iff β + A·tanh(ℓ_k/2) + B·ε′ > 0, where A = 2m²τ²/(1−τ)² and B = 2mτ/(1−τ).

A minority mode the teacher has fully committed to survives with probability **R\* = Φ((A − β)/B)**. That is a ceiling set by the restart level τ and the student's bias β, and more teacher steps cannot raise it. The derivation and the novelty audit are in the [setup log](../../../docs/progress/log/2026-11-14-exp026-transmissibility-test.md).

## What runs

All arms come from one teacher trajectory per seed, so the comparisons are exact:

| Arm | What it is | Passes |
|---|---|---:|
| W4, W8 | student restarted at τ = 0.0761 from Step 4 / Step 8; W8 is the 0.25 reference | 13, 21 |
| W4R2, W8R2 | student restarted at τ = 0.2929 from Step 4 / Step 8 | 12, 20 |
| T16 | the guided teacher alone, run to the end on the released 16-interval grid | 32 |

- **Pitch variety:** CVf0 across 20 seeds (EXP-020's seeds) on 20 development speakers, clean prompts. The released W4 and W8 CVf0 are also checked against EXP-020's stored values.
- **Quality:** W4R2, W8R2 and T16 on 40 speakers, scored against the stored reference (EXP-018).
- **Smoke test:** the teacher loop must match the released teacher (≤ 1e-4), and the stored W4 and W8 scores must be reproduced.

## Hypotheses and verdict rule (frozen)

| | Prediction | Test |
|---|---|---|
| H1 | The reference has less pitch variety than the pure teacher | CVf0(T16) − CVf0(W8): bootstrap lower bound > 0 |
| H2 | A later restart recovers it | CVf0(W8R2) − CVf0(W8): lower bound > 0, and W8R2 is closer to T16 than W8 is |
| H3 | At τ = 0.0761, extra teacher steps add nothing once the ceiling is reached | \|CVf0(W8) − CVf0(W4)\| ≤ 0.005 |
| H4 (exploratory) | Speakers whose student loses more variety also lose more at the reference | Spearman correlation > 0 |

**Verdict:**
- **Supported** if H1 and H2 hold. The next step is then an algorithm aimed at the ceiling.
- **Refuted** if H1 fails. The theory's central prediction is then dropped.
- **Mixed** otherwise.
- No retuning on these data.

**Power:** on EXP-020's 20 speakers, the 95% half-width of a CVf0 difference was:
- 0.0016 between two handovers from the same trajectory;
- 0.0083 for Voice-Swap against the reference;
- 0.0215 for the student alone against the reference.

Differences smaller than these may go undetected.

## Novelty of the theory

Searches on 2026-11-14 found no published two-model analysis and no retention ceiling for a biased student after re-noising. The closest work:
- speciation theory, which covers a single model;
- [Gandikota & Bau](https://arxiv.org/pdf/2503.10637) and [First-Step Bias](https://arxiv.org/html/2610.37110), which are empirical;
- [speculative sampling with reflection coupling](https://www.alphaxiv.org/abs/2501.05370), which is a different purpose.

The ingredients are textbook, and novelty is not certified. It still needs this test and expert review.

## Data and evidence status

Development evidence only, on speakers that designed earlier experiments, with clean prompts. Results will be stored under `results/` with a SHA-256 manifest. Audio is not committed.
