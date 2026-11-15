# EXP-026: Transmissibility test (a theory's predictions)

**Status:** run on 2026-11-14 (Colab Tesla T4, thesis commit `9998709`). Verdict under the frozen rule: **supported**; see [Results](#results-2026-11-14). Preregistration frozen 2026-11-14 before any synthesis (`78f3432d54ebe994...`). Task EVAL-019 (review). The run needs a Colab Tesla T4: [notebook](../../../code/colab/DMOSpeech2_EXP026_Transmissibility_Test.ipynb), [runner](../../../code/colab/run_transmissibility_test.py), [tests](../../../code/tests/test_transmissibility.py).

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

## Results (2026-11-14)

Files: [report](results/TRANSMISSIBILITY_REPORT.md), [results](results/transmissibility_results.json), [manifest](results/MANIFEST.json), [diagnostics](results/diagnostics.json).
- **Checks before the verdict:**
  - the smoke test reproduced the stored W4 and W8 scores exactly, and the teacher loop matched the released teacher (difference 0.0);
  - the full run's W4 and W8 CVf0 equal EXP-020's stored values;
  - recomputing the analysis locally from the two CSV files gave identical results.

**Verdict (frozen rule): supported.**

| | Result | Mean difference [95% CI] |
|---|---|---|
| H1 | Yes: the reference has less pitch variety than the pure teacher | T16 − W8: +0.0226 [+0.0089, +0.0447] |
| H2 | Yes: the restart at τ = 0.2929 from Step 8 recovers it | W8R2 − W8: +0.0223 [+0.0084, +0.0419]; mean gap to T16: W8 0.0249, W8R2 0.0148 |
| H3 | Yes, but not new evidence (below) | W8 − W4: +0.0007 [−0.0008, +0.0024] |
| H4 (exploratory) | Positive | Spearman +0.60 |

| Arm | Passes | CVf0 | ΔWER | ΔSimilarity | ΔUTMOS (vs reference, 40 speakers) |
|---|---:|---:|---|---|---|
| W4 | 13 | 0.0942 | | | |
| W8 (reference) | 21 | 0.0948 | | | |
| W4R2 | 12 | 0.1036 | +0.0001 [−0.0068, +0.0071] | −0.0078 [−0.0157, −0.0010] | −0.427 [−0.588, −0.291] |
| W8R2 | 20 | 0.1171 | +0.0018 [+0.0000, +0.0054] | −0.0015 [−0.0065, +0.0034] | −0.169 [−0.310, −0.042] |
| T16 | 32 | 0.1175 | −0.0010 [−0.0083, +0.0054] | −0.0131 [−0.0207, −0.0054] | −0.594 [−0.751, −0.458] |

**What it shows:**
- **Where the variety goes:** the reference keeps about 81% of the pure teacher's mean CVf0. The restart at τ = 0.2929 from Step 8 recovers essentially all of it (T16 − W8R2: +0.0004 [−0.0108, +0.0099]).
- **Variety against quality:** the recovered variety comes with the teacher's lower UTMOS. On these data, pitch variety and UTMOS trade against each other through the restart level. W8R2 saves 1 pass against the reference.
- **The Step-4 restart at τ = 0.2929 did not reliably recover variety:** W4R2 − W4 was +0.0094 [−0.0041, +0.0259], with 11 of 20 speakers above zero. This was not predicted in advance. It fits the theory's R_k, where commitment matters more as A grows, but only as a post hoc reading.

**Limits:**
- **H3 is not new evidence:** its W4 and W8 values are EXP-020's stored data, which had already informed the preregistration's power figures.
- **Not yet separated from a simpler view:** H1 and H2 also follow from the simpler view that a later restart lets the student overwrite less of the teacher's sample. The test supports the theory's direction, not its specific Φ ceiling. A sharper test would preregister the interaction the ceiling predicts: Step 4 against Step 8 matters at τ = 0.2929 but not at τ = 0.0761, on new speakers.
- **One speaker carries much of the effect.** 18 of 20 speakers are above zero for both H1 and H2 (sign test p = 0.0004). One speaker (`common_voice_en_15735839`) has the largest difference in both, and without the two largest speakers the means roughly halve (+0.0120 and +0.0107).
- **Pitch tracker (post hoc):** T16 keeps fewer frames voiced in at least half the seeds (0.52 against 0.63).
  - **T16:** its per-speaker gain is unrelated to that change (Spearman +0.02) and only weakly related to its UTMOS (+0.24).
  - **W8R2:** its gain is moderately related to its smaller voiced-share change (−0.46).
  - **Verdict:** with no audio stored, a tracker contribution cannot be excluded.
- **Data:** development speakers only, clean prompts, one run. Not held-out evidence and not evidence for a method.

## Novelty of the theory

Searches on 2026-11-14 found no published two-model analysis and no retention ceiling for a biased student after re-noising. The closest work:
- speciation theory, which covers a single model;
- [Gandikota & Bau](https://arxiv.org/pdf/2503.10637) and [First-Step Bias](https://arxiv.org/html/2610.37110), which are empirical;
- [speculative sampling with reflection coupling](https://www.alphaxiv.org/abs/2501.05370), which is a different purpose.

The ingredients are textbook, and novelty is not certified. It still needs this test and expert review.

## Data and evidence status

Development evidence only, on speakers that designed earlier experiments, with clean prompts. Results are stored under `results/` with a SHA-256 manifest. Audio is not committed.
