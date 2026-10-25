# EXP-014: Preliminary test of three zero-cost handover operators

**Status:** designed 2026-10-25, not run. Notebook `code/colab/DMOSpeech2_Handover_Operators_Prelim.ipynb`, runner `code/colab/run_handover_operators_prelim.py` (Colab T4; same setup cells as EXP-013).

**Why:** EXP-012 showed that gates cannot beat a calibrated SNR threshold, and EXP-013 that the Step-8 endpoint cannot be predicted by extrapolation. EXP-013 also showed what fails: 96 of 126 noisy Step-4 failures are speaker-similarity (SIM) failures, not word errors. These operators change what the student receives at the Step-4 handover (13 NFE), with no extra network evaluation, and target that failure.

| Operator | Definition |
|---|---|
| Teacher-Coupled Restart (`tcr`) | The student's first input is the teacher's state x(t₄) instead of (1 − τ)ε + τ·x̂₁(t₄) with fresh noise ε. Because τ₁ = t₄ = 0.0761, x(t₄) = (1 − τ)ε̂ + τ·x̂₁(t₄) with the teacher's implied noise ε̂, so the student continues the teacher's trajectory. x(t₄) is recovered exactly from the endpoints (tested). |
| Coherence-gated DEH (`cdeh`) | x̂₁(t₄) + S(ρ, 4)·Δ₄ if cos(Δ₄, Δ₃) ≥ 0.2, else x̂₁(t₄) (the EXP-013 exploratory finding, now fixed in advance). |
| Speech-Envelope Anchoring (`sea`) | Per mel band b, the generated frames move by 0.5·(μ_prompt[b] − μ_gen[b]); both means use only the louder half of the frames (speech, not the noise floor). |

**Data:** the EXP-012 Quick set (60 development speakers × clean, noisy15, Gaussian 18/20/22/25 dB = 360 cases); the plain Step-4 and Step-8 labels are reused, and only the three operator outputs are synthesized and scored against the stored Step-8 reference. The smoke test re-scores Step 4 and Step 8 through this runner's own student loop to check that the reuse holds.

**Preregistered:** C1 noisy Step-4 failures drop by at least 25% with p < 0.05; C2 clean failures rise by at most 1; C3 the calibrated `<op>_theta` policy has lower mean NFE than the calibrated SNR gate with no more failures. Best operator: the one passing the most criteria, ties broken by lower `<op>_theta` NFE.

**Cost:** about 1.5 hours on a T4 (estimate).

**Prior work:** a coupled (deterministic) restart resembles DDIM-style use of the predicted noise; spectral mean normalization is a classic feature-domain technique. No source was found that applies either at a teacher-to-student handover, or that gates endpoint extrapolation by direction coherence; recorded as "not found", not proof of absence.

**Caveats:** development evidence only, Gaussian noise only. The operators were chosen after looking at EXP-012 and EXP-013 data, so a positive result needs a held-out test.
