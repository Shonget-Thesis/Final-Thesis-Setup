# EXP-013: Preliminary test of Drift-Extrapolated Handover (DEH)

**Status:** designed 2026-10-25, not run. Notebook `code/colab/DMOSpeech2_Endpoint_Extrapolation_Prelim.ipynb`, runner `code/colab/run_endpoint_extrapolation_prelim.py` (Colab T4; same setup cells as EXP-011, without the DEMAND download).

**Why:** EXP-012 found that no gate beats a calibrated SNR threshold: gates only choose *when* to hand over. DEH changes *what* is handed over. In this pipeline the student always restarts at τ = 0.0761, which equals the teacher's Step-4 time t₄, so a Step-4 handover (13 NFE) and the reference Step-8 handover (21 NFE) differ only in the endpoint estimate given to the student. The EXP-012 drift values decay roughly geometrically from Step 3 on (median d₅/d₄ 0.79 on clean prompts, 0.82–0.98 on noisy ones), so the endpoint is still moving at Step 4, most of all on noisy prompts.

**Operator** (generated frames only; prompt frames unchanged; no extra network evaluation):

- Δₖ = x̂₁(tₖ) − x̂₁(tₖ₋₁), the change whose relative norm is the drift dₖ
- ρₖ = min(‖Δₖ‖_M / ‖Δₖ₋₁‖_M, 0.95), the measured contraction rate
- S(ρ, m) = ρ + ρ² + … + ρᵐ = ρ(1 − ρᵐ)/(1 − ρ), the predicted remaining change in units of Δₖ
- T = x̂₁(t₄) + β · S(ρ₄, 4) · Δ₄, handed to the student at Step 4; β = 1 (DEH) or 0.5 (DEH-half)

On an exactly geometric trajectory T equals x̂₁(t₈) (checked in `code/tests/test_endpoint_extrapolation.py`).

**Data:** the EXP-012 Quick set (first 60 development speakers × clean, noisy15, Gaussian 18/20/22/25 dB = 360 cases). The plain Step-4 and Step-8 labels are reused from the T4 grid, the EXP-011 pilot and EXP-012; only the two DEH outputs per case are synthesized and scored against the stored Step-8 reference with the frozen failure rule. The smoke test re-scores Step 4 and Step 8 for 2 speakers to check that reuse.

**Reported:** Step-4 failures with the plain endpoint vs DEH vs DEH-half per condition (exact McNemar; the pooled noisy test is primary); endpoint diagnostics that need no scoring (how much closer T is to x̂₁(t₈), direction coherence cos(Δ₄, Δ₃) and cos(Δ₄, x̂₁(t₈) − x̂₁(t₄))); out-of-fold policies (5-fold speaker CV, pooled and stratified LTT, α = 0.05, δ = 0.10): SNR-only, calibrated SNR gate, drift window, and the same gates handing over the DEH endpoint, plus DEH for every input.

**Preregistered criteria:** C1 noisy Step-4 failures drop by at least 25% with p < 0.05; C2 clean Step-4 failures rise by at most 1; C3 pooled out-of-fold DEH with a calibrated SNR gate has lower mean NFE than the calibrated SNR gate alone, with no more failures.

**Cost:** about 1 hour on a T4 (estimate: one teacher pass and two scored student outputs per case).

**Prior work checked (2026-10-25 web search):** extrapolation is used in diffusion sampling (RX-DPM, Richardson extrapolation over several ODE solutions; multistep solvers combine past data predictions; feature forecasting for caching), and DMOSpeech 2 introduced the teacher-to-student handover. No source was found that extrapolates the teacher's endpoint to a later horizon at the handover, using a contraction rate measured from the drift. This is "not found", not proof of absence.

**Caveats:** development evidence only, Gaussian noise only. The extrapolation idea came from looking at the EXP-012 drift values, so a positive result motivates a held-out test, not a claim.
