# EXP-019: Guidance-Decomposed Trajectory Telemetry and Risk-Controlled Handover (`GPC`)

**Status:** Preregistration frozen on 2026-11-07 (`e907af77e5cd6a34...`). Ready for execution on a Colab Tesla T4 GPU. Task EVAL-012 (todo).
- Notebook: [DMOSpeech2_EXP019_Guidance_Decomposition.ipynb](../../../code/colab/DMOSpeech2_EXP019_Guidance_Decomposition.ipynb)
- Runner: [run_guidance_work_prelim.py](../../../code/colab/run_guidance_work_prelim.py)
- Tests: [test_guidance_work.py](../../../code/tests/test_guidance_work.py)
- Preregistration: [preregistration.json](preregistration.json)

---

## 1. Mathematical Foundation and Mechanism

In teacher-guided flow-matching inference (such as DMOSpeech 2 / F5-TTS), the teacher executes an Euler trajectory with classifier-free guidance scale $w = 2.0$:
$$v(x_t, t) = v_{\text{cond}}(x_t, t) + w \underbrace{\bigl(v_{\text{cond}}(x_t, t) - v_{\text{uncond}}(x_t, t)\bigr)}_{g(x_t, t)}$$
The projected endpoint at step $j$ is $\hat{x}_1(t_j) = x_{t_j} + (1 - t_j) v(x_{t_j}, t_j)$.

### Lemma 1 (Exact Euler Trajectory Curvature Identity)
Substituting the Euler forward step $x_{t_j} = x_{t_{j-1}} + (t_j - t_{j-1}) v(x_{t_{j-1}}, t_{j-1})$ into the step difference yields an identical cancellation of $x_{t_{j-1}}$:
$$\hat{x}_1(t_j) - \hat{x}_1(t_{j-1}) = (1 - t_j)\bigl(v(x_{t_j}, t_j) - v(x_{t_{j-1}}, t_{j-1})\bigr)$$
This proves that step-to-step endpoint drift is **not** an estimation error, but the **horizon-weighted trajectory curvature (acceleration)** induced by the model.

### Exact Guidance Decomposition
Decomposing velocity change into conditional velocity update $\delta c_j = (1 - t_j)(v_{\text{cond}, j} - v_{\text{cond}, j-1})$ and guidance update $\delta g_j = w(1 - t_j)(g_j - g_{j-1})$:
$$\hat{x}_1(t_j) - \hat{x}_1(t_{j-1}) = \delta c_j + \delta g_j$$
Squaring the Frobenius norm over generated speech frames $[L_{\text{prompt}} : L]$ yields an exact quadratic decomposition of drift $d_j^2$:
$$d_j^2 = p^c_j + p^g_j + p^x_j$$
where:
- $p^c_j = \frac{\|\delta c_j\|_M^2}{\|\hat{x}_1(t_j)\|_M^2}$ is the **conditional velocity acceleration power**,
- $p^g_j = \frac{\|\delta g_j\|_M^2}{\|\hat{x}_1(t_j)\|_M^2}$ is the **guidance acceleration power**, and
- $p^x_j = \frac{2 \langle \delta c_j, \delta g_j \rangle_M}{\|\hat{x}_1(t_j)\|_M^2}$ is the **interaction cross-term**.

Because $v_{\text{cond}}$ and $v_{\text{uncond}}$ are already evaluated at every teacher step for CFG, computing this decomposition requires **zero additional neural function evaluations (0 extra NFE)** and adds only $\sim 0.12$ ms of scalar linear algebra.

---

## 2. Preregistered Hypotheses

1. **H1 (Mathematical Exactness):** The Lemma 1 residual is $\le 10^{-3}$ on all Tesla T4 trajectories.
2. **H2 (Premature Stabilization Mechanism):** Guidance share $S_g = \frac{\sum (\Delta p^g + 0.5 \Delta p^x)}{\sum \Delta(d^2)}$ determines whether noise-induced velocity damping is driven by guidance contrast collapse or conditional drift suppression.
3. **H3 (Out-of-Sample Prediction):** On 40 fresh speakers (development speakers 61–100 by utterance ID, uninspected by EXP-012 through EXP-018), the selected guidance signal achieves higher mean early-exit AUC than STAC's scalar action $A_k = \sum d_j$.
4. **H4 (Risk-Controlled Handover & Ablation Necessity):** The proposed **GPC** policy (Guidance-Pareto-Cascade: selected signal + Pareto marginal-efficiency path + split-path monotone conformal risk control) passes continuous TOST equivalence, bounds test risk $\le \alpha = 0.05$, avoids UTMOS degradation, and matches or exceeds STAC's savings while ablations demonstrate that each component is necessary.
5. **H5 (Zero-Shot OOD Generalization):** On unseen degraded acoustic channels (reverberation, telephone bandpass), GPC maintains $\Delta\text{WER} \le +0.01$ and equals or exceeds Stage-Wise SNR Cascade savings.

---

## 3. Evaluated Methods and Ablations

- **Proposed `GPC`:** Selected guidance signal ($A_k / gc_0$ or $\sum \sqrt{p^g_j}$) with Pareto marginal-efficiency path and split-path monotone Conformal Risk Control ($\alpha = 0.05$).
- **Ablation `gpc_A`:** Tests signal necessity by replacing the guidance signal with scalar cumulative action $A_k$.
- **Ablation `gpc_greedy`:** Tests path necessity by replacing the Pareto marginal-efficiency path with STAC's greedy stage-by-stage ordering.
- **Ablation `gpc_nocrc`:** Tests calibration necessity by optimizing empirical risk without the monotone-envelope conformal guarantee.
- **Baselines:** Fixed Step-8 / Step-14 references, Fixed Step-7, STAC, 1D-CRC-STAC, Stage-Wise SNR Cascade, and Calibrated SNR-only.

