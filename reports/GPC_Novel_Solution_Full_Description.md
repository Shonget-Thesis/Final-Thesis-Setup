# The Guidance-Pareto-Cascade (GPC): A Novel Inference Framework for Hybrid Flow-Matching Text-to-Speech

**Author / Thesis Project:** Rhenel Jhon R. Sajol  
**Scope:** Frozen Teacher (CFG w=2.0, 2 passes/step), Frozen Distilled Student (unguided, 1 pass/step), Fixed Duration Prediction & Architecture  
**Hardware & Runtime Overhead:** NVIDIA Tesla T4 GPU, 0 extra forward passes (0 NFE), 0.12 ms linear algebra overhead per utterance  

---

## Executive Summary

The **Guidance-Pareto-Cascade (GPC)** is a training-free, zero-forward-overhead adaptive inference algorithm that dynamically determines the optimal, per-utterance switching time from a heavy multi-step teacher flow-matching model to a fast distilled few-step student model. 

Unlike previous heuristic early-stopping rules that fail under noise or rely on brittle external waveform SNR meters, GPC is established across three formal scientific layers:
1. **A Derived Mathematical Law (Lemma 1):** Proves that step-to-step endpoint differences in Euler flow matching are identically horizon-weighted trajectory curvature, and derives an exact quadratic decomposition of trajectory drift into conditional acceleration power, classifier-free guidance (CFG) power, and their cross-alignment.
2. **An Internal Guidance Signal Functional:** Normalizes cumulative trajectory action by the DiT's initial guidance contrast shock ($A_k / gc_0$), separating noise-induced guidance starvation from true post-alignment convergence.
3. **A Finite-Horizon Optimal-Stopping Solver:** Computes a global Pareto marginal-efficiency path across discrete candidate stages and applies Split-Path Monotone Conformal Risk Control (CRC) to guarantee distribution-free multi-metric risk control ($\mathbb{E}[L] \le 0.05$) on unseen speakers.

---

## 1. The Scientific Problem: The "Premature Stabilization" Paradox

In teacher-guided flow-matching distillation (e.g., DMOSpeech 2 / F5-TTS), the teacher model integrates a guided vector field:
$$v(x_t, t) = v_{\text{cond}}(x_t, t) + w \cdot \underbrace{\bigl(v_{\text{cond}}(x_t, t) - v_{\text{uncond}}(x_t, t)\bigr)}_{g(x_t, t)}, \qquad w = 2.0$$
The fast student model is distilled to run without classifier-free guidance (1 pass per step instead of 2).

### Why Standard ODE Early-Exit Literature Fails
Classical adaptive ODE solvers and diffusion early-stopping rules evaluate step-to-step variation:
$$d_j = \frac{\|\hat{x}_1(t_j) - \hat{x}_1(t_{j-1})\|_M}{\|\hat{x}_1(t_j)\|_M}$$
and stop when drift falls below a threshold ($d_k \le \lambda$), assuming that small $d_j$ indicates trajectory straightness and convergence.

**The Paradox:** When speech prompts are degraded by background noise, reverberation, or bandpass filtering, the conditioning embeddings are corrupted, causing $v_{\text{cond}} \approx v_{\text{uncond}} \implies g(x_t, t) \approx 0$. 
- The guided teacher experiences **guidance starvation**, drifting along the unguided prior with near-zero trajectory velocity change.
- On the Tesla T4 benchmark, median $d_4$ drops from $0.61$ on clean speech to $0.17$ on 15 dB white noise, while Word Error Rate (WER) remains over $50\%$.
- A standard stopping rule ($d_k \le \lambda$) prematurely switches to the student, catastrophic acoustic collapse ensues, and risk-calibrating the rule forces $\hat{\lambda} \le 0$, resulting in **0.0% compute savings**.

### Why Naive Waveform SNR Meters Fail
Using an external speech signal-to-noise ratio estimator ($\hat{S} \ge \theta$) adds 1.85 ms of wall-clock latency, fails on white noise between 18–22 dB, and fails catastrophically on telephone-band speech (300–3400 Hz). In telephone speech, silent pauses between words fool energy-based SNR meters into reporting $44.2\text{ dB}$ (clean speech), triggering premature early exit at Step 4 and adding speech recognition errors.

---

## 2. Layer 1: Mathematical Theorem & Exact Guidance Decomposition

### Lemma 1 (Exact Euler Trajectory Curvature Identity)
Let $x_{t_j} = x_{t_{j-1}} + (t_j - t_{j-1}) v(x_{t_{j-1}}, t_{j-1})$ denote the discrete Euler update, and let $\hat{x}_1(t_j) = x_{t_j} + (1 - t_j) v(x_{t_j}, t_j)$ denote the projected mel-spectrogram endpoint at step $j$.

**Theorem:** For any discrete step $j \ge 1$:
$$\hat{x}_1(t_j) - \hat{x}_1(t_{j-1}) \equiv (1 - t_j)\bigl(v(x_{t_j}, t_j) - v(x_{t_{j-1}}, t_{j-1})\bigr)$$

**Proof:**
Substitute the Euler step $x_{t_j}$ into the projected endpoint $\hat{x}_1(t_j)$:
$$\hat{x}_1(t_j) = x_{t_{j-1}} + (t_j - t_{j-1})v_{j-1} + (1 - t_j)v_j$$
Subtract the previous projected endpoint $\hat{x}_1(t_{j-1}) = x_{t_{j-1}} + (1 - t_{j-1})v_{j-1}$:
$$\hat{x}_1(t_j) - \hat{x}_1(t_{j-1}) = \bigl[x_{t_{j-1}} - x_{t_{j-1}}\bigr] + (1 - t_j)v_j + \bigl[(t_j - t_{j-1}) - (1 - t_{j-1})\bigr]v_{j-1}$$
Simplifying the second coefficient:
$$(t_j - t_{j-1}) - (1 - t_{j-1}) = t_j - 1 = -(1 - t_j)$$
Factoring out $(1 - t_j)$:
$$\hat{x}_1(t_j) - \hat{x}_1(t_{j-1}) = (1 - t_j)v_j - (1 - t_j)v_{j-1} = (1 - t_j)(v_j - v_{j-1}) \quad \blacksquare$$

**Significance:** Intermediate spatial coordinates $x_{t_{j-1}}$ cancel completely. Endpoint drift is proven to be the **horizon-weighted acceleration (curvature)** of the vector field.

---

### Exact Quadratic Guidance Power Decomposition
Because $v_j = v_{\text{cond}, j} + w \cdot g_j$, the horizon-weighted velocity difference decomposes into:
$$\delta c_j = (1 - t_j)(v_{\text{cond}, j} - v_{\text{cond}, j-1}), \qquad \delta g_j = w (1 - t_j)(g_j - g_{j-1})$$
$$\hat{x}_1(t_j) - \hat{x}_1(t_{j-1}) = \delta c_j + \delta g_j$$

Squaring the Frobenius norm over generated speech frames $M = [L_{\text{prompt}} : L]$ and normalizing by $\|\hat{x}_1(t_j)\|_M^2$:
$$d_j^2 = p^c_j + p^g_j + p^x_j$$
where:
- $p^c_j = \frac{\|\delta c_j\|_M^2}{\|\hat{x}_1(t_j)\|_M^2}$: **Conditional Acceleration Power**
- $p^g_j = \frac{\|\delta g_j\|_M^2}{\|\hat{x}_1(t_j)\|_M^2}$: **Guidance Acceleration Power** (work performed by CFG)
- $p^x_j = \frac{2 \langle \delta c_j, \delta g_j \rangle_M}{\|\hat{x}_1(t_j)\|_M^2}$: **Interaction Cross-Term**

Because $v_{\text{cond}}$ and $v_{\text{uncond}}$ are already evaluated during normal teacher CFG passes, these components are extracted with **0 extra neural evaluations** and negligible scalar algebra ($\sim 0.12\text{ ms}$).

---

## 3. Layer 2: The Domain-Specific Signal Functional

Scalar path length ($A_k = \sum d_j$) cannot distinguish random Brownian velocity jitter from coherent conditioning alignment, which caused ultra-early false exits at Step 1 (clean UTMOS drop to 3.06).

GPC introduces the **Guidance-Normalized Action**:
$$A_k^{\text{norm}}(x) = \frac{\sum_{j=1}^k d_j(x)}{gc_0(x) + \epsilon}$$
where $gc_0 = \frac{\|v_{\text{cond}}(x_0, 0) - v_{\text{uncond}}(x_0, 0)\|_M}{\|v_{\text{cond}}(x_0, 0)\|_M + \epsilon}$ is the **initial guidance contrast shock** at $t_0 = 0$.

### Physical Interpretation:
- On clean, intelligible prompts, the DiT recognizes the phonemic conditioning at $t_0$, producing a clean initial contrast $gc_0$ that scales $A_k^{\text{norm}}$ into a stable trajectory odometer.
- On corrupted or OOD prompts, guidance starvation suppresses intermediate drift, keeping $A_k^{\text{norm}}$ far below the threshold and preventing premature student handover.

---

## 4. Layer 3: The Decision Algorithm (Pareto Path + Conformal Risk Control)

The handover decision is formulated as a **Finite-Horizon Constrained Optimal Stopping Problem**:
$$\min_{\bm{\lambda}} \mathbb{E}\left[\text{NFE}(\pi_{\bm{\lambda}}(X))\right] \quad \text{subject to} \quad \mathbb{E}\left[L(X, \pi_{\bm{\lambda}}(X))\right] \le \alpha$$

The multi-metric loss $L(x, k) \in [0, 1]$ simultaneously bounds degradation across Word Error Rate, Speaker Similarity, and Perceptual Naturalness (UTMOS):
$$L(x, k) = \min\left(1.0, \;\max\left( \frac{\Delta\text{WER} - 0.01}{0.10}, \; \frac{-\Delta\text{SIM} - 0.015}{0.025}, \; \frac{-\Delta\text{UTMOS}_{\text{clean}} - 0.15}{0.50} \right)_+\right)$$

```
                       CALIBRATION POPULATION (D_cal)
                       +----------------------------+
                       |  1/3 Path-Design Speakers  |
                       +--------------+-------------+
                                      |
                                      v  (Greedy Marginal Search)
                       +----------------------------+
                       | Pareto Path: λ(0) ... λ(T) |
                       | m* = argmax ΔNFE / (ΔRisk) |
                       +--------------+-------------+
                                      |
                       +--------------v-------------+
                       |  2/3 Conformal Cal. Spks.  |
                       +--------------+-------------+
                                      |
                                      v  (Monotone Loss Envelope)
                       +----------------------------+
                       | Find largest t̂ where:      |
                       | (1 + ∑ L_s(t̂)) / (n+1) ≤ α |
                       +--------------+-------------+
                                      |
                                      v
                       FROZEN THRESHOLD VECTOR λ* = λ(t̂)
```

### Step A: Pareto Marginal-Efficiency Stopping Path
Instead of greedy stage-by-stage tuning, GPC constructs a 1D path of threshold vectors $\bm{\lambda}(0), \bm{\lambda}(1), \dots, \bm{\lambda}(T)$ starting from $\bm{\lambda}(0) = (+\infty, \dots, +\infty)$ (always reference Step 8). At each step:
$$m^* = \arg\max_{m} \frac{\Delta\text{NFE}_m}{\Delta\text{Risk}_m + \epsilon}$$
Stage $m^*$ is advanced to its next empirical quantile. This dynamically prioritizes stages that offer large compute reductions with minimal quality degradation, while keeping sensitive ultra-early stages locked.

### Step B: Split-Path Monotone Conformal Risk Control
1. Calibration speakers $\mathcal{D}_{\text{cal}}$ are split into **Path-Design** ($1/3$) and **Exchangeable Calibration** ($2/3$) cohorts.
2. The Pareto path $\bm{\lambda}(t)$ is generated exclusively on the design set.
3. For each calibration speaker $s$, the non-decreasing monotone loss envelope is evaluated:
   $$\overline{L}_s(t) = \max_{t' \le t} \frac{1}{|U_s|} \sum_{u \in U_s} L(u, \pi_{\bm{\lambda}(t')}(u))$$
4. The stopping index $\hat{t}$ is calibrated via conformal risk control:
   $$\hat{t} = \max \left\{ t : \frac{1}{n_{\text{cal}} + 1}\sum_{s=1}^{n_{\text{cal}}} \overline{L}_s(t) + \frac{1}{n_{\text{cal}} + 1} \le \alpha = 0.05 \right\}$$

**Statistical Guarantee:** For any unseen exchangeable test speaker $X_{n+1}$:
$$\mathbb{E}\left[L(X_{n+1}, \pi_{\hat{\bm{\lambda}}}(X_{n+1}))\right] \le \alpha = 0.05$$

---

## 5. Performance and Telemetry Summary

When benchmarked on an NVIDIA Tesla T4 GPU across 360 development cases, 40 unseen validation speakers, and zero-shot OOD corruptions:

| Handover Policy | Gate Overhead | 16-Grid NFE (Saved) | E2E Speedup | TOST Equiv. ($p_{\text{WER}} / p_{\text{SIM}}$) | Clean UTMOS | Test Risk ($\le 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed Step 8 Reference** | 0.00 ms | 21.00 (0.0%) | 1.00x | Reference | 3.924 | 0.0000 |
| **Calibrated SNR-Only** | 1.85 ms | 20.69 (1.5%) | 1.01x | Passed ($< 10^{-20}$) | 3.907 | 0.0021 |
| **Stage-Wise SNR Cascade** | 1.85 ms | 18.13 (13.7%) | 1.15x | Passed ($< 10^{-20}$) | 3.858 | 0.0152 |
| **Raw STAC ($A_k \ge \hat{\lambda}_k$)** | 0.12 ms | 16.82 (19.9%) | 1.25x | Passed ($1.2 \times 10^{-7}$) | 3.643 (leaks Step 1) | 0.0410 |
| **GPC (Proposed Novel Solution)** | **0.12 ms** | **17.21 (18.0%)** | **1.22x** | **Passed ($1.1 \times 10^{-6}$)** | **3.849 (preserved)** | **0.0247 (guaranteed)** |

- **Live Speech Synthesis:** Reduces end-to-end synthesis latency from 3.23s to 2.60s (1.79x faster than real-time) on clean speech with identical voice quality, while automatically falling back 100% of the time to Step 8 on noisy speech.
- **Zero-Shot OOD Generalization:** Accurately routes telephone-band speech (44.2 dB false SNR) to late teacher steps, preventing the acoustic degradation that defeats waveform SNR baselines.
