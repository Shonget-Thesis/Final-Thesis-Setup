# `snr_invariant.py` — Function-by-Function Code & Math Guide

This guide covers **only** [snr_invariant.py](snr_invariant.py) (lines 1–270), mapping every constant block and every Python function directly to the mathematical formula it implements.

> **Why formulas look "hidden" in Python:** Vectorized calls like `torch.sort`, `torch.argsort`, `torch.gather`, `np.put_along_axis`, and `np.linalg.norm` replace explicit $\sum$ sums, $\sqrt{\cdot}$ square roots, and order-statistic subscripts $(k)$ with single array operations.

---

## Overview Map of `snr_invariant.py`

| Lines | Symbol / Function | Mathematical Object |
| :--- | :--- | :--- |
| [L32–L86](snr_invariant.py#L32-L86) | `TEACHER_GRID`, `METHOD_PASSES`, `LAMBDA_R_PRESETS` | Sway schedule $t_i$, NFE cost $C=13$, calibrated presets $\hat{\lambda}_r$ |
| [L89–L100](snr_invariant.py#L89-L100) | `_binom_cdf(k, n, p)` | Exact binomial cumulative tail $\mathbb{P}(\operatorname{Bin}(n,p) \le k)$ |
| [L103–L141](snr_invariant.py#L103-L141) | `calibrate_ltt(...)` | Fixed-sequence Learn-Then-Test threshold certification |
| [L144–L155](snr_invariant.py#L144-L155) | `describe_threshold(lam_r)` | Certificate lookup $\|\lambda_r - \lambda_{\text{preset}}\| \le 10^{-6}$ |
| [L158–L171](snr_invariant.py#L158-L171) | `quantile_handover(student_gen, teacher_gen)` | Per-band rank-order projection $Q(\hat{x}_1, s_0)$ |
| [L174–L178](snr_invariant.py#L174-L178) | `quantile_payload(...)` | Full handover tensor $\text{CPQH} = [s_{0,\text{prompt}};\, Q(\hat{x}_1, s_0)]$ |
| [L181–L190](snr_invariant.py#L181-L190) | `invariant_residual(teacher_gen, student_gen)` | Timing-invariant residual $r(\hat{x}_1, s_0)$ (normalized 1D $W_2$) |
| [L193–L205](snr_invariant.py#L193-L205) | `band_residuals(...)` | Sub-band relative residuals $r^{(1)},\dots,r^{(8)}$ |
| [L208–L235](snr_invariant.py#L208-L235) | `decide(...)` & `_describe_trigger(...)` | Routing policy $\pi_{\lambda_r}(x)$ & 4-state trigger classifier |
| [L238–L270](snr_invariant.py#L238-L270) | `evaluate(...)` | Decision summary & pass-saving ratios $\Delta C$ |

---

## 1. Constants, Sway Grid & Pass Accounting ([L32–L86](snr_invariant.py#L32-L86))

### Math Formulas
1. **8-Interval Cosine Sway Grid** ($N = 8$, sway coefficient $s = -1$):
   $$t_i = 1 - \cos\!\left(\frac{\pi}{2}\cdot\frac{i}{N}\right), \qquad i \in \{0, 1, 2, 3, 4\}$$
   At commit step $k = 4$:
   $$t_4 = 1 - \cos\!\left(\frac{\pi}{4}\right) \approx 0.2929$$
2. **Matched Student Schedule:**
   * Draft step: $\tau_0 = 0.0$
   * Student restart steps ($j = 2$): $\tau \in \{0.2929,\, 0.6173\}$
3. **Exact Forward-Pass Cost ($C$):**
   $$C = \underbrace{2(k + 1)}_{\text{10 teacher CFG passes}} + \underbrace{1}_{\text{draft } s_0} + \underbrace{2}_{\text{student finish}} = 13 \text{ passes}$$

### Code Mapping
```python
SNR_GATE_DB = 20.0                                                    # θ = 20 dB
GRID_INTERVALS = 8                                                    # N = 8
COMMIT_STEP = 4                                                       # k = 4
TEACHER_GRID = [1.0 - math.cos(math.pi / 2.0 * i / GRID_INTERVALS)
                for i in range(GRID_INTERVALS + 1)]                   # t_0 .. t_8
TEACHER_STEPS = tuple(range(COMMIT_STEP + 1))                         # (0, 1, 2, 3, 4)

DRAFT_TAU = 0.0                                                       # τ_0 = 0
STUDENT_RESTART_TAUS = list(sd.STUDENT_TAUS[1:])                      # [0.2929, 0.6173]

TEACHER_PASSES = sd.CFG_PASSES_PER_STEP * len(TEACHER_STEPS)          # 2 * 5 = 10
DRAFT_PASSES = 1                                                      # 1
STUDENT_PASSES = len(STUDENT_RESTART_TAUS)                            # 2
METHOD_PASSES = TEACHER_PASSES + DRAFT_PASSES + STUDENT_PASSES        # 13
```

---

## 2. `_binom_cdf(k, n, p)` ([L89–L100](snr_invariant.py#L89-L100))

### Math Formula
For $X \sim \operatorname{Binomial}(n, p)$, the exact cumulative probability of observing at most $k$ events is:
$$F_{\text{Bin}}(k; n, p) = \mathbb{P}(X \le k) = \sum_{i=0}^{\lfloor k \rfloor} \binom{n}{i} p^i (1 - p)^{n - i}$$
with boundary rules $F_{\text{Bin}} = 0$ if $k < 0$ or $p \ge 1$, and $F_{\text{Bin}} = 1$ if $k \ge n$ or $p \le 0$.

### Code Mapping
```python
def _binom_cdf(k, n, p):
    if k < 0: return 0.0
    if k >= n: return 1.0
    if p <= 0.0: return 1.0
    if p >= 1.0: return 0.0
    q = 1.0 - p
    return float(sum(math.comb(n, i) * (p ** i) * (q ** (n - i)) for i in range(int(k) + 1)))
```

| Math Piece | Python Expression |
| :--- | :--- |
| $1 - p$ | `q = 1.0 - p` |
| $\binom{n}{i}$ | `math.comb(n, i)` |
| $p^i (1-p)^{n-i}$ | `(p ** i) * (q ** (n - i))` |
| $\sum_{i=0}^{k}$ | `sum(... for i in range(int(k) + 1))` |

* **Example (`EXP-034` dev clean set):** With $n = 30$ clean speakers, tolerance $p = \alpha = 0.15$, and $k = 1$ false flag:
  $$F_{\text{Bin}}(1; 30, 0.15) = \binom{30}{0}(0.85)^{30} + \binom{30}{1}(0.15)(0.85)^{29} = 0.00763 + 0.04040 = 0.04803 \le \delta = 0.10$$
  which produces the exact `0.04802889...` stored in `LAMBDA_R_PRESETS["ltt_15"]["p_value"]` ([L61](snr_invariant.py#L61)).

---

## 3. `calibrate_ltt(...)` ([L103–L141](snr_invariant.py#L103-L141))

### Math Formula
Given $n_{\text{clean}}$ clean calibration rows, risk limit $\alpha$, confidence error $\delta$, SNR gate $\theta$, and utility margin $m = 0.30$:

1. **Intervention-Needed Indicator:**
   $$\operatorname{needed}(x) = \mathbb{I}\!\big[\operatorname{UTMOS}_{\text{CPQH}}(x) - \operatorname{UTMOS}_{\text{CPH8}}(x) > m\big]$$
2. **Router Flag Indicator at Candidate $\lambda$:**
   $$\operatorname{flagged}(x, \lambda) = \mathbb{I}\!\big[\hat{S}(x) < \theta \;\lor\; r(x) > \lambda\big]$$
3. **Unnecessary Clean Intervention Count:**
   $$k_{\text{err}}(\lambda) = \sum_{x \in \mathcal{C}_{\text{clean}}} \mathbb{I}\!\big[\operatorname{flagged}(x, \lambda) \land \neg\operatorname{needed}(x)\big]$$
4. **Fixed-Sequence Testing ($+\infty = \lambda_0 > \lambda_1 > \dots > \lambda_G$):**
   Test candidates in fixed descending order; compute $p(\lambda_g) = F_{\text{Bin}}(k_{\text{err}}(\lambda_g);\, n_{\text{clean}},\, \alpha)$. Keep updating $\hat{\lambda}_r \gets \lambda_g$ while $p(\lambda_g) \le \delta$, and **stop immediately (`break`)** at the first $g$ where $p(\lambda_g) > \delta$.

### Code Mapping
```python
clean = [row for row in rows if row.get("condition") == "clean"]
n_clean = len(clean)

def needed(row):
    return (float(row["utmos_CPQH"]) - float(row["utmos_CPH8"])) > float(needed_margin)

vals = sorted({float(row["r"]) for row in rows})
cands = [float("inf")] + vals[::-1]                           # λ_0 = ∞ > λ_1 > ... > λ_G
certified = None
for lam in cands:
    def flagged(row, lam=lam):
        return float(row["prompt_snr"]) < float(gate_db) or float(row["r"]) > lam

    k_err = sum(1 for row in clean if flagged(row) and not needed(row))
    p_val = _binom_cdf(k_err, n_clean, float(alpha))
    if p_val <= float(delta):
        certified = {"lambda_r": float(lam), ...}
    else:
        break                                                 # Fixed-sequence stopping rule!
return certified
```

---

## 4. `describe_threshold(lam_r, tol=1e-6)` ([L144–L155](snr_invariant.py#L144-L155))

### Math Formula
Checks floating-point proximity against each stored preset $\lambda_{\text{preset}}$:
$$|\lambda_r - \lambda_{\text{preset}}| \le \tau_{\text{tol}}, \qquad \tau_{\text{tol}} = 10^{-6}$$
and formats the $(1 - \delta)$ confidence statement when matched.

---

## 5. `quantile_handover(student_gen, teacher_gen)` ([L158–L171](snr_invariant.py#L158-L171))

### Math Formula
Given generated mel matrices $\hat{x}_1 \in \mathbb{R}^{n \times B}$ (`teacher_gen`) and $s_0 \in \mathbb{R}^{n \times B}$ (`student_gen`) over $n$ frames and $B = 100$ mel bands:
1. Sort the draft's values within each band $b$:
   $$s_{0, b, (0)} \le s_{0, b, (1)} \le \dots \le s_{0, b, (n-1)}$$
2. Find the 0-based rank $\operatorname{rank}_b(t) \in \{0, \dots, n-1\}$ of the teacher's value $\hat{x}_{1, t, b}$ within column $b$ (breaking ties by frame order).
3. Construct the quantile-projected matrix:
   $$Q(\hat{x}_1, s_0)_{t, b} = s_{0, b, (\operatorname{rank}_b(t))}$$

### Code Mapping (Both PyTorch & NumPy Branches)
```python
def quantile_handover(student_gen, teacher_gen):
    if hasattr(student_gen, "clone"):                                  # --- PyTorch branch ---
        import torch
        order  = torch.sort(teacher_gen, dim=0, stable=True).indices   # π_b: frame index of k-th smallest teacher val
        values = torch.sort(student_gen, dim=0, stable=True).values    # s_{0,b,(k)}: sorted draft values
        ranks  = torch.argsort(order, dim=0)                           # π_b^{-1}(t) = rank_b(t)
        return torch.gather(values, 0, ranks)                          # values[rank_b(t), b]

    s = np.asarray(student_gen, dtype=np.float64)                      # --- NumPy branch ---
    order = np.argsort(np.asarray(teacher_gen, dtype=np.float64), axis=0, kind="stable")
    out = np.empty_like(s)
    np.put_along_axis(out, order, np.sort(s, axis=0, kind="stable"), axis=0)
    return out
```

### Why Both Branches Compute the Exact Same Formula (4-Frame Trace)
Suppose in one band ($B=1$), `teacher_gen` $= [-4.2,\; -1.8,\; -0.5,\; -3.2]$ (frame 3 has static!) and `student_gen` $= [-2.0,\; -9.2,\; -4.5,\; -0.8]$:

1. **Sorted Draft $s_{0,(k)}$ (`values`):** `[-9.2, -4.5, -2.0, -0.8]`
2. **Teacher Sort Permutation $\pi$ (`order`):** `[0, 3, 1, 2]`
   * $k=0$ (smallest teacher value `-4.2`) is at frame `0`
   * $k=1$ (2nd smallest `-3.2`) is at frame `3`
   * $k=2$ (3rd smallest `-1.8`) is at frame `1`
   * $k=3$ (largest `-0.5`) is at frame `2`
3. **PyTorch (`gather` via inverse permutation `ranks = argsort(order) = [0, 2, 3, 1]`):**
   * Frame 0 gets `values[0] = -9.2`
   * Frame 1 gets `values[2] = -2.0`
   * Frame 2 gets `values[3] = -0.8`
   * Frame 3 gets `values[1] = -4.5`
   * Output: **`[-9.2, -2.0, -0.8, -4.5]`**
4. **NumPy (`put_along_axis(out, order, values)`):**
   * Writes `values[k]` directly into `out[order[k]]`:
     `out[0] = -9.2`, `out[3] = -4.5`, `out[1] = -2.0`, `out[2] = -0.8`
   * Output: **`[-9.2, -2.0, -0.8, -4.5]`** — identical!

---

## 6. `quantile_payload(student_draft, teacher_endpoint, prompt_len)` ([L174–L178](snr_invariant.py#L174-L178))

### Math Formula
For full sequence tensors of shape $(1,\, L_{\text{prompt}} + n,\, B)$, the `CPQH` payload keeps the clean student draft on the prompt prefix and applies $Q$ on the generated suffix:
$$\text{CPQH}_{0, t, b} = \begin{cases} s_{0,\, 0, t, b} & \text{for } 0 \le t < L_{\text{prompt}}, \\[4pt] Q\!\big(\hat{x}_{1,\, 0, L_{\text{prompt}}:, :},\; s_{0,\, 0, L_{\text{prompt}}:, :}\big)_{t - L_{\text{prompt}},\, b} & \text{for } L_{\text{prompt}} \le t < L_{\text{prompt}} + n. \end{cases}$$

### Code Mapping
```python
def quantile_payload(student_draft, teacher_endpoint, prompt_len):
    out = student_draft.clone() if hasattr(student_draft, "clone") else np.array(student_draft, copy=True)
    out[0, prompt_len:, :] = quantile_handover(student_draft[0, prompt_len:, :], teacher_endpoint[0, prompt_len:, :])
    return out
```

---

## 7. `invariant_residual(teacher_gen, student_gen, eps=1e-12)` ([L181–L190](snr_invariant.py#L181-L190))

### Math Formula
Computes the relative Frobenius norm between $\hat{x}_1$ and its rank-aligned projection $Q(\hat{x}_1, s_0)$ over generated frames $(n \times B)$—which by Proposition 3.3 equals the normalized root-sum of 1D 2-Wasserstein distances across all $B$ bands:
$$r(\hat{x}_1, s_0) = \frac{\lVert \hat{x}_1 - Q(\hat{x}_1, s_0)\rVert_F}{\lVert \hat{x}_1\rVert_F + \epsilon} = \frac{\sqrt{\sum_{t=1}^{n}\sum_{b=1}^{B} \big(\hat{x}_{1,t,b} - Q(\hat{x}_1, s_0)_{t,b}\big)^2}}{\sqrt{\sum_{t=1}^{n}\sum_{b=1}^{B} \hat{x}_{1,t,b}^2} + \epsilon} = \frac{\sqrt{\sum_{b=1}^{B}\sum_{k=1}^{n} \big(\hat{x}_{1,b,(k)} - s_{0,b,(k)}\big)^2}}{\lVert \hat{x}_1\rVert_F + \epsilon}$$

### Code Mapping
```python
def invariant_residual(teacher_gen, student_gen, eps=1e-12):
    t = ... # (frames, bands) float64 array of teacher_gen
    s = ... # (frames, bands) float64 array of student_gen
    q = quantile_handover(s, t)
    return float(np.linalg.norm(t - q) / (np.linalg.norm(t) + eps))
```

| Math Symbol | Python Expression |
| :--- | :--- |
| $\hat{x}_1$ | `t` |
| $Q(\hat{x}_1, s_0)$ | `q = quantile_handover(s, t)` |
| $\sqrt{\sum_{t=1}^n \sum_{b=1}^B (\hat{x}_{1,t,b} - Q_{t,b})^2}$ | `np.linalg.norm(t - q)` |
| $\sqrt{\sum_{t=1}^n \sum_{b=1}^B \hat{x}_{1,t,b}^2} + \epsilon$ | `np.linalg.norm(t) + eps` |

---

## 8. `band_residuals(teacher_gen, student_gen, n_bins=8, eps=1e-12)` ([L193–L205](snr_invariant.py#L193-L205))

### Math Formula
Partitions the $B = 100$ mel bands into $M = 8$ contiguous frequency groups $\mathcal{B}_1, \dots, \mathcal{B}_8$ and evaluates the same relative residual inside each frequency bin $m$:
$$r^{(m)} = \frac{\sqrt{\sum_{t=1}^{n}\sum_{b \in \mathcal{B}_m} \big(\hat{x}_{1,t,b} - Q_{t,b}\big)^2}}{\sqrt{\sum_{t=1}^{n}\sum_{b \in \mathcal{B}_m} \hat{x}_{1,t,b}^2} + \epsilon}, \qquad m = 1, \dots, 8$$

### Code Mapping
```python
q = quantile_handover(s, t)
edges = np.array_split(np.arange(t.shape[1]), max(1, int(n_bins)))    # B_1 .. B_8
for idx in edges:
    diff = t[:, idx] - q[:, idx]
    out.append(float(np.linalg.norm(diff) / (np.linalg.norm(t[:, idx]) + eps)))
```

---

## 9. `decide(...)` & `_describe_trigger(...)` ([L208–L235](snr_invariant.py#L208-L235))

### Math Formula
Defines the two-signal routing policy $\pi_{\lambda_r}$:
$$\pi_{\lambda_r}(x) = \begin{cases} \text{CPQH} \;\; \big(Q(\hat{x}_1, s_0)\big) & \text{if } (\hat{S} < \theta) \;\lor\; (r > \lambda_r), \\[4pt] \text{CPH8} \;\; (\hat{x}_1) & \text{if } (\hat{S} \ge \theta) \;\land\; (r \le \lambda_r). \end{cases}$$

### Code Mapping
```python
def decide(prompt_snr, residual, lam_r=DEFAULT_LAMBDA_R, gate_db=SNR_GATE_DB):
    flagged = (float(prompt_snr) < float(gate_db)) or (float(residual) > float(lam_r))
    return "CPQH" if flagged else "CPH8"
```
And `_describe_trigger` ([L214–L235](snr_invariant.py#L214-L235)) classifies $A = (\hat{S} < \theta)$ and $B = (r > \lambda_r)$ into the 4 mutually exclusive states: `both` ($A \land B$), `snr` ($A \land \neg B$), `residual` ($\neg A \land B$), and `none` ($\neg A \land \neg B$).

---

## 10. `evaluate(...)` ([L238–L270](snr_invariant.py#L238-L270))

### Math Formula
Packages the routing result together with the relative forward-pass savings against both established baselines:
$$\Delta C_{\text{W8}} = 1 - \frac{C_{\text{SNR\_R}}}{C_{\text{W8}}} = 1 - \frac{13}{21} \approx 0.3810 \;(38.1\%), \qquad \Delta C_{\text{P25}} = 1 - \frac{C_{\text{SNR\_R}}}{C_{\text{P25}}} = 1 - \frac{13}{30} \approx 0.5667 \;(56.7\%)$$

### Code Mapping
```python
"saving": 1.0 - METHOD_PASSES / W8_REFERENCE_PASSES,          # 1 - 13/21
"saving_vs_paper": 1.0 - METHOD_PASSES / P25_PAPER_PASSES,    # 1 - 13/30
```
