# 2026-11-07: EXP-019 Guidance-Decomposed Trajectory Telemetry & Risk-Controlled Handover Setup

**Task:** EVAL-012 (Guidance-decomposed trajectory telemetry and risk-controlled handover).

## Context & Motivation
Following the analysis of CS novelty criteria, STAC was audited against the 8 requirements of genuine scientific contribution. While STAC's empirical results on multi-grid benchmarks, live GPU speedups, and OOD generalization are verified, its formulation ($A_k = \sum d_j \ge \hat{\lambda}_k$) combines existing operations.

`EXP-019` elevates this into a formal, falsifiable scientific contribution across three distinct layers:
1. **Mathematical Theorem (Lemma 1):** Proves that step-to-step projected endpoint differences equal horizon-weighted velocity curvature:
   $$\hat{x}_1(t_j) - \hat{x}_1(t_{j-1}) = (1 - t_j)\bigl(v(x_{t_j}, t_j) - v(x_{t_{j-1}}, t_{j-1})\bigr)$$
   and deconstructs trajectory drift $d_j^2$ into exact conditional velocity power ($p^c_j$), classifier-free guidance acceleration power ($p^g_j$), and interaction cross-term ($p^x_j$) with $0$ extra forward passes.
2. **Domain-Specific Signal Functional:** Extracts internal DiT guidance contrast and net manifold displacement $D_k = \|\hat{x}_1(t_k) - \hat{x}_1(t_0)\|_M / \|\hat{x}_1(t_k)\|_M$ to separate guidance starvation from true post-alignment convergence.
3. **Rigorous Decision Algorithm (GPC):** Implements the Pareto Marginal-Efficiency Stopping Path and Split-Path Monotone Conformal Risk Control to simultaneously control multi-metric degradation (WER, SIM, and clean-prompt UTMOS) with finite-sample statistical guarantees.

## Prepared Infrastructure
- **Preregistration:** `research/experiments/EXP-019-guidance-decomposition-prelim/preregistration.json` frozen with SHA-256 `e907af77e5cd6a34...`, pre-specifying hypotheses H1–H5, data splits, and candidate signals before GPU execution.
- **Runner Script:** `code/colab/run_guidance_work_prelim.py` implementing `freeze`, `smoke`, `dev`, `fresh`, `ood`, and `analyze` subcommands.
- **Turnkey Colab Notebook:** `code/colab/DMOSpeech2_EXP019_Guidance_Decomposition.ipynb` with step-by-step T4 GPU execution.
- **Unit Tests:** `code/tests/test_guidance_work.py` covering Lemma 1 algebraic identities, AUC calculations, Pareto path monotonicity, and preregistration verification.
- **Task Tracking:** Registered task `EVAL-012` in `docs/progress/tasks.json` and rendered `docs/progress/TASKS.md`.

## Verification Checks
- `python scripts/check_workspace.py`: PASSED.
- `python -m unittest discover -s code/tests`: PASSED (Ran 73 tests in ~31s, OK).

