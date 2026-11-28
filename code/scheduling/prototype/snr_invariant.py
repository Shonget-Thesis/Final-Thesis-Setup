"""SNR + Invariant Residual (SNR_R): risk-calibrated adaptive handover and condition-safe routing (EXP-034 / EXP-035).

Implements the model-free SNR_R routing rule with fixed-sequence Learn-Then-Test (LTT) risk calibration:

1. Draft & Coarse Teacher (t_commit = 0.2929, 8-interval sway grid Steps 0..4):
   - One-pass student draft s_0 at tau = 0.0 (1 pass) with clean prompt frames preserved.
   - Five guided teacher Euler steps on the 8-interval sway grid (Steps 0..4, 10 CFG passes) -> endpoint x_hat_1(t_4).
2. Gate + Invariant Residual over generated frames [prompt_len:]:
   - Compute Q(x_hat_1, s_0): per-mel-band sorted values of s_0 placed at the time ranks of x_hat_1.
   - Compute r = ||x_hat_1 - Q(x_hat_1, s_0)||_F / (||x_hat_1||_F + eps).
   - Route:
       flag = (prompt_snr < 20 dB) or (r > lambda_r)
       * Flagged -> CPQH (Q(x_hat_1, s_0) generated frames + s_0 prompt frames; condition-safe)
       * Safe    -> CPH8 (teacher endpoint x_hat_1 directly; preserves full clean prosody)
3. Learn-Then-Test (LTT) Risk Calibration (`calibrate_ltt`):
   - Orders candidate thresholds lambda_r descending (conservative -> aggressive) and accepts while the exact binomial
     p-value BinomCDF(k_wrong_clean; n_clean, alpha) <= delta (controls unnecessary clean interventions <= alpha with
     confidence >= 1 - delta).
4. Student completion:
   - Two student restart steps at tau in {0.2929, 0.6173} (2 passes) -> 13 passes total (vs 21 W8 / 30 paper P25).

The arithmetic matches code/colab/run_invariant_residual.py and code/colab/run_method_comparison.py;
code/tests/test_prototype_snr_invariant.py verifies exact numerical agreement and LTT calibration. NumPy and the Python
standard library are the only required runtime dependencies (PyTorch tensors are accepted cleanly where noted).
"""
import math

import numpy as np

import snr_drift as sd

SNR_GATE_DB = 20.0
GRID_INTERVALS = 8
COMMIT_STEP = 4
TEACHER_GRID = [1.0 - math.cos(math.pi / 2.0 * i / GRID_INTERVALS) for i in range(GRID_INTERVALS + 1)]
TEACHER_STEPS = tuple(range(COMMIT_STEP + 1))

DRAFT_TAU = 0.0
STUDENT_TAUS = list(sd.STUDENT_TAUS)
STUDENT_RESTART_TAUS = list(sd.STUDENT_TAUS[1:])

TEACHER_PASSES = sd.CFG_PASSES_PER_STEP * len(TEACHER_STEPS)          # 10 CFG passes (steps 0..4)
DRAFT_PASSES = 1                                                      # s_0 at tau = 0.0
STUDENT_PASSES = len(STUDENT_RESTART_TAUS)                            # 2 restart passes (0.2929, 0.6173)
METHOD_PASSES = TEACHER_PASSES + DRAFT_PASSES + STUDENT_PASSES        # 13 total passes
CPH8_PASSES = TEACHER_PASSES + STUDENT_PASSES                         # 12 passes without draft/r check

W8_REFERENCE_STEP = sd.REFERENCE_STEP                                 # 8 on 16-interval grid (21 passes)
W8_REFERENCE_PASSES = sd.REFERENCE_PASSES                             # 21 passes
P25_PAPER_PASSES = 30                                                 # 28 teacher + 2 student

EMPIRICAL_DEV_LAMBDA_R = 0.34004843191300677
LTT_15_LAMBDA_R = 0.34161274793996665
LTT_10_LAMBDA_R = 0.3816179751252351

LAMBDA_R_PRESETS = {
    "ltt_15": {
        "value": LTT_15_LAMBDA_R,
        "alpha": 0.15,
        "delta": 0.10,
        "p_value": 0.04802889862602791,
        "short": "0.342 · LTT (15%)",
        "label": "0.342 — LTT certified on EXP-034 dev (alpha=0.15, delta=0.10, p=0.048; EXP-035 unseen: 10.0% clean flags, 90/90 spectral)",
    },
    "ltt_10": {
        "value": LTT_10_LAMBDA_R,
        "alpha": 0.10,
        "delta": 0.10,
        "p_value": 0.042391158275216195,
        "short": "0.382 · LTT (10%)",
        "label": "0.382 — Strict LTT on EXP-034 dev (alpha=0.10, delta=0.10, p=0.042; EXP-035 unseen: 3.3% clean flags, 89/90 spectral)",
    },
    "empirical": {
        "value": EMPIRICAL_DEV_LAMBDA_R,
        "short": "0.340 · Dev Emp",
        "label": "0.340 — EXP-033 empirical constraint (<=15% clean flags without finite-sample LTT correction)",
    },
    "snr_only": {
        "value": 10.0,
        "short": "Off · SNR only",
        "label": "10.00 — Ablation (r disabled): acts as 20 dB SNR-only gate (misses high-SNR spectral noise)",
    },
}
DEFAULT_LAMBDA_R = LAMBDA_R_PRESETS["ltt_15"]["value"]

estimate_prompt_snr = sd.estimate_prompt_snr


def _binom_cdf(k, n, p):
    """Exact cumulative binomial probability P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    q = 1.0 - p
    return float(sum(math.comb(n, i) * (p ** i) * (q ** (n - i)) for i in range(int(k) + 1)))


def calibrate_ltt(rows, alpha=0.15, delta=0.10, gate_db=SNR_GATE_DB, needed_margin=0.30):
    """Fixed-sequence Learn-Then-Test (LTT) calibration of `lambda_r` controlling unnecessary clean switches <= `alpha`.

    Expects CSV-like row dicts with `condition`, `prompt_snr`, `r`, `utmos_CPH8`, `utmos_CPQH`. Evaluates candidate
    thresholds from conservative (inf / highest `r`) to aggressive, stopping as soon as BinomCDF(k_wrong, n_clean,
    alpha) exceeds `delta`. Returns the tightest certified dictionary or `None` if no finite threshold is certified.
    """
    clean = [row for row in rows if row.get("condition") == "clean"]
    n_clean = len(clean)
    if n_clean == 0:
        raise ValueError("Cannot run LTT calibration without clean prompt rows.")

    def needed(row):
        return (float(row["utmos_CPQH"]) - float(row["utmos_CPH8"])) > float(needed_margin)

    vals = sorted({float(row["r"]) for row in rows})
    cands = [float("inf")] + vals[::-1]
    certified = None
    for lam in cands:
        def flagged(row, lam=lam):
            return float(row["prompt_snr"]) < float(gate_db) or float(row["r"]) > lam

        k_err = sum(1 for row in clean if flagged(row) and not needed(row))
        p_val = _binom_cdf(k_err, n_clean, float(alpha))
        if p_val <= float(delta):
            mean_u = sum(float(row["utmos_CPQH" if flagged(row) else "utmos_CPH8"]) for row in rows) / len(rows)
            certified = {
                "lambda_r": float(lam),
                "alpha": float(alpha),
                "delta": float(delta),
                "k_err": int(k_err),
                "n_clean": int(n_clean),
                "cal_wrong_rate": float(k_err / n_clean),
                "p_value": float(p_val),
                "cal_mean_utmos": float(mean_u),
            }
        else:
            break
    return certified


def describe_threshold(lam_r, tol=1e-6):
    """Identify the active risk certificate for `lam_r` when it matches a calibrated/LTT preset."""
    val = float(lam_r)
    for key, preset in LAMBDA_R_PRESETS.items():
        if abs(val - float(preset["value"])) <= tol:
            if "alpha" in preset:
                return (
                    f"LTT risk certificate: clean false-flag risk <= {preset['alpha']:.0%} "
                    f"at {1.0 - preset['delta']:.0%} confidence (EXP-034 dev, p={preset['p_value']:.3f})"
                )
            return preset["label"]
    return f"Custom lambda_r = {val:.3f} (uncalibrated threshold)"


def quantile_handover(student_gen, teacher_gen):
    """Per-band sorted student values placed at the teacher's time ranks over `(frames, bands)` arrays."""
    if hasattr(student_gen, "clone"):
        import torch

        order = torch.sort(teacher_gen, dim=0, stable=True).indices
        values = torch.sort(student_gen, dim=0, stable=True).values
        ranks = torch.argsort(order, dim=0)
        return torch.gather(values, 0, ranks)
    s = np.asarray(student_gen, dtype=np.float64)
    order = np.argsort(np.asarray(teacher_gen, dtype=np.float64), axis=0, kind="stable")
    out = np.empty_like(s)
    np.put_along_axis(out, order, np.sort(s, axis=0, kind="stable"), axis=0)
    return out


def quantile_payload(student_draft, teacher_endpoint, prompt_len):
    """CPQH payload over `(1, frames, bands)`: student draft prompt `[:prompt_len]` + `Q(t_gen, s0_gen)`."""
    out = student_draft.clone() if hasattr(student_draft, "clone") else np.array(student_draft, copy=True)
    out[0, prompt_len:, :] = quantile_handover(student_draft[0, prompt_len:, :], teacher_endpoint[0, prompt_len:, :])
    return out


def invariant_residual(teacher_gen, student_gen, eps=1e-12):
    """r = ||x_hat - Q(x_hat, s_0)||_F / (||x_hat||_F + eps) over `(frames, bands)` generated segments."""
    if hasattr(teacher_gen, "detach") and hasattr(student_gen, "detach"):
        t = teacher_gen.detach().cpu().numpy().astype(np.float64)
        s = student_gen.detach().cpu().numpy().astype(np.float64)
    else:
        t = np.asarray(teacher_gen, dtype=np.float64)
        s = np.asarray(student_gen, dtype=np.float64)
    q = quantile_handover(s, t)
    return float(np.linalg.norm(t - q) / (np.linalg.norm(t) + eps))


def band_residuals(teacher_gen, student_gen, n_bins=8, eps=1e-12):
    """Mean per-bin relative residual `||t_b - Q(t_b, s0_b)|| / (||t_b|| + eps)` across grouped mel bands."""
    t = teacher_gen.detach().cpu().numpy().astype(np.float64) if hasattr(teacher_gen, "detach") else np.asarray(teacher_gen, dtype=np.float64)
    s = student_gen.detach().cpu().numpy().astype(np.float64) if hasattr(student_gen, "detach") else np.asarray(student_gen, dtype=np.float64)
    q = quantile_handover(s, t)
    edges = np.array_split(np.arange(t.shape[1]), max(1, int(n_bins)))
    out = []
    for idx in edges:
        if len(idx) == 0:
            continue
        diff = t[:, idx] - q[:, idx]
        out.append(float(np.linalg.norm(diff) / (np.linalg.norm(t[:, idx]) + eps)))
    return out


def decide(prompt_snr, residual, lam_r=DEFAULT_LAMBDA_R, gate_db=SNR_GATE_DB):
    """Return `'CPQH'` when `prompt_snr < gate_db` or `residual > lam_r`, else `'CPH8'`."""
    flagged = (float(prompt_snr) < float(gate_db)) or (float(residual) > float(lam_r))
    return "CPQH" if flagged else "CPH8"


def _describe_trigger(prompt_snr, residual, lam_r, gate_db):
    snr_low = prompt_snr < gate_db
    r_high = residual > lam_r
    if snr_low and r_high:
        return "both", (
            f"SNR {prompt_snr:.1f} dB < {gate_db:.0f} dB and r = {residual:.3f} > lambda_r = {lam_r:.3f}: "
            "switch to condition-safe CPQH"
        )
    if snr_low:
        return "snr", (
            f"SNR {prompt_snr:.1f} dB < {gate_db:.0f} dB (gate closed, r = {residual:.3f}): "
            "switch to condition-safe CPQH"
        )
    if r_high:
        return "residual", (
            f"SNR {prompt_snr:.1f} dB >= {gate_db:.0f} dB passed the SNR gate, but r = {residual:.3f} > "
            f"lambda_r = {lam_r:.3f} caught non-monotone residual: switch to CPQH"
        )
    return "none", (
        f"SNR {prompt_snr:.1f} dB >= {gate_db:.0f} dB and r = {residual:.3f} <= lambda_r = {lam_r:.3f}: "
        "safe to hand over teacher endpoint (CPH8)"
    )


def evaluate(prompt_snr, residual, lam_r=DEFAULT_LAMBDA_R, gate_db=SNR_GATE_DB, r_trace=None, band_r=None):
    """Build the full decision summary dict returned by `engine.py` and consumed by `cli.py` and `web/index.html`."""
    prompt_snr = float(prompt_snr)
    residual = float(residual)
    lam_r = float(lam_r)
    gate_db = float(gate_db)
    choice = decide(prompt_snr, residual, lam_r=lam_r, gate_db=gate_db)
    trigger, reason = _describe_trigger(prompt_snr, residual, lam_r, gate_db)
    return {
        "method": "SNR_R",
        "prompt_snr_db": prompt_snr,
        "gate_db": gate_db,
        "gate_open": prompt_snr >= gate_db,
        "residual_r": residual,
        "lambda_r": lam_r,
        "risk_certificate": describe_threshold(lam_r),
        "r_trace": {str(k): float(v) for k, v in sorted((r_trace or {}).items())},
        "band_residuals": [float(v) for v in (band_r or [])],
        "choice": choice,
        "flagged": choice == "CPQH",
        "trigger": trigger,
        "reason": reason,
        "handover_step": COMMIT_STEP,
        "handover_time": TEACHER_GRID[COMMIT_STEP],
        "passes": METHOD_PASSES,
        "teacher_passes": TEACHER_PASSES,
        "draft_passes": DRAFT_PASSES,
        "student_passes": STUDENT_PASSES,
        "reference_passes": W8_REFERENCE_PASSES,
        "paper_passes": P25_PAPER_PASSES,
        "saving": 1.0 - METHOD_PASSES / W8_REFERENCE_PASSES,
        "saving_vs_paper": 1.0 - METHOD_PASSES / P25_PAPER_PASSES,
    }
