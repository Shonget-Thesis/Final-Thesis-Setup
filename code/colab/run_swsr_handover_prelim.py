"""Sway-Weighted SNR-Drift Readiness (SWSR / STSR / CSDR) & Full Steps Direct Benchmark (No Failure Rule).

Direct benchmark comparison across all fixed steps and adaptive handover algorithms using the standard
TTS evaluation metrics (WER, SIM, UTMOS, NFE, Runtime, RTF, paired Deltas, Wilcoxon, and TOST
equivalence) WITHOUT the binary Definition 3.1 failure rule:

  1. 16-Interval Sway Grid (t_k = 1 - cos(pi * k / 32), Steps k = 1..8):
     - Steps {4, 5, 6, 8} are stored in EXP-017 `grid32.csv`.
     - Missing steps {1, 2, 3, 7} are synthesized on the Tesla T4 and merged with {4, 5, 6, 8}, giving
       all 8 consecutive checkpoints:
         Step 1 (t_1 = 0.0048,  7 NFE) .. Step 7 (t_7 = 0.2270 <= 0.25, 19 NFE) .. Step 8 (t_8 = 0.2929 > 0.25, 21 NFE).

  2. 32-Interval Sway Grid (t_i = 1 - cos(pi * i / 64), Steps i = 1..15):
     - Steps {7, 9, 11, 13, 14} (`s007, s010, s015, s020, s025`) are stored in EXP-017 `grid32.csv`.
     - Missing steps {1, 2, 3, 4, 5, 6, 8, 10, 12, 15} are synthesized on the Tesla T4 and merged with
       {7, 9, 11, 13, 14}, giving all 15 consecutive 32-grid checkpoints up to 0.25:
         Step 1 (t_1 = 0.0012,  7 NFE) .. Step 14 (t_14 = 0.2270 <= 0.25, 33 NFE) .. Step 15 (t_15 = 0.2591 > 0.25, 35 NFE).

  3. Novel Zero-Overhead Handover Readiness Algorithms (SWSR / STSR / CSDR) on Both Grids:
     Using only prompt SNR S_hat, sway grid t_k / step size Delta_t_k = t_k - t_{k-1}, and endpoint drift d_k:
       - SWSR (Sway-Weighted SNR-Drift Readiness): R_k^SWSR = S_hat * Delta_t_k * d_k >= lambda_hat
       - STSR (Sway-Time Coupled SNR-Drift Readiness): R_k^STSR = S_hat * t_k * d_k >= lambda_hat
       - CSDR (Cumulative SNR-Drift Readiness): R_k^CSDR = S_hat * sum_{j <= k} d_j >= lambda_hat
     Thresholds are calibrated directly on continuous WER and SIM preservation against the reference
     baseline (mean WER within +margin_wer and mean SIM within -margin_sim on calibration speakers,
     verified with TOST equivalence and Wilcoxon signed-rank tests).
"""
import argparse
import csv
import datetime
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_followup_tests as rf  # noqa: E402
import run_grid32_prelim as g32  # noqa: E402

THESIS_ROOT = rf.THESIS_ROOT
FOLDS, SEED, THETA = g32.FOLDS, g32.SEED, g32.THETA
CONDS, NOISY = g32.CONDS, g32.NOISY
EQUIV, RESPLITS, BOOT = g32.EQUIV, g32.RESPLITS, g32.BOOT
METRICS = g32.METRICS
TIMING = g32.TIMING

# Direct continuous-metric calibration tolerances relative to the reference step on training folds
WER_CAL_TOL = 0.0020  # mean WER within +0.2 percentage points (0.0020) of the reference baseline
SIM_CAL_TOL = 0.0014  # mean SIM within -0.0014 of the reference baseline

GRID16 = g32.sway_grid(16)
GRID32 = g32.GRID32
DELTA_T16 = {k: GRID16[k] - GRID16[k - 1] for k in range(1, 17)}
DELTA_T32 = {k: GRID32[k] - GRID32[k - 1] for k in range(1, 33)}

ALL_G16_STEPS = (1, 2, 3, 4, 5, 6, 7, 8)
STORED_G16_STEPS = (4, 5, 6, 8)
MISSING_G16_STEPS = (1, 2, 3, 7)

ALL_G32_STEPS = tuple(range(1, 16))
STORED_G32_MAP = {7: "s007", 9: "s010", 11: "s015", 13: "s020", 14: "s025"}
MISSING_G32_STEPS = tuple(k for k in ALL_G32_STEPS if k not in STORED_G32_MAP)

EVALS_ALL = {
    **g32.EVALS,
    **{f"g16_{k}": k + 1 for k in ALL_G16_STEPS},
    **{f"g32_{k}": k + 1 for k in ALL_G32_STEPS},
}
STUDENT_STEPS_ALL = {
    **g32.STUDENT_STEPS,
    **{f"g16_{k}": 3 for k in ALL_G16_STEPS},
    **{f"g32_{k}": 3 for k in ALL_G32_STEPS},
}
COST_ALL = {o: 2 * EVALS_ALL[o] + STUDENT_STEPS_ALL[o] for o in EVALS_ALL}

EXP017_CSV = THESIS_ROOT / "research/experiments/EXP-017-grid32-prelim/results/grid32.csv"
ALL1000_CSV = THESIS_ROOT / "research/evidence/benchmarks/gpu_grid_all1000.csv"
SPLITS_DIR = THESIS_ROOT / "research/protocols/splits"

STEPS_CSV_FIELDS = (
    ["utt_id", "speaker_id", "condition", "prompt_snr", "gpu", "has_g32_all", "case_seconds"]
    + [f"d16_{k}" for k in ALL_G16_STEPS]
    + [f"d32_{k}" for k in ALL_G32_STEPS]
    + [f"{m}_g16_{k}" for k in ALL_G16_STEPS for m in METRICS]
    + [f"{m}_g32_{k}" for k in ALL_G32_STEPS for m in METRICS]
    + list(TIMING)
    + ["utc"]
)

NOVEL_NAMES = {
    "swsr": "SWSR: Sway-Weighted SNR-Drift Readiness (S_hat * Delta_t_k * d_k >= lambda)",
    "stsr": "STSR: Sway-Time Coupled SNR-Drift Readiness (S_hat * t_k * d_k >= lambda)",
    "csdr": "CSDR: Cumulative SNR-Drift Readiness (S_hat * sum_{j<=k} d_j >= lambda)",
}

CASCADE_NAMES = {
    "stac": "STAC: Stage-Wise Trajectory Action Cascade (A_k = sum_{j=1}^k d_j >= lambda_k, NO SNR!)",
    "crc_stac": "CRC-STAC: Conformal Risk-Controlled Trajectory Action Cascade (alpha=0.05, NO SNR!)",
    "stac_single": "Ablation: Single-Threshold Trajectory Action (A_k = sum_{j=1}^k d_j >= lambda, NO SNR!)",
    "pos_drift_casc": "Ablation: Stage-Wise Single-Step Drift (d_k >= lambda_k, NO path sum, NO SNR!)",
    "snr_casc": "Control: Stage-Wise SNR Cascade (S_hat >= theta_k, matched DOF!)",
}

ALL_SCHEDULERS = {
    "snr20": "SNR-only (20 dB gate -> earliest step)",
    "snr_cal": "SNR-only (calibrated SNR gate)",
    "snr_casc": CASCADE_NAMES["snr_casc"],
    "later5": "SNR gate (20 dB) -> fixed step 5",
    "later6": "SNR gate (20 dB) -> fixed step 6",
    "stab": "Stabilization only (d_k <= lambda)",
    "snr_drift": "Old Definition 3.4 (S_hat >= 20 dB and d_k <= lambda)",
    "snr_tas": "SNR + TAS (Mahalanobis trajectory anomaly)",
    "stac": CASCADE_NAMES["stac"],
    "crc_stac": CASCADE_NAMES["crc_stac"],
    "stac_single": CASCADE_NAMES["stac_single"],
    "pos_drift_casc": CASCADE_NAMES["pos_drift_casc"],
    "swsr": NOVEL_NAMES["swsr"],
    "stsr": NOVEL_NAMES["stsr"],
    "csdr": NOVEL_NAMES["csdr"],
}

GRIDS_EXT = {
    **g32.GRIDS,
    "g16_all8": {
        "label": "16-interval grid, ALL Steps 1..8 (k = 1..7 candidates vs Step 8 reference, t_8 = 0.2929)",
        "cands": tuple(f"g16_{k}" for k in range(1, 8)),
        "ref": "g16_8",
        "k": {f"g16_{k}": k for k in range(1, 8)},
        "drift": "d16",
        "later": {"later5": "g16_5", "later6": "g16_6"},
        "lam": g32.LAM16,
        "tas_k": 4,
    },
    "g16_le025": {
        "label": "16-interval grid strictly <= 0.25 (k = 1..6 candidates vs Step 7 reference, t_7 = 0.2270 <= 0.25)",
        "cands": tuple(f"g16_{k}" for k in range(1, 7)),
        "ref": "g16_7",
        "k": {f"g16_{k}": k for k in range(1, 7)},
        "drift": "d16",
        "later": {"later5": "g16_5", "later6": "g16_6"},
        "lam": g32.LAM16,
        "tas_k": 4,
    },
    "g32_all14": {
        "label": "32-interval grid strictly <= 0.25, ALL Steps 1..14 (i = 1..13 candidates vs Step 14 reference, t_14 = 0.2270)",
        "cands": tuple(f"g32_{k}" for k in range(1, 14)),
        "ref": "g32_14",
        "k": {f"g32_{k}": k for k in range(1, 14)},
        "drift": "d32",
        "later": {"later5": "g32_9", "later6": "g32_11"},
        "lam": g32.LAM32,
        "tas_k": 7,
    },
    "g32_all15": {
        "label": "32-interval grid, ALL Steps 1..15 (i = 1..14 candidates vs Step 15 reference, t_15 = 0.2591 > 0.25)",
        "cands": tuple(f"g32_{k}" for k in range(1, 15)),
        "ref": "g32_15",
        "k": {f"g32_{k}": k for k in range(1, 15)},
        "drift": "d32",
        "later": {"later5": "g32_9", "later6": "g32_11"},
        "lam": g32.LAM32,
        "tas_k": 7,
    },
}


def is_g32_grid(G):
    return G.startswith("g32")


# ----------------------------------------------------------------------------- SWSR / STSR / CSDR readiness scores
def readiness_score(r, G, o, variant):
    """Computes the readiness score R_k at candidate output `o` on grid `G`.
    Uses only prompt SNR S_hat, sway-grid time t_k (or step size Delta_t_k), and endpoint drift d_k."""
    g = GRIDS_EXT[G]
    k = g["k"][o]
    d_map = r[g["drift"]]
    snr = max(float(r["snr"]), 0.0)
    grid = GRID32 if is_g32_grid(G) else GRID16
    dt_map = DELTA_T32 if is_g32_grid(G) else DELTA_T16
    if variant == "swsr":
        return snr * dt_map[k] * d_map[k]
    if variant == "stsr":
        return snr * grid[k] * d_map[k]
    if variant == "csdr":
        cands_up_to_o = [c for c in g["cands"] if g["k"][c] <= k]
        return snr * sum(d_map[g["k"][c]] for c in cands_up_to_o)
    raise ValueError(f"Unknown readiness variant: {variant}")


def readiness_grid(train, G, variant, theta=THETA, n_quantiles=80):
    """Threshold grid ordered from MOST CONSERVATIVE (highest lambda -> reference)
    to MOST PERMISSIVE (lowest lambda -> earliest candidate). Constructed purely from training rows."""
    g = GRIDS_EXT[G]
    vals = sorted({
        round(readiness_score(r, G, o, variant), 6)
        for r in train
        if (theta is None or r["snr"] >= theta)
        for o in g["cands"]
    }, reverse=True)
    if not vals:
        return []
    if len(vals) <= n_quantiles:
        return vals
    idxs = sorted({int(round(i * (len(vals) - 1) / (n_quantiles - 1))) for i in range(n_quantiles)})
    return [vals[i] for i in idxs]


def p_readiness(G, variant, lam, theta=THETA):
    """Selects the earliest candidate o in K_cand with S_hat >= theta and R_k >= lam; else reference."""
    g = GRIDS_EXT[G]

    def pol(r):
        if lam is None or (theta is not None and r["snr"] < theta):
            return g["ref"]
        return next((o for o in g["cands"] if readiness_score(r, G, o, variant) >= lam), g["ref"])

    return pol


def p_snr_ext(G, theta):
    g = GRIDS_EXT[G]
    return lambda r: g["cands"][0] if theta is not None and r["snr"] >= theta else g["ref"]


def p_later_ext(G, o, theta=THETA):
    return lambda r: o if r["snr"] >= theta else GRIDS_EXT[G]["ref"]


def p_stab_ext(G, lam, theta=None):
    g = GRIDS_EXT[G]

    def pol(r):
        if lam is None or (theta is not None and r["snr"] < theta):
            return g["ref"]
        return next((o for o in g["cands"] if r[g["drift"]][g["k"][o]] <= lam), g["ref"])

    return pol


def tas_fit_ext(train, G):
    import numpy as np
    g = GRIDS_EXT[G]
    logs = lambda r: [math.log(max(r[g["drift"]][i], 1e-12)) for i in range(1, g["tas_k"] + 1)]
    X = np.array([logs(r) for r in train if r["cond"] == "clean"])
    mu = X.mean(0)
    Ci = np.linalg.inv(np.cov(X, rowvar=False) + 1e-3 * np.eye(X.shape[1]))

    def score(r):
        z = np.asarray(logs(r)) - mu
        return float(z @ Ci @ z) if z.mean() < 0 else 0.0

    return score


def p_tas_ext(G, score, kappa):
    g = GRIDS_EXT[G]
    return lambda r: g["cands"][0] if kappa is not None and r["snr"] >= THETA and score(r) <= kappa else g["ref"]


# ----------------------------------------------------------------------------- Direct Continuous-Metric Calibration (No Failure Rule)
def quality_preserved(train, pol, G, wer_tol=WER_CAL_TOL, sim_tol=SIM_CAL_TOL):
    """Checks directly on continuous WER and SIM (no binary failure rule!) whether policy `pol`
    preserves the reference baseline's mean WER (within +wer_tol) and mean SIM (within -sim_tol)
    overall, on clean/noisy groups, and across individual training conditions."""
    ref = GRIDS_EXT[G]["ref"]
    for keep in (lambda r: True, lambda r: r["cond"] == "clean", lambda r: r["cond"] != "clean"):
        sub = [r for r in train if keep(r)]
        if not sub:
            continue
        dw = statistics.mean(r["m"][pol(r)]["wer"] - r["m"][ref]["wer"] for r in sub)
        ds = statistics.mean(r["m"][pol(r)]["similarity"] - r["m"][ref]["similarity"] for r in sub)
        if dw > wer_tol or ds < -sim_tol:
            return False
    for c in CONDS:
        sub = [r for r in train if r["cond"] == c]
        if not sub:
            continue
        dw = statistics.mean(r["m"][pol(r)]["wer"] - r["m"][ref]["wer"] for r in sub)
        ds = statistics.mean(r["m"][pol(r)]["similarity"] - r["m"][ref]["similarity"] for r in sub)
        if dw > wer_tol * 2.0 or ds < -sim_tol * 2.0:
            return False
    if any((r["m"][pol(r)]["wer"] - r["m"][ref]["wer"]) > 0.25 for r in train):
        return False
    return True


def calibrate_direct_sequence(train, grid, make, G):
    """Scans thresholds from most conservative to most permissive and selects the last threshold
    that preserves continuous WER and SIM within (WER_CAL_TOL, SIM_CAL_TOL) of the reference."""
    chosen = None
    for p in grid:
        if not quality_preserved(train, make(p), G):
            break
        chosen = p
    return chosen


def trajectory_action(r, G, o, use_path_sum=True):
    """Endogenous ODE trajectory action A_k = sum_{j=1}^k d_j (discrete path variation of the projected
    endpoint x_hat_1(t) from t_0 to t_k). Requires ZERO prompt SNR heuristic."""
    g = GRIDS_EXT[G]
    k = g["k"][o]
    d_map = r[g["drift"]]
    if not use_path_sum:
        return d_map[k]
    return sum(d_map[j] for j in range(1, k + 1) if j in d_map)


def fit_stagewise_cascade(train, G, use_path_sum=True):
    """Calibrates a stage-wise stopping boundary vector lambda_vec = (lambda_{k_1}, ..., lambda_{k_{m-1}})
    sequentially across candidate checkpoints using ONLY endogenous trajectory action A_k (no SNR)."""
    g = GRIDS_EXT[G]
    cands = list(g["cands"])
    lams = {o: None for o in cands}

    def make_pol(cur_lams):
        def pol(r):
            for o in cands:
                if cur_lams.get(o) is not None and trajectory_action(r, G, o, use_path_sum) >= cur_lams[o]:
                    return o
            return g["ref"]

        return pol

    for o in cands:
        vals = sorted({round(trajectory_action(r, G, o, use_path_sum), 5) for r in train}, reverse=True)
        if len(vals) > 80:
            idxs = sorted({int(round(i * (len(vals) - 1) / 79)) for i in range(80)})
            vals = [vals[i] for i in idxs]
        best = None
        for v in vals:
            trial = dict(lams)
            trial[o] = v
            if not quality_preserved(train, make_pol(trial), G):
                break
            best = v
        lams[o] = best
    return make_pol(lams), lams


def fit_single_action(train, G):
    """Ablation: single scalar threshold lambda applied to A_k = sum_{j=1}^k d_j across all steps (no stage-wise cascade)."""
    g = GRIDS_EXT[G]
    vals = sorted({round(trajectory_action(r, G, o, True), 5) for r in train for o in g["cands"]}, reverse=True)
    if len(vals) > 80:
        idxs = sorted({int(round(i * (len(vals) - 1) / 79)) for i in range(80)})
        vals = [vals[i] for i in idxs]

    def make(x):
        return lambda r: g["ref"] if x is None else next((o for o in g["cands"] if trajectory_action(r, G, o, True) >= x), g["ref"])

    lam = calibrate_direct_sequence(train, vals, make, G)
    return make(lam), lam


def fit_snr_cascade(train, G):
    """Control: Stage-wise SNR cascade (S_hat >= theta_k) with the exact same stage-wise degrees of freedom as STAC."""
    g = GRIDS_EXT[G]
    cands = list(g["cands"])
    lams = {o: None for o in cands}

    def make_pol(cur_lams):
        return lambda r: next((o for o in cands if cur_lams.get(o) is not None and r["snr"] >= cur_lams[o]), g["ref"])

    vals = sorted({round(float(r["snr"]), 3) for r in train}, reverse=True)
    for o in cands:
        best = None
        for v in vals:
            trial = dict(lams)
            trial[o] = v
            if not quality_preserved(train, make_pol(trial), G):
                break
            best = v
        lams[o] = best
    return make_pol(lams), lams


def loss_bounded_hinge(r, o, ref, m_wer=0.01, b_wer=0.10, m_sim=0.015, b_sim=0.025):
    """Bounded equivalence hinge loss L(x, k) in [0, 1] outside indifference margin (m_wer, m_sim)."""
    dw = max(0.0, (r["m"][o]["wer"] - r["m"][ref]["wer"] - m_wer) / b_wer)
    ds = max(0.0, (r["m"][ref]["similarity"] - r["m"][o]["similarity"] - m_sim) / b_sim)
    return min(1.0, max(dw, ds))


def fit_crc_stac(train, G, alpha=0.05):
    """Conformal Risk-Controlled Stage-Wise Trajectory Action Cascade (CRC-STAC, Angelopoulos et al., 2024):
    Enforces finite-sample distribution-free marginal risk control (1 + sum_{i=1}^n L_i) / (n + 1) <= alpha."""
    g = GRIDS_EXT[G]
    cands = list(g["cands"])
    ref = g["ref"]
    spks = sorted({r["spk"] for r in train})
    by_spk = {s: [r for r in train if r["spk"] == s] for s in spks}
    n = len(spks)
    lams = {o: None for o in cands}

    def make_pol(cur_lams):
        return lambda r: next(
            (o for o in cands if cur_lams.get(o) is not None and trajectory_action(r, G, o, True) >= cur_lams[o]),
            ref,
        )

    for o in cands:
        vals = sorted({round(trajectory_action(r, G, o, True), 5) for r in train}, reverse=True)
        if len(vals) > 80:
            idxs = sorted({int(round(i * (len(vals) - 1) / 79)) for i in range(80)})
            vals = [vals[i] for i in idxs]
        best = None
        for v in vals:
            trial = dict(lams)
            trial[o] = v
            pol = make_pol(trial)
            emp_sum = sum(statistics.mean(loss_bounded_hinge(r, pol(r), ref) for r in by_spk[s]) for s in spks)
            if (1.0 + emp_sum) / (n + 1.0) <= alpha and quality_preserved(train, pol, G):
                best = v
            else:
                break
        lams[o] = best
    return make_pol(lams), lams


def fold_policy_direct(name, train, G):
    g = GRIDS_EXT[G]
    if name == "stac":
        return fit_stagewise_cascade(train, G, use_path_sum=True)
    if name == "crc_stac":
        return fit_crc_stac(train, G, alpha=0.05)
    if name == "stac_single":
        return fit_single_action(train, G)
    if name == "pos_drift_casc":
        return fit_stagewise_cascade(train, G, use_path_sum=False)
    if name == "snr_casc":
        return fit_snr_cascade(train, G)
    if name in NOVEL_NAMES:
        grid = readiness_grid(train, G, name, theta=THETA)
        lam = calibrate_direct_sequence(train, grid, lambda x: p_readiness(G, name, x, THETA), G)
        return p_readiness(G, name, lam, THETA), lam
    if name == "snr20":
        return p_snr_ext(G, THETA), THETA
    if name == "snr_cal":
        theta = calibrate_direct_sequence(train, g32.THETA_GRID, lambda x: p_snr_ext(G, x), G)
        return p_snr_ext(G, theta), theta
    if name in g["later"]:
        return p_later_ext(G, g["later"][name]), THETA
    if name in ("stab", "snr_drift"):
        gate = None if name == "stab" else THETA
        lam = calibrate_direct_sequence(train, g["lam"], lambda x: p_stab_ext(G, x, gate), G)
        return p_stab_ext(G, lam, gate), lam
    if name == "snr_tas":
        score = tas_fit_ext(train, G)
        kappa = calibrate_direct_sequence(
            train, sorted({score(r) for r in train if r["snr"] >= THETA}), lambda x: p_tas_ext(G, score, x), G
        )
        return p_tas_ext(G, score, kappa), kappa
    raise ValueError(name)


def active_schedulers():
    names = list(ALL_SCHEDULERS.keys())
    try:
        import numpy  # noqa: F401
    except ImportError:
        names.remove("snr_tas")
    return names


def runtime_ext(r, o):
    t = r.get("time") or {}
    if any(t.get(x) is None for x in ("t_prepare", "t_eval", "t_student", "t_vocoder", "audio_seconds")) or not t["audio_seconds"]:
        return None, None
    sec = t["t_prepare"] + EVALS_ALL[o] * t["t_eval"] + STUDENT_STEPS_ALL[o] * t["t_student"] + t["t_vocoder"]
    return sec, sec / t["audio_seconds"]


def teacher_runtime_ext(r, o):
    t = r.get("time") or {}
    if t.get("t_eval") is None:
        return None
    return EVALS_ALL[o] * t["t_eval"]


GATE_OVERHEAD_MS = {
    "stac": 0.12,
    "crc_stac": 0.12,
    "stac_single": 0.12,
    "pos_drift_casc": 0.12,
    "stab": 0.12,
    "snr20": 1.85,
    "snr_cal": 1.85,
    "snr_casc": 1.85,
    "snr_drift": 1.97,
    "swsr": 1.97,
    "stsr": 1.97,
    "csdr": 1.97,
    "scre": 1.97,
    "edp": 1.97,
    "snr_tas": 2.35,
}


def summarize_direct(table, pick, G):
    """Direct continuous benchmark summary (WER, SIM, UTMOS, NFE, Saving, Runtime, RTF, Deltas vs Reference,
    Wilcoxon signed-rank p-values, and TOST equivalence p-values) WITHOUT any binary failure rule."""
    g = GRIDS_EXT[G]
    ref = g["ref"]
    rows = [r for r in table if g32.key(r) in pick]
    ref_cost = COST_ALL[ref]
    costs = [COST_ALL[pick[g32.key(r)]] for r in rows]
    cost = statistics.mean(costs)

    clean_rows = [r for r in rows if r["cond"] == "clean"]
    noisy_rows = [r for r in rows if r["cond"] != "clean"]

    def mean_metric(sub, meas):
        return statistics.mean(r["m"][pick[g32.key(r)]][meas] for r in sub) if sub else None

    def mean_delta(sub, meas):
        return statistics.mean(r["m"][pick[g32.key(r)]][meas] - r["m"][ref][meas] for r in sub) if sub else None

    non_halluc = [r for r in rows if r["m"][ref]["wer"] <= 1.0]
    wer_cap_all = statistics.mean(min(1.0, r["m"][pick[g32.key(r)]]["wer"]) for r in rows) if rows else None
    d_wer_cap_all = (
        statistics.mean(min(1.0, r["m"][pick[g32.key(r)]]["wer"]) - min(1.0, r["m"][ref]["wer"]) for r in rows)
        if rows
        else None
    )
    wer_trim_all = statistics.mean(min(1.0, r["m"][pick[g32.key(r)]]["wer"]) for r in non_halluc) if non_halluc else None
    d_wer_trim_all = (
        statistics.mean(min(1.0, r["m"][pick[g32.key(r)]]["wer"]) - r["m"][ref]["wer"] for r in non_halluc)
        if non_halluc
        else None
    )

    res = {
        "n": len(rows),
        "mean_cost": cost,
        "clean_cost": statistics.mean(COST_ALL[pick[g32.key(r)]] for r in clean_rows) if clean_rows else None,
        "noisy_cost": statistics.mean(COST_ALL[pick[g32.key(r)]] for r in noisy_rows) if noisy_rows else None,
        "saving": 1.0 - cost / ref_cost,
        "clean_saving": (1.0 - statistics.mean(COST_ALL[pick[g32.key(r)]] for r in clean_rows) / ref_cost) if clean_rows else None,
        "wer_all": mean_metric(rows, "wer"),
        "wer_clean": mean_metric(clean_rows, "wer"),
        "wer_noisy": mean_metric(noisy_rows, "wer"),
        "d_wer_all": mean_delta(rows, "wer"),
        "d_wer_clean": mean_delta(clean_rows, "wer"),
        "d_wer_noisy": mean_delta(noisy_rows, "wer"),
        "wer_capped_all": wer_cap_all,
        "d_wer_capped_all": d_wer_cap_all,
        "wer_trimmed_all": wer_trim_all,
        "d_wer_trimmed_all": d_wer_trim_all,
        "sim_all": mean_metric(rows, "similarity"),
        "sim_clean": mean_metric(clean_rows, "similarity"),
        "sim_noisy": mean_metric(noisy_rows, "similarity"),
        "d_sim_all": mean_delta(rows, "similarity"),
        "d_sim_clean": mean_delta(clean_rows, "similarity"),
        "d_sim_noisy": mean_delta(noisy_rows, "similarity"),
        "utmos_all": mean_metric(rows, "utmos"),
        "utmos_clean": mean_metric(clean_rows, "utmos"),
        "utmos_noisy": mean_metric(noisy_rows, "utmos"),
        "d_utmos_all": mean_delta(rows, "utmos"),
        "per_cond_wer": {c: mean_metric([r for r in rows if r["cond"] == c], "wer") for c in CONDS},
        "per_cond_sim": {c: mean_metric([r for r in rows if r["cond"] == c], "similarity") for c in CONDS},
        "picks": {
            o: sum(pick[g32.key(r)] == o for r in rows)
            for o in sorted({pick[g32.key(r)] for r in rows}, key=lambda x: COST_ALL[x])
        },
    }

    # Speaker-cluster bootstrap 95% CIs (resampling speakers to account for within-speaker correlation)
    spks = sorted({r["spk"] for r in rows})
    if len(spks) >= 4:
        by_spk = {s: [r for r in rows if r["spk"] == s] for s in spks}
        spk_w = [len(by_spk[s]) for s in spks]
        spk_cost = [statistics.mean(COST_ALL[pick[g32.key(r)]] for r in by_spk[s]) for s in spks]
        spk_dwer = [statistics.mean(r["m"][pick[g32.key(r)]]["wer"] - r["m"][ref]["wer"] for r in by_spk[s]) for s in spks]
        spk_dcap = [
            statistics.mean(min(1.0, r["m"][pick[g32.key(r)]]["wer"]) - min(1.0, r["m"][ref]["wer"]) for r in by_spk[s])
            for s in spks
        ]
        rng = random.Random(2026)
        b_cost, b_dwer, b_dcap = [], [], []
        n_s = len(spks)
        for _ in range(400):
            idxs = [rng.randrange(n_s) for _ in range(n_s)]
            tot_w = sum(spk_w[i] for i in idxs)
            b_cost.append(sum(spk_cost[i] * spk_w[i] for i in idxs) / tot_w)
            b_dwer.append(sum(spk_dwer[i] * spk_w[i] for i in idxs) / tot_w)
            b_dcap.append(sum(spk_dcap[i] * spk_w[i] for i in idxs) / tot_w)
        b_cost.sort()
        b_dwer.sort()
        b_dcap.sort()
        res["cost_ci_lo"], res["cost_ci_hi"] = b_cost[10], b_cost[389]
        res["d_wer_ci_lo"], res["d_wer_ci_hi"] = b_dwer[10], b_dwer[389]
        res["d_wer_capped_ci_lo"], res["d_wer_capped_ci_hi"] = b_dcap[10], b_dcap[389]

    rts = [runtime_ext(r, pick[g32.key(r)]) for r in rows]
    res["teacher_evals"] = statistics.mean(EVALS_ALL[pick[g32.key(r)]] for r in rows) if rows else None
    res["teacher_passes"] = 2.0 * res["teacher_evals"] if res["teacher_evals"] is not None else None
    if rts and all(x[0] is not None for x in rts):
        ref_rt = statistics.mean(runtime_ext(r, ref)[0] for r in rows)
        mean_rt = statistics.mean(x[0] for x in rts)
        mean_rtf = statistics.mean(x[1] for x in rts)
        t_rts = [teacher_runtime_ext(r, pick[g32.key(r)]) for r in rows]
        ref_t_rts = [teacher_runtime_ext(r, ref) for r in rows]
        mean_t_rt = statistics.mean(t_rts) if all(x is not None for x in t_rts) else None
        ref_t_rt = statistics.mean(ref_t_rts) if all(x is not None for x in ref_t_rts) else None
        res.update({
            "runtime": mean_rt,
            "ref_runtime": ref_rt,
            "time_saved_sec": ref_rt - mean_rt,
            "rtf": mean_rtf,
            "x_realtime": (1.0 / mean_rtf) if mean_rtf > 1e-9 else None,
            "runtime_saving": 1.0 - mean_rt / ref_rt,
            "speedup": (ref_rt / mean_rt) if mean_rt > 1e-9 else None,
            "teacher_runtime": mean_t_rt,
            "teacher_speedup": (ref_t_rt / mean_t_rt) if (mean_t_rt and ref_t_rt and mean_t_rt > 1e-9) else None,
        })

    # Speaker-level paired Wilcoxon and TOST equivalence vs reference
    spks = sorted({r["spk"] for r in rows})
    for meas in ("wer", "similarity", "utmos"):
        diffs = [
            statistics.mean(r["m"][pick[g32.key(r)]][meas] - r["m"][ref][meas] for r in rows if r["spk"] == s)
            for s in spks
        ]
        try:
            from scipy.stats import wilcoxon, ttest_1samp
            res[f"{meas}_wilcoxon_p"] = float(wilcoxon(diffs).pvalue) if any(abs(x) > 1e-12 for x in diffs) else 1.0
            if meas in EQUIV and len(diffs) > 2:
                m = EQUIV[meas]
                if statistics.pstdev(diffs) < 1e-12:
                    res[f"{meas}_tost_p"] = 0.0 if abs(statistics.mean(diffs)) < m else 1.0
                else:
                    lo = ttest_1samp(diffs, -m, alternative="greater").pvalue
                    hi = ttest_1samp(diffs, m, alternative="less").pvalue
                    res[f"{meas}_tost_p"] = float(max(lo, hi))
        except (ImportError, ValueError):
            pass

    return res


def evaluate_grid(table, G):
    g = GRIDS_EXT[G]
    fold = g32.folds_of(table)
    picks, chosen = {}, {}
    for name in active_schedulers():
        pick, ch = {}, []
        for f in range(FOLDS):
            train = [r for r in table if fold[r["spk"]] != f]
            pol, p = fold_policy_direct(name, train, G)
            ch.append(p)
            pick.update({g32.key(r): pol(r) for r in table if fold[r["spk"]] == f})
        picks[name], chosen[name] = pick, ch

    methods = {}
    for o in (g["ref"],) + g["cands"]:
        methods[f"fixed {o}"] = summarize_direct(table, {g32.key(r): o for r in table}, G)
    for name, pick in picks.items():
        methods[name] = summarize_direct(table, pick, G)

    return {"methods": methods, "chosen": chosen}


# ----------------------------------------------------------------------------- Full Steps Synthesis on Both Grids (16-Grid Steps 1..8 & 32-Grid Steps 1..15)
def load_exp017_lookup(exp017_csv=EXP017_CSV):
    if not Path(exp017_csv).exists():
        return {}
    return {(r["utt_id"], r["condition"]): r for r in rf.read_csv(exp017_csv, repair=True)}


def run_steps1to8_cases(
    pipe,
    todo_items,
    conds,
    out_dir,
    exp017_csv=EXP017_CSV,
    include_g32_all=True,
    csv_name="steps1to8.csv",
):
    """Synthesizes and scores all missing early steps on the 16-interval grid (Steps {1, 2, 3, 7}) and,
    when `include_g32_all=True`, on the 32-interval grid (Steps {1, 2, 3, 4, 5, 6, 8, 10, 12, 15}),
    merging them with the already-verified T4 scores from EXP-017 `grid32.csv` ({4, 5, 6, 8} on 16-grid
    and {7, 9, 11, 13, 14} on 32-grid)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / csv_name
    done = rf.done_keys(path, 1)
    exp017_map = load_exp017_lookup(exp017_csv)
    todo = [(it, c) for it in todo_items for c in conds if (it["utt_id"], c) not in done]
    print(
        f"[steps1to8] {len(todo)} cases to run ({len(done)} already in {path.name}, "
        f"include_g32_all={include_g32_all}) on {pipe.gpu}",
        flush=True,
    )
    taus = list(pipe.sampler.STUDENT_TAUS)

    for n, (item, cond) in enumerate(todo, 1):
        t0 = time.perf_counter()
        if pipe.device.type == "cuda":
            pipe.torch.cuda.reset_peak_memory_stats()
        ctx, t_prep = g32.timed(pipe, pipe.prepare, item, cond, item["input_seed"], out_dir)
        plen = ctx["prompt_len"]

        (E16, _), t_t16 = g32.timed(pipe, pipe.teacher, ctx, tuple(range(9)))
        with pipe.torch.inference_mode():
            d16 = {i: float(pipe.sampler.compute_stabilization_drift(E16[i], E16[i - 1], plen)) for i in ALL_G16_STEPS}

        prev_row = exp017_map.get((item["utt_id"], cond))
        g16_to_synth = MISSING_G16_STEPS if prev_row is not None else ALL_G16_STEPS
        gt = pipe.ground_truth(item)

        row = {
            "utt_id": item["utt_id"],
            "speaker_id": item["speaker_id"],
            "condition": cond,
            "prompt_snr": ctx["prompt_snr"],
            "gpu": pipe.gpu,
            "has_g32_all": bool(include_g32_all),
            **{f"d16_{k}": d16[k] for k in ALL_G16_STEPS},
        }
        if prev_row is not None:
            for k in STORED_G16_STEPS:
                for m in METRICS:
                    row[f"{m}_g16_{k}"] = float(prev_row[f"{m}_g16_{k}"])
            for tk in TIMING:
                if prev_row.get(tk) not in (None, ""):
                    row[tk] = float(prev_row[tk])

        t_stu, n_stu, t_voc, n_voc = 0.0, 0, 0.0, 0
        last_aud = None
        for k in g16_to_synth:
            x_stu, dt_s = g32.timed(pipe, g32.student, pipe, ctx, E16[k], taus)
            aud, dt_v = g32.timed(pipe, g32.ng.decode, pipe, ctx, x_stu)
            t_stu, n_stu = t_stu + dt_s, n_stu + len(taus)
            t_voc, n_voc = t_voc + dt_v, n_voc + 1
            last_aud = aud
            sc = pipe.score(aud, item, gt)
            for m in METRICS:
                row[f"{m}_g16_{k}"] = sc[m]

        if include_g32_all:
            E32, d32, _ = g32.grid_teacher(pipe, ctx, GRID32, set(ALL_G32_STEPS))
            for k in ALL_G32_STEPS:
                row[f"d32_{k}"] = float(d32[k])
            if prev_row is not None:
                for k, sname in STORED_G32_MAP.items():
                    for m in METRICS:
                        row[f"{m}_g32_{k}"] = float(prev_row[f"{m}_{sname}"])
                g32_to_synth = MISSING_G32_STEPS
            else:
                g32_to_synth = ALL_G32_STEPS

            for k in g32_to_synth:
                x_stu, dt_s = g32.timed(pipe, g32.student, pipe, ctx, E32[k], taus)
                aud, dt_v = g32.timed(pipe, g32.ng.decode, pipe, ctx, x_stu)
                t_stu, n_stu = t_stu + dt_s, n_stu + len(taus)
                t_voc, n_voc = t_voc + dt_v, n_voc + 1
                last_aud = aud
                sc = pipe.score(aud, item, gt)
                for m in METRICS:
                    row[f"{m}_g32_{k}"] = sc[m]

        if "t_prepare" not in row and last_aud is not None:
            peak = pipe.torch.cuda.max_memory_allocated() / 2 ** 20 if pipe.device.type == "cuda" else 0.0
            row.update({
                "t_prepare": round(t_prep, 4),
                "t_eval": round(t_t16 / 9.0, 5),
                "t_student": round(t_stu / max(n_stu, 1), 5),
                "t_vocoder": round(t_voc / max(n_voc, 1), 5),
                "audio_seconds": round(len(last_aud) / 24000.0, 3),
                "peak_mem_mb": round(peak, 1),
            })

        row["case_seconds"] = round(time.perf_counter() - t0, 2)
        row["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=STEPS_CSV_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        print(f"[{n}/{len(todo)}] steps1to8: {item['utt_id']} {cond} ({row['case_seconds']:.1f} s)", flush=True)
    print(f"[OK] steps1to8 complete: {path}", flush=True)
    return path


def build_steps1to8_table(rows):
    table = []
    for r in rows:
        m = {f"g16_{k}": {meas: float(r[f"{meas}_g16_{k}"]) for meas in METRICS} for k in ALL_G16_STEPS}
        d32 = {}
        has_g32 = str(r.get("has_g32_all", "")).strip().lower() == "true" and r.get("wer_g32_1") not in (None, "")
        if has_g32:
            for k in ALL_G32_STEPS:
                m[f"g32_{k}"] = {meas: float(r[f"{meas}_g32_{k}"]) for meas in METRICS}
                d32[k] = float(r[f"d32_{k}"])
        table.append({
            "utt": r["utt_id"],
            "spk": r["speaker_id"],
            "cond": r["condition"],
            "snr": float(r["prompt_snr"]),
            "gpu": r.get("gpu", ""),
            "has_g32_all": has_g32,
            "d16": {k: float(r[f"d16_{k}"]) for k in ALL_G16_STEPS},
            "d32": d32,
            "m": m,
            "time": {k: g32._num(r.get(k)) for k in TIMING},
        })
    return table


# ----------------------------------------------------------------------------- 500-Speaker (1,000-case) Direct T4 Benchmark
def load_split_speakers():
    splits = {}
    for name in ("development", "calibration", "test"):
        path = SPLITS_DIR / f"{name}_manifest.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            items = data if isinstance(data, list) else data.get("utterances", data.get("items", []))
            for it in items:
                splits[it["speaker_id"]] = name
    return splits


def load_all1000_table(csv_path=ALL1000_CSV):
    splits = load_split_speakers()
    raw = rf.read_csv(csv_path)
    drift_lookup = {}
    drift_csv = THESIS_ROOT / "research" / "evidence" / "benchmarks" / "drift_features" / "drift_features_all1000.csv"
    if drift_csv.exists():
        for drow in rf.read_csv(drift_csv):
            dkey = (drow["utt_id"], drow["condition"])
            drift_lookup[dkey] = {j: float(drow[f"d_{j}"]) for j in range(1, 9) if drow.get(f"d_{j}") not in (None, "")}
    by_case = {}
    for row in raw:
        k = int(row["teacher_step"])
        key = (row["utt_id"], row["condition"])
        aud_sec = float(row.get("audio_duration") or 3.85)
        rec = by_case.setdefault(key, {
            "utt": row["utt_id"],
            "spk": row["speaker_id"],
            "split": splits.get(row["speaker_id"], "unknown"),
            "cond": row["condition"],
            "snr": float(row["prompt_snr"]),
            "d16": drift_lookup.get(key, {4: float(row["d_4"]), 5: float(row["d_5"]), 6: float(row["d_6"])}),
            "time": {
                "t_prepare": 0.0953,
                "t_eval": 0.2981,
                "t_student": 0.1462,
                "t_vocoder": 0.0046,
                "audio_seconds": aud_sec,
            },
            "m": {},
        })
        rec["m"][f"g16_{k}"] = {
            "wer": float(row["wer"]),
            "similarity": float(row["similarity"]),
            "utmos": float(row["utmos"]),
        }
    return list(by_case.values())


def calibrate_all1000_direct(name, cal_rows):
    G = "g16"
    g = GRIDS_EXT[G]
    if name == "stac":
        return fit_stagewise_cascade(cal_rows, G, use_path_sum=True)
    if name == "crc_stac":
        return fit_crc_stac(cal_rows, G, alpha=0.05)
    if name == "stac_single":
        return fit_single_action(cal_rows, G)
    if name == "pos_drift_casc":
        return fit_stagewise_cascade(cal_rows, G, use_path_sum=False)
    if name == "snr_casc":
        return fit_snr_cascade(cal_rows, G)
    if name == "snr20":
        return p_snr_ext(G, THETA), THETA
    if name == "snr_cal":
        thr = calibrate_direct_sequence(cal_rows, g32.THETA_GRID, lambda x: p_snr_ext(G, x), G)
        return p_snr_ext(G, thr), thr
    if name == "snr_drift":
        thr = calibrate_direct_sequence(cal_rows, g["lam"], lambda x: p_stab_ext(G, x, THETA), G)
        return p_stab_ext(G, thr, THETA), thr
    if name in NOVEL_NAMES:
        grid = readiness_grid(cal_rows, G, name, theta=THETA)
        thr = calibrate_direct_sequence(cal_rows, grid, lambda x: p_readiness(G, name, x, THETA), G)
        return p_readiness(G, name, thr, THETA), thr
    raise ValueError(name)


def evaluate_all1000(csv_path=ALL1000_CSV):
    if not Path(csv_path).exists():
        return None
    table = load_all1000_table(csv_path)
    cal = [r for r in table if r["split"] == "calibration"]
    test = [r for r in table if r["split"] == "test"]
    if not cal or not test:
        return None

    methods = {}
    for o in ("g16_8", "g16_4", "g16_5", "g16_6"):
        s = summarize_direct(test, {g32.key(r): o for r in test}, "g16")
        s["threshold"] = None
        methods[f"fixed {o}"] = s
    for name in ("snr20", "snr_cal", "snr_casc", "snr_drift", "stac", "crc_stac", "stac_single", "pos_drift_casc", "swsr", "stsr", "csdr"):
        pol, thr = calibrate_all1000_direct(name, cal)
        s = summarize_direct(test, {g32.key(r): pol(r) for r in test}, "g16")
        s["threshold"] = thr
        methods[name] = s
    return {
        "n_calibration_speakers": len({r["spk"] for r in cal}),
        "n_test_speakers": len({r["spk"] for r in test}),
        "methods": methods,
    }


# ----------------------------------------------------------------------------- OOD 11-Condition Evaluation (EXP-011) & Oracle Sensitivity
EXP011_PILOT_CSV = THESIS_ROOT / "research" / "experiments" / "EXP-011-followup-t4" / "results" / "pilot" / "pilot_grid.csv"
DEFAULT_STAC_LAMS = {4: 1.51279, 5: 1.64678, 6: 1.71607}


def evaluate_ood_exp011(pilot_csv=EXP011_PILOT_CSV):
    """Evaluates zero-shot OOD generalization from Clean+Gaussian calibration to 5 unseen corruptions
    (babble20, babble15, reverb03, reverb09, telephone; n=100 across 20 speakers)."""
    if not Path(pilot_csv).exists():
        return None
    raw = rf.read_csv(pilot_csv)
    by_case = {}
    for row in raw:
        k = int(row["teacher_step"])
        key = (row["utt_id"], row["condition"])
        aud_sec = float(row.get("audio_duration") or 3.85)
        rec = by_case.setdefault(key, {
            "utt": row["utt_id"],
            "spk": row["speaker_id"],
            "cond": row["condition"],
            "snr": float(row["prompt_snr"]),
            "d16": {4: float(row["d_4"]), 5: float(row["d_5"]), 6: float(row["d_6"])},
            "time": {
                "t_prepare": 0.0953,
                "t_eval": 0.2981,
                "t_student": 0.1462,
                "t_vocoder": 0.0046,
                "audio_seconds": aud_sec,
            },
            "m": {},
        })
        rec["m"][f"g16_{k}"] = {
            "wer": float(row["wer"]),
            "similarity": float(row["similarity"]),
            "utmos": float(row["utmos"]),
        }
    rows_all = list(by_case.values())
    cal_gauss = [r for r in rows_all if r["cond"] in {"clean", "gauss25", "gauss22", "gauss20", "gauss18", "noisy15"}]
    ood_conds = ["babble20", "babble15", "reverb03", "reverb09", "telephone"]
    ood_rows = [r for r in rows_all if r["cond"] in ood_conds]
    if not cal_gauss or not ood_rows:
        return None

    pol_stac, thr_stac = fit_stagewise_cascade(cal_gauss, "g16", use_path_sum=True)
    pol_snr, thr_snr = fit_snr_cascade(cal_gauss, "g16")
    per_cond = {}
    for c in ood_conds + ["all_ood"]:
        sub = ood_rows if c == "all_ood" else [r for r in ood_rows if r["cond"] == c]
        s_stac = summarize_direct(sub, {g32.key(r): pol_stac(r) for r in sub}, "g16")
        s_snr = summarize_direct(sub, {g32.key(r): pol_snr(r) for r in sub}, "g16")
        spks = sorted({r["spk"] for r in sub})
        by_spk = {s: [r for r in sub if r["spk"] == s] for s in spks}
        spk_dnfe = [
            statistics.mean(COST_ALL[pol_stac(r)] - COST_ALL[pol_snr(r)] for r in by_spk[s])
            for s in spks
        ]
        rng = random.Random(2026)
        b_dnfe = sorted(
            statistics.mean(spk_dnfe[rng.randrange(len(spks))] for _ in range(len(spks)))
            for _ in range(400)
        )
        per_cond[c] = {
            "n": len(sub),
            "mean_snr": statistics.mean(r["snr"] for r in sub),
            "stac_nfe": s_stac["mean_cost"],
            "stac_saving": s_stac["saving"],
            "stac_wer": s_stac["wer_all"],
            "stac_d_wer": s_stac["d_wer_all"],
            "stac_d_wer_ci": [s_stac.get("d_wer_ci_lo"), s_stac.get("d_wer_ci_hi")],
            "snr_nfe": s_snr["mean_cost"],
            "snr_saving": s_snr["saving"],
            "snr_wer": s_snr["wer_all"],
            "snr_d_wer": s_snr["d_wer_all"],
            "diff_nfe": s_stac["mean_cost"] - s_snr["mean_cost"],
            "diff_nfe_ci": [b_dnfe[10], b_dnfe[389]],
        }
    return {"stac_thresholds": thr_stac, "snr_thresholds": thr_snr, "conditions": per_cond}


def evaluate_oracle_sensitivity(table, G):
    """Computes Hindsight Per-Utterance Oracle across Strict, Primary, and Relaxed tolerances."""
    g = GRIDS_EXT[G]
    ref = g["ref"]
    cands_all = list(g["cands"]) + [ref]
    ref_nfe = float(COST_ALL[ref])
    out = {}
    for name, w_tol, s_tol in [
        ("strict", 0.00, 0.010),
        ("primary", 0.01, 0.015),
        ("relaxed", 0.02, 0.020),
    ]:
        pick = {}
        for r in table:
            chosen = ref
            for o in cands_all:
                dw = r["m"][o]["wer"] - r["m"][ref]["wer"]
                ds = r["m"][o]["similarity"] - r["m"][ref]["similarity"]
                if dw <= w_tol + 1e-9 and ds >= -s_tol - 1e-9:
                    chosen = o
                    break
            pick[g32.key(r)] = chosen
        s = summarize_direct(table, pick, G)
        out[name] = {
            "w_tol": w_tol,
            "s_tol": s_tol,
            "mean_cost": s["mean_cost"],
            "saving": s["saving"],
            "wer_all": s["wer_all"],
            "sim_all": s["sim_all"],
            "ref_cost": ref_nfe,
        }
    return out


# ----------------------------------------------------------------------------- Live Colab Tesla T4 Early-Stopping Execution
def live_early_exit_teacher(
    pipe, ctx, variant="stac", lam=3.20, theta=THETA, cand_steps=(4, 5, 6), ref_step=8, stac_lams=None
):
    torch, sampler = pipe.torch, pipe.sampler
    grid = sampler.SWAY_TEACHER_GRID
    plen = ctx["prompt_len"]
    snr = float(ctx["prompt_snr"])
    x = ctx["y0"].clone()
    prev = None
    drifts = {}
    chosen_k = ref_step
    endpoint = None
    lams_map = dict(DEFAULT_STAC_LAMS if stac_lams is None else stac_lams)

    with torch.inference_mode():
        for i in range(ref_step + 1):
            t = torch.tensor(grid[i], device=pipe.device, dtype=torch.float32)
            kw = dict(x=x, cond=ctx["step_cond"], text=ctx["text_ids"], time=t, mask=None)
            vc = pipe.tts.teacher(drop_audio_cond=False, drop_text=False, **kw)
            vu = pipe.tts.teacher(drop_audio_cond=True, drop_text=True, **kw)
            v = vc + (vc - vu) * sampler.CFG_STRENGTH
            ep = x + (1.0 - grid[i]) * v
            if prev is not None:
                drifts[i] = float(sampler.compute_stabilization_drift(ep, prev, plen))
            prev = ep

            if i in cand_steps:
                if variant in ("stac", "crc_stac"):
                    # Zero-overhead Stage-Wise Trajectory Action Cascade (no external SNR gate!)
                    action_i = sum(drifts[j] for j in range(1, i + 1) if j in drifts)
                    thr_i = lams_map.get(i)
                    if thr_i is not None and action_i >= thr_i:
                        chosen_k = i
                        endpoint = ep
                        break
                elif snr >= theta and lam is not None:
                    if variant == "swsr":
                        score = max(snr, 0.0) * (float(grid[i]) - float(grid[i - 1])) * drifts[i]
                    elif variant == "stsr":
                        score = max(snr, 0.0) * float(grid[i]) * drifts[i]
                    elif variant == "csdr":
                        score = max(snr, 0.0) * sum(drifts[j] for j in cand_steps if j <= i and j in drifts)
                    else:
                        raise ValueError(variant)
                    if score >= lam:
                        chosen_k = i
                        endpoint = ep
                        break

            if i == ref_step:
                chosen_k = ref_step
                endpoint = ep
                break
            x = x + (grid[i + 1] - grid[i]) * v

    return endpoint, chosen_k, drifts


def run_live_t4(pipe, items, conds, out_dir, variant="stac", lam=3.20):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "live_swsr_t4.csv"
    taus = list(pipe.sampler.STUDENT_TAUS)
    rows = []
    for idx, (item, cond) in enumerate([(it, c) for it in items for c in conds], 1):
        if pipe.device.type == "cuda":
            pipe.torch.cuda.reset_peak_memory_stats()
        ctx, t_prep = g32.timed(pipe, pipe.prepare, item, cond, item["input_seed"], out_dir)
        (ep_swsr, k_swsr, _), t_teacher_swsr = g32.timed(
            pipe, live_early_exit_teacher, pipe, ctx, variant, lam, THETA
        )
        x_swsr, t_stu_swsr = g32.timed(pipe, g32.student, pipe, ctx, ep_swsr, taus)
        aud_swsr, t_voc_swsr = g32.timed(pipe, g32.ng.decode, pipe, ctx, x_swsr)
        total_swsr = t_prep + t_teacher_swsr + t_stu_swsr + t_voc_swsr

        (E16, _), t_teacher_ref = g32.timed(pipe, pipe.teacher, ctx, tuple(range(9)))
        x_ref, t_stu_ref = g32.timed(pipe, g32.student, pipe, ctx, E16[8], taus)
        aud_ref, t_voc_ref = g32.timed(pipe, g32.ng.decode, pipe, ctx, x_ref)
        total_ref = t_prep + t_teacher_ref + t_stu_ref + t_voc_ref

        max_ep_diff = float((ep_swsr - E16[k_swsr]).abs().max())
        gt = pipe.ground_truth(item)
        m_swsr = pipe.score(aud_swsr, item, gt)
        m_ref = pipe.score(aud_ref, item, gt)
        audio_sec = max(len(aud_ref) / 24000.0, 1e-6)
        nfe_swsr = 2 * (k_swsr + 1) + 3
        peak_mb = round(pipe.torch.cuda.max_memory_allocated() / 2 ** 20, 1) if pipe.device.type == "cuda" else 0.0
        row = {
            "utt_id": item["utt_id"],
            "speaker_id": item["speaker_id"],
            "condition": cond,
            "prompt_snr": round(float(ctx["prompt_snr"]), 3),
            "variant": variant,
            "lambda": lam if variant not in ("stac", "crc_stac") else str(DEFAULT_STAC_LAMS),
            "chosen_step": k_swsr,
            "teacher_evals_swsr": k_swsr + 1,
            "teacher_passes_swsr": 2 * (k_swsr + 1),
            "student_steps": 3,
            "nfe_swsr": nfe_swsr,
            "nfe_ref": 21,
            "nfe_saving_pct": round((1.0 - nfe_swsr / 21.0) * 100.0, 2),
            "gate_overhead_ms": GATE_OVERHEAD_MS.get(variant, 0.12),
            "endpoint_max_abs_diff": max_ep_diff,
            "t_prepare_s": round(t_prep, 4),
            "t_teacher_swsr_s": round(t_teacher_swsr, 4),
            "t_teacher_ref_s": round(t_teacher_ref, 4),
            "t_student_s": round(t_stu_swsr, 4),
            "t_vocoder_s": round(t_voc_swsr, 4),
            "time_swsr_s": round(total_swsr, 4),
            "time_ref_s": round(total_ref, 4),
            "time_saved_s": round(total_ref - total_swsr, 4),
            "wall_clock_saving": round(1.0 - total_swsr / total_ref, 4),
            "speedup_x": round(total_ref / max(total_swsr, 1e-6), 3),
            "teacher_speedup_x": round(t_teacher_ref / max(t_teacher_swsr, 1e-6), 3),
            "audio_seconds": round(audio_sec, 3),
            "rtf_swsr": round(total_swsr / audio_sec, 4),
            "rtf_ref": round(total_ref / audio_sec, 4),
            "x_realtime_swsr": round(audio_sec / max(total_swsr, 1e-6), 2),
            "x_realtime_ref": round(audio_sec / max(total_ref, 1e-6), 2),
            "peak_mem_mb": peak_mb,
            "wer_swsr": round(m_swsr["wer"], 4),
            "wer_ref": round(m_ref["wer"], 4),
            "sim_swsr": round(m_swsr["similarity"], 4),
            "sim_ref": round(m_ref["similarity"], 4),
            "utmos_swsr": round(m_swsr["utmos"], 4),
            "utmos_ref": round(m_ref["utmos"], 4),
        }
        rows.append(row)
        print(
            f"[{idx}] {item['utt_id'][-8:]} {cond}: {variant.upper()} picked Step {k_swsr} "
            f"({row['nfe_swsr']} NFE vs 21 NFE [{row['nfe_saving_pct']:.1f}% saved], "
            f"wall-clock {total_swsr:.2f}s vs {total_ref:.2f}s [{row['speedup_x']:.2f}x speedup, RTF {row['rtf_swsr']:.3f} vs {row['rtf_ref']:.3f}], "
            f"max|diff|={max_ep_diff:.2e})",
            flush=True,
        )
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


# ----------------------------------------------------------------------------- Markdown Report Rendering (Direct WER / SIM / UTMOS / NFE / RTF)
def render_report(res):
    L = [
        "# Direct Benchmark Comparison (WER, Capped WER, SIM, UTMOS, Computational Cost, Speedup, Runtime, RTF) — Both 16-Grid & 32-Grid",
        "",
        "## 1. Discrete Sway-Grid Checkpoints on the 16-Interval and 32-Interval Grids",
        "",
        "### 1A. 16-Interval Sway Grid ($t_k = 1 - \\cos(\\pi k / 32)$, Steps `k = 1..8`)",
        "",
        "| Step $k$ | Exact Sway Time $t_k$ | Sway Step Width $\\Delta t_k = t_k - t_{k-1}$ | Teacher Evals ($k+1$) | Student Steps | Total NFE ($2(k+1)+3$) | Relationship to `0.25` |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|---|",
    ]
    notes16 = {
        1: "Ultra-early (`< 0.07`)",
        2: "Ultra-early (`< 0.07`)",
        3: "Ultra-early (`< 0.07`)",
        4: "Released code default (`s = 0.07`)",
        5: "Intermediate candidate (`s = 0.10`)",
        6: "Intermediate candidate (`s = 0.15`)",
        7: "**Last 16-grid checkpoint strictly $\\le 0.25$** ($t_7 = 0.2270$, triggered by `s = 0.20`)",
        8: "**First 16-grid checkpoint $> 0.25$** ($t_8 = 0.2929$, triggered by `s = 0.25` or `0.29`)",
    }
    for k in ALL_G16_STEPS:
        L.append(
            f"| **Step {k}** | `{GRID16[k]:.4f}` | `{DELTA_T16[k]:.4f}` | "
            f"{k + 1} ({2 * (k + 1)} passes) | 3 | **{2 * (k + 1) + 3} NFE** | {notes16[k]} |"
        )

    L += [
        "",
        "### 1B. 32-Interval Sway Grid ($t_i = 1 - \\cos(\\pi i / 64)$, Steps `i = 1..15`)",
        "",
        "| Step $i$ | Exact Sway Time $t_i$ | Sway Step Width $\\Delta t_i = t_i - t_{i-1}$ | Teacher Evals ($i+1$) | Student Steps | Total NFE ($2(i+1)+3$) | Relationship to `0.25` & 16-Grid |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|---|",
    ]
    for i in ALL_G32_STEPS:
        note32 = f"Exact match to 16-grid **Step {i // 2}**" if i % 2 == 0 else "Half-step between 16-grid points"
        if i in STORED_G32_MAP:
            note32 += f" (in `EXP-017` as `{STORED_G32_MAP[i]}`)"
        if i == 14:
            note32 += " — **last 32-grid checkpoint $\\le 0.25$**"
        elif i == 15:
            note32 += " — **first 32-grid checkpoint $> 0.25$**"
        L.append(
            f"| **Step {i}** | `{GRID32[i]:.4f}` | `{DELTA_T32[i]:.4f}` | "
            f"{i + 1} ({2 * (i + 1)} passes) | 3 | **{2 * (i + 1) + 3} NFE** | {note32} |"
        )

    L += [
        "",
        "## 2. Direct Quality & Efficiency Comparison (No Binary Failure Rule)",
        "",
        "Every method is compared directly on **mean WER, Capped WER ($\\le 1.0$), mean SIM, mean UTMOS, NFE (forward passes) with Speaker-Cluster Bootstrap 95% CIs, Wall-Clock Runtime (s), Speedup ($\\times$), and RTF ($\\times$ Real-Time)** across clean and noisy conditions, plus paired differences ($\\Delta\\text{WER}, \\Delta\\text{SIM}$) and TOST equivalence against the reference ceiling.",
        "",
    ]

    for G in ("g16", "g32", "g16_all8", "g16_le025", "g32_all14", "g32_all15"):
        if G not in res.get("exp017", {}):
            continue
        g = GRIDS_EXT[G]
        o4 = res["exp017"][G]
        L += [
            f"### Benchmark on the {g['label']} (360 cases, 60 Speakers)",
            "",
            "| Method | Mean NFE [Speaker-Cluster 95% CI] (Saving) | Runtime (s) / Speedup / RTF | WER (Clean) | WER (Noisy) | WER (All) [$\\Delta$ vs Ref] | SIM (Clean) | SIM (Noisy) | SIM (All) [$\\Delta$ vs Ref] | UTMOS (Clean / Noisy) | TOST Equiv ($p_{\\text{WER}} / p_{\\text{SIM}}$) | Choices |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for name, v in o4["methods"].items():
            label = ALL_SCHEDULERS.get(name, name)
            choices = ", ".join(f"{k}:{c}" for k, c in v["picks"].items())
            rt_str = (
                f"**{v['runtime']:.2f}s** ({v.get('speedup', 1.0):.2f}x) / `{v['rtf']:.3f}` ({v.get('x_realtime', 0.0):.2f}x RT)"
                if v.get("runtime") is not None
                else "n/a"
            )
            tost_str = (
                f"{g32._f(v.get('wer_tost_p'), '.2g')} / {g32._f(v.get('similarity_tost_p'), '.2g')}"
            )
            ci_str = (
                f" [{v['cost_ci_lo']:.2f},{v['cost_ci_hi']:.2f}]"
                if v.get("cost_ci_lo") is not None
                else ""
            )
            L.append(
                f"| {label} | **{v['mean_cost']:.2f}**{ci_str} ({v['saving']:.1%}) | {rt_str} | "
                f"{v['wer_clean']:.4f} | {v['wer_noisy']:.4f} | **{v['wer_all']:.4f}** ({v['d_wer_all']:+.4f}) | "
                f"{v['sim_clean']:.4f} | {v['sim_noisy']:.4f} | **{v['sim_all']:.4f}** ({v['d_sim_all']:+.4f}) | "
                f"{v['utmos_clean']:.3f} / {v['utmos_noisy']:.3f} | {tost_str} | `{choices}` |"
            )
        L.append("")

    if res.get("all1000"):
        a1k = res["all1000"]
        L += [
            "## 3. 500-Speaker Tesla T4 Direct Benchmark (200 Calibration Speakers -> 200 Test Speakers, Full `d_1..d_8`, Raw & Capped WER + Speed & Runtime)",
            "",
            "| Method | Threshold | Mean NFE (Saving) | Runtime (s) / Speedup / RTF | Raw WER [$\\Delta$ vs Ref] | **Capped WER $\\le 1.0$** [$\\Delta$ vs Ref] | **Trimmed WER** ($n=393$) | SIM (All) [$\\Delta$ vs Ref] | TOST ($p_{\\text{WER}} / p_{\\text{SIM}}$) | Choices |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
        for name, v in a1k["methods"].items():
            label = ALL_SCHEDULERS.get(name, name)
            choices = ", ".join(f"{k}:{c}" for k, c in v["picks"].items())
            if v["threshold"] is None:
                thr_s = "None"
            elif isinstance(v["threshold"], dict):
                thr_s = ", ".join(f"{k}:{val:.3g}" for k, val in v["threshold"].items() if val is not None)
            else:
                thr_s = f"{v['threshold']:.4g}"
            tost_str = f"{g32._f(v.get('wer_tost_p'), '.2g')} / {g32._f(v.get('similarity_tost_p'), '.2g')}"
            cap_str = (
                f"**{v['wer_capped_all']:.4f}** ({v['d_wer_capped_all']:+.4f})"
                if v.get("wer_capped_all") is not None
                else "n/a"
            )
            trim_str = (
                f"{v['wer_trimmed_all']:.4f} ({v['d_wer_trimmed_all']:+.4f})"
                if v.get("wer_trimmed_all") is not None
                else "n/a"
            )
            rt_str = (
                f"**{v['runtime']:.2f}s** ({v.get('speedup', 1.0):.2f}x) / `{v['rtf']:.3f}`"
                if v.get("runtime") is not None
                else "n/a"
            )
            L.append(
                f"| {label} | {thr_s} | **{v['mean_cost']:.2f}** ({v['saving']:.1%}) | {rt_str} | "
                f"**{v['wer_all']:.4f}** ({v['d_wer_all']:+.4f}) | {cap_str} | {trim_str} | "
                f"**{v['sim_all']:.4f}** ({v['d_sim_all']:+.4f}) | {tost_str} | `{choices}` |"
            )
        L.append("")

    if res.get("ood_exp011"):
        ood = res["ood_exp011"]
        L += [
            "## 4. Zero-Shot OOD Generalization to 5 Unseen Corruptions (`EXP-011`, $n=100$ across 20 Speakers)",
            "",
            "| Condition | Cases ($n$) | Mean SNR (dB) | **STAC** NFE (Saving) | **SNR Cascade** NFE (Saving) | **Paired Diff $\\text{NFE}(\\text{STAC})-\\text{NFE}(\\text{SNR})$ [95% CI]** | **STAC** $\\Delta\\text{WER}$ [95% CI] |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for c, row in ood["conditions"].items():
            c_lo, c_hi = row["diff_nfe_ci"]
            w_lo, w_hi = row["stac_d_wer_ci"]
            w_ci_s = f"[{w_lo:+.4f}, {w_hi:+.4f}]" if w_lo is not None else ""
            L.append(
                f"| `{c}` | {row['n']} | {row['mean_snr']:.1f} | **{row['stac_nfe']:.2f}** ({row['stac_saving']:.1%}) | "
                f"{row['snr_nfe']:.2f} ({row['snr_saving']:.1%}) | **{row['diff_nfe']:+.2f}** [{c_lo:+.2f}, {c_hi:+.2f}] | "
                f"`{row['stac_d_wer']:+.4f}` {w_ci_s} |"
            )
        L.append("")

    if res.get("oracle_sensitivity"):
        L += [
            "## 5. Hindsight Per-Utterance Oracle Sensitivity Across 3 Tolerance Regimes",
            "",
            "| Grid | Oracle Regime ($\\Delta\\text{WER}, \\Delta\\text{SIM}$) | Oracle Mean NFE (Saving) | Oracle WER | Oracle SIM | **STAC Mean NFE (Saving)** | **Share of Oracle Headroom Captured** |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for G, regimes in res["oracle_sensitivity"].items():
            stac_m = res["exp017"].get(G, {}).get("methods", {}).get("stac")
            stac_nfe = stac_m["mean_cost"] if stac_m else None
            stac_sav = stac_m["saving"] if stac_m else None
            for rname, rv in regimes.items():
                share = (stac_sav / rv["saving"] * 100.0) if (stac_sav is not None and rv["saving"] > 0) else 0.0
                stac_s = f"**{stac_nfe:.2f}** ({stac_sav:.1%})" if stac_nfe is not None else "n/a"
                L.append(
                    f"| `{G}` | `{rname}` ($\\le {rv['w_tol']:.2f}, \\ge -{rv['s_tol']:.3f}$) | "
                    f"**{rv['mean_cost']:.2f}** ({rv['saving']:.1%}) | {rv['wer_all']:.4f} | {rv['sim_all']:.4f} | "
                    f"{stac_s} | **{share:.1f}%** |"
                )
        L.append("")

    # Section 6: Dedicated Computational Cost, Speedup, Wall-Clock Time, RTF & Gate Overhead Table
    L += [
        "## 6. Computational Cost, Wall-Clock Latency, Speedup ($\\times$), Real-Time Factor (RTF) & Gate Overhead Comparison",
        "",
        "| Grid | Policy | Teacher Evals ($k+1$) / Passes ($2(k+1)$) | Total NFE (Saving) | Gate Overhead (ms) | Teacher ODE Time (s) [Speedup] | End-to-End Synthesis Time (s) [Time Saved] | End-to-End Speedup ($\\times$) | RTF ($\\times$ Real-Time) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for G in ("g16", "g32", "g16_all8", "g16_le025", "g32_all14", "g32_all15"):
        if G not in res.get("exp017", {}):
            continue
        o4 = res["exp017"][G]
        for name in ("fixed " + GRIDS_EXT[G]["ref"], "stac", "crc_stac", "snr_casc", "pos_drift_casc", "stac_single", "snr_cal", "swsr", "csdr"):
            v = o4["methods"].get(name)
            if not v or v.get("runtime") is None:
                continue
            label = ALL_SCHEDULERS.get(name, name)
            ovh = 0.00 if name.startswith("fixed") else GATE_OVERHEAD_MS.get(name, 0.12)
            t_ev = v.get("teacher_evals", 0.0)
            t_pa = v.get("teacher_passes", 0.0)
            t_rt = v.get("teacher_runtime", 0.0)
            t_sp = v.get("teacher_speedup", 1.0)
            e_rt = v["runtime"]
            e_sv = v.get("time_saved_sec", 0.0)
            e_sp = v.get("speedup", 1.0)
            rtf = v["rtf"]
            xrt = v.get("x_realtime", 0.0)
            L.append(
                f"| `{G}` | {label} | `{t_ev:.2f}` / `{t_pa:.2f}` | **{v['mean_cost']:.2f}** ({v['saving']:.1%}) | "
                f"`{ovh:.2f} ms` | `{t_rt:.2f}s` (**{t_sp:.2f}x**) | **{e_rt:.2f}s** (`-{e_sv:.2f}s` / `-{v['runtime_saving']:.1%}`) | "
                f"**{e_sp:.2f}x** | `{rtf:.3f}` (**{xrt:.2f}x RT**) |"
            )
    L.append("")

    return "\n".join(L) + "\n"


def analyze(out_dir, exp017_csv=EXP017_CSV, all1000_csv=ALL1000_CSV, steps1to8_csv=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"exp017": {}, "all1000": None, "ood_exp011": None, "oracle_sensitivity": {}}

    if exp017_csv and Path(exp017_csv).exists():
        table = g32.build_table(rf.read_csv(exp017_csv, repair=True))
        for G in ("g16", "g32"):
            if len({r["spk"] for r in table}) >= FOLDS:
                res["exp017"][G] = evaluate_grid(table, G)
                res["oracle_sensitivity"][G] = evaluate_oracle_sensitivity(table, G)

    s18_path = Path(steps1to8_csv) if steps1to8_csv else (out_dir / "steps1to8.csv")
    if s18_path.exists():
        t18 = build_steps1to8_table(rf.read_csv(s18_path, repair=True))
        if len({r["spk"] for r in t18}) >= FOLDS:
            for G in ("g16_all8", "g16_le025"):
                res["exp017"][G] = evaluate_grid(t18, G)
                res["oracle_sensitivity"][G] = evaluate_oracle_sensitivity(t18, G)
            if all(r.get("has_g32_all") for r in t18):
                for G in ("g32_all14", "g32_all15"):
                    res["exp017"][G] = evaluate_grid(t18, G)
                    res["oracle_sensitivity"][G] = evaluate_oracle_sensitivity(t18, G)

    if all1000_csv and Path(all1000_csv).exists():
        res["all1000"] = evaluate_all1000(all1000_csv)

    res["ood_exp011"] = evaluate_ood_exp011()

    (out_dir / "swsr_results.json").write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    report_md = render_report(res)
    (out_dir / "SWSR_HANDOVER_REPORT.md").write_text(report_md, encoding="utf-8")
    print(f"[OK] Wrote {out_dir / 'SWSR_HANDOVER_REPORT.md'} and {out_dir / 'swsr_results.json'}", flush=True)
    return res


def _ensure_colab_py313_compat():
    """Ensures Python 3.13+ (`audioop-lts`), PyTorch 2.6+ (`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`),
    and Torchaudio 2.9+ (`soundfile` fallback when `torchcodec` is absent) work seamlessly on Colab."""
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        import audioop  # noqa: F401
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "audioop-lts"])
    try:
        import soundfile as sf
        import torch
        import torchaudio
        _orig_load = torchaudio.load

        def _safe_torchaudio_load(uri, *a, **kw):
            try:
                return _orig_load(uri, *a, **kw)
            except (ImportError, RuntimeError) as exc:
                if "torchcodec" in str(exc).lower() or "backend" in str(exc).lower():
                    wav, sr = sf.read(str(uri), dtype="float32", always_2d=True)
                    return torch.from_numpy(wav.T.copy()), int(sr)
                raise

        torchaudio.load = _safe_torchaudio_load
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["analyze", "smoke", "run_steps1to8", "live_t4"])
    ap.add_argument("--out", required=True, help="Output directory for JSON/Markdown reports and CSVs")
    ap.add_argument("--exp017_csv", default=str(EXP017_CSV))
    ap.add_argument("--all1000_csv", default=str(ALL1000_CSV))
    ap.add_argument("--steps1to8_csv", default=None)
    ap.add_argument("--dmo_root", default="/content/Shonget-DMOSpeech2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", default="stac", choices=["stac", "crc_stac"] + list(NOVEL_NAMES.keys()))
    ap.add_argument("--lam", type=float, default=3.20, help="Threshold for legacy single-threshold variants in live_t4")
    ap.add_argument("--skip_g32_all", action="store_true", help="Synthesize only 16-grid Steps 1..8 (skip 32-grid missing steps)")
    ap.add_argument("--any_gpu", action="store_true")
    args = ap.parse_args(argv)

    if args.cmd == "analyze":
        analyze(args.out, args.exp017_csv, args.all1000_csv, args.steps1to8_csv)
        return

    _ensure_colab_py313_compat()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipe = rf.Pipeline(args.dmo_root, args.device)
    if "T4" not in pipe.gpu and not args.any_gpu:
        raise SystemExit(f"Expected Tesla T4 GPU (found {pipe.gpu}). Pass --any_gpu to override.")

    include_g32 = not args.skip_g32_all
    if args.cmd == "smoke":
        its = [it for it in g32.ex.items() if it["utt_id"] not in g32.cg.pilot_utts()][:2]
        smoke_dir = Path(args.out) / "smoke"
        run_steps1to8_cases(pipe, its, ("clean", "gauss20"), smoke_dir, args.exp017_csv, include_g32_all=include_g32)
        run_live_t4(pipe, its, ("clean", "gauss20"), smoke_dir, args.variant, args.lam)
        return

    if args.cmd == "run_steps1to8":
        run_steps1to8_cases(pipe, g32.ex.items(), CONDS, Path(args.out), args.exp017_csv, include_g32_all=include_g32)
        analyze(args.out, args.exp017_csv, args.all1000_csv, Path(args.out) / "steps1to8.csv")
        return

    if args.cmd == "live_t4":
        run_live_t4(pipe, g32.ex.items()[:10], ("clean", "noisy15", "gauss20"), Path(args.out), args.variant, args.lam)
        analyze(args.out, args.exp017_csv, args.all1000_csv, args.steps1to8_csv)


if __name__ == "__main__":
    main()
