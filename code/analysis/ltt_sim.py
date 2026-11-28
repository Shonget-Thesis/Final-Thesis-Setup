"""Sample simulation for Appendix B: Learn-Then-Test selection of teacher path and residual threshold.

Synthetic data only (no speech is synthesized). Mirrors Algorithm 3.4 and Proposition 3.8 of Chapter 3:
three candidate teacher paths k in K = {2, 4, 6} on the 8-interval grid (costs C(k) = 10, 13, 16 passes;
released baseline fallback = 21 passes), fixed prompt SNR gate theta = 20 dB, ordered residual grid
Lambda = {0.00, 0.05, ..., 1.00}, tolerance alpha = 0.05, and confidence parameter delta = 0.10 shared
equally across the three paths (delta / 3 per path).
"""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import binom

PATHS = (2, 4, 6)
COST = {2: 10.0, 4: 13.0, 6: 16.0, "ref": 21.0}
THETA = 20.0
GRID = np.round(np.arange(0.00, 1.0001, 0.05), 2)
ALPHA, DELTA = 0.05, 0.10
LEVEL_PER_PATH = DELTA / len(PATHS)


def population(N, rng):
    """Synthetic 8-condition mixture where k=2 (10p) is slightly unsafe (min excess risk > 0.05),
     while k=4 (13p) and k=6 (16p) are safe up to interior residual thresholds."""
    cond = rng.integers(0, 8, size=N)
    is_clean = cond == 0
    is_low_snr = np.isin(cond, (1, 7))      # white 15 dB, babble 15 dB
    is_mod25 = np.isin(cond, (2, 3, 4))     # 25 dB white, pink, high-pass (passes SNR gate!)
    # remaining cond in {5, 6}: reverb, telephone

    snr = np.empty(N)
    snr[is_clean] = rng.normal(35.0, 4.5, is_clean.sum())
    snr[is_low_snr] = rng.normal(17.5, 1.4, is_low_snr.sum())
    snr[is_mod25] = rng.normal(26.2, 1.8, is_mod25.sum())
    other = ~(is_clean | is_low_snr | is_mod25)
    snr[other] = rng.normal(28.5, 5.0, other.sum())
    snr = np.clip(snr, 0.0, 50.0)

    # Released 21-pass baseline can itself fail under Definition 3.1 (nonzero F_ref)
    p_ref = np.where(is_clean, 0.05, np.where(is_mod25, 0.55, np.where(is_low_snr, 0.65, 0.70)))
    f_ref = rng.random(N) < p_ref

    r_score, f_teach, f_proj = {}, {}, {}
    for k in PATHS:
        # Residual rises sharply when teacher copies moderate/heavy noise
        mu_r = np.where(is_clean, 0.24, np.where(is_mod25, 0.54, np.where(is_low_snr, 0.58, 0.34)))
        r_k = np.clip(rng.normal(mu_r + 0.01 * (k - 4), 0.085, N), 0.02, 0.98)
        r_score[k] = r_k

        # Projected payload Q: safe on k=4,6, but early commit k=2 suffers extra excess failures (>5%)
        base_q_ex = {2: 0.135, 4: 0.042, 6: 0.026}[k]
        p_q_fail_given_ref_ok = np.where(is_clean, base_q_ex * 0.7, base_q_ex * 1.15)
        # Unprojected teacher payload x_hat_1: fails heavily when r_k is large on gate-open noise
        p_t_fail_given_ref_ok = np.clip(
            {2: 0.145, 4: 0.032, 6: 0.020}[k] + 0.42 * np.maximum(0.0, r_k - 0.30) + 0.28 * is_mod25,
            0.0,
            0.95,
        )
        u_q = rng.random(N)
        u_t = rng.random(N)
        f_proj[k] = np.where(f_ref, rng.random(N) < 0.75, u_q < p_q_fail_given_ref_ok)
        f_teach[k] = np.where(f_ref, rng.random(N) < 0.88, u_t < p_t_fail_given_ref_ok)

    return dict(snr=snr, f_ref=f_ref, r=r_score, f_teach=f_teach, f_proj=f_proj)


def excess_and_routing(pop, k, lam, idx=slice(None)):
    """Excess-failure rate R(k, lam) and projection routing rate rho(k, lam)."""
    snr = pop["snr"][idx]
    r_k = pop["r"][k][idx]
    use_q = (snr < THETA) | (r_k > lam)
    f_out = np.where(use_q, pop["f_proj"][k][idx], pop["f_teach"][k][idx])
    f_ref = pop["f_ref"][idx]
    excess = f_out & (~f_ref)
    return float(excess.mean()), float(use_q.mean())


def ltt_select(pop, idx, alpha=ALPHA, level=LEVEL_PER_PATH, paths=PATHS):
    """Algorithm 3.4: fixed-sequence LTT per path at delta/|K|, then cheapest accepted path."""
    n = len(idx)
    accepted = {}
    for k in paths:
        lam_k = None
        for lam in GRID:
            r_hat, _ = excess_and_routing(pop, k, lam, idx)
            if binom.cdf(round(n * r_hat), n, alpha) <= level:
                lam_k = float(lam)
            else:
                break
        if lam_k is not None:
            accepted[k] = lam_k
    if not accepted:
        return None, None
    k_hat = min(accepted)
    return k_hat, accepted[k_hat]


def naive_select(pop, idx, alpha=ALPHA, paths=PATHS):
    """Uncalibrated rule: cheapest path with empirical excess risk <= alpha, highest such threshold."""
    accepted = {}
    for k in paths:
        ok = [float(lam) for lam in GRID if excess_and_routing(pop, k, lam, idx)[0] <= alpha]
        if ok:
            accepted[k] = max(ok)
    if not accepted:
        return None, None
    k_hat = min(accepted)
    return k_hat, accepted[k_hat]


def fitted_exp034_factory():
    """Bootstrap/parametric generator fitted to the 240 real Tesla T4 rows of EXP-034 (k=4 path)."""
    root = Path(__file__).resolve().parents[2]
    rows = list(csv.DictReader(open(root / "research/experiments/EXP-034-invariant-residual/results/quality.csv", encoding="utf-8")))
    clean = [r for r in rows if r["condition"] == "clean"]
    s_min = float(np.quantile([float(r["similarity_W8"]) for r in clean], 0.05))
    u_min = float(np.quantile([float(r["utmos_W8"]) for r in clean], 0.05))

    def fails(r, arm):
        return float(r[f"wer_{arm}"]) > 0.125 or float(r[f"similarity_{arm}"]) < s_min or float(r[f"utmos_{arm}"]) < u_min

    snr_arr = np.array([float(r["prompt_snr"]) for r in rows])
    r_arr = np.array([float(r["r"]) for r in rows])
    f_ref_arr = np.array([fails(r, "W8") for r in rows], dtype=bool)
    f_t_arr = np.array([fails(r, "CPH8") for r in rows], dtype=bool)
    f_q_arr = np.array([fails(r, "CPQH") for r in rows], dtype=bool)

    def make_fitted(N, rng):
        pick = rng.integers(0, len(rows), size=N)
        jitter = rng.normal(0.0, 0.015, size=N)
        return dict(
            snr=np.clip(snr_arr[pick] + rng.normal(0.0, 0.4, size=N), 0.0, 50.0),
            f_ref=f_ref_arr[pick],
            r={4: np.clip(r_arr[pick] + jitter, 0.0, 1.0)},
            f_teach={4: f_t_arr[pick]},
            f_proj={4: f_q_arr[pick]},
        )

    return make_fitted, dict(n_rows=len(rows), s_min=round(s_min, 4), u_min=round(u_min, 4))


def main():
    rng = np.random.default_rng(2026)
    big = population(400_000, rng)
    true_table = {k: {float(lam): excess_and_routing(big, k, float(lam)) for lam in GRID} for k in PATHS}

    print("True excess risk R(k, lam) and routing rate rho(k, lam) across paths:")
    for k in PATHS:
        safe = [lam for lam in GRID if true_table[k][float(lam)][0] <= ALPHA]
        best = max(safe) if safe else None
        print(f"  Path k={k} (C={COST[k]:.0f}): min_R={min(v[0] for v in true_table[k].values()):.4f}, largest safe lam={best}")
        for lam in (0.00, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.80, 1.00):
            rk, rh = true_table[k][lam]
            print(f"    lam={lam:.2f}: R={rk:.4f}, rho={rh:.3f}")

    print("\nCalibration simulation over 1,000 repetitions per size (alpha=0.05, delta=0.10, delta/3=0.0333):")
    print("   n  method   P(R<=a)  meanR   meanC  fallback%  k2%    k4%    k6%   med_lam(k4)")
    for n in (50, 75, 100, 200, 400):
        for name, fn in (("LTT", ltt_select), ("Naive", naive_select)):
            R_list, C_list, k_list, lam4_list = [], [], [], []
            for _ in range(1000):
                cal = population(n, rng)
                kh, lh = fn(cal, np.arange(n))
                if kh is None:
                    R_list.append(0.0)
                    C_list.append(COST["ref"])
                    k_list.append(0)
                else:
                    rk, _ = true_table[kh][lh]
                    R_list.append(rk)
                    C_list.append(COST[kh])
                    k_list.append(kh)
                    if kh == 4:
                        lam4_list.append(lh)
            R_a = np.array(R_list)
            C_a = np.array(C_list)
            k_a = np.array(k_list)
            med_l4 = float(np.median(lam4_list)) if lam4_list else float("nan")
            print(
                f"{n:4d}  {name:6s}   {(R_a <= ALPHA).mean():.3f}   {R_a.mean():.4f}  {C_a.mean():5.2f}   "
                f"{100*(k_a==0).mean():5.1f}%   {100*(k_a==2).mean():4.1f}%  {100*(k_a==4).mean():4.1f}%  {100*(k_a==6).mean():4.1f}%   {med_l4:.2f}"
            )


if __name__ == "__main__":
    import sys
    if "--fitted" in sys.argv:
        rng = np.random.default_rng(2026)
        make_f, meta = fitted_exp034_factory()
        print("EXP-034 k=4 fitted metadata:", meta)
        big_f = make_f(400_000, rng)
        true_f = {float(lam): excess_and_routing(big_f, 4, float(lam)) for lam in GRID}
        safe_f = [float(lam) for lam in GRID if true_f[float(lam)][0] <= ALPHA]
        print("k=4 true risk range:", round(true_f[0.0][0], 4), "to", round(true_f[1.0][0], 4), "| largest safe lam:", max(safe_f))
        for n in (75, 100, 200, 400):
            for name, fn in (("LTT(d/3)", lambda p, idx: ltt_select(p, idx, level=LEVEL_PER_PATH, paths=(4,))),
                             ("Naive", lambda p, idx: naive_select(p, idx, paths=(4,)))):
                Rs, fb, lams = [], 0, []
                for _ in range(1000):
                    cal = make_f(n, rng)
                    kh, lh = fn(cal, np.arange(n))
                    if kh is None:
                        Rs.append(0.0)
                        fb += 1
                    else:
                        Rs.append(true_f[lh][0])
                        lams.append(lh)
                Ra = np.array(Rs)
                print(f"  n={n:3d} {name:8s}: P(R<=0.05)={(Ra<=ALPHA).mean():.3f}, meanR={Ra.mean():.4f}, fallback={fb/10:.1f}%, med_lam={np.median(lams) if lams else float('nan'):.2f}")
    else:
        main()
