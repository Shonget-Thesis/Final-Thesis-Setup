"""Sample simulation for Appendix B: does fixed-sequence Learn-Then-Test keep its promise?

Synthetic data only (no speech is synthesized). Each simulated utterance has
endpoint-stabilization scores d_4, d_5, d_6 at the three candidate steps, a prompt SNR,
and, for every candidate, an added-failure indicator f(x,k) whose probability rises with d_k.
The step-8 reference never fails by definition (Definition 3.1).
"""
import numpy as np
from scipy.stats import binom

rng = np.random.default_rng(2026)
CAND = [4, 5, 6]                    # candidate stop indices
COST = {4: 13, 5: 15, 6: 17, 8: 21}  # Proposition 3.1
THETA = 20.0                         # SNR gate (dB)
GRID = np.round(np.arange(0.20, 1.2001, 0.05), 2)   # conservative -> aggressive
ALPHA, DELTA = 0.05, 0.10


def population(N):
    noisy = rng.random(N) < 0.25
    snr = np.where(noisy, rng.normal(16, 2, N), rng.normal(33, 4, N))
    hard = rng.lognormal(0.0, 0.45, N)                  # per-utterance difficulty
    d = {k: hard * np.exp(-0.35 * (k - 4)) * rng.lognormal(-0.55, 0.25, N) for k in CAND}
    fail = {}
    for k in CAND:
        logit = -4.4 + 3.2 * d[k] + 1.6 * noisy - 0.45 * (k - 4)
        fail[k] = rng.random(N) < 1 / (1 + np.exp(-logit))
    fail[8] = np.zeros(N, bool)
    return dict(snr=snr, d=d, fail=fail)


def policy(pop, lam, idx=slice(None)):
    """stop index chosen by Definition 3.4 for every utterance in idx"""
    snr = pop["snr"][idx]
    n = len(snr)
    stop = np.full(n, 8)
    undecided = np.ones(n, bool)
    for k in CAND:
        ok = undecided & (pop["d"][k][idx] <= lam) & (snr >= THETA)
        stop[ok] = k
        undecided &= ~ok
    return stop


def risk_cost(pop, lam, idx=slice(None)):
    stop = policy(pop, lam, idx)
    f = np.zeros(len(stop), bool)
    c = np.zeros(len(stop))
    for k in COST:
        m = stop == k
        f[m] = pop["fail"][k][idx][m]
        c[m] = COST[k]
    return f.mean(), c.mean()


def ltt(pop, idx):
    n = len(idx)
    lam_hat = None
    for lam in GRID:
        r, _ = risk_cost(pop, lam, idx)
        p = binom.cdf(round(n * r), n, ALPHA)
        if p <= DELTA:
            lam_hat = lam
        else:
            break
    return lam_hat


def naive(pop, idx):
    """largest grid value whose empirical risk is <= alpha (no test)"""
    best = None
    for lam in GRID:
        r, _ = risk_cost(pop, lam, idx)
        if r <= ALPHA:
            best = lam
    return best


def fitted_population_factory():
    """Population with parameters fitted to the T4 development split (research/evidence/benchmarks).

    Half of the utterances are noisy15, as in the splits. Prompt SNR is normal per condition; log d_4 is
    normal per condition and each later step adds a normal log-change per condition; failure follows a
    logistic model in (d_k, noisy, k - 4) fitted by maximum likelihood to the development rows.
    """
    import csv
    import json
    import math
    from pathlib import Path
    from scipy.optimize import minimize
    root = Path(__file__).resolve().parents[2]
    dev = {r["utt_id"] for r in json.loads((root / "research/protocols/splits/development_manifest.json").read_text(encoding="utf-8"))}
    rows = {}
    with open(root / "research/evidence/benchmarks/gpu_grid_all1000.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["utt_id"] in dev:
                rows.setdefault((r["utt_id"], r["condition"]), {})[int(r["teacher_step"])] = r
    fit, X, y = {}, [], []
    for c in ("clean", "noisy15"):
        cs = [v for (u, cc), v in rows.items() if cc == c]
        snr = np.array([float(v[8]["prompt_snr"]) for v in cs])
        logd = np.array([[math.log(float(v[8][f"d_{k}"])) for k in CAND] for v in cs])
        fit[c] = dict(snr=(snr.mean(), snr.std(ddof=1)), d4=(logd[:, 0].mean(), logd[:, 0].std(ddof=1)),
                      step=(np.diff(logd, axis=1).mean(), np.diff(logd, axis=1).std(ddof=1)))
        for v in cs:
            for k in CAND:
                X.append([1.0, float(v[8][f"d_{k}"]), float(c == "noisy15"), k - 4.0])
                y.append(float(v[k]["is_failure"].strip().lower() == "true"))
    X, y = np.array(X), np.array(y)
    w = minimize(lambda w: np.sum(np.logaddexp(0, X @ w) - y * (X @ w)), np.zeros(4), method="BFGS").x

    def fitted_population(N):
        noisy = rng.random(N) < 0.5
        pick = lambda key, i: np.where(noisy, fit["noisy15"][key][i], fit["clean"][key][i])
        snr = rng.normal(pick("snr", 0), pick("snr", 1))
        d = {4: np.exp(rng.normal(pick("d4", 0), pick("d4", 1)))}
        for k in (5, 6):
            d[k] = d[k - 1] * np.exp(rng.normal(pick("step", 0), pick("step", 1)))
        fail = {k: rng.random(N) < 1 / (1 + np.exp(-(w[0] + w[1] * d[k] + w[2] * noisy + w[3] * (k - 4)))) for k in CAND}
        fail[8] = np.zeros(N, bool)
        return dict(snr=snr, d=d, fail=fail)
    return fitted_population, fit, w


def main(make_population=population):
    big = make_population(400_000)                # stands in for the true distribution
    true = {lam: risk_cost(big, lam) for lam in GRID}
    print("true risk / mean passes per lambda")
    for lam in GRID:
        print(f"  {lam:.2f}  R={true[lam][0]:.4f}  C={true[lam][1]:.2f}")
    oracle_lam = max([l for l in GRID if true[l][0] <= ALPHA])
    print("largest lambda with true risk <= alpha:", oracle_lam, true[oracle_lam])

    rows = []
    for n in (50, 100, 200, 400):
        res = {"ltt": [], "naive": []}
        for rep in range(1000):
            cal = make_population(n)
            for name, fn in (("ltt", ltt), ("naive", naive)):
                lam = fn(cal, np.arange(n))
                if lam is None:
                    res[name].append((0.0, 21.0, None))
                else:
                    r, c = true[lam]
                    res[name].append((r, c, lam))
        for name in ("ltt", "naive"):
            R = np.array([x[0] for x in res[name]])
            C = np.array([x[1] for x in res[name]])
            none = sum(x[2] is None for x in res[name])
            lams = [x[2] for x in res[name] if x[2] is not None]
            med = np.median(lams) if lams else float("nan")
            rows.append((n, name, (R <= ALPHA).mean(), R.mean(), C.mean(), none / 1000, med))
    print("\n n  method  P(R<=a)  meanR  meanC  P(ref only)  median lam")
    for r in rows:
        print(f"{r[0]:4d} {r[1]:6s} {r[2]:.3f}  {r[3]:.4f} {r[4]:.2f}  {r[5]:.3f}  {r[6]:.2f}")


if __name__ == "__main__":
    import sys
    if "--fitted" in sys.argv:
        # Parameters fitted to the T4 development split (Appendix B, last paragraph).
        make, fit, w = fitted_population_factory()
        print("fitted:", {c: {k: [round(float(x), 3) for x in v] for k, v in f.items()} for c, f in fit.items()})
        print("failure logit a, b(d_k), c(noisy), e(k-4):", [round(float(x), 3) for x in w])
        main(make)
    else:
        main()
