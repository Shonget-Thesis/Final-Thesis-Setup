"""Appendix D.4 on the T4 results: cross-validated threshold for the stabilization policy.

Same procedure as the team's earlier CPU experiment: four-fold speaker-grouped cross-validation;
in each fold the threshold lambda is the largest value in {0.20, 0.25, ..., 0.60} with zero added
failures on the training speakers, frozen, then scored on the held-out speakers. Policy of
Definition 3.4 with candidates Steps 4/5/6, reference Step 8 and SNR gate 20 dB. Data: the 100
development speakers of research/evidence/benchmarks/gpu_grid_all1000.csv (Tesla T4), clean and
noisy15 prompts. Standard library only.

    python code/analysis/cv_stabilization_t4.py            # prints the results and writes the JSON
"""
import csv
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "research/evidence/benchmarks/gpu_grid_all1000.csv"
DEV = ROOT / "research/protocols/splits/development_manifest.json"
OUT = ROOT / "research/experiments/EXP-009-cv-stabilization-t4/results.json"
COST = {4: 13, 5: 15, 6: 17, 8: 21}
LAMBDAS = [round(0.20 + 0.05 * i, 2) for i in range(9)]       # 0.20 ... 0.60, as in the CPU run
WIDE = [round(0.20 + 0.05 * i, 2) for i in range(21)]         # 0.20 ... 1.20, the proposal's grid
THETA, FOLDS, SEED = 20.0, 4, 2026


def load():
    dev = {r["utt_id"] for r in json.loads(DEV.read_text(encoding="utf-8"))}
    cases = {}
    with GRID.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["utt_id"] in dev:
                cases.setdefault((r["speaker_id"], r["utt_id"], r["condition"]), {})[int(r["teacher_step"])] = r
    return cases


def step(case, lam, theta=THETA):
    r = case[8]
    if float(r["prompt_snr"]) < theta:
        return 8
    for k in (4, 5, 6):
        if float(r[f"d_{k}"]) <= lam:
            return k
    return 8


def failed(case, k):
    return case[k]["is_failure"].strip().lower() == "true"


def select(cases, keys, grid, theta=THETA):
    """Largest lambda in the grid with zero added failures on these instances (None: reference)."""
    ok = [lam for lam in grid if not any(failed(cases[c], step(cases[c], lam, theta)) for c in keys)]
    return max(ok) if ok else None


def cross_validate(cases, grid, theta=THETA):
    speakers = sorted({s for (s, _, _) in cases})
    random.Random(SEED).shuffle(speakers)
    fold_of = {s: i % FOLDS for i, s in enumerate(speakers)}
    folds, fails, nfe, chosen = [], 0, [], {}
    for f in range(FOLDS):
        train = [c for c in cases if fold_of[c[0]] != f]
        test = [c for c in cases if fold_of[c[0]] == f]
        lam = select(cases, train, grid, theta)
        steps = [step(cases[c], lam if lam is not None else -1.0, theta) for c in test]
        k = sum(failed(cases[c], s) for c, s in zip(test, steps))
        folds.append({"fold": f, "train_speakers": len({c[0] for c in train}), "train_instances": len(train),
                      "test_speakers": len({c[0] for c in test}), "test_instances": len(test), "lambda": lam,
                      "test_failures": k, "test_mean_nfe": sum(COST[s] for s in steps) / len(steps)})
        fails += k
        nfe += [COST[s] for s in steps]
        chosen.update(zip(test, steps))
    return folds, fails, sum(nfe) / len(nfe), len(nfe), chosen


def clopper_pearson_upper(failures, n, alpha=0.05):
    """Upper end of the two-sided exact (Clopper-Pearson) 1 - alpha interval for a failure rate."""
    if failures >= n:
        return 1.0
    cdf = lambda p: sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(failures + 1))
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) > alpha / 2 else (lo, mid)
    return hi


def value(case, k, measure):
    """Per-instance measure at step k: forward passes, WER (percent), SIM or UTMOS."""
    if measure == "nfe":
        return COST[k]
    v = float(case[k]["wer" if measure == "wer" else "similarity" if measure == "sim" else "utmos"])
    return 100 * v if measure == "wer" else v


def summarize(cases, chosen):
    """Added failures, cost and mean quality of a per-instance step choice, by prompt condition."""
    out = {}
    for cond in ("clean", "noisy15", "both"):
        keys = [c for c in cases if cond == "both" or c[2] == cond]
        nfe = sum(COST[chosen[c]] for c in keys) / len(keys)
        k = sum(failed(cases[c], chosen[c]) for c in keys)
        out[cond] = {"failures": k, "instances": len(keys), "failure_rate_cp95_upper": clopper_pearson_upper(k, len(keys)),
                     "mean_nfe": nfe, "saving_vs_21": 1 - nfe / 21}
        for m in ("wer", "sim", "utmos"):
            out[cond][f"{m}_mean"] = sum(value(cases[c], chosen[c], m) for c in keys) / len(keys)
    return out


def mcnemar_exact(cases, a, b):
    """Exact two-sided McNemar test on the paired failures of step choices a and b."""
    x = sum(failed(cases[c], a[c]) and not failed(cases[c], b[c]) for c in cases)
    y = sum(failed(cases[c], b[c]) and not failed(cases[c], a[c]) for c in cases)
    n = x + y
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return {"only_first_fails": x, "only_second_fails": y, "p": p}


def wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank test, zeros dropped, normal approximation with tie correction."""
    d = [x for x in diffs if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return {"n_nonzero": 0, "p": 1.0}
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks, i, ties = [0.0] * n, 0, 0.0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(d[order[j + 1]]) - abs(d[order[i]])) < 1e-12:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        ties += (j - i + 1) ** 3 - (j - i + 1)
        i = j + 1
    w = sum(r for r, x in zip(ranks, d) if x > 0)
    mean, var = n * (n + 1) / 4, n * (n + 1) * (2 * n + 1) / 24 - ties / 48
    z = (w - mean) / math.sqrt(var) if var > 0 else 0.0
    return {"n_nonzero": n, "w_plus": w, "z": z, "p": math.erfc(abs(z) / math.sqrt(2))}


def speaker_diffs(cases, chosen, reference, measure):
    """Speaker-level differences (policy minus reference), averaging each speaker's instances."""
    per = {}
    for c in cases:
        per.setdefault(c[0], []).append(value(cases[c], chosen[c], measure) - value(cases[c], reference[c], measure))
    return [sum(v) / len(v) for v in per.values()]


def holm(pvals):
    """Holm-adjusted p-values for a dict of raw p-values."""
    out, run = {}, 0.0
    for i, name in enumerate(sorted(pvals, key=pvals.get)):
        run = max(run, min(1.0, (len(pvals) - i) * pvals[name]))
        out[name] = run
    return out


def cost_matched_random(cases, chosen, draws=2000):
    """Permute the policy's chosen steps among the gated instances: same cost, steps assigned at random."""
    gated = [c for c in cases if float(cases[c][8]["prompt_snr"]) >= THETA]
    picks = [chosen[c] for c in gated]
    own = sum(failed(cases[c], chosen[c]) for c in cases)
    rng, counts = random.Random(SEED), []
    for _ in range(draws):
        rng.shuffle(picks)
        counts.append(sum(failed(cases[c], k) for c, k in zip(gated, picks)))
    return {"draws": draws, "mean_failures": sum(counts) / draws, "policy_failures": own,
            "share_of_draws_with_at_most_policy_failures": sum(k <= own for k in counts) / draws,
            "mean_nfe": sum(COST[chosen[c]] for c in cases) / len(cases)}


def main():
    cases = load()
    folds, fails, mean_nfe, n, chosen = cross_validate(cases, LAMBDAS)
    noisy = [c for c in cases if c[2] == "noisy15"]
    fixed6 = sum(failed(cases[c], 6) for c in cases)
    wide_folds, wide_fails, wide_nfe, _, _ = cross_validate(cases, WIDE)
    # Same 200 instances under the compared methods of Chapter 3 (Appendix D.4 tables).
    gate = lambda c: float(cases[c][8]["prompt_snr"]) >= THETA
    policies = {f"fixed_step_{k}": {c: k for c in cases} for k in (4, 5, 6, 8)}
    policies["snr_only_step4"] = {c: 4 if gate(c) else 8 for c in cases}
    for k in (5, 6):
        policies[f"snr_fixed_later_step_{k}"] = {c: k if gate(c) else 8 for c in cases}
    policies["stabilization_only_out_of_fold"] = cross_validate(cases, LAMBDAS, theta=-math.inf)[4]
    policies["snr_drift_out_of_fold"] = chosen
    policies["oracle_hindsight"] = {c: next(k for k in (4, 5, 6, 8) if not failed(cases[c], k)) for c in cases}
    reference = policies["fixed_step_8"]
    compared = [name for name in policies if name not in ("fixed_step_8", "oracle_hindsight")]
    tests = {"mcnemar_vs_snr_drift": {name: mcnemar_exact(cases, policies[name], chosen)
                                      for name in compared if name != "snr_drift_out_of_fold"},
             "wilcoxon_vs_reference_holm": {}}
    for m in ("nfe", "wer", "sim", "utmos"):
        raw = {name: wilcoxon(speaker_diffs(cases, policies[name], reference, m)) for name in compared}
        adj = holm({name: r["p"] for name, r in raw.items()})
        tests["wilcoxon_vs_reference_holm"][m] = {name: raw[name] | {"p_holm": adj[name]} for name in compared}
    train_spk = folds[0]["train_speakers"]
    res = {
        "data": "T4 grid, development split", "speakers": len({c[0] for c in cases}), "instances": n,
        "conditions": sorted({c[2] for c in cases}), "folds": folds,
        "out_of_fold": {"failures": fails, "instances": n, "mean_nfe": mean_nfe, "saving_vs_21": 1 - mean_nfe / 21},
        "fixed_s015_step6": {"failures": fixed6, "instances": n, "nfe": 17},
        "noisy_prompts_sent_to_reference": sum(float(cases[c][8]["prompt_snr"]) < THETA for c in noisy), "noisy_prompts": len(noisy),
        "lambda_at_grid_edge_in_folds": sum(f["lambda"] == LAMBDAS[-1] for f in folds),
        "smallest_certifiable_risk_zero_failures_delta_0.10": {
            "per_training_speaker": 1 - 0.10 ** (1 / train_spk), "training_speakers": train_spk},
        "sensitivity_grid_0.20_1.20": {"lambdas": [f["lambda"] for f in wide_folds], "failures": wide_fails, "mean_nfe": wide_nfe},
        "policy_comparison": {name: summarize(cases, pick) for name, pick in policies.items()},
        "cost_matched_random": cost_matched_random(cases, chosen),
        "tests": tests,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
