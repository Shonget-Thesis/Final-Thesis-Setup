"""Recompute calibration and test outcomes from the committed T4 grid (standard library only).

Usage, from the repository root:
    python research/audits/2026-10-25/recompute_gpu_grid.py

Reads research/evidence/benchmarks/gpu_grid_all1000.csv and the split manifests in
research/protocols/splits/, applies the policy of Definition 3.4 (candidates 4, 5, 6;
reference 8; SNR gate 20 dB), counts a speaker as failing when its clean or noisy15
case fails, and runs fixed-sequence Learn-Then-Test (alpha 0.05, delta 0.10) over the
manuscript's grid 0.20, 0.25, ..., 1.20. Failure labels are the grid's is_failure column.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRID = ROOT / "research/evidence/benchmarks/gpu_grid_all1000.csv"
SPLITS = ROOT / "research/protocols/splits"
COST = {4: 13, 5: 15, 6: 17, 8: 21}
LAMBDAS = [round(0.20 + 0.05 * i, 2) for i in range(21)]
ALPHA, DELTA, THETA = 0.05, 0.10, 20.0


def load():
    split_of = {}
    for name in ("development", "calibration", "test"):
        for row in json.loads((SPLITS / f"{name}_manifest.json").read_text(encoding="utf-8")):
            split_of[row["utt_id"]] = name
    cases = defaultdict(dict)  # (utt_id, condition) -> teacher_step -> row
    with GRID.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cases[(row["utt_id"], row["condition"])][int(row["teacher_step"])] = row
    return split_of, cases


def choose(steps, lam, theta):
    row = steps[8]
    if float(row["prompt_snr"]) < theta:
        return 8
    for k in (4, 5, 6):
        if float(row[f"d_{k}"]) <= lam:
            return k
    return 8


def evaluate(split_of, cases, split, lam, theta=THETA):
    """Return (speakers, speaker-level failures, clean failures, noisy failures, mean NFE per case)."""
    by_utt = defaultdict(dict)
    for (utt, cond), steps in cases.items():
        if split_of[utt] == split:
            by_utt[utt][cond] = steps
    robust = clean = noisy = 0
    nfe = []
    for conds in by_utt.values():
        failed = {}
        for cond, steps in conds.items():
            k = choose(steps, lam, theta)
            failed[cond] = steps[k]["is_failure"].strip().lower() == "true"
            nfe.append(COST[k])
        robust += any(failed.values())
        clean += failed.get("clean", False)
        noisy += failed.get("noisy15", False)
    return len(by_utt), robust, clean, noisy, sum(nfe) / len(nfe)


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def main():
    split_of, cases = load()
    print("Fixed-sequence LTT on the calibration split (T4 labels):")
    lam_hat = None
    for lam in LAMBDAS:
        n, k, _, _, nfe = evaluate(split_of, cases, "calibration", lam)
        p = binom_cdf(k, n, ALPHA)
        print(f"  lambda={lam:.2f}  failures={k}/{n}  p={p:.4f}  mean NFE={nfe:.2f}")
        if p > DELTA:
            break
        lam_hat = lam
    print(f"lambda_hat = {lam_hat}\n")
    print("Test split: (speakers, speaker failures, clean, noisy15, mean NFE)")
    rows = [("stabilization + SNR, lambda=0.50", 0.50, THETA),
            (f"stabilization + SNR, lambda_hat={lam_hat}", lam_hat, THETA),
            ("SNR-only (lambda=inf)", math.inf, THETA),
            ("fixed step 4", math.inf, -math.inf),
            ("reference step 8", -1.0, THETA)]
    for label, lam, theta in rows:
        if lam is not None:
            print(f"  {label:36s} {evaluate(split_of, cases, 'test', lam, theta)}")


if __name__ == "__main__":
    main()
