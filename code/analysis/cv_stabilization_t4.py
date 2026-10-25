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


def step(case, lam):
    r = case[8]
    if float(r["prompt_snr"]) < THETA:
        return 8
    for k in (4, 5, 6):
        if float(r[f"d_{k}"]) <= lam:
            return k
    return 8


def failed(case, k):
    return case[k]["is_failure"].strip().lower() == "true"


def select(cases, keys, grid):
    """Largest lambda in the grid with zero added failures on these instances (None: reference)."""
    ok = [lam for lam in grid if not any(failed(cases[c], step(cases[c], lam)) for c in keys)]
    return max(ok) if ok else None


def cross_validate(cases, grid):
    speakers = sorted({s for (s, _, _) in cases})
    random.Random(SEED).shuffle(speakers)
    fold_of = {s: i % FOLDS for i, s in enumerate(speakers)}
    folds, fails, nfe = [], 0, []
    for f in range(FOLDS):
        train = [c for c in cases if fold_of[c[0]] != f]
        test = [c for c in cases if fold_of[c[0]] == f]
        lam = select(cases, train, grid)
        steps = [step(cases[c], lam if lam is not None else -1.0) for c in test]
        k = sum(failed(cases[c], s) for c, s in zip(test, steps))
        folds.append({"fold": f, "train_speakers": len({c[0] for c in train}), "train_instances": len(train),
                      "test_speakers": len({c[0] for c in test}), "test_instances": len(test), "lambda": lam,
                      "test_failures": k, "test_mean_nfe": sum(COST[s] for s in steps) / len(steps)})
        fails += k
        nfe += [COST[s] for s in steps]
    return folds, fails, sum(nfe) / len(nfe), len(nfe)


def main():
    cases = load()
    folds, fails, mean_nfe, n = cross_validate(cases, LAMBDAS)
    noisy = [c for c in cases if c[2] == "noisy15"]
    fixed6 = sum(failed(cases[c], 6) for c in cases)
    wide_folds, wide_fails, wide_nfe, _ = cross_validate(cases, WIDE)
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
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
