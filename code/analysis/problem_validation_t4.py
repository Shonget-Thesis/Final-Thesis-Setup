"""Problem validation on the T4 grid: does one fixed switching time suit every input?

The T4 counterpart of the team's earlier CPU "fixed handover-timing" proof (12 speakers, Appendix D.2),
computed from research/evidence/benchmarks/gpu_grid_all1000.csv (500 speakers x clean/noisy15,
Steps 4/5/6/8, Tesla T4) with the added-failure rule of Definition 3.1. Standard library only.

    python code/analysis/problem_validation_t4.py      # prints the results and writes the JSON

Questions:
  1. How does the added-failure rate change with a fixed switching time (per condition and split)?
  2. Per input, what is the cheapest switching time that does not fail (the oracle), and does it vary?
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "research/evidence/benchmarks/gpu_grid_all1000.csv"
SPLITS = ROOT / "research/protocols/splits"
OUT = ROOT / "research/experiments/EXP-001-problem-validation-t4/results.json"
COST = {4: 13, 5: 15, 6: 17, 8: 21}
SWITCH = {4: 0.07, 5: 0.10, 6: 0.15, 8: 0.25}


def main():
    split_of = {}
    for name in ("development", "calibration", "test"):
        for r in json.loads((SPLITS / f"{name}_manifest.json").read_text(encoding="utf-8")):
            split_of[r["utt_id"]] = name
    cases = {}
    with GRID.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            cases.setdefault((r["utt_id"], r["condition"]), {})[int(r["teacher_step"])] = r
    failed = lambda case, k: case[k]["is_failure"].strip().lower() == "true"
    oracle = lambda case: next(k for k in (4, 5, 6, 8) if not failed(case, k))

    res = {"data": "T4 grid, 500 speakers x clean/noisy15", "fixed_switching": {}, "oracle": {}}
    for cond in ("clean", "noisy15", "both"):
        sel = [c for c in cases.values() if cond == "both" or c[8]["condition"] == cond]
        res["fixed_switching"][cond] = {
            f"s={SWITCH[k]} (Step {k}, {COST[k]} NFE)": {"failures": sum(failed(c, k) for c in sel), "n": len(sel),
                                                          "rate": sum(failed(c, k) for c in sel) / len(sel)}
            for k in (4, 5, 6, 8)}
        picks = [oracle(c) for c in sel]
        counts = Counter(picks)
        res["oracle"][cond] = {"n": len(sel), "cheapest_safe_step_counts": {f"Step {k}": counts.get(k, 0) for k in (4, 5, 6, 8)},
                               "share_earlier_than_reference": sum(p < 8 for p in picks) / len(sel),
                               "oracle_mean_nfe": sum(COST[p] for p in picks) / len(sel),
                               "distinct_oracle_steps": len(counts)}
    # Per split, both conditions: the same picture should hold on every split.
    res["fixed_switching_by_split"] = {
        sp: {f"Step {k}": sum(failed(c, k) for (u, _), c in cases.items() if split_of[u] == sp) for k in (4, 5, 6)}
        | {"n": sum(1 for (u, _) in cases if split_of[u] == sp)} for sp in ("development", "calibration", "test")}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
