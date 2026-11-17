"""Pre-run power check for EXP-024 (Analytic Unconditional Guidance pilot). Exploratory, stored development data only.

Written 2026-11-14, before EXP-024's preregistration was frozen and before any EXP-024 synthesis, to size the pilot's
speaker count and quality bounds. AUG is meant to leave the teacher's output almost unchanged, so the question is how
often a change that is known to be small would pass the bounds. The proxy is the 16-interval grid against the
32-interval grid at the same sway time (EXP-018: g16_k against g32_2k), which differs only in the number of teacher
steps. For random N-speaker subsets of the 60 Quick-set speakers it computes the speaker-bootstrap 95% interval of the
paired difference (capped WER, similarity, UTMOS) and checks the bounds.

Run: python code/analysis/exp024_power_check.py [thesis_root]   (writes research/experiments/EXP-024-*/power_check.json)
"""
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

R = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
STEPS_CSV = R / "research/experiments/EXP-018-swsr-handover-prelim/results/steps1to8.csv"
OUT = R / "research/experiments/EXP-024-analytic-unconditional-guidance-pilot/power_check.json"
SUBSETS, BOOT_B, SEED = 300, 2000, 2026
NS = (20, 30, 40)
# (step k of the 16-interval grid, condition, bounds); k = 7 stands in for Step 8, whose 32-interval twin was not stored
CELLS = [(4, "clean", {"wer": 0.01, "similarity": 0.01, "utmos": 0.10}),
         (7, "clean", {"wer": 0.01, "similarity": 0.01, "utmos": 0.10}),
         (7, "clean", {"wer": 0.015, "similarity": 0.01, "utmos": 0.10}),
         (4, "noisy15", {"wer": 0.025, "similarity": 0.01, "utmos": 0.10}),
         (7, "noisy15", {"wer": 0.025, "similarity": 0.01, "utmos": 0.10})]
METRICS = ("wer", "similarity", "utmos")


def load():
    with open(STEPS_CSV, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def differences(rows, k, cond):
    """Per speaker: g16_k minus g32_2k (capped WER), one case per speaker and condition in the Quick set."""
    out = {}
    for r in rows:
        if r["condition"] != cond:
            continue
        d = {}
        for x in METRICS:
            a, b = float(r[f"{x}_g16_{k}"]), float(r[f"{x}_g32_{2 * k}"])
            d[x] = (min(1.0, a) - min(1.0, b)) if x == "wer" else a - b
        out[r["speaker_id"]] = d
    return out


def interval(values, rng, B=BOOT_B):
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), size=(B, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def passes(d, speakers, bounds, rng):
    _, _, wer_hi = interval([d[s]["wer"] for s in speakers], rng)
    _, sim_lo, _ = interval([d[s]["similarity"] for s in speakers], rng)
    _, ut_lo, _ = interval([d[s]["utmos"] for s in speakers], rng)
    ok = {"wer": wer_hi <= bounds["wer"], "similarity": sim_lo >= -bounds["similarity"], "utmos": ut_lo >= -bounds["utmos"]}
    ok["all"] = all(ok.values())
    return ok


def main():
    rows = load()
    rng, pick = np.random.default_rng(SEED), random.Random(SEED)
    res = {"proxy": "16-interval grid (g16_k) against the 32-interval grid at the same sway time (g32_2k), EXP-018 stored "
                    "outputs of the 60 Quick-set development speakers; k = 7 stands in for Step 8",
           "subsets": SUBSETS, "bootstrap": {"B": BOOT_B, "seed": SEED, "unit": "speaker"}, "effects_60": {}, "pass_rates": []}
    for k in (4, 7):
        for cond in ("clean", "noisy15"):
            d = differences(rows, k, cond)
            sp = sorted(d)
            res["effects_60"][f"k{k}_{cond}"] = {x: dict(zip(("mean", "lo", "hi"), interval([d[s][x] for s in sp], rng)))
                                                 for x in METRICS}
    for k, cond, bounds in CELLS:
        d = differences(rows, k, cond)
        sp = sorted(d)
        for n in NS:
            tally = {"wer": 0, "similarity": 0, "utmos": 0, "all": 0}
            for _ in range(SUBSETS):
                ok = passes(d, pick.sample(sp, n), bounds, rng)
                for key in tally:
                    tally[key] += ok[key]
            res["pass_rates"].append({"k": k, "condition": cond, "speakers": n, "bounds": bounds,
                                      **{key: round(v / SUBSETS, 3) for key, v in tally.items()}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    for p in res["pass_rates"]:
        print(f"k={p['k']} {p['condition']:8} N={p['speakers']} WER<=+{p['bounds']['wer']}: pass WER {p['wer']:.2f} "
              f"SIM {p['similarity']:.2f} UTMOS {p['utmos']:.2f} all {p['all']:.2f}")
    for key, v in res["effects_60"].items():
        print(key, "  ".join(f"{x} {v[x]['mean']:+.4f} [{v[x]['lo']:+.4f}, {v[x]['hi']:+.4f}]" for x in METRICS))
    print(f"[OK] wrote {OUT}")


if __name__ == "__main__":
    main()
