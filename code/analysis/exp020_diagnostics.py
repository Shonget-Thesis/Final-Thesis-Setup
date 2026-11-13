"""Descriptive diagnostics for EXP-020, not part of its preregistration.

Recomputes the failure counts from the raw CSV files without importing the runner, then prints the follow-up comparisons
quoted in the EXP-020 README: which arm of Definition 3.1 fails, the reference-against-reference floor, failure rates by
prompt SNR, student-only against STT, and whether later teacher steps help once the voice part is swapped.

These were computed after the results were seen, so they explain the verdict and cannot change it.

Usage: python code/analysis/exp020_diagnostics.py
"""
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "research/experiments"
RESULTS = EXP / "EXP-020-selective-handover-prelim/results"
CONDS = ["clean", "noisy15", "gauss18", "gauss20", "gauss22", "gauss25"]
NOISY = CONDS[1:]
METRICS = ("wer", "similarity", "utmos")
OUTS = ["STT_4", "TST_4", "TTS_4", "SST_4", "STS_4", "TSS_4", "STT_5", "STT_6", "STT_7"]
NAMES = ["student4", "plain_4", "plain_5", "plain_6", "plain_7", "plain_8", "s025"] + OUTS
BANDS = [(30, 999), (25, 30), (22, 25), (20, 22), (18, 20), (-99, 18)]
BOOT_B = 2000


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def arms(early, ref, th):
    """Definition 3.1, one flag per arm: (word arm, voice arm)."""
    return (early["wer"] > max(th["absolute_wer_ceiling"], ref["wer"] + th["relative_wer_margin"]),
            early["similarity"] < ref["similarity"] - th["similarity_margin"])


def load():
    th = json.loads((ROOT / "code/scheduling/failure_config.json").read_text(encoding="utf-8"))
    sel = read(RESULTS / "selective.csv")
    steps = {(r["utt_id"], r["condition"]): r for r in read(EXP / "EXP-018-swsr-handover-prelim/results/steps1to8.csv")}
    grid = {(r["utt_id"], r["condition"]): r for r in read(EXP / "EXP-017-grid32-prelim/results/grid32.csv")}
    table = []
    for r in sel:
        key = (r["utt_id"], r["condition"])
        m = {f"plain_{i}": {x: float(steps[key][f"{x}_g16_{i}"]) for x in METRICS} for i in (4, 5, 6, 7, 8)}
        m["student4"] = {x: float(grid[key][f"{x}_student4"]) for x in METRICS}
        m["s025"] = {x: float(grid[key][f"{x}_s025"]) for x in METRICS}
        m.update({o: {x: float(r[f"{x}_{o}"]) for x in METRICS} for o in OUTS})
        row = {"spk": r["speaker_id"], "cond": r["condition"], "snr": float(steps[key]["prompt_snr"]), "m": m}
        row["arms"] = {n: arms(m[n], m["plain_8"], th) for n in NAMES}
        row["fail"] = {n: any(row["arms"][n]) for n in NAMES}
        row["reverse_floor"] = any(arms(m["plain_8"], m["s025"], th))
        table.append(row)
    return sel, read(RESULTS / "diversity.csv"), table


def mcnemar(table, a, b, conds):
    rows = [r for r in table if r["cond"] in conds]
    x = sum(r["fail"][a] and not r["fail"][b] for r in rows)
    y = sum(r["fail"][b] and not r["fail"][a] for r in rows)
    n = x + y
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0
    return {"a": sum(r["fail"][a] for r in rows), "b": sum(r["fail"][b] for r in rows), "only_a": x, "only_b": y, "p": p}


def boot(values):
    rng = random.Random(0)
    runs = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(BOOT_B))
    return f"{sum(values) / len(values):+.4f} [{runs[int(.025 * BOOT_B)]:+.4f}, {runs[int(.975 * BOOT_B)]:+.4f}]"


def _value(row, name, metric):
    v = row["m"][name][metric]
    return min(1.0, v) if metric == "wer" else v


def paired(table, a, b, metric, conds):
    groups = defaultdict(list)
    for r in table:
        if r["cond"] in conds:
            groups[r["spk"]].append(_value(r, a, metric) - _value(r, b, metric))
    return boot([sum(v) / len(v) for v in groups.values()])


def mean(table, name, metric, conds):
    values = [_value(r, name, metric) for r in table if r["cond"] in conds]
    return sum(values) / len(values)


def main():
    sel, div, table = load()
    seconds, div_seconds = [float(r["case_seconds"]) for r in sel], [float(r["case_seconds"]) for r in div]
    print(f"cases {len(table)}, unique {len({(r['utt_id'], r['condition']) for r in sel})}, speakers {len({r['spk'] for r in table})}, "
          f"GPU {dict(Counter(r['gpu'] for r in sel))}, diversity pairs {len(div)}")
    print(f"per-case time: quality {sum(seconds) / 3600:.2f} h, diversity {sum(div_seconds) / 3600:.2f} h, total {(sum(seconds) + sum(div_seconds)) / 3600:.2f} h")

    print("\nfailures against Step 8 (" + ", ".join(CONDS) + ", total)")
    for n in NAMES:
        counts = [sum(r["fail"][n] for r in table if r["cond"] == c) for c in CONDS]
        print(f"  {n:9s} {counts} {sum(counts)}")

    print("\nnoisy failures by arm of Definition 3.1")
    for n in ("student4", "plain_4", "plain_7", "STT_4", "STT_5", "STT_6", "STT_7", "s025"):
        flags = [r["arms"][n] for r in table if r["cond"] in NOISY]
        word, voice, both = sum(a and not b for a, b in flags), sum(b and not a for a, b in flags), sum(a and b for a, b in flags)
        print(f"  {n:9s} word only {word:3d}  voice only {voice:3d}  both {both:3d}  word arm {word + both:3d}  voice arm {voice + both:3d}  total {word + voice + both}")

    print(f"\nreference floor on noisy prompts: s025 against Step 8 {sum(r['fail']['s025'] for r in table if r['cond'] in NOISY)} of 300, "
          f"Step 8 against s025 {sum(r['reverse_floor'] for r in table if r['cond'] in NOISY)} of 300")

    print("\nexact McNemar on noisy prompts")
    for a, b in (("plain_4", "STT_4"), ("student4", "STT_4"), ("STT_4", "STT_5"), ("STT_4", "STT_6"), ("STT_4", "STT_7")):
        print(f"  {a} against {b}: {mcnemar(table, a, b, NOISY)}")

    print("\nmeans: capped WER, similarity, UTMOS")
    for n in ("student4", "plain_4", "plain_8", "STT_4", "STT_5", "STT_6", "STT_7"):
        print(f"  {n:9s} noisy {mean(table, n, 'wer', NOISY):.3f} {mean(table, n, 'similarity', NOISY):.3f} {mean(table, n, 'utmos', NOISY):.2f}"
              f"   clean {mean(table, n, 'wer', ['clean']):.3f} {mean(table, n, 'similarity', ['clean']):.3f} {mean(table, n, 'utmos', ['clean']):.2f}")

    print("\npaired differences, speaker bootstrap: mean [95% CI]")
    pairs = [("STT_4", "plain_4", NOISY), ("STT_4", "plain_8", NOISY), ("STT_4", "student4", NOISY), ("STT_5", "STT_4", NOISY), ("STT_6", "STT_4", NOISY),
             ("STT_7", "STT_4", NOISY), ("STT_4", "plain_4", ["clean"]), ("STT_4", "plain_8", ["clean"]), ("STT_4", "student4", ["clean"])]
    for a, b, conds in pairs:
        where = "clean" if conds == ["clean"] else "noisy"
        print(f"  {where} {a} - {b}: " + "  ".join(f"{x} {paired(table, a, b, x, conds)}" for x in METRICS))

    print("\nfailure rate by measured prompt SNR")
    head = ("plain_4", "STT_4", "plain_5", "STT_5", "plain_6", "STT_6", "plain_7", "STT_7", "student4", "s025")
    print("  band (dB)      n " + " ".join(f"{h:>8s}" for h in head))
    for lo, hi in BANDS:
        rows = [r for r in table if lo <= r["snr"] < hi]
        print(f"  {lo:>4} to {hi:<4} {len(rows):4d} " + " ".join(f"{sum(r['fail'][h] for r in rows) / len(rows):8.3f}" for h in head))

    print("\nCVf0 on clean prompts, 20 speakers, paired: mean [95% CI]")
    for a, b in (("STT_4", "plain_8"), ("STT_4", "plain_4"), ("STT_4", "student4"), ("plain_8", "student4")):
        print(f"  {a} - {b}: {boot([float(r[f'cvf0_{a}']) - float(r[f'cvf0_{b}']) for r in div])}")


if __name__ == "__main__":
    main()
