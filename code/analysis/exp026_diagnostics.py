"""Descriptive diagnostics for EXP-026, not part of its preregistration.

These were chosen after the results were seen. They describe the verdict and cannot change it. Committed files are
only read (no CSV repair), and the runner is not imported.

- Per contrast: the per-speaker mean, median, the number of the 20 speakers above zero, an exact two-sided sign test,
  and the mean without the two largest speakers, to show whether one speaker carries the result.
- Pitch-tracker check: the teacher alone scores lower on UTMOS, and noisier audio can raise pYIN's frame-wise spread
  without any change in prosody. CVf0 keeps only frames voiced in at least half the seeds, so the share of frames kept
  (voiced_*) is reported per arm, with the Spearman correlation of the CVf0 gain against the change in that share and
  against the arm's single-seed UTMOS on the same speaker.
- H3 provenance: the run's W4 and W8 CVf0 equal EXP-020's stored values, so H3 re-reads EXP-020's data.

Usage: python code/analysis/exp026_diagnostics.py [results dir]
"""
import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "research/experiments/EXP-026-transmissibility-test/results"
EXP020 = ROOT / "research/experiments/EXP-020-selective-handover-prelim/results/diversity.csv"
ARMS = ("W4", "W8", "W4R2", "W8R2", "T16")
CONTRASTS = (("T16", "W8"), ("W8R2", "W8"), ("W4R2", "W4"), ("T16", "W8R2"), ("W8", "W4"))


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    a, b = ranks(x), ranks(y)
    ma, mb = st.mean(a), st.mean(b)
    den = math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    return sum((p - ma) * (q - mb) for p, q in zip(a, b)) / den if den else None


def sign_test(n_pos, n):
    tail = sum(math.comb(n, k) for k in range(0, min(n_pos, n - n_pos) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def contrast(rows, x, y):
    d = sorted(((float(r[f"cvf0_{x}"]) - float(r[f"cvf0_{y}"])), r["speaker_id"]) for r in rows)
    v = [a for a, _ in d]
    pos = sum(a > 0 for a in v)
    return {"mean": st.mean(v), "median": st.median(v), "positive": pos, "n": len(v),
            "sign_test_p": sign_test(pos, len(v)), "mean_without_top2": st.mean(v[:-2]),
            "largest": {"speaker_id": d[-1][1], "difference": d[-1][0]}}


def main():
    div = read(RES / "diversity.csv")
    qual = {r["utt_id"]: r for r in read(RES / "quality.csv")}
    out = {"note": "post hoc and descriptive; outside the frozen verdict rule (EXP-026 preregistration 78f3432d...)",
           "speakers": len(div)}
    out["voiced_share"] = {a: {"mean": st.mean(float(r[f"voiced_{a}"]) for r in div),
                               "min": min(float(r[f"voiced_{a}"]) for r in div)} for a in ARMS}
    out["contrasts"] = {f"{x}_minus_{y}": contrast(div, x, y) for x, y in CONTRASTS}
    both = [r for r in div if r["utt_id"] in qual]
    tracker = {"speakers_in_both_files": len(both)}
    for a in ("T16", "W8R2"):
        gain = [float(r[f"cvf0_{a}"]) - float(r["cvf0_W8"]) for r in both]
        tracker[a] = {
            "spearman_gain_vs_voiced_share_change": spearman(gain, [float(r[f"voiced_{a}"]) - float(r["voiced_W8"]) for r in both]),
            "spearman_gain_vs_utmos": spearman(gain, [float(qual[r["utt_id"]][f"utmos_{a}"]) for r in both])}
    out["pitch_tracker_check"] = tracker
    if EXP020.exists():
        stored = {r["speaker_id"]: r for r in read(EXP020) if r.get("condition", "clean") == "clean"}
        gaps = [max(abs(float(r["cvf0_W4"]) - float(stored[r["speaker_id"]]["cvf0_plain_4"])),
                    abs(float(r["cvf0_W8"]) - float(stored[r["speaker_id"]]["cvf0_plain_8"])))
                for r in div if r["speaker_id"] in stored]
        out["h3_provenance"] = {"speakers_matched_to_exp020": len(gaps), "max_abs_difference": max(gaps) if gaps else None,
                                "meaning": "W4 and W8 CVf0 are EXP-020's stored values, so H3 is not new evidence"}
    path = RES / "diagnostics.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
