"""Re-reading the stored handover arms against the paper's t_switch = 0.25 instead of the thesis's W8 (post hoc).

EXP-032 showed that the thesis's W8 (j = 1 restart) keeps about a third of the paper setting's (P25) aligned pitch-variety
gain, and its EXP-026 join showed that W8R2 (Step 8, student restarted at 0.2929, 20 passes) matches P25 (30 passes) in
UTMOS on 30 shared clean cases and roughly in CVf0. EXP-028 and EXP-029 judged their cheaper arms against W8, whose UTMOS
is high partly because it has less variety. This script re-reads every stored arm against W8R2 as the stand-in for the
paper's setting, paired within each experiment (same speakers, seeds and cases). No synthesis; committed CSVs only.

Cohorts (clean prompts):
- EXP-027: fresh development speakers 61-100 (W4, W4R2, W8R2, W8, T16, student alone).
- EXP-028 `unseen_cal`: 20 calibration-split speakers x 15 seeds, 30 quality cases (every arm, including W8R2).
- EXP-028 development: speakers 61-100, its arms joined to EXP-027's stored W4, W8R2 and W8 (the runner reused them).
- EXP-029: development speakers 61-90 (20 x 12 seeds; 30 quality cases per condition).

For each arm: CVf0 minus W8R2 (speaker-paired), its share of W8R2's CVf0 gain over W4, and UTMOS, WER and SIM minus
W8R2 (case-paired), with the runners' bootstrap (B = 2000, seed 2026). EXP-029's gauss18 and noisy15 rows are reported
too, with the number of runaway outputs (WER > 1).

Usage: python code/analysis/paper_anchor_rescreen.py
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "colab"))
import run_transmissibility_confirm as tc  # noqa: E402

EXPS = ROOT / "research/experiments"
OUT = EXPS / "EXP-032-variety-per-pass/offline/paper_anchor_rescreen.json"
PASSES = {"student4": 4, "W4": 13, "W4R2": 12, "W5R2": 14, "W6R2": 16, "W8R2": 20, "W8": 21, "T16": 32,
          "TBRG": 13, "STS2": 13, "BLND": 13, "SHUF": 13, "TB25": 13, "TB50": 13, "TB75": 13}
METRICS = ("utmos", "wer", "similarity")
ANCHOR = "W8R2"


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(r, k):
    try:
        return float(r[k])
    except (KeyError, ValueError):
        return float("nan")


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for m in range(i, j + 1):
                r[order[m]] = (i + j) / 2.0
            i = j + 1
        return r
    a, b = ranks(x), ranks(y)
    ma, mb = st.mean(a), st.mean(b)
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** 0.5
    return sum((p - ma) * (q - mb) for p, q in zip(a, b)) / den if den else None


def mean(vals):
    """Mean over defined values (a speaker's CVf0 is undefined when pYIN voices too few frames)."""
    vals = [v for v in vals if v == v]
    return st.mean(vals) if vals else float("nan")


def boot(vals):
    vals = [v for v in vals if v == v]
    return tc.boot_ci(vals, B=2000, seed=2026) if len(vals) > 1 else None


def arms_of(rows, prefix):
    return [c[len(prefix):] for c in rows[0] if c.startswith(prefix) and not c[len(prefix):].startswith("common")]


def cohort(div, qual, label):
    arms = [a for a in arms_of(div, "cvf0_") if a in PASSES]
    out = {"label": label, "speakers": len(div), "quality_cases": len(qual), "arms": {}}
    w4 = mean(num(r, "cvf0_W4") for r in div) if "W4" in arms else None
    anchor = mean(num(r, f"cvf0_{ANCHOR}") for r in div)
    for a in sorted(arms, key=lambda x: PASSES[x]):
        e = {"passes": PASSES[a], "cvf0": mean(num(r, f"cvf0_{a}") for r in div),
             "cvf0_defined": sum(num(r, f"cvf0_{a}") == num(r, f"cvf0_{a}") for r in div)}
        if a != ANCHOR:
            e["cvf0_minus_W8R2"] = boot(num(r, f"cvf0_{a}") - num(r, f"cvf0_{ANCHOR}") for r in div)
            if w4 is not None and anchor - w4 > 0:
                e["share_of_W8R2_gain_over_W4"] = (e["cvf0"] - w4) / (anchor - w4)
        if qual and f"utmos_{a}" in qual[0]:
            e["utmos"] = st.mean(num(r, f"utmos_{a}") for r in qual)
            if a != ANCHOR:
                e["quality_minus_W8R2"] = {m: boot(num(r, f"{m}_{a}") - num(r, f"{m}_{ANCHOR}") for r in qual)
                                           for m in METRICS}
        out["arms"][a] = e
    return out


def main():
    res = {"kind": "post hoc re-reading of stored arms against W8R2 (stand-in for the paper's 0.25); no synthesis",
           "cohorts": {}}
    e27 = EXPS / "EXP-027-transmissibility-confirmation/results"
    res["cohorts"]["EXP-027"] = cohort(read(e27 / "diversity.csv"), read(e27 / "quality.csv"), "development 61-100")

    e28 = EXPS / "EXP-028-transmissibility-bridge/results"
    res["cohorts"]["EXP-028_unseen_cal"] = cohort(read(e28 / "unseen_diversity.csv"), read(e28 / "unseen_quality.csv"),
                                                   "calibration-split speakers")
    d27 = {r["utt_id"]: r for r in read(e27 / "diversity.csv")}
    q27 = {r["utt_id"]: r for r in read(e27 / "quality.csv")}
    keep = ("W4", "W8", "W8R2")
    div28 = [{**r, **{f"cvf0_{a}": d27[r["utt_id"]][f"cvf0_{a}"] for a in keep}} for r in read(e28 / "diversity.csv")
             if r["utt_id"] in d27]
    qual28 = [{**r, **{f"{m}_{a}": q27[r["utt_id"]][f"{m}_{a}"] for a in keep for m in METRICS}}
              for r in read(e28 / "quality.csv") if r["utt_id"] in q27]
    res["cohorts"]["EXP-028_dev"] = cohort(div28, qual28, "development 61-100, joined to EXP-027's stored references")

    e29 = EXPS / "EXP-029-damped-transmissibility-stac/results"
    q29 = read(e29 / "quality.csv")
    res["cohorts"]["EXP-029"] = cohort(read(e29 / "diversity.csv"), [r for r in q29 if r["condition"] == "clean"],
                                       "development 61-90")
    degraded = {}
    for cond in ("gauss18", "noisy15"):
        rows = [r for r in q29 if r["condition"] == cond]
        arms = [a for a in arms_of(rows, "utmos_") if a in PASSES]
        degraded[cond] = {a: {"passes": PASSES[a], "utmos": st.mean(num(r, f"utmos_{a}") for r in rows),
                              "wer_mean": st.mean(num(r, f"wer_{a}") for r in rows),
                              "runaway": sum(num(r, f"wer_{a}") > 1.0 for r in rows)}
                          for a in sorted(arms, key=lambda x: PASSES[x])}
    res["exp029_degraded"] = degraded

    # Can drift tell which inputs may stop early under the j = 2 restart? EXP-029 logged d_1..d_8 on each clean quality
    # case (the input seed); variety is per speaker over 12 seeds, so this is a per-speaker proxy.
    d29 = {r["utt_id"]: r for r in read(e29 / "diversity.csv")}
    c29 = {r["utt_id"]: r for r in q29 if r["condition"] == "clean"}
    ids = [u for u in d29 if u in c29]
    drift = {}
    for k, arm in ((5, "W5R2"), (6, "W6R2")):
        deficit = [num(d29[u], f"cvf0_{arm}") - num(d29[u], f"cvf0_{ANCHOR}") for u in ids]
        dk = [num(c29[u], f"d_{k}") for u in ids]
        drift[arm] = {"speakers": len(ids), "spearman_d_k_vs_cvf0_minus_W8R2": spearman(dk, deficit),
                      "speakers_within_0.005_or_above": sum(x >= -0.005 for x in deficit),
                      "d_k_median": st.median(dk)}
    res["exp029_drift_vs_early_stop_deficit"] = drift
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {OUT}")
    return res


if __name__ == "__main__":
    main()
