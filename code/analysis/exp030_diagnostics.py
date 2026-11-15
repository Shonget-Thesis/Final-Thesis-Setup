"""Descriptive diagnostics for EXP-030, not part of its preregistration.

These were chosen after the results were seen. They describe the results and cannot change the frozen verdict.
Committed files are only read, and the runner is not imported.

- Mirror control: TOH3 minus MIR3 per condition and pooled over the two noise conditions per speaker (UTMOS,
  similarity, WER), with a percentile bootstrap 95% CI (B = 2000, seed 2026) and the number of speakers above zero.
  MIR3 has the same amount and kind of re-timing as TOH3, placed where the teacher did not put it.
- Degraded prompts: median WER, runaway outputs (WER > 1), and failures against W8 under the thesis's Definition 3.1
  (code/scheduling/failure_config.json: WER > max(0.125, WER_W8 + 0.01) or SIM < SIM_W8 - 0.02).
- Time map: mean and largest displacement per condition, and how often the largest displacement reaches the band.
- Variety and displacement: Spearman correlation across the 20 diversity speakers between the mean map displacement
  and each arm's CVf0 gain over the student alone. CVf0 is computed frame by frame across seeds without alignment
  (run_selective_handover_prelim.cvf0), so seeds whose timing differs raise it even when their pitch contours agree.

Usage: python code/analysis/exp030_diagnostics.py [results dir]
"""
import csv
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "research/experiments/EXP-030-timing-only-handover"
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP / "results"
FAIL_CFG = ROOT / "code/scheduling/failure_config.json"
ARMS = ("STU", "W2", "W3", "W4", "TOH2", "TOH3", "MIR3", "W8")
CONDS = ("clean", "gauss25", "noisy15", "telephone")
NOISE = ("gauss25", "noisy15")
METRICS = ("utmos", "similarity", "wer")
B, SEED = 2000, 2026


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def boot(v):
    rng = random.Random(SEED)
    means = sorted(st.mean(rng.choices(v, k=len(v))) for _ in range(B))
    return {"mean": st.mean(v), "lo": means[int(0.025 * B)], "hi": means[int(0.975 * B) - 1],
            "positive": sum(x > 0 for x in v), "n": len(v)}


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


def failures(rows, arm, cfg):
    wer_f = sim_f = both = 0
    for r in rows:
        w, wr = float(r[f"wer_{arm}"]), float(r["wer_W8"])
        s, sr = float(r[f"similarity_{arm}"]), float(r["similarity_W8"])
        fw = w > max(cfg["absolute_wer_ceiling"], wr + cfg["relative_wer_margin"])
        fs = s < sr - cfg["similarity_margin"]
        wer_f += fw
        sim_f += fs
        both += fw or fs
    return {"failures": both, "wer_failures": wer_f, "sim_failures": sim_f}


def per_speaker(rows, fn):
    acc = {}
    for r in rows:
        acc.setdefault(r["speaker_id"], []).append(fn(r))
    return [st.mean(v) for _, v in sorted(acc.items())]


def main():
    qual = read(RES / "quality.csv")
    div = read(RES / "diversity.csv")
    results = json.loads((RES / "timing_only_handover_results.json").read_text(encoding="utf-8"))
    prereg = json.loads((EXP / "preregistration.json").read_text(encoding="utf-8"))
    cfg = json.loads(FAIL_CFG.read_text(encoding="utf-8"))
    out = {"note": f"post hoc and descriptive; outside the frozen verdict rule (EXP-030 preregistration "
                   f"{prereg['sha256'][:12]}...)",
           "stage": {"quality_rows": len(qual), "diversity_rows": len(div), "frozen_verdict": results["verdict"],
                     "secondary_verdict_TOH2": results["secondary_verdict_TOH2"], "partial": results["partial"]}}

    def diff(a, b, m):
        return lambda r: float(r[f"{m}_{a}"]) - float(r[f"{m}_{b}"])

    mirror = {}
    for cond in CONDS:
        sub = [r for r in qual if r["condition"] == cond]
        mirror[cond] = {m: boot([diff("TOH3", "MIR3", m)(r) for r in sub]) for m in METRICS}
    noise = [r for r in qual if r["condition"] in NOISE]
    mirror["noise_pooled_per_speaker"] = {m: boot(per_speaker(noise, diff("TOH3", "MIR3", m))) for m in METRICS}
    out["toh3_minus_mir3"] = mirror

    cond_out = {}
    for cond in CONDS:
        sub = [r for r in qual if r["condition"] == cond]
        arms = {}
        for a in ARMS:
            w = [float(r[f"wer_{a}"]) for r in sub]
            arms[a] = {"median_wer": st.median(w), "runaway_wer_gt_1": sum(x > 1 for x in w),
                       "mean_utmos": st.mean(float(r[f"utmos_{a}"]) for r in sub),
                       "mean_similarity": st.mean(float(r[f"similarity_{a}"]) for r in sub)}
            if a != "W8":
                arms[a].update(failures(sub, a, cfg))
        cond_out[cond] = {"n": len(sub), "arms": arms}
    out["definition_3_1_vs_W8"] = {"rule": "WER > max(0.125, WER_W8 + 0.01) or SIM < SIM_W8 - 0.02",
                                   "by_condition": cond_out}

    maps = {}
    for cond in CONDS:
        sub = [r for r in qual if r["condition"] == cond]
        entry = {}
        for k in (2, 3):
            rms = [float(r[f"warp_rms_{k}"]) for r in sub]
            mx = [float(r[f"warp_max_{k}"]) for r in sub]
            entry[f"TOH{k}"] = {"mean_rms_frames": st.mean(rms), "mean_max_frames": st.mean(mx),
                                "cases_max_at_band": sum(m >= float(r["warp_band"]) - 1e-9 for m, r in zip(mx, sub))}
        entry["mean_band_frames"] = st.mean(float(r["warp_band"]) for r in sub)
        entry["mean_frames"] = st.mean(float(r["frames"]) for r in sub)
        maps[cond] = entry
    out["time_map_by_condition"] = maps

    gains = {}
    for arm, disp in (("TOH3", "warp_rms_3_mean"), ("MIR3", "warp_rms_3_mean"), ("W3", "warp_rms_3_mean"),
                      ("TOH2", "warp_rms_2_mean"), ("W2", "warp_rms_2_mean")):
        x = [float(r[disp]) for r in div]
        y = [float(r[f"cvf0_{arm}"]) - float(r["cvf0_STU"]) for r in div]
        gains[f"{arm}_minus_STU_vs_{disp}"] = spearman(x, y)
    out["cvf0_gain_vs_map_displacement_spearman_20_speakers"] = gains

    path = RES / "diagnostics.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # LF on every platform: results/ is stored as -text
        f.write(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
