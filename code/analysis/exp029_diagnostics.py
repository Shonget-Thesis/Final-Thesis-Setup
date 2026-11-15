"""Descriptive diagnostics for EXP-029, not part of its preregistration.

These were chosen after the quality-stage results were seen. They describe the results and cannot change the
preregistered verdict. Committed files are only read, and the runner is not imported.

- Stage: which stages have rows. With no diversity.csv, the frozen rule treats the diversity condition as met, so the
  "passed" arms passed on quality alone.
- Clean prompts: each arm minus W4 (same 13-NFE budget as the bridges) and minus W8, per speaker, with a percentile
  bootstrap 95% CI (B = 2000, seed 2026) and the number of the 30 speakers above zero.
- Degraded prompts: failures against W8 under the thesis's Definition 3.1 (code/scheduling/failure_config.json:
  WER > max(0.125, WER_W8 + 0.01) or SIM < SIM_W8 - 0.02), the median WER, and runaway outputs (WER > 1), because
  the mean WER is dominated by a few runaway cases.
- Routing: which frozen criterion each non-SNR W8-reference router missed, and how closely the teacher features track
  prompt SNR across all 90 cases.

Usage: python code/analysis/exp029_diagnostics.py [results dir]
"""
import csv
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "research/experiments/EXP-029-damped-transmissibility-stac"
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else EXP / "results"
FAIL_CFG = ROOT / "code/scheduling/failure_config.json"
ARMS = ("W4", "TB25", "TB50", "TB75", "TBRG", "BLND", "W5R2", "W6R2", "W8R2", "W8")
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


def main():
    qual = read(RES / "quality.csv")
    div_path = RES / "diversity.csv"
    div = read(div_path) if div_path.exists() else []
    results = json.loads((RES / "damped_transmissibility_results.json").read_text(encoding="utf-8"))
    prereg = json.loads((EXP / "preregistration.json").read_text(encoding="utf-8"))
    cfg = json.loads(FAIL_CFG.read_text(encoding="utf-8"))
    out = {"note": "post hoc and descriptive; outside the frozen verdict rule (EXP-029 preregistration e58f3856ad20...)",
           "stage": {"quality_rows": len(qual), "diversity_rows": len(div),
                     "frozen_verdict": results["verdict"], "partial": results["partial"],
                     "meaning": ("diversity.csv is missing, so determine_verdict treated the CVf0 condition as met: "
                                 "the passed arms passed on quality only") if not div else "both stages present"}}

    clean = [r for r in qual if r["condition"] == "clean"]
    out["clean_utmos"] = {
        f"{a}_minus_{ref}": boot([float(r[f"utmos_{a}"]) - float(r[f"utmos_{ref}"]) for r in clean])
        for ref in ("W4", "W8") for a in ARMS if a != ref}

    cond_out = {}
    for cond in ("clean", "gauss18", "noisy15"):
        sub = [r for r in qual if r["condition"] == cond]
        arms = {}
        for a in ARMS:
            w = [float(r[f"wer_{a}"]) for r in sub]
            arms[a] = {"median_wer": st.median(w), "runaway_wer_gt_1": sum(x > 1 for x in w),
                       "mean_similarity": st.mean(float(r[f"similarity_{a}"]) for r in sub),
                       "mean_utmos": st.mean(float(r[f"utmos_{a}"]) for r in sub)}
            if a != "W8":
                arms[a].update(failures(sub, a, cfg))
        cond_out[cond] = {"n": len(sub), "arms": arms}
    out["definition_3_1_vs_W8"] = {"rule": "WER > max(0.125, WER_W8 + 0.01) or SIM < SIM_W8 - 0.02", "by_condition": cond_out}

    snr = [float(r["prompt_snr"]) for r in qual]
    out["telemetry_vs_prompt_snr_spearman_all_90"] = {f: spearman([float(r[f]) for r in qual], snr)
                                                     for f in ("A_4", "A_4_norm", "gc_0")}
    misses = {}
    for key, r in results["routing_5fold"].items():
        if r is None or r["use_snr"] or not key.endswith(("_to_W8_via_A_4", "_to_W8_via_A_4_norm")):
            continue
        cu = r["clean_dutmos_vs_ref"]["mean"] if r["clean_dutmos_vs_ref"] else 0.0
        not_passed = "W4_is_the_baseline_not_a_candidate" if r["cheap_arm"] == "W4" else "arm_not_in_passed_clean_arms"
        missed = [name for name, ok in (
            (not_passed, r["cheap_arm"] in results["passed_clean_arms"]),
            ("early_frac", r["early_frac"] >= prereg["route_min_early_frac"]),
            ("nfe_saving_vs_W8", r["saving_vs_W8"] >= prereg["route_min_nfe_saving_vs_w8"]),
            ("clean_dutmos_mean", cu >= prereg["clean_utmos_bar"]),
            ("pooled_dwer_hi", r["pooled_dwer_vs_ref"]["hi"] <= prereg["wer_bar"]),
            ("pooled_dsim_lo", r["pooled_dsim_vs_ref"]["lo"] >= prereg["sim_bar"])) if not ok]
        misses[key] = {"missed": missed, "early_by_condition": r["cond_early_frac"],
                       "pooled_dsim_lo": r["pooled_dsim_vs_ref"]["lo"], "saving_vs_W8": r["saving_vs_W8"]}
    out["router_criteria_missed"] = misses

    path = RES / "diagnostics.json"
    with open(path, "w", encoding="utf-8", newline="\n") as f:  # LF on every platform: results/ is stored as -text
        f.write(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
