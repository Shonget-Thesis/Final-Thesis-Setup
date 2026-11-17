"""Descriptive diagnostics for EXP-033, not part of its preregistration.

These were chosen after the results were seen. They describe the results and cannot change the frozen labels.
Committed files are only read. The runner (code/colab/run_cheaper_teacher_path.py) is imported read-only for its
bootstrap and share functions, so every interval here is computed exactly as the frozen analysis computes them.

- Against W8R2: each arm minus W8R2 on variety and clean quality, per speaker or case. W8R2 is the 16-interval setting
  that matches the paper's P25, so this asks whether a cheaper arm differs from the 20-pass way of getting there.
- Direction per speaker: how many of the 20 speakers have each arm at or above P25, per variety measure.
- How fragile the shares are: each arm's share of P25's gain with one speaker left out (smallest and largest of the 20),
  and without the two speakers whose P25 CVf0 is largest.
- The grid's coarseness: CPH12, CPH8 and CPH6 side by side (distance of the handed-over estimate from the 16-interval
  Step 8 estimate, melody agreement with P25, variety, quality), and CPH12 minus CPH8.
- CPH8's clean WER: the cases behind the upper bound that sits close to the +0.01 bar.
- noisy15: runaway outputs per arm, and SST against the student alone.
- The thesis's current policy on the same 60 cases: Chapter 3's SNR + drift rule (lambda = 0.35 and the LTT-certified
  1.20), its step recomputed from this run's recorded drifts and prompt SNR, its scores read from EXP-031's stored
  W4, W5, W6 and W8 rows (same cases; W8 and the drifts are checked to match). Set beside "SNR gate, then CPH8".
  Variety is not available for the policy; W4 and W8 bound it (EXP-032).
- Time: seconds per forward pass from the timing stage (least squares through the arms' medians).

Usage: python code/analysis/exp033_diagnostics.py [results dir]
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "colab"))
import run_cheaper_teacher_path as ct  # noqa: E402
import run_method_comparison as mc  # noqa: E402
import run_variety_per_pass as vp  # noqa: E402

EXP = ROOT / "research/experiments/EXP-033-cheaper-teacher-path"
RES = EXP / "results"
EXP031_QUAL = ROOT / "research/experiments/EXP-031-snr-toh-method-comparison/results/quality.csv"
VARIETY = ("cvf0", "cvf0_aligned")
METRICS = ("utmos", "wer", "similarity")
ANCHOR = "W8R2"
MATCH_ARMS = ("W8R2", "CPH12", "CPH8", "CPH6", "GI")


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def num(r, key):
    try:
        return float(r[key])
    except (KeyError, ValueError):
        return float("nan")


def boot(vals):
    vals = [v for v in vals if v == v]
    out = vp._boot(vals)
    out["positive"] = sum(v > 0 for v in vals)
    return out


def share(rows, arm, measure):
    s = vp.share_ci(rows, arm, measure, ref=ct.REFERENCE, base=ct.BASE)
    return s["point"]


def policy(rows, pick, scores):
    """Mean passes and quality of a per-case choice: pick(row) names the arm, scores(row, arm, metric) reads it."""
    out = {"mean_passes": st.mean(pick(r)[1] for r in rows)}
    for cond in ct.CONDS:
        sub = [r for r in rows if r["condition"] == cond]
        out[cond] = {"passes": st.mean(pick(r)[1] for r in sub),
                     **{m: st.mean(scores(r, pick(r)[0], m) for r in sub) for m in METRICS},
                     "choices": {a: sum(pick(r)[0] == a for r in sub) for a in sorted({pick(r)[0] for r in sub})}}
    return out


def main(res_dir=None):
    res = Path(res_dir) if res_dir else RES
    qual, div, timing = read(res / "quality.csv"), read(res / "diversity.csv"), read(res / "timing.csv")
    clean = [r for r in qual if r["condition"] == "clean"]
    noisy = [r for r in qual if r["condition"] == "noisy15"]
    out = {"experiment": "EXP-033", "kind": "post hoc descriptive diagnostics (not preregistered)"}

    # each arm against W8R2
    out["minus_W8R2"] = {
        a: {"passes": ct.COST[a],
            "variety": {m: boot(num(r, f"{m}_{a}") - num(r, f"{m}_{ANCHOR}") for r in div)
                        for m in VARIETY + ("timing_spread", "melody")},
            "quality_clean": {m: boot(num(r, f"{m}_{a}") - num(r, f"{m}_{ANCHOR}") for r in clean) for m in METRICS}}
        for a in ct.ARMS if a != ANCHOR}

    # direction per speaker against P25
    out["speakers_at_or_above_P25"] = {a: {m: sum(num(r, f"{m}_{a}") >= num(r, f"{m}_P25") for r in div)
                                           for m in VARIETY} for a in ct.ARMS if a != "P25"}

    # fragility of the shares
    top2 = {r["utt_id"] for r in sorted(div, key=lambda r: -num(r, "cvf0_P25"))[:2]}
    frag = {"top2_speakers_by_P25_cvf0": sorted(top2)}
    for a in ct.TEST_ARMS:
        e = {}
        for m in VARIETY:
            loo = [share([r for r in div if r is not skip], a, m) for skip in div]
            loo = [x for x in loo if x is not None]
            e[m] = {"all": share(div, a, m), "leave_one_out_min": min(loo), "leave_one_out_max": max(loo),
                    "leave_one_out_at_or_above_90": sum(x >= ct.SHARE_BAR for x in loo),
                    "without_top2": share([r for r in div if r["utt_id"] not in top2], a, m)}
        frag[a] = e
    out["share_fragility"] = frag

    # the grid's coarseness
    grid = {}
    for a in ("W8R2", "CPH12", "CPH8", "CPH6"):
        grid[a] = {"passes": ct.COST[a], "teacher_steps_to_commit": {"W8R2": 8, "CPH12": 6, "CPH8": 4, "CPH6": 3}[a],
                   "est_dist": 0.0 if a == "W8R2" else st.mean(num(r, f"est_dist_{a}") for r in div),
                   "melody": st.mean(num(r, f"melody_{a}") for r in div),
                   "cvf0": st.mean(num(r, f"cvf0_{a}") for r in div),
                   "cvf0_aligned": st.mean(num(r, f"cvf0_aligned_{a}") for r in div),
                   "timing_spread": st.mean(num(r, f"timing_spread_{a}") for r in div),
                   "utmos": st.mean(num(r, f"utmos_{a}") for r in clean),
                   "wer": st.mean(num(r, f"wer_{a}") for r in clean),
                   "similarity": st.mean(num(r, f"similarity_{a}") for r in clean)}
    out["grid_coarseness"] = grid
    out["CPH12_minus_CPH8"] = {
        "variety": {m: boot(num(r, f"{m}_CPH12") - num(r, f"{m}_CPH8") for r in div)
                    for m in VARIETY + ("timing_spread", "melody")},
        "quality_clean": {m: boot(num(r, f"{m}_CPH12") - num(r, f"{m}_CPH8") for r in clean) for m in METRICS}}

    # CPH8's clean WER against P25
    out["CPH8_clean_wer_cases"] = [
        {"utt_id": r["utt_id"], "wer_CPH8": num(r, "wer_CPH8"), "wer_P25": num(r, "wer_P25"),
         "hypothesis_CPH8": r["hypothesis_CPH8"], "hypothesis_P25": r["hypothesis_P25"]}
        for r in clean if num(r, "wer_CPH8") != num(r, "wer_P25")]

    # noisy15
    out["noisy15"] = {
        "runaway": {a: sum(num(r, f"wer_{a}") > ct.RUNAWAY_WER for r in noisy) for a in ct.ARMS},
        "wer_median": {a: st.median(num(r, f"wer_{a}") for r in noisy) for a in ct.ARMS},
        "SST_minus_STU": {m: boot(num(r, f"{m}_SST") - num(r, f"{m}_STU") for r in noisy) for m in METRICS},
        "SST_minus_W8R2": {m: boot(num(r, f"{m}_SST") - num(r, f"{m}_W8R2") for r in noisy) for m in METRICS}}

    # Chapter 3's SNR + drift policy on the same cases, scores from EXP-031's stored rows
    s31 = {(r["utt_id"], r["condition"]): r for r in read(EXP031_QUAL)}
    shared = [r for r in qual if (r["utt_id"], r["condition"]) in s31]
    join = {"cases": len(shared),
            "max_abs_d4_d5_d6": max(abs(num(r, f"d_{k}") - num(s31[(r["utt_id"], r["condition"])], f"d_{k}"))
                                    for r in shared for k in (4, 5, 6)),
            "max_abs_W8": {m: max(abs(num(r, f"{m}_W8") - num(s31[(r["utt_id"], r["condition"])], f"{m}_W8"))
                                  for r in shared) for m in METRICS}}

    def ch3(lam):
        def pick(r):
            k = mc.ch3_step({j: num(r, f"d_{j}") for j in mc.CH3_CANDIDATES}, num(r, "prompt_snr"), lam)
            return f"W{k}", 2 * k + 5
        return pick

    def routed(arm):
        def pick(r):
            return (ct.BASE, ct.COST[ct.BASE]) if num(r, "prompt_snr") < ct.THETA else (arm, ct.COST[arm])
        return pick

    stored = lambda r, a, m: num(s31[(r["utt_id"], r["condition"])], f"{m}_{a}")   # noqa: E731
    fresh = lambda r, a, m: num(r, f"{m}_{a}")                                      # noqa: E731
    out["policies_on_the_60_cases"] = {
        "join_check": join,
        "CH3_lambda_0.35": policy(shared, ch3(mc.CH3_LAMBDA), stored),
        "CH3_LTT_lambda_1.20": policy(shared, ch3(mc.CH3_LAMBDA_LTT), stored),
        "fixed_W8": policy(shared, lambda r: ("W8", ct.COST["W8"]), fresh),
        "fixed_P25": policy(shared, lambda r: ("P25", ct.COST["P25"]), fresh),
        "SNR_then_W8R2": policy(shared, routed("W8R2"), fresh),
        "SNR_then_CPH8": policy(shared, routed("CPH8"), fresh),
        "SNR_then_GI": policy(shared, routed("GI"), fresh)}

    # time per forward pass
    med = {a: st.median(num(r, f"seconds_{a}") for r in timing) for a in ct.ARMS}
    xs, ys = [ct.COST[a] for a in ct.ARMS], [med[a] for a in ct.ARMS]
    slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    out["timing"] = {"cases": len(timing), "median_seconds": med, "seconds_per_pass_through_origin": slope,
                     "largest_relative_misfit": max(abs(y - slope * x) / y for x, y in zip(xs, ys)),
                     "saving_vs_P25_seconds": {a: 1.0 - med[a] / med["P25"] for a in ct.ARMS}}

    path = res / "diagnostics.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(out, indent=2) + "\n")
    print(f"[OK] wrote {path}")
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
